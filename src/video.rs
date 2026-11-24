//! Restoring a video: decode with ffmpeg (deinterlace, denoise, square pixels),
//! improve and enlarge every frame with the engine, encode with ffmpeg.
//!
//! The work is cut into **chunks** of a few seconds. Each chunk is decoded,
//! processed and encoded on its own into `<output>.work/chunk_NNNNN.mkv`, and a
//! `state.json` next to them lists the chunks that are complete. Stopping
//! (Ctrl-C) discards only the chunk in progress; running the same command again
//! carries on with the first chunk that is not complete. When every chunk is
//! done they are joined without re-encoding and the original audio is added.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Instant, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use clap::ValueEnum;
use indicatif::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};

use crate::engine::Engine;
use crate::ffmpeg::{Tools, VideoInfo};

/// Set by Ctrl-C (or in tests): the chunk in progress is dropped and the run ends.
pub static STOP: AtomicBool = AtomicBool::new(false);

/// The run was stopped; `chunks_done` chunks are kept for the next run.
#[derive(Debug)]
pub struct Stopped {
    pub chunks_done: usize,
    pub chunks_total: usize,
}

impl std::fmt::Display for Stopped {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "stopped with {} of {} chunks done",
            self.chunks_done, self.chunks_total
        )
    }
}

impl std::error::Error for Stopped {}

/// How much noise to remove before enlarging (ffmpeg `hqdn3d`: spatial and
/// temporal, which is what tape and camcorder noise calls for).
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum Denoise {
    Off,
    Light,
    Medium,
    Strong,
}

impl Denoise {
    fn filter(self) -> Option<&'static str> {
        match self {
            Denoise::Off => None,
            Denoise::Light => Some("hqdn3d=1.5:1.5:4:4"),
            Denoise::Medium => Some("hqdn3d=4:3:8:6"),
            Denoise::Strong => Some("hqdn3d=7:6:12:10"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum Deinterlace {
    /// When the file says its fields are interlaced.
    Auto,
    On,
    Off,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum Codec {
    H265,
    H264,
    Av1,
    Prores,
}

#[derive(Clone, Debug)]
pub struct Options {
    pub input: PathBuf,
    pub output: PathBuf,
    pub tools: Tools,
    pub denoise: Denoise,
    pub deinterlace: Deinterlace,
    /// Keep both fields of interlaced video (twice the frame rate).
    pub double_rate: bool,
    pub codec: Codec,
    pub crf: Option<u32>,
    pub preset: Option<String>,
    pub chunk_secs: f64,
    /// Ignore and delete an unfinished earlier run.
    pub restart: bool,
    pub keep_work: bool,
    /// For tests: stop (as if by Ctrl-C) after this many chunks of this run.
    pub stop_after_chunks: Option<usize>,
}

#[derive(Debug)]
pub struct Summary {
    pub frames: u64,
    pub chunks: usize,
    /// Chunks found complete from an earlier run.
    pub resumed: usize,
    pub seconds: f64,
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct State {
    version: u32,
    input_size: u64,
    input_mtime: u64,
    settings: String,
    chunks: usize,
    done: Vec<usize>,
}

fn work_dir(output: &Path) -> PathBuf {
    let mut name = output.file_name().unwrap_or_default().to_os_string();
    name.push(".work");
    output.with_file_name(name)
}

fn chunk_name(i: usize) -> String {
    format!("chunk_{i:05}.mkv")
}

fn ext(p: &Path) -> String {
    p.extension()
        .map(|e| e.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default()
}

impl Options {
    fn interlaced(&self, info: &VideoInfo) -> bool {
        match self.deinterlace {
            Deinterlace::On => true,
            Deinterlace::Off => false,
            Deinterlace::Auto => info.interlaced,
        }
    }

    /// The size of the frames the engine receives: square pixels, even sides.
    fn frame_size(info: &VideoInfo) -> (usize, usize) {
        let (w, h) = info.square_size();
        ((w & !1) as usize, (h & !1) as usize)
    }

    /// The frame rate of the result.
    fn out_fps(&self, info: &VideoInfo) -> (u32, u32) {
        let mul = if self.interlaced(info) && self.double_rate {
            2
        } else {
            1
        };
        (info.fps_num * mul, info.fps_den)
    }

    /// ffmpeg's `-vf` for decoding: deinterlace first (everything after wants
    /// whole frames), then denoise, then square pixels, then raw RGB.
    pub fn decode_filters(&self, info: &VideoInfo) -> String {
        let mut f: Vec<String> = Vec::new();
        if self.interlaced(info) {
            let mode = if self.double_rate {
                "send_field"
            } else {
                "send_frame"
            };
            f.push(format!("bwdif=mode={mode}:parity=auto:deint=all"));
        }
        if let Some(d) = self.denoise.filter() {
            f.push(d.to_string());
        }
        let (sw, sh) = info.square_size();
        if (sw, sh) != (info.width, info.height) {
            f.push(format!("scale={sw}:{sh}:flags=lanczos"));
        }
        let (ew, eh) = Self::frame_size(info);
        if (ew as u32, eh as u32) != (sw, sh) {
            f.push(format!("crop={ew}:{eh}:0:0"));
        }
        f.push("setsar=1".into());
        f.push("format=rgb24".into());
        f.join(",")
    }

    fn codec_args(&self) -> Vec<String> {
        let s = |v: &str| v.to_string();
        match self.codec {
            Codec::H264 => vec![
                s("-c:v"),
                s("libx264"),
                s("-preset"),
                self.preset.clone().unwrap_or_else(|| "slow".into()),
                s("-crf"),
                self.crf.unwrap_or(16).to_string(),
                s("-pix_fmt"),
                s("yuv420p"),
            ],
            Codec::H265 => vec![
                s("-c:v"),
                s("libx265"),
                s("-preset"),
                self.preset.clone().unwrap_or_else(|| "medium".into()),
                s("-crf"),
                self.crf.unwrap_or(18).to_string(),
                s("-pix_fmt"),
                s("yuv420p10le"),
                s("-x265-params"),
                s("log-level=error"),
            ],
            Codec::Av1 => vec![
                s("-c:v"),
                s("libsvtav1"),
                s("-preset"),
                self.preset.clone().unwrap_or_else(|| "6".into()),
                s("-crf"),
                self.crf.unwrap_or(28).to_string(),
                s("-pix_fmt"),
                s("yuv420p10le"),
            ],
            Codec::Prores => vec![
                s("-c:v"),
                s("prores_ks"),
                s("-profile:v"),
                s("3"),
                s("-pix_fmt"),
                s("yuv422p10le"),
            ],
        }
    }

    /// Everything that must be the same for chunks of two runs to belong together.
    fn settings_key(&self, info: &VideoInfo, engine: &dyn Engine) -> String {
        format!(
            "v1|{}|{:?}|{:?}|{}|{:?}|{:?}|{:?}|{}|{}",
            self.input.display(),
            self.denoise,
            self.deinterlace,
            self.double_rate,
            self.codec,
            self.crf,
            self.preset,
            self.chunk_secs,
            format_args!("{} {}x{}", engine.describe(), info.width, info.height),
        )
    }
}

fn file_stamp(p: &Path) -> Result<(u64, u64)> {
    let m = std::fs::metadata(p).with_context(|| format!("cannot read {}", p.display()))?;
    let t = m
        .modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_secs());
    Ok((m.len(), t))
}

fn save_state(work: &Path, state: &State) -> Result<()> {
    let tmp = work.join("state.json.tmp");
    std::fs::write(&tmp, serde_json::to_vec_pretty(state)?)?;
    std::fs::rename(&tmp, work.join("state.json"))?;
    Ok(())
}

/// Reads one raw frame; false at a clean end of the stream.
fn read_frame(r: &mut impl Read, buf: &mut [u8]) -> Result<bool> {
    let mut n = 0;
    while n < buf.len() {
        let k = r.read(&mut buf[n..])?;
        if k == 0 {
            if n == 0 {
                return Ok(false);
            }
            bail!("the decoder stopped in the middle of a frame");
        }
        n += k;
    }
    Ok(true)
}

fn kill(children: &mut [&mut Child]) {
    for c in children {
        let _ = c.kill();
        let _ = c.wait();
    }
}

/// Decodes, improves and encodes one chunk into `part`; returns its frame count.
#[allow(clippy::too_many_arguments)]
fn run_chunk(
    o: &Options,
    info: &VideoInfo,
    engine: &mut dyn Engine,
    start: f64,
    dur: f64,
    part: &Path,
    bar: &ProgressBar,
) -> Result<u64> {
    let (w, h) = Options::frame_size(info);
    let s = engine.scale();
    let (ow, oh) = (w * s, h * s);
    let (fn_, fd) = o.out_fps(info);
    let mut dec = Command::new(&o.tools.ffmpeg)
        .args([
            "-v",
            "error",
            "-nostdin",
            "-ss",
            &format!("{start:.6}"),
            "-t",
            &format!("{dur:.6}"),
        ])
        .arg("-i")
        .arg(&o.input)
        .args(["-an", "-sn", "-vf", &o.decode_filters(info)])
        .args(["-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .context("cannot start the ffmpeg decoder")?;
    let mut enc = Command::new(&o.tools.ffmpeg)
        .args(["-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24"])
        .args([
            "-s",
            &format!("{ow}x{oh}"),
            "-r",
            &format!("{fn_}/{fd}"),
            "-i",
            "-",
        ])
        .args(o.codec_args())
        .args(["-an", "-f", "matroska"])
        .arg(part)
        .stdin(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .context("cannot start the ffmpeg encoder")?;

    let mut out = dec.stdout.take().context("no decoder output")?;
    let mut sink = enc.stdin.take().context("no encoder input")?;
    let mut frame = vec![0u8; w * h * 3];
    let mut frames = 0u64;
    loop {
        if STOP.load(Ordering::Relaxed) {
            kill(&mut [&mut dec, &mut enc]);
            let _ = std::fs::remove_file(part);
            return Err(Stopped {
                chunks_done: 0,
                chunks_total: 0,
            }
            .into());
        }
        match read_frame(&mut out, &mut frame) {
            Ok(true) => {}
            Ok(false) => break,
            Err(e) => {
                kill(&mut [&mut dec, &mut enc]);
                return Err(e);
            }
        }
        let better = match engine.process(&frame, w, h) {
            Ok(b) => b,
            Err(e) => {
                kill(&mut [&mut dec, &mut enc]);
                return Err(e);
            }
        };
        if sink.write_all(&better).is_err() {
            // The encoder died: its own message is on screen.
            break;
        }
        frames += 1;
        bar.inc(1);
    }
    drop(sink);
    let d = dec.wait()?;
    let e = enc.wait()?;
    if !d.success() {
        bail!("the ffmpeg decoder failed ({d})");
    }
    if !e.success() {
        bail!("the ffmpeg encoder failed ({e})");
    }
    Ok(frames)
}

/// Joins the chunks without re-encoding and adds the original audio.
fn assemble(o: &Options, info: &VideoInfo, work: &Path, chunks: usize) -> Result<()> {
    let list = work.join("list.txt");
    let mut text = String::new();
    for i in 0..chunks {
        text.push_str(&format!("file '{}'\n", chunk_name(i)));
    }
    std::fs::write(&list, text)?;
    let container = ext(&o.output);
    let mp4ish = matches!(container.as_str(), "mp4" | "m4v" | "mov");
    let mut cmd = Command::new(&o.tools.ffmpeg);
    cmd.args(["-v", "error", "-y", "-f", "concat", "-safe", "0", "-i"])
        .arg(&list)
        .arg("-i")
        .arg(&o.input)
        .args(["-map", "0:v:0", "-map", "1:a?", "-c:v", "copy"]);
    if mp4ish && o.codec == Codec::H265 {
        cmd.args(["-tag:v", "hvc1"]);
    }
    if info.has_audio {
        if mp4ish {
            cmd.args(["-c:a", "aac", "-b:a", "192k"]);
        } else {
            cmd.args(["-c:a", "copy"]);
        }
    }
    if mp4ish {
        cmd.args(["-movflags", "+faststart"]);
    }
    let status = cmd
        .arg(&o.output)
        .stderr(Stdio::inherit())
        .status()
        .context("cannot start ffmpeg to join the chunks")?;
    if !status.success() {
        bail!("ffmpeg could not join the chunks ({status})");
    }
    Ok(())
}

/// Restores `o.input` into `o.output` with `engine`, resuming an unfinished run.
pub fn run(o: &Options, engine: &mut dyn Engine, info: &VideoInfo) -> Result<Summary> {
    let started = Instant::now();
    if o.chunk_secs < 1.0 {
        bail!("--chunk-seconds must be at least 1");
    }
    if o.input == o.output {
        bail!("the output must not be the input");
    }
    let chunks = ((info.duration / o.chunk_secs).ceil() as usize).max(1);
    let work = work_dir(&o.output);
    let (size, mtime) = file_stamp(&o.input)?;
    let settings = o.settings_key(info, engine);

    if o.restart && work.exists() {
        std::fs::remove_dir_all(&work)?;
    }
    std::fs::create_dir_all(&work)?;
    let state_path = work.join("state.json");
    let mut state = match std::fs::read(&state_path) {
        Ok(b) => {
            let s: State = serde_json::from_slice(&b)
                .context("the unfinished run's state is unreadable; use --restart")?;
            if s.settings != settings
                || s.input_size != size
                || s.input_mtime != mtime
                || s.chunks != chunks
            {
                bail!(
                    "{} holds an unfinished run with other settings or another input; \
                     run with --restart to throw it away and start over",
                    work.display()
                );
            }
            s
        }
        Err(_) => State {
            version: 1,
            input_size: size,
            input_mtime: mtime,
            settings,
            chunks,
            done: Vec::new(),
        },
    };
    // A chunk counts only if its file is there.
    state.done.retain(|&i| work.join(chunk_name(i)).exists());
    let resumed = state.done.len();
    save_state(&work, &state)?;

    let fps = o.out_fps(info).0 as f64 / o.out_fps(info).1 as f64;
    let total_frames = (info.duration * fps).round() as u64;
    let bar = ProgressBar::new(total_frames);
    bar.set_style(
        ProgressStyle::with_template(
            "{bar:40.cyan/blue} {pos}/{len} frames  {per_sec}  eta {eta}  {msg}",
        )
        .unwrap_or_else(|_| ProgressStyle::default_bar()),
    );
    let per_chunk = (o.chunk_secs * fps).round() as u64;
    bar.set_position(resumed as u64 * per_chunk);
    if resumed > 0 {
        eprintln!("Carrying on: {resumed} of {chunks} chunks were already done.");
    }

    let mut frames = 0u64;
    let mut this_run = 0usize;
    for i in 0..chunks {
        if state.done.contains(&i) {
            continue;
        }
        if STOP.load(Ordering::Relaxed) {
            bar.abandon();
            return Err(Stopped {
                chunks_done: state.done.len(),
                chunks_total: chunks,
            }
            .into());
        }
        let start = i as f64 * o.chunk_secs;
        let dur = (info.duration - start).min(o.chunk_secs);
        bar.set_message(format!("chunk {}/{chunks}", i + 1));
        let part = work.join(format!("chunk_{i:05}.part.mkv"));
        match run_chunk(o, info, engine, start, dur, &part, &bar) {
            Ok(n) => frames += n,
            Err(e) if e.downcast_ref::<Stopped>().is_some() => {
                bar.abandon();
                return Err(Stopped {
                    chunks_done: state.done.len(),
                    chunks_total: chunks,
                }
                .into());
            }
            Err(e) => {
                let _ = std::fs::remove_file(&part);
                return Err(e);
            }
        }
        std::fs::rename(&part, work.join(chunk_name(i)))?;
        state.done.push(i);
        save_state(&work, &state)?;
        this_run += 1;
        if o.stop_after_chunks == Some(this_run) {
            STOP.store(true, Ordering::Relaxed);
        }
    }
    bar.finish_with_message("joining");
    assemble(o, info, &work, chunks)?;
    if !o.keep_work {
        let _ = std::fs::remove_dir_all(&work);
    }
    Ok(Summary {
        frames,
        chunks,
        resumed,
        seconds: started.elapsed().as_secs_f64(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts() -> Options {
        Options {
            input: "in.avi".into(),
            output: "out.mp4".into(),
            tools: Tools {
                ffmpeg: "ffmpeg".into(),
                ffprobe: "ffprobe".into(),
            },
            denoise: Denoise::Medium,
            deinterlace: Deinterlace::Auto,
            double_rate: false,
            codec: Codec::H265,
            crf: None,
            preset: None,
            chunk_secs: 20.0,
            restart: false,
            keep_work: false,
            stop_after_chunks: None,
        }
    }

    fn dv() -> VideoInfo {
        VideoInfo {
            width: 720,
            height: 480,
            fps_num: 30000,
            fps_den: 1001,
            duration: 60.0,
            interlaced: true,
            sar: (8, 9),
            has_audio: true,
            frames: None,
        }
    }

    #[test]
    fn a_dv_tape_is_deinterlaced_denoised_and_made_square_in_that_order() {
        let f = opts().decode_filters(&dv());
        assert_eq!(
            f,
            "bwdif=mode=send_frame:parity=auto:deint=all,hqdn3d=4:3:8:6,\
             scale=640:480:flags=lanczos,setsar=1,format=rgb24"
        );
    }

    #[test]
    fn options_change_the_chain() {
        let mut o = opts();
        o.double_rate = true;
        assert!(o.decode_filters(&dv()).starts_with("bwdif=mode=send_field"));
        assert_eq!(o.out_fps(&dv()), (60000, 1001));
        o.deinterlace = Deinterlace::Off;
        o.denoise = Denoise::Off;
        let mut v = dv();
        v.sar = (1, 1);
        v.width = 161;
        v.height = 99;
        // Odd sides are cropped to even for the encoder.
        assert_eq!(
            o.decode_filters(&v),
            "crop=160:98:0:0,setsar=1,format=rgb24"
        );
        assert_eq!(o.out_fps(&v), (30000, 1001));
        assert_eq!(Options::frame_size(&v), (160, 98));
    }

    #[test]
    fn state_round_trips_and_work_dir_sits_next_to_the_output() {
        assert_eq!(
            work_dir(Path::new("/a/b/out.mp4")),
            Path::new("/a/b/out.mp4.work")
        );
        assert_eq!(chunk_name(7), "chunk_00007.mkv");
        let s = State {
            version: 1,
            input_size: 5,
            input_mtime: 6,
            settings: "x".into(),
            chunks: 3,
            done: vec![0, 1],
        };
        let j = serde_json::to_string(&s).unwrap();
        assert_eq!(serde_json::from_str::<State>(&j).unwrap(), s);
    }

    #[test]
    fn frames_are_read_whole_or_the_stream_ends_cleanly() {
        let data = [1u8; 10];
        let mut buf = [0u8; 5];
        let mut r = &data[..];
        assert!(read_frame(&mut r, &mut buf).unwrap());
        assert!(read_frame(&mut r, &mut buf).unwrap());
        assert!(!read_frame(&mut r, &mut buf).unwrap());
        let mut half = &data[..7];
        assert!(read_frame(&mut half, &mut buf).unwrap());
        assert!(read_frame(&mut half, &mut buf).is_err());
    }
}
