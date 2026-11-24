//! ffmpeg and ffprobe as helper programs: probing a video and its frames,
//! and the command lines that decode and encode it. Frames travel through
//! pipes as raw RGB, so no picture files are written (an hour of DV is about
//! 100,000 frames).

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};
use serde_json::Value;

/// Where ffmpeg and ffprobe are.
#[derive(Clone, Debug)]
pub struct Tools {
    pub ffmpeg: PathBuf,
    pub ffprobe: PathBuf,
}

impl Tools {
    /// `ffmpeg` as given, or the one on the search path; `ffprobe` beside it
    /// when it is there, else on the search path.
    pub fn locate(ffmpeg: Option<&Path>) -> Result<Tools> {
        let ffmpeg = ffmpeg.map_or_else(|| PathBuf::from("ffmpeg"), Path::to_path_buf);
        let sibling = ffmpeg
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .map(|p| {
                p.join(if cfg!(windows) {
                    "ffprobe.exe"
                } else {
                    "ffprobe"
                })
            })
            .filter(|p| p.exists());
        let tools = Tools {
            ffprobe: sibling.unwrap_or_else(|| PathBuf::from("ffprobe")),
            ffmpeg,
        };
        for (name, p) in [("ffmpeg", &tools.ffmpeg), ("ffprobe", &tools.ffprobe)] {
            Command::new(p)
                .arg("-version")
                .output()
                .with_context(|| {
                    format!(
                        "cannot run {name} ({}): install FFmpeg (https://ffmpeg.org) or pass --ffmpeg PATH",
                        p.display()
                    )
                })?;
        }
        Ok(tools)
    }
}

/// What the first video stream and the file say about it.
#[derive(Clone, Debug, PartialEq)]
pub struct VideoInfo {
    pub width: u32,
    pub height: u32,
    pub fps_num: u32,
    pub fps_den: u32,
    /// Seconds.
    pub duration: f64,
    /// Tape-style interlaced fields (top or bottom field first).
    pub interlaced: bool,
    /// Pixel shape (DV is not square: 720x480 is shown at 4:3 or 16:9).
    pub sar: (u32, u32),
    pub has_audio: bool,
    pub frames: Option<u64>,
}

impl VideoInfo {
    pub fn fps(&self) -> f64 {
        f64::from(self.fps_num) / f64::from(self.fps_den.max(1))
    }

    /// The size of a frame once its pixels are made square, rounded to even.
    pub fn square_size(&self) -> (u32, u32) {
        let (n, d) = self.sar;
        if n == d || n == 0 || d == 0 {
            return (self.width, self.height);
        }
        let w = (f64::from(self.width) * f64::from(n) / f64::from(d) / 2.0).trunc() as u32 * 2;
        (w.max(2), self.height)
    }
}

fn ratio(s: &str) -> Option<(u32, u32)> {
    let (a, b) = s.split_once([':', '/'])?;
    Some((a.trim().parse().ok()?, b.trim().parse().ok()?))
}

/// Limits on what a video may declare about itself. The numbers come from the
/// file, which may be hostile or corrupt; they size memory and loop counts, so
/// they are bounded here, before anything is allocated from them.
pub const MAX_SIDE: u32 = 16_384;
pub const MAX_PIXELS: u64 = 70_000_000;
pub const MAX_SECONDS: f64 = 7.0 * 86_400.0;

fn checked(info: VideoInfo) -> Result<VideoInfo> {
    let (w, h) = (info.width, info.height);
    if !(16..=MAX_SIDE).contains(&w) || !(16..=MAX_SIDE).contains(&h) {
        bail!(
            "the video declares an unusable size ({w}x{h}); 16 to {MAX_SIDE} pixels a side are supported"
        );
    }
    if u64::from(w) * u64::from(h) > MAX_PIXELS {
        bail!("the video is too large ({w}x{h}); at most {MAX_PIXELS} pixels per frame");
    }
    let fps = info.fps();
    if !(0.1..=1000.0).contains(&fps) {
        bail!("the video declares an unusable frame rate ({fps:.3})");
    }
    if !info.duration.is_finite() || info.duration <= 0.0 || info.duration > MAX_SECONDS {
        bail!(
            "the video declares an unusable length ({} s)",
            info.duration
        );
    }
    let (n, d) = info.sar;
    if n > 10_000 || d > 10_000 || !(0.1..=10.0).contains(&(f64::from(n) / f64::from(d.max(1)))) {
        bail!("the video declares an unusable pixel shape ({n}:{d})");
    }
    let (sw, sh) = info.square_size();
    if sw > MAX_SIDE || sh > MAX_SIDE || u64::from(sw) * u64::from(sh) > MAX_PIXELS {
        bail!("the video is too large once its pixels are made square ({sw}x{sh})");
    }
    Ok(info)
}

/// Reads ffprobe's JSON (`-show_streams -show_format`).
pub fn parse_probe(json: &str) -> Result<VideoInfo> {
    let v: Value = serde_json::from_str(json).context("ffprobe gave no JSON")?;
    let streams = v["streams"].as_array().context("no streams in the file")?;
    let video = streams
        .iter()
        .find(|s| s["codec_type"] == "video")
        .context("the file has no video stream")?;
    let side = |k: &str| -> Result<u32> {
        u32::try_from(video[k].as_u64().with_context(|| format!("no {k}"))?)
            .map_err(|_| anyhow::anyhow!("the video declares an impossible {k}"))
    };
    let (width, height) = (side("width")?, side("height")?);
    let (fps_num, fps_den) = video["avg_frame_rate"]
        .as_str()
        .and_then(ratio)
        .filter(|&(n, d)| n > 0 && d > 0)
        .or_else(|| video["r_frame_rate"].as_str().and_then(ratio))
        .filter(|&(n, d)| n > 0 && d > 0)
        .context("no frame rate")?;
    let num = |x: &Value| x.as_str().and_then(|s| s.parse::<f64>().ok());
    let duration = num(&video["duration"])
        .or_else(|| num(&v["format"]["duration"]))
        .context("no duration")?;
    let interlaced = matches!(
        video["field_order"].as_str(),
        Some("tt" | "bb" | "tb" | "bt")
    );
    let sar = video["sample_aspect_ratio"]
        .as_str()
        .and_then(ratio)
        .filter(|&(n, d)| n > 0 && d > 0)
        .unwrap_or((1, 1));
    checked(VideoInfo {
        width,
        height,
        fps_num,
        fps_den,
        duration,
        interlaced,
        sar,
        has_audio: streams.iter().any(|s| s["codec_type"] == "audio"),
        frames: video["nb_frames"].as_str().and_then(|s| s.parse().ok()),
    })
}

pub fn probe(tools: &Tools, input: &Path) -> Result<VideoInfo> {
    let out = Command::new(&tools.ffprobe)
        .args([
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
        ])
        .arg(input)
        .output()
        .context("cannot run ffprobe")?;
    if !out.status.success() {
        bail!(
            "ffprobe could not read {}: {}",
            input.display(),
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    let json = String::from_utf8_lossy(&out.stdout);
    let mut info = parse_probe(&json)?;
    // Captures often leave the header silent about fields, even though every
    // frame is interlaced; ask the frames, then look at the pictures.
    if !field_order_declared(&json) {
        info.interlaced = detect_interlace(tools, input);
    }
    Ok(info)
}

/// True when the video's header says what its field order is (or that it has none).
fn field_order_declared(json: &str) -> bool {
    serde_json::from_str::<Value>(json)
        .ok()
        .and_then(|v| {
            v["streams"]
                .as_array()?
                .iter()
                .find(|s| s["codec_type"] == "video")?["field_order"]
                .as_str()
                .map(|f| matches!(f, "tt" | "bb" | "tb" | "bt" | "progressive"))
        })
        .unwrap_or(false)
}

/// Most of the first frames flagged as interlaced by the decoder.
fn mostly_flagged(csv: &str) -> bool {
    let flags: Vec<bool> = csv
        .lines()
        .filter_map(|l| match l.split(',').next()?.trim() {
            "1" => Some(true),
            "0" => Some(false),
            _ => None,
        })
        .collect();
    !flags.is_empty() && flags.iter().filter(|&&f| f).count() * 2 > flags.len()
}

/// Reads ffmpeg's `idet` summary: interlaced when combed frames clearly
/// outnumber progressive ones **and one field order dominates** (real tape has
/// one; fine moving detail in progressive video fools `idet` both ways at once).
/// Static scenes are "undetermined" and not counted.
fn combed(log: &str) -> bool {
    let count = |line: &str, key: &str| -> u64 {
        line.split_once(key)
            .and_then(|(_, rest)| rest.split_whitespace().next()?.parse().ok())
            .unwrap_or(0)
    };
    log.lines()
        .filter(|l| l.contains("Multi frame detection:"))
        .map(|l| {
            let (t, b) = (count(l, "TFF:"), count(l, "BFF:"));
            (t.max(b), t.min(b), count(l, "Progressive:"))
        })
        .max_by_key(|&(a, b, p)| a + b + p)
        .is_some_and(|(main, other, progressive)| {
            main >= 10 && main >= 4 * other && main > 2 * progressive
        })
}

fn detect_interlace(tools: &Tools, input: &Path) -> bool {
    let flags = Command::new(&tools.ffprobe)
        .args(["-v", "error", "-select_streams", "v:0"])
        .args(["-show_entries", "frame=interlaced_frame", "-of", "csv=p=0"])
        .args(["-read_intervals", "%+#60"])
        .arg(input)
        .output();
    if flags.is_ok_and(|o| mostly_flagged(&String::from_utf8_lossy(&o.stdout))) {
        return true;
    }
    Command::new(&tools.ffmpeg)
        .args(["-nostdin", "-v", "info", "-i"])
        .arg(input)
        .args([
            "-an",
            "-sn",
            "-vf",
            "idet",
            "-frames:v",
            "300",
            "-f",
            "null",
            "-",
        ])
        .output()
        .is_ok_and(|o| combed(&String::from_utf8_lossy(&o.stderr)))
}

#[cfg(test)]
mod tests {
    use super::*;

    const DV: &str = r#"{"streams":[
      {"codec_type":"video","width":720,"height":480,"avg_frame_rate":"30000/1001",
       "r_frame_rate":"30000/1001","field_order":"bb","sample_aspect_ratio":"8:9",
       "duration":"61.061","nb_frames":"1830"},
      {"codec_type":"audio"}],"format":{"duration":"61.1"}}"#;

    fn with(width: &str, height: &str, fps: &str, dur: &str, sar: &str) -> String {
        format!(
            r#"{{"streams":[{{"codec_type":"video","width":{width},"height":{height},
            "avg_frame_rate":"{fps}","sample_aspect_ratio":"{sar}","duration":"{dur}"}}],
            "format":{{}}}}"#
        )
    }

    #[test]
    fn hostile_or_corrupt_numbers_are_refused_before_they_size_anything() {
        assert!(parse_probe(&with("1920", "1080", "25/1", "60", "1:1")).is_ok());
        for (w, h, f, d, sar, why) in [
            (
                "4294968016",
                "480",
                "25/1",
                "10",
                "1:1",
                "width that wraps to 720 in 32 bits",
            ),
            ("100000", "100000", "25/1", "10", "1:1", "huge frame"),
            ("0", "480", "25/1", "10", "1:1", "zero width"),
            (
                "720",
                "480",
                "4000000000/1",
                "10",
                "1:1",
                "absurd frame rate",
            ),
            (
                "720",
                "480",
                "1/4000000000",
                "10",
                "1:1",
                "frame rate near zero",
            ),
            ("720", "480", "25/1", "inf", "1:1", "infinite length"),
            ("720", "480", "25/1", "nan", "1:1", "NaN length"),
            ("720", "480", "25/1", "-5", "1:1", "negative length"),
            (
                "720",
                "480",
                "25/1",
                "99999999999",
                "1:1",
                "length of millennia",
            ),
            ("720", "480", "25/1", "10", "4000000:1", "stretched pixels"),
            (
                "16384",
                "16384",
                "25/1",
                "10",
                "1:1",
                "over the pixel budget",
            ),
        ] {
            assert!(parse_probe(&with(w, h, f, d, sar)).is_err(), "{why}");
        }
    }

    #[test]
    fn a_silent_header_is_told_apart_from_a_declared_one() {
        let with = |f: &str| format!(r#"{{"streams":[{{"codec_type":"video"{f}}}]}}"#);
        assert!(field_order_declared(&with(r#","field_order":"tt""#)));
        assert!(field_order_declared(&with(
            r#","field_order":"progressive""#
        )));
        assert!(!field_order_declared(&with(r#","field_order":"unknown""#)));
        assert!(!field_order_declared(&with("")));
    }

    #[test]
    fn frame_flags_and_idet_output_are_read() {
        assert!(mostly_flagged("1\n1\n0\n1,\n"));
        assert!(!mostly_flagged("0\n0\n1\n"));
        assert!(!mostly_flagged(""));
        let log = "[idet] Multi frame detection: TFF:     0 BFF:     0 Progressive:     0 Undetermined:     0
[idet] Multi frame detection: TFF:   799 BFF:     4 Progressive:     0 Undetermined:     1";
        assert!(combed(log));
        assert!(!combed(
            "Multi frame detection: TFF: 3 BFF: 2 Progressive: 200 Undetermined: 9"
        ));
        assert!(!combed(
            "Multi frame detection: TFF: 4 BFF: 0 Progressive: 0 Undetermined: 900"
        ));
        assert!(!combed(""));
    }

    #[test]
    fn reads_a_dv_camcorder_file() {
        let v = parse_probe(DV).unwrap();
        assert_eq!((v.width, v.height), (720, 480));
        assert_eq!((v.fps_num, v.fps_den), (30000, 1001));
        assert!(v.interlaced && v.has_audio);
        assert_eq!(v.sar, (8, 9));
        assert_eq!(v.frames, Some(1830));
        // 720 x 8/9 = 640: the picture is 4:3 once the pixels are square.
        assert_eq!(v.square_size(), (640, 480));
        assert!((v.fps() - 29.97).abs() < 0.01);
    }

    #[test]
    fn progressive_square_and_missing_values_have_defaults() {
        let v = parse_probe(
            r#"{"streams":[{"codec_type":"video","width":161,"height":99,
                "avg_frame_rate":"0/0","r_frame_rate":"25/1","field_order":"progressive",
                "sample_aspect_ratio":"N/A"}],"format":{"duration":"3.0"}}"#,
        )
        .unwrap();
        assert!(!v.interlaced && !v.has_audio);
        assert_eq!((v.fps_num, v.fps_den), (25, 1));
        assert_eq!(v.sar, (1, 1));
        assert_eq!(v.square_size(), (161, 99));
        assert_eq!(v.frames, None);
        assert!(parse_probe("{}").is_err());
        assert!(parse_probe(r#"{"streams":[{"codec_type":"audio"}]}"#).is_err());
    }
}
