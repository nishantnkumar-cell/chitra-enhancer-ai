use std::path::PathBuf;
use std::sync::atomic::Ordering;

use anyhow::{Result, bail};
use clap::Parser;

use chitra_enhance::engine::{Engine, Lanczos};
use chitra_enhance::ffmpeg::{self, Tools};
use chitra_enhance::video::{self, Codec, Deinterlace, Denoise, Options, Stopped};

/// Restore and enlarge old, noisy video (camcorder, tape) on your own computer.
#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    /// The video to restore.
    #[arg(short, long)]
    input: PathBuf,
    /// Where to write the result (never the input). .mp4, .mkv or .mov.
    #[arg(short, long)]
    output: PathBuf,
    /// How many times larger the picture becomes.
    #[arg(long, default_value_t = 2)]
    scale: usize,
    #[arg(long, value_enum, default_value_t = Denoise::Medium)]
    denoise: Denoise,
    #[arg(long, value_enum, default_value_t = Deinterlace::Auto)]
    deinterlace: Deinterlace,
    /// Keep both fields of interlaced video: smoother motion, twice the frames.
    #[arg(long)]
    double_rate: bool,
    #[arg(long, value_enum, default_value_t = Codec::H265)]
    codec: Codec,
    /// Quality (lower is better; codec default if not given).
    #[arg(long)]
    crf: Option<u32>,
    #[arg(long)]
    preset: Option<String>,
    /// Seconds per chunk: the unit that is kept when you stop and resume.
    #[arg(long, default_value_t = 30.0)]
    chunk_seconds: f64,
    /// Throw away an unfinished earlier run and start over.
    #[arg(long)]
    restart: bool,
    /// Keep the work folder after finishing.
    #[arg(long)]
    keep_work: bool,
    /// Path to ffmpeg (ffprobe is looked for beside it).
    #[arg(long)]
    ffmpeg: Option<PathBuf>,
    /// Only report what was found in the input and what would be done.
    #[arg(long)]
    dry_run: bool,
    #[arg(long, hide = true)]
    stop_after_chunks: Option<usize>,
}

fn main() {
    if let Err(e) = real_main() {
        if let Some(s) = e.downcast_ref::<Stopped>() {
            eprintln!("\n{s}. Run the same command again to carry on.");
            std::process::exit(130);
        }
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
}

fn real_main() -> Result<()> {
    let a = Args::parse();
    if !a.input.is_file() {
        bail!("{} is not a file", a.input.display());
    }
    if a.output.exists() && !a.restart && a.output == a.input {
        bail!("the output must not be the input");
    }
    if !(1..=4).contains(&a.scale) {
        bail!("--scale must be 1 to 4");
    }
    let tools = Tools::locate(a.ffmpeg.as_deref())?;
    let info = ffmpeg::probe(&tools, &a.input)?;
    eprintln!(
        "Input: {}x{} at {:.3} fps, {:.1} s, {}{}",
        info.width,
        info.height,
        info.fps(),
        info.duration,
        if info.interlaced {
            "interlaced"
        } else {
            "progressive"
        },
        if info.has_audio { ", with audio" } else { "" }
    );
    let o = Options {
        input: a.input,
        output: a.output,
        tools,
        denoise: a.denoise,
        deinterlace: a.deinterlace,
        double_rate: a.double_rate,
        codec: a.codec,
        crf: a.crf,
        preset: a.preset,
        chunk_secs: a.chunk_seconds,
        restart: a.restart,
        keep_work: a.keep_work,
        stop_after_chunks: a.stop_after_chunks,
    };
    let mut engine: Box<dyn Engine> = Box::new(Lanczos::new(a.scale));
    eprintln!("Filters: {}", o.decode_filters(&info));
    eprintln!("Engine: {}", engine.describe());
    if a.dry_run {
        return Ok(());
    }
    ctrlc::set_handler(|| video::STOP.store(true, Ordering::Relaxed))?;
    let s = video::run(&o, engine.as_mut(), &info)?;
    eprintln!(
        "Done: {} frames in {} chunks ({} resumed) in {:.0} s -> {}",
        s.frames,
        s.chunks,
        s.resumed,
        s.seconds,
        o.output.display()
    );
    Ok(())
}
