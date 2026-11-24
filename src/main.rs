use std::path::PathBuf;
use std::sync::atomic::Ordering;

use anyhow::{Result, bail};
use clap::Parser;

use chitra_enhance::engine::{Engine, Lanczos, Rescaled};
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
    /// How many times larger the picture becomes. Default: the model's own
    /// scale, or 2 without a model. A different number resizes the model's result.
    #[arg(long)]
    scale: Option<usize>,
    /// Film grain to put back after cleaning (0 = none, 3 = fine, 8 = strong).
    /// Fully clean video can look plastic. Default 3 when denoising, else 0.
    #[arg(long)]
    grain: Option<u32>,
    /// An ONNX upscaling model (FP16 or FP32, RGB in 0..1, NCHW). Its own scale
    /// is used. Needs ORT_DYLIB_PATH and a build with `--features onnx`.
    #[arg(long)]
    model: Option<PathBuf>,
    /// Which processor runs the model.
    #[cfg(feature = "onnx")]
    #[arg(long, value_enum, default_value_t = chitra_enhance::onnx::Device::Auto)]
    device: chitra_enhance::onnx::Device,
    /// Graphics memory the model may use, in GB (sets the tile size).
    #[arg(long, default_value_t = 8)]
    vram_budget: usize,
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
    if let Some(s) = a.scale
        && !(1..=4).contains(&s)
    {
        bail!("--scale must be 1 to 4");
    }
    if a.grain.is_some_and(|g| g > 30) {
        bail!("--grain must be 0 to 30");
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
        grain: a
            .grain
            .unwrap_or(if a.denoise == Denoise::Off { 0 } else { 3 }),
        stop_after_chunks: a.stop_after_chunks,
    };
    let mut engine: Box<dyn Engine> = match &a.model {
        #[cfg(feature = "onnx")]
        Some(m) => Box::new(chitra_enhance::onnx::OnnxEngine::load(
            m,
            a.device,
            a.vram_budget * 1024,
        )?),
        #[cfg(not(feature = "onnx"))]
        Some(_) => bail!(
            "this build has no AI model support: rebuild with `cargo build --release --features onnx`"
        ),
        None => Box::new(Lanczos::new(a.scale.unwrap_or(2))),
    };
    if let Some(t) = a.scale {
        engine = Rescaled::wrap(engine, t);
    }
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
