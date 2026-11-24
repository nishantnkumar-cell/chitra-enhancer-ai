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
    /// A five-frame video denoising ONNX model (FastDVDnet; see
    /// scripts/export_fastdvdnet.py). It looks at neighbouring frames, so it
    /// removes camcorder noise far better than the single-frame `--denoise`
    /// filter; with it, consider `--denoise off`. Video only.
    #[arg(long)]
    temporal_model: Option<PathBuf>,
    /// Noise level for --temporal-model on a 0..255 scale: 5 faint, 20 clearly
    /// noisy, 40 very noisy. Too high smooths real detail. Default: measured
    /// from the video itself.
    #[arg(long)]
    temporal_sigma: Option<f32>,
    /// Try the settings on just this many seconds from the middle of the video:
    /// writes the short clip to -o plus a before/after picture next to it.
    #[arg(long, value_name = "SECONDS")]
    preview: Option<f64>,
    /// Print `PROGRESS done total` lines on stdout (used by the GUI).
    #[arg(long, hide = true)]
    progress: bool,
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
    /// Video quality (lower is better; codec default if not given).
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
    /// Photos: JPEG XL quality as a distance (1.0 is visually lossless; 0.5
    /// is better, 2.0 smaller).
    #[arg(long, default_value_t = 1.0)]
    jxl_distance: f32,
    /// Replace an existing output file (never the input).
    #[arg(long)]
    force: bool,
    /// Photos: a separate model for heavily compressed JPEGs (quality below
    /// about 85), for instance a JPEG-artifact remover.
    #[arg(long)]
    model_compressed: Option<PathBuf>,
    /// Starting tile size in pixels (default from --vram-budget).
    #[arg(long)]
    tile_size: Option<usize>,
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
    if a.temporal_sigma
        .is_some_and(|v| !(0.0..=100.0).contains(&v))
    {
        bail!("--temporal-sigma must be 0 to 100");
    }
    if a.grain.is_some_and(|g| g > 30) {
        bail!("--grain must be 0 to 30");
    }
    video::REPORT_PROGRESS.store(a.progress, Ordering::Relaxed);
    if is_photo(&a.input) {
        return photo_mode(&a);
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
        Some(m) => load_model(m, a.device, a.vram_budget, a.tile_size)?,
        #[cfg(not(feature = "onnx"))]
        Some(_) => bail!(
            "this build has no AI model support: rebuild with `cargo build --release --features onnx`"
        ),
        None => Box::new(Lanczos::new(a.scale.unwrap_or(2))),
    };
    if let Some(t) = a.scale {
        engine = Rescaled::wrap(engine, t);
    }
    #[cfg_attr(not(feature = "onnx"), allow(unused_variables))]
    let sigma = match a.temporal_sigma {
        Some(v) => v,
        None if a.temporal_model.is_some() || a.dry_run => {
            match chitra_enhance::noise::estimate_video(&o, &info)? {
                Some(v) => {
                    // FastDVDnet does best when told the noise is somewhat
                    // stronger than it is (tuned on three test clips: the best
                    // setting was 1.5-2x the measured level).
                    let used = (v * 1.6).clamp(3.0, 60.0);
                    eprintln!(
                        "Measured noise level: about {v:.0} (0-255 scale); using {used:.0} for the denoiser"
                    );
                    used
                }
                None => {
                    eprintln!("Could not measure the noise level; using 15");
                    15.0
                }
            }
        }
        None => 15.0,
    };
    let mut temporal: Option<Box<dyn chitra_enhance::temporal::Temporal>> = match &a.temporal_model
    {
        #[cfg(feature = "onnx")]
        Some(m) => Some(Box::new(chitra_enhance::onnx::OnnxDenoiser::load(
            m,
            a.device,
            a.vram_budget * 1024,
            a.tile_size,
            sigma,
        )?)),
        #[cfg(not(feature = "onnx"))]
        Some(_) => bail!(
            "this build has no AI model support: rebuild with `cargo build --release --features onnx`"
        ),
        None => None,
    };
    eprintln!("Filters: {}", o.decode_filters(&info));
    if let Some(t) = &temporal {
        eprintln!("Temporal denoiser: {}", t.describe());
    }
    eprintln!("Engine: {}", engine.describe());
    if a.dry_run {
        return Ok(());
    }
    ctrlc::set_handler(|| video::STOP.store(true, Ordering::Relaxed))?;
    if let Some(secs) = a.preview {
        if o.output.exists() && !a.force {
            bail!(
                "{} exists; a preview would replace it (use --force, or another name)",
                o.output.display()
            );
        }
        let p = video::preview(&o, engine.as_mut(), temporal.as_deref_mut(), &info, secs)?;
        eprintln!(
            "Preview clip: {}\nBefore/after picture: {}",
            p.clip.display(),
            p.compare.display()
        );
        return Ok(());
    }
    let s = video::run(&o, engine.as_mut(), temporal.as_deref_mut(), &info)?;
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

fn is_photo(p: &std::path::Path) -> bool {
    matches!(
        p.extension()
            .map(|e| e.to_string_lossy().to_ascii_lowercase())
            .as_deref(),
        Some("jpg" | "jpeg" | "png" | "webp" | "bmp" | "gif" | "tif" | "tiff")
    )
}

#[cfg(feature = "onnx")]
fn load_model(
    m: &std::path::Path,
    device: chitra_enhance::onnx::Device,
    vram_gb: usize,
    tile: Option<usize>,
) -> Result<Box<dyn Engine>> {
    Ok(Box::new(chitra_enhance::onnx::OnnxEngine::load(
        m,
        device,
        vram_gb * 1024,
        tile,
    )?))
}

fn photo_mode(a: &Args) -> Result<()> {
    use chitra_enhance::{inspector, photo};
    let bytes = std::fs::read(&a.input)?;
    let mut model = a.model.clone();
    if let Some(info) = inspector::inspect(&bytes) {
        let tier = info.tier();
        eprintln!(
            "JPEG {}x{}, quality about {}, {:?} chroma: {}",
            info.width,
            info.height,
            info.estimated_quality(),
            info.subsampling,
            match tier {
                inspector::Tier::Clean => "clean, upscaling only",
                inspector::Tier::Compressed => "compressed, artifacts will be cleaned",
            }
        );
        if tier == inspector::Tier::Compressed
            && let Some(m) = &a.model_compressed
        {
            model = Some(m.clone());
        }
    }
    let mut engine: Box<dyn Engine> = match &model {
        #[cfg(feature = "onnx")]
        Some(m) => load_model(m, a.device, a.vram_budget, a.tile_size)?,
        #[cfg(not(feature = "onnx"))]
        Some(_) => bail!(
            "this build has no AI model support: rebuild with `cargo build --release --features onnx`"
        ),
        None => Box::new(Lanczos::new(a.scale.unwrap_or(2))),
    };
    if let Some(t) = a.scale {
        engine = Rescaled::wrap(engine, t);
    }
    eprintln!("Engine: {}", engine.describe());
    let o = photo::Options {
        input: a.input.clone(),
        output: a.output.clone(),
        jxl_distance: a.jxl_distance,
        force: a.force,
    };
    if a.dry_run {
        photo::check_paths(&o.input, &o.output, o.force)?;
        return Ok(());
    }
    let s = photo::run(&o, &bytes, engine.as_mut())?;
    eprintln!(
        "Done: {}x{} -> {}x{}, {} KB -> {}",
        s.in_size.0,
        s.in_size.1,
        s.out_size.0,
        s.out_size.1,
        s.bytes / 1024,
        a.output.display()
    );
    Ok(())
}
