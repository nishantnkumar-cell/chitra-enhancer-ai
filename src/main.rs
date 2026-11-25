use std::path::PathBuf;
use std::sync::atomic::Ordering;

use anyhow::{Context, Result, bail};
use clap::Parser;

use chitra_enhance::engine::{Engine, Lanczos, Rescaled};
use chitra_enhance::ffmpeg::{self, Tools};
use chitra_enhance::video::{self, Codec, Deinterlace, Denoise, Options, Stopped};
use chitra_enhance::{analyze, disc};

/// Accepts an empty path (meaning "not given"), which clap's own parser refuses.
fn any_path(s: &str) -> Result<PathBuf, std::convert::Infallible> {
    Ok(PathBuf::from(s))
}

#[derive(Clone, Copy, Debug)]
struct Edges([u32; 4]);
#[derive(Clone, Copy, Debug)]
struct Shift(i32, i32);

fn parse_crop(s: &str) -> Result<Edges, String> {
    let v: Vec<u32> = s
        .split(':')
        .map(|x| x.parse().map_err(|_| format!("\"{x}\" is not a number")))
        .collect::<Result<_, _>>()?;
    <[u32; 4]>::try_from(v)
        .map(Edges)
        .map_err(|_| "give four numbers, left:right:top:bottom".into())
}

fn parse_shift(s: &str) -> Result<Shift, String> {
    let (a, b) = s.split_once(':').ok_or("give two numbers, blue:red")?;
    let n = |x: &str| {
        x.parse::<i32>()
            .map_err(|_| format!("\"{x}\" is not a number"))
    };
    Ok(Shift(n(a)?, n(b)?))
}

/// Restore and enlarge old, noisy video (camcorder, tape) on your own computer.
#[derive(Parser, Debug, Clone)]
#[command(
    version,
    about,
    after_help = "Also: `chitra-enhance models` manages the AI models; `chitra-enhance licences` shows\nthe licence. Free software (AGPL-3.0-or-later) with no warranty."
)]
struct Args {
    /// The video or photo to restore, a folder of them, or a DVD (an .iso image,
    /// a drive, or a folder holding VIDEO_TS).
    #[arg(short, long)]
    input: PathBuf,
    /// Where to write the result (never the input). .mp4, .mkv or .mov; a folder
    /// for a batch or for every title of a DVD. Not needed with --analyze.
    #[arg(short, long, default_value = "", value_parser = any_path)]
    output: PathBuf,
    /// Look at a few places in the video and suggest settings, changing nothing.
    #[arg(long)]
    analyze: bool,
    /// Cut black edges: pixels from the left, right, top and bottom (even
    /// numbers), for example 10:6:0:0. --analyze suggests values.
    #[arg(long, value_name = "L:R:T:B", value_parser = parse_crop)]
    crop: Option<Edges>,
    /// Remove a colour cast: amounts (-40..40) taken from the blue and red
    /// colour channels, for example 6:2. --analyze reports what it sees.
    #[arg(long, value_name = "BLUE:RED", value_parser = parse_shift, allow_hyphen_values = true)]
    colour_shift: Option<Shift>,
    /// Line colour up with the picture: lines (-8..8) the blue and red colour
    /// sit too low, for example 2:3. Common on tape. --analyze suggests values.
    #[arg(long, value_name = "BLUE:RED", value_parser = parse_shift, allow_hyphen_values = true)]
    colour_align: Option<Shift>,
    /// Soften the bright outlines a camera's edge sharpening leaves, 0..100.
    /// About 40 suits typical camcorder tape; higher also softens detail.
    #[arg(long, default_value_t = 0)]
    dehalo: u32,
    /// DVD only: show the titles on the disc.
    #[arg(long)]
    list_titles: bool,
    /// DVD only: a title number, or "all". Default: the longest title.
    #[arg(long, value_name = "N|all")]
    title: Option<String>,
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
    /// Measure the video's noise, print `NOISE <measured> <suggested>` and stop.
    /// The suggested value is what the five-frame denoiser would use.
    #[arg(long)]
    measure_noise: bool,
    /// Try the settings on 1 to 30 seconds from the middle of the video:
    /// writes the short clip to -o plus a before/after picture next to it.
    #[arg(long, value_name = "SECONDS")]
    preview: Option<f64>,
    /// Print `PROGRESS done total` lines on stdout (used by the GUI).
    #[arg(long, hide = true)]
    progress: bool,
    /// Accepted but unused in a build without AI support, so that the same
    /// command works with either build.
    #[cfg(not(feature = "onnx"))]
    #[arg(long = "device", hide = true)]
    _device: Option<String>,
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

/// Checks that a model file can be used and says what it is.
#[cfg(feature = "onnx")]
fn check_model(p: &std::path::Path) -> Result<String> {
    println!("Checking {} ...", p.display());
    Ok(chitra_enhance::onnx::inspect(p)
        .with_context(|| format!("{} cannot be used as a model", p.display()))?
        .describe())
}

#[cfg(not(feature = "onnx"))]
fn check_model(_: &std::path::Path) -> Result<String> {
    bail!("checking a model needs a build with `--features onnx`")
}

/// `chitra-enhance models list | get [name...] | add <file> | remove <name> | runtime`
fn models_command(args: &[String]) -> Result<()> {
    use chitra_enhance::models;
    let dir = chitra_enhance::job::models_dir().context("cannot work out where models are kept")?;
    match args.first().map(String::as_str) {
        Some("list") | None => {
            println!("Models folder: {}", dir.display());
            let mine: Vec<_> = models::library(&dir)
                .into_iter()
                .filter(|p| {
                    !models::MODELS
                        .iter()
                        .any(|m| p.file_name().is_some_and(|n| n == m.file))
                })
                .collect();
            for m in models::MODELS {
                let state = if models::is_installed(&dir, m) {
                    "installed"
                } else {
                    "not installed"
                };
                println!(
                    "  {:34} {:>5.1} MB  {state}  - {}",
                    m.file,
                    m.bytes as f64 / 1e6,
                    m.what
                );
            }
            if !mine.is_empty() {
                println!("\nYour own models:");
                for p in &mine {
                    println!("  {}", p.display());
                }
            }
            println!("\nFetch with: chitra-enhance models get [all | part of a name]");
            println!(
                "Add your own: chitra-enhance models add <file.onnx>  (remove: models remove <name>)"
            );
            Ok(())
        }
        Some("get") => {
            for m in models::select(&args[1..])? {
                if models::is_installed(&dir, m) {
                    println!("{} is already here", m.file);
                    continue;
                }
                print!("Downloading {} ... ", m.file);
                std::io::Write::flush(&mut std::io::stdout())?;
                let p = models::fetch(&dir, m)?;
                println!("ok, checksum verified: {}", p.display());
            }
            println!(
                "Models are licensed by their authors: see docs/MODEL_LICENCES.md in the repository."
            );
            Ok(())
        }
        Some("add") => {
            let force = args.iter().any(|a| a == "--force");
            let mut rest = args[1..].iter().filter(|a| !a.starts_with("--"));
            let src = PathBuf::from(
                rest.next()
                    .context("usage: models add <file.onnx> [name] [--force]")?,
            );
            if !src.is_file() {
                bail!("{} is not a file", src.display());
            }
            let name = match rest.next() {
                Some(n) => n.clone(),
                None => src
                    .file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_default(),
            };
            let kind = check_model(&src)?;
            let dest = models::add_to_library(&dir, &src, &name, force)?;
            println!(
                "Added {} - {}.\nUse it with {} {}",
                dest.display(),
                kind,
                if kind.contains("denoiser") {
                    "--temporal-model"
                } else {
                    "--model"
                },
                dest.display()
            );
            Ok(())
        }
        Some("remove") => {
            let name = args.get(1).context("usage: models remove <name>")?;
            let p = models::remove_custom(&dir, name)?;
            println!("Removed {}", p.display());
            Ok(())
        }
        Some("runtime") => {
            match models::find_runtime() {
                Some(p) => println!("ONNX Runtime found: {}", p.display()),
                None => println!("{}", models::RUNTIME_HELP),
            }
            Ok(())
        }
        Some(other) => {
            bail!("unknown models command \"{other}\"; use list, get, add, remove or runtime")
        }
    }
}

fn real_main() -> Result<()> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.first().map(String::as_str) == Some("models") {
        return models_command(&argv[1..]);
    }
    if matches!(
        argv.first().map(String::as_str),
        Some("licences" | "licenses")
    ) {
        println!("{}", chitra_enhance::legal::notice());
        return Ok(());
    }
    let a = Args::parse();
    if let Some(src) = disc::source(&a.input) {
        return disc_mode(a, src);
    }
    if a.list_titles || a.title.is_some() {
        bail!("--title and --list-titles are for DVDs");
    }
    if a.analyze {
        return analyze_file(&a);
    }
    need_output(&a)?;
    if a.input.is_dir() {
        return batch(a);
    }
    run_one(a)
}

fn need_output(a: &Args) -> Result<()> {
    if a.output.as_os_str().is_empty() {
        bail!("say where to write the result with -o");
    }
    Ok(())
}

/// Which of the downloaded models could be suggested.
fn installed_models() -> (Option<PathBuf>, Option<PathBuf>) {
    use chitra_enhance::job;
    let Some(dir) = job::models_dir() else {
        return (None, None);
    };
    // The CPU build of a model is the faster one without a GPU.
    let fp16 = chitra_enhance::models::likely_gpu();
    (
        job::find_model(&dir, "realesr", fp16),
        job::find_model(&dir, "fastdvdnet", fp16),
    )
}

fn analyze_file(a: &Args) -> Result<()> {
    if !a.input.is_file() || is_photo(&a.input) {
        bail!("--analyze looks at one video or a DVD");
    }
    let tools = Tools::locate(a.ffmpeg.as_deref())?;
    let input = std::path::absolute(&a.input)?;
    let info = ffmpeg::probe(&tools, &input)?;
    let samples: Vec<_> = analyze::positions(info.duration)
        .into_iter()
        .map(|p| (input.clone(), p))
        .collect();
    show_analysis(a, &tools, &info, &samples)
}

fn show_analysis(
    a: &Args,
    tools: &Tools,
    info: &ffmpeg::VideoInfo,
    samples: &[(PathBuf, f64)],
) -> Result<()> {
    use chitra_enhance::job;
    let (upscaler, denoiser) = installed_models();
    let have = analyze::Available {
        upscaler: upscaler.is_some(),
        denoiser: denoiser.is_some(),
    };
    let r = analyze::run(tools, info, samples, have)?;
    println!(
        "Looked at {} places: {}x{} at {:.3} fps, {:.0} s, {}; noise about {:.1} of 255.",
        r.samples,
        r.width,
        r.height,
        r.fps,
        r.seconds,
        if r.interlaced {
            "interlaced"
        } else {
            "progressive"
        },
        r.noise
    );
    println!("\nSuggested:");
    for line in &r.reasons {
        println!("  - {line}");
    }
    if !r.notes.is_empty() {
        println!("\nAlso seen:");
        for line in &r.notes {
            println!("  - {line}");
        }
    }
    let mut s = job::Settings {
        input: a.input.clone(),
        output: if a.output.as_os_str().is_empty() {
            job::default_output(&a.input)
        } else {
            a.output.clone()
        },
        title: a.title.clone(),
        ..job::Settings::default()
    };
    s.apply(&r.suggestion, upscaler, denoiser);
    println!("\nCommand with these settings (change any of them):");
    println!("  {}", job::command_line(&s.args()));
    println!("Add --preview 5 to try them on a few seconds first.");
    if a.progress {
        println!("ANALYSIS {}", serde_json::to_string(&r)?);
    }
    Ok(())
}

/// A DVD: list, analyse, preview or restore its titles. A title is first copied
/// off the disc without re-encoding, then handled like any video.
fn disc_mode(a: Args, src: PathBuf) -> Result<()> {
    let tools = Tools::locate(a.ffmpeg.as_deref())?;
    let src = std::path::absolute(src)?;
    disc::check_support(&tools)?;
    disc::check_unprotected(&src)?;
    let titles = disc::titles(&tools, &src)?;
    if a.list_titles {
        for t in &titles {
            if a.progress {
                println!("TITLE {} {:.0}", t.number, t.seconds);
            } else {
                println!(
                    "Title {:2}: {}:{:02}",
                    t.number,
                    t.seconds as u64 / 60,
                    t.seconds as u64 % 60
                );
            }
        }
        return Ok(());
    }
    let all = a.title.as_deref() == Some("all");
    let chosen: Vec<disc::Title> = match a.title.as_deref() {
        None => disc::longest(&titles).cloned().into_iter().collect(),
        Some("all") => titles
            .iter()
            .filter(|t| t.seconds >= disc::MIN_TITLE_SECONDS)
            .cloned()
            .collect(),
        Some(n) => {
            let n: u32 = n.parse().context("--title takes a number or \"all\"")?;
            let t = titles.iter().find(|t| t.number == n).with_context(|| {
                format!("the disc has no title {n}; it has 1 to {}", titles.len())
            })?;
            vec![t.clone()]
        }
    };
    let first = chosen.first().context("no usable title on the disc")?;
    eprintln!(
        "DVD with {} title(s); using {}",
        titles.len(),
        if all {
            format!("{} of them", chosen.len())
        } else {
            format!("title {} ({:.0} s)", first.number, first.seconds)
        }
    );

    if a.analyze || a.measure_noise || a.preview.is_some() {
        if all {
            bail!("pick one title with --title for this");
        }
        let tmp = tempfile::tempdir()?;
        if a.analyze {
            let mut samples = Vec::new();
            for (i, at) in analyze::positions(first.seconds).into_iter().enumerate() {
                let f = tmp.path().join(format!("sample{i}.mkv"));
                disc::import(&tools, &src, first.number, Some((at, 3.0)), &f)?;
                samples.push((f, 0.0));
            }
            let mut info = ffmpeg::probe(&tools, &samples[0].0)?;
            info.duration = first.seconds;
            return show_analysis(&a, &tools, &info, &samples);
        }
        let len = a.preview.map_or(20.0, |p| p.clamp(1.0, 30.0) + 2.0);
        let start = ((first.seconds - len) / 2.0).max(0.0);
        let f = tmp.path().join("sample.mkv");
        disc::import(&tools, &src, first.number, Some((start, len)), &f)?;
        let mut one = a.clone();
        one.input = f;
        if one.output.as_os_str().is_empty() {
            if a.preview.is_some() {
                bail!("say where to write the preview with -o");
            }
            one.output = tmp.path().join("unused.mp4");
        }
        return run_one(one);
    }

    need_output(&a)?;
    let output = std::path::absolute(&a.output)?;
    if all {
        std::fs::create_dir_all(&output).context("--title all needs a folder for -o")?;
    }
    let stem = src
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "dvd".into());
    let mut failed = Vec::new();
    for t in &chosen {
        let out = if all {
            output.join(format!("{stem}-title{:02}.mp4", t.number))
        } else {
            output.clone()
        };
        if out.exists() && !a.force {
            if all {
                eprintln!("Title {} is already done, skipping", t.number);
                continue;
            }
            bail!(
                "{} already exists; use --force to replace it, or choose another name",
                out.display()
            );
        }
        match disc_title(&a, &tools, &src, t, &out) {
            Ok(()) => {}
            Err(e) if !all || e.downcast_ref::<Stopped>().is_some() => return Err(e),
            Err(e) => {
                eprintln!("  title {} FAILED: {e:#}", t.number);
                failed.push(t.number.to_string());
            }
        }
    }
    if !failed.is_empty() {
        bail!("these titles failed: {}", failed.join(", "));
    }
    Ok(())
}

fn disc_title(
    a: &Args,
    tools: &Tools,
    src: &std::path::Path,
    t: &disc::Title,
    out: &std::path::Path,
) -> Result<()> {
    // Kept beside the result until it is finished, so a stopped run resumes
    // without reading the disc again.
    let copy = out.with_extension(format!("title{:02}.source.mkv", t.number));
    let reusable = copy.is_file()
        && ffmpeg::probe(tools, &copy).is_ok_and(|i| (i.duration - t.seconds).abs() < 2.0);
    if !reusable {
        eprintln!("Copying title {} off the disc ...", t.number);
        disc::import(tools, src, t.number, None, &copy)?;
    }
    let mut one = a.clone();
    one.input = copy.clone();
    one.output = out.to_path_buf();
    run_one(one)?;
    if !a.keep_work && !a.dry_run && a.preview.is_none() {
        let _ = std::fs::remove_file(&copy);
    }
    Ok(())
}

/// Every photo and video in a folder, one after the other, into another folder.
/// A file whose result already exists is skipped, so an interrupted batch picks
/// up where it stopped (and an interrupted video resumes inside its own work folder).
fn batch(a: Args) -> Result<()> {
    use chitra_enhance::job;
    if a.preview.is_some() || a.measure_noise {
        bail!("--preview and --measure-noise work on one file at a time");
    }
    if a.output.is_file() {
        bail!(
            "{} is a file; a folder is needed for a batch",
            a.output.display()
        );
    }
    std::fs::create_dir_all(&a.output)?;
    if a.output.canonicalize()? == a.input.canonicalize()? {
        bail!("the output folder must not be the input folder");
    }
    let mut files: Vec<PathBuf> = std::fs::read_dir(&a.input)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_file() && (job::is_photo(p) || job::is_video(p)))
        .collect();
    files.sort();
    if files.is_empty() {
        bail!("no photos or videos in {}", a.input.display());
    }
    let (mut done, mut skipped, mut failed) = (0, 0, Vec::new());
    for (i, f) in files.iter().enumerate() {
        let name = job::default_output(f);
        let out = a.output.join(name.file_name().unwrap_or_default());
        eprintln!("\n[{}/{}] {}", i + 1, files.len(), f.display());
        if out.exists() && !a.force {
            eprintln!("  already done ({}), skipping", out.display());
            skipped += 1;
            continue;
        }
        let mut one = a.clone();
        one.input = f.clone();
        one.output = out;
        match run_one(one) {
            Ok(()) => done += 1,
            Err(e) if e.downcast_ref::<Stopped>().is_some() => return Err(e),
            Err(e) => {
                eprintln!("  FAILED: {e:#}");
                failed.push(f.display().to_string());
            }
        }
    }
    eprintln!(
        "\nBatch finished: {done} done, {skipped} skipped, {} failed",
        failed.len()
    );
    if !failed.is_empty() {
        bail!(
            "{} of {} files failed: {}",
            failed.len(),
            files.len(),
            failed.join(", ")
        );
    }
    Ok(())
}

fn run_one(mut a: Args) -> Result<()> {
    if !a.input.is_file() {
        bail!("{} is not a file", a.input.display());
    }
    // Absolute paths can never be mistaken for an ffmpeg option (a name that
    // starts with "-"), and make "output is the input" comparisons reliable.
    a.input = std::path::absolute(&a.input)?;
    a.output = std::path::absolute(&a.output)?;
    if a.output.exists() && !a.restart && a.output == a.input {
        bail!("the output must not be the input");
    }
    if let Some(s) = a.scale
        && !(1..=4).contains(&s)
    {
        bail!("--scale must be 1 to 4");
    }
    if a.preview.is_some_and(|p| !(1.0..=30.0).contains(&p)) {
        bail!("--preview must be 1 to 30 seconds");
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
        crop: a.crop.map(|c| c.0),
        colour_shift: a.colour_shift.map(|s| (s.0, s.1)),
        colour_align: a.colour_align.map(|s| (s.0, s.1)),
        dehalo: a.dehalo,
    };
    o.check_picture(&info)?;
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
    let measured = if a.temporal_sigma.is_none()
        && (a.measure_noise || a.temporal_model.is_some() || a.dry_run)
    {
        chitra_enhance::noise::estimate_video(&o, &info)?
    } else {
        None
    };
    // FastDVDnet does best when told the noise is somewhat stronger than it is
    // (tuned on three test clips: the best setting was 1.5-2x the measured level).
    let suggested = measured.map(|v| (v * 1.6).clamp(3.0, 60.0));
    if a.measure_noise {
        match (measured, suggested) {
            (Some(m), Some(u)) => println!("NOISE {m:.1} {u:.0}"),
            _ => println!("NOISE none"),
        }
        return Ok(());
    }
    #[cfg_attr(not(feature = "onnx"), allow(unused_variables))]
    let sigma = match (a.temporal_sigma, measured, suggested) {
        (Some(v), ..) => v,
        (None, Some(m), Some(u)) => {
            eprintln!(
                "Measured noise level: about {m:.0} (0-255 scale); using {u:.0} for the denoiser"
            );
            u
        }
        (None, ..) if a.temporal_model.is_some() || a.dry_run => {
            eprintln!("Could not measure the noise level; using 15");
            15.0
        }
        _ => 15.0,
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
    // Once per process: a batch runs this for every file.
    let _ = ctrlc::set_handler(|| video::STOP.store(true, Ordering::Relaxed));
    if let Some(secs) = a.preview {
        if o.output.exists() && !a.force {
            bail!(
                "{} exists; a preview would replace it (use --force, or another name)",
                o.output.display()
            );
        }
        let p = video::preview(&o, engine.as_mut(), temporal.as_deref_mut(), &info, secs)?;
        let mins = p.whole.as_secs().div_ceil(60);
        eprintln!(
            "Preview clip: {}\nBefore/after picture: {}\nThe whole video would take about {} h {:02} min at this speed.",
            p.clip.display(),
            p.compare.display(),
            mins / 60,
            mins % 60
        );
        return Ok(());
    }
    if o.output.exists() && !a.force {
        bail!(
            "{} already exists; use --force to replace it, or choose another name",
            o.output.display()
        );
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
