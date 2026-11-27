//! The `serve`, `remote` and `new-key` commands.

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result, bail};
use clap::Parser;

use chitra_enhance::job;
use chitra_enhance::server::client::{Client, Remote, Wanted};
use chitra_enhance::server::{self, Config};

/// Enhance pictures and short clips for other computers on your Tailscale
/// network. It listens on this computer only unless told otherwise.
#[derive(Parser, Debug)]
#[command(name = "chitra-enhance serve")]
struct Serve {
    /// A JSON configuration file; the options below replace what it says.
    #[arg(long)]
    config: Option<PathBuf>,
    /// An address and port, such as this computer's Tailscale address:
    /// 100.101.102.103:8765. May be given more than once.
    #[arg(long)]
    listen: Vec<SocketAddr>,
    /// A file holding the key (see `chitra-enhance new-key`).
    #[arg(long)]
    key_file: Option<PathBuf>,
    /// The folder of upscaling models. Default: where `models get` puts them.
    #[arg(long)]
    models: Option<PathBuf>,
    /// The model used when none is asked for, by file name.
    #[arg(long)]
    model: Option<String>,
    /// Which processor runs the model: auto, tensorrt, cuda, directml, coreml
    /// or cpu.
    #[arg(long)]
    device: Option<String>,
    /// Run under the Windows service manager (see
    /// scripts/install-windows-service.ps1).
    #[arg(long, requires = "config")]
    windows_service: bool,
}

/// Send pictures or clips to a `chitra-enhance serve` on another computer.
#[derive(Parser, Debug)]
#[command(name = "chitra-enhance remote")]
struct SendTo {
    /// Pictures or short videos. With none, shows what the service offers.
    files: Vec<PathBuf>,
    /// The service, such as 100.101.102.103:8765. Remembered with --save;
    /// otherwise CHITRA_SERVER, or what was saved.
    #[arg(long)]
    server: Option<String>,
    /// A file holding the key. Otherwise CHITRA_KEY, or what was saved.
    #[arg(long)]
    key_file: Option<PathBuf>,
    /// Remember the server and key for next time.
    #[arg(long)]
    save: bool,
    /// Where to save the result of one file. Default: beside it, as
    /// `name-enhanced.jxl` or `.mp4`.
    #[arg(short, long)]
    output: Option<PathBuf>,
    /// 1, 2 or 4 times larger.
    #[arg(long)]
    scale: Option<usize>,
    /// A model the service offers, by name.
    #[arg(long)]
    model: Option<String>,
    /// Replace an existing result.
    #[arg(long)]
    force: bool,
}

pub fn command(name: &str, argv: &[String]) -> Result<()> {
    // The first word is the command itself, which the parsers take as their name.
    match name {
        "new-key" => {
            if argv.len() > 1 {
                bail!("usage: chitra-enhance new-key");
            }
            println!("{}", server::new_key()?);
            Ok(())
        }
        "serve" => serve(Serve::parse_from(argv)),
        _ => remote(SendTo::parse_from(argv)),
    }
}

fn key_from(file: &Path) -> Result<String> {
    Ok(std::fs::read_to_string(file)
        .with_context(|| format!("cannot read {}", file.display()))?
        .trim_start_matches('\u{feff}')
        .trim()
        .to_string())
}

fn serve(a: Serve) -> Result<()> {
    #[cfg(windows)]
    if a.windows_service {
        return server::winservice::run(a.config.context("--config is needed")?);
    }
    if a.windows_service {
        bail!("--windows-service is for Windows");
    }
    let mut config = match &a.config {
        Some(file) => Config::read(file)?,
        None => Config::default(),
    };
    if !a.listen.is_empty() {
        config.listen = a.listen;
    }
    if let Some(file) = &a.key_file {
        config.key = key_from(file)?;
    }
    if a.models.is_some() {
        config.models = a.models;
    }
    if a.model.is_some() {
        config.model = a.model;
    }
    if let Some(device) = a.device {
        config.device = device;
    }
    if config.models.is_none() {
        config.models = job::models_dir().filter(|d| d.is_dir());
    }
    // The downloaded upscaler, if the service was not told which to use.
    if config.model.is_none()
        && let Some(folder) = &config.models
    {
        let fp16 = chitra_enhance::models::likely_gpu();
        config.model = job::find_model(folder, "realesr", fp16)
            .and_then(|p| Some(p.file_stem()?.to_str()?.to_string()));
    }
    let stop = Arc::new(AtomicBool::new(false));
    let asked = stop.clone();
    ctrlc::set_handler(move || asked.store(true, Ordering::Relaxed))
        .context("cannot watch for Ctrl-C")?;
    server::run(config, stop, |_| eprintln!("Press Ctrl-C to stop."))
}

fn remote(a: SendTo) -> Result<()> {
    // Kept beside the models folder.
    let saved = job::models_dir().and_then(|m| Some(m.parent()?.join("remote.json")));
    let before = saved.as_deref().and_then(Remote::read).unwrap_or_default();
    let given = |flag: Option<String>, variable: &str, kept: String| {
        flag.or_else(|| std::env::var(variable).ok().filter(|v| !v.is_empty()))
            .unwrap_or(kept)
    };
    let remote = Remote {
        server: given(a.server, "CHITRA_SERVER", before.server),
        key: given(
            a.key_file.as_deref().map(key_from).transpose()?,
            "CHITRA_KEY",
            before.key,
        ),
    };
    if remote.server.is_empty() || remote.key.is_empty() {
        bail!(
            "say which service and key to use, once: chitra-enhance remote --server ADDRESS:8765 --key-file FILE --save"
        );
    }
    if a.output.is_some() && a.files.len() != 1 {
        bail!("-o is for exactly one file");
    }
    let client = Client::new(&remote);
    // Checks the address and key before anything is saved or sent.
    let about = client.about()?;
    if a.save {
        let file = saved.context("cannot work out where to keep the settings")?;
        remote.save(&file)?;
        println!("Saved to {}", file.display());
    }
    if a.files.is_empty() {
        println!("{}", serde_json::to_string_pretty(&about)?);
        return Ok(());
    }
    let wanted = Wanted {
        scale: a.scale,
        model: a.model,
    };
    let mut failed = 0;
    for file in &a.files {
        let photo = job::is_photo(file);
        let out = match &a.output {
            Some(o) => o.clone(),
            None => job::default_output(file),
        };
        if out.exists() && !a.force {
            eprintln!("{}: {} exists; use --force", file.display(), out.display());
            failed += 1;
            continue;
        }
        if std::fs::canonicalize(file).ok() == std::fs::canonicalize(&out).ok() && out.exists() {
            eprintln!(
                "{}: the result must not replace the original",
                file.display()
            );
            failed += 1;
            continue;
        }
        let started = std::time::Instant::now();
        let done = if photo {
            client.photo(file, &wanted, &out)
        } else {
            let mut shown = 0;
            client.clip(file, &wanted, &out, |done, total| {
                let percent = done * 100 / total;
                if percent >= shown + 10 {
                    shown = percent;
                    eprintln!("  {percent}%");
                }
            })
        };
        match done {
            Ok(()) => println!(
                "{} -> {} ({:.1} s)",
                file.display(),
                out.display(),
                started.elapsed().as_secs_f64()
            ),
            Err(e) => {
                eprintln!("{}: {e:#}", file.display());
                failed += 1;
            }
        }
    }
    if failed > 0 {
        bail!("{failed} of {} files failed", a.files.len());
    }
    Ok(())
}
