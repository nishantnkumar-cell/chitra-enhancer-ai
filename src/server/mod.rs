//! A small service that enhances pictures and short clips for other computers
//! on your Tailscale network: `chitra-enhance serve`.
//!
//! It listens only on this computer's own and Tailscale addresses, asks for a
//! secret key with every request, takes only picture and video files (never
//! names or folders from the sender), runs one job at a time, and removes what
//! was sent once it has been dealt with.

pub mod client;
mod guard;
mod web;
#[cfg(windows)]
pub mod winservice;
mod work;

use std::io::Write;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

pub use guard::{allowed_address, new_key};

/// Jobs that may wait for the worker.
const WAITING_AT_MOST: usize = 8;

/// What the service is told, from a JSON file.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct Config {
    /// Addresses and ports, such as "100.101.102.103:8765".
    pub listen: Vec<SocketAddr>,
    /// The secret every request must carry.
    pub key: String,
    /// Where uploads and results are held while they are worked on.
    pub data: PathBuf,
    /// The folder of upscaling models senders may choose from.
    pub models: Option<PathBuf>,
    /// The model used when none is asked for: a file name in `models`.
    pub model: Option<String>,
    /// A different one for pictures; clips keep `model` (a model that suits
    /// pictures is often too slow, or flickers, on video).
    pub photo_model: Option<String>,
    /// How many times larger, when none is asked for.
    pub scale: usize,
    pub device: String,
    pub vram_gb: usize,
    /// ffmpeg, if it is not on the search path.
    pub ffmpeg: Option<PathBuf>,
    /// The ONNX Runtime library, if it is not found by itself.
    pub runtime: Option<PathBuf>,
    pub max_photo_mb: usize,
    pub max_clip_mb: usize,
    pub max_clip_seconds: u32,
    /// Where to note what happened (without file contents or keys).
    pub log: Option<PathBuf>,
    /// The program that restores clips; this one unless told otherwise.
    pub program: Option<PathBuf>,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            listen: vec![SocketAddr::from(([127, 0, 0, 1], 8765))],
            key: String::new(),
            data: std::env::temp_dir().join("chitra-enhance-service"),
            models: None,
            model: None,
            photo_model: None,
            scale: 2,
            device: "auto".into(),
            vram_gb: 8,
            ffmpeg: None,
            runtime: None,
            max_photo_mb: 50,
            max_clip_mb: 500,
            max_clip_seconds: 60,
            log: None,
            program: None,
        }
    }
}

impl Config {
    pub fn read(file: &Path) -> Result<Config> {
        let text = std::fs::read_to_string(file)
            .with_context(|| format!("cannot read {}", file.display()))?;
        // Windows editors like to start a file with a byte order mark.
        serde_json::from_str(text.trim_start_matches('\u{feff}'))
            .with_context(|| format!("{} is not a valid service configuration", file.display()))
    }

    /// Refuses anything that would open the service wider than intended.
    pub fn check(&self) -> Result<()> {
        if self.listen.is_empty() {
            bail!("give at least one address to listen on");
        }
        if let Some(a) = self.listen.iter().find(|a| !allowed_address(a.ip())) {
            bail!(
                "{a} cannot be used: the service listens only on this computer (127.0.0.1) or its Tailscale address (100.x.y.z; see `tailscale ip`)"
            );
        }
        guard::Key::new(&self.key).map_err(anyhow::Error::msg)?;
        if ![1, 2, 4].contains(&self.scale) {
            bail!("scale must be 1, 2 or 4");
        }
        if !(1..=1024).contains(&self.max_photo_mb)
            || !(1..=4096).contains(&self.max_clip_mb)
            || !(1..=600).contains(&self.max_clip_seconds)
        {
            bail!("the size and length limits are out of range");
        }
        #[cfg(feature = "onnx")]
        self.onnx_device()?;
        for name in [&self.model, &self.photo_model].into_iter().flatten() {
            if !cfg!(feature = "onnx") {
                bail!(
                    "a model is set up, but this build has no AI model support (feature \"onnx\")"
                );
            }
            if self.find_model(Some(name)).is_none() {
                bail!("the default model \"{name}\" is not in the models folder");
            }
        }
        Ok(())
    }

    #[cfg(feature = "onnx")]
    pub(crate) fn onnx_device(&self) -> Result<crate::onnx::Device> {
        <crate::onnx::Device as clap::ValueEnum>::from_str(&self.device, true)
            .map_err(|_| anyhow::anyhow!("unknown device \"{}\"", self.device))
    }

    /// The names senders may ask for: the models folder's files, without
    /// ".onnx", leaving out the video denoiser.
    pub fn model_names(&self) -> Vec<String> {
        let Some(folder) = &self.models else {
            return Vec::new();
        };
        let mut names: Vec<String> = std::fs::read_dir(folder)
            .into_iter()
            .flatten()
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.is_file() && p.extension().is_some_and(|e| e == "onnx"))
            .filter_map(|p| Some(p.file_stem()?.to_str()?.to_string()))
            .filter(|n| !n.starts_with("fastdvdnet"))
            .collect();
        names.sort();
        names
    }

    /// The model file for a name a sender gave, or the default one. `None`
    /// means the name is unknown; `Some(None)` that no model is set up (plain
    /// enlarging). Only names from the list are ever turned into files.
    pub fn find_model(&self, wanted: Option<&str>) -> Option<Option<PathBuf>> {
        let Some(name) = wanted.or(self.model.as_deref()) else {
            return Some(None);
        };
        let name = name.strip_suffix(".onnx").unwrap_or(name);
        let folder = self.models.as_ref()?;
        self.model_names()
            .iter()
            .find(|n| *n == name)
            .map(|n| Some(folder.join(format!("{n}.onnx"))))
    }

    /// `find_model` for a picture: its own usual model when there is one.
    pub fn find_photo_model(&self, wanted: Option<&str>) -> Option<Option<PathBuf>> {
        self.find_model(wanted.or(self.photo_model.as_deref()))
    }

    /// Folders that error messages must not mention.
    pub(crate) fn private_folders(&self) -> Vec<PathBuf> {
        let mut all = vec![self.data.clone()];
        all.extend(self.models.clone());
        all
    }

    pub fn log(&self, message: &str) {
        let when = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        eprintln!("{message}");
        if let Some(file) = &self.log
            && let Ok(mut f) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(file)
        {
            let _ = writeln!(f, "{when} {message}");
        }
    }
}

/// Removes what an earlier run left behind: only folders this service names.
fn clear_old_jobs(data: &Path) {
    for entry in std::fs::read_dir(data).into_iter().flatten().flatten() {
        let name = entry.file_name();
        if name.to_str().is_some_and(guard::is_job_id)
            && entry.file_type().is_ok_and(|t| t.is_dir())
        {
            let _ = std::fs::remove_dir_all(entry.path());
        }
    }
}

/// Runs the service until `stop` is set. `ready` is told the addresses once it
/// is listening. An address that is not there yet (Tailscale still starting)
/// is tried again until it is.
pub fn run(
    config: Config,
    stop: Arc<AtomicBool>,
    ready: impl FnOnce(&[SocketAddr]) + Send,
) -> Result<()> {
    config.check()?;
    std::fs::create_dir_all(&config.data)
        .with_context(|| format!("cannot use {}", config.data.display()))?;
    clear_old_jobs(&config.data);
    let key = guard::Key::new(&config.key).map_err(anyhow::Error::msg)?;
    let config = Arc::new(config);
    let clips = Arc::new(work::Clips::default());
    let (queue, jobs) = std::sync::mpsc::sync_channel(WAITING_AT_MOST);
    let worker = {
        let (config, clips, stop) = (config.clone(), clips.clone(), stop.clone());
        std::thread::spawn(move || work::run(config, jobs, clips, stop))
    };
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()?;
    let served: Result<()> = runtime.block_on(async {
        let app = web::App::new(config.clone(), key, queue, clips.clone());
        let mut listeners = Vec::new();
        for address in &config.listen {
            let mut waited = 0u32;
            loop {
                match tokio::net::TcpListener::bind(address).await {
                    Ok(l) => break listeners.push(l),
                    Err(e) if e.kind() == std::io::ErrorKind::AddrNotAvailable => {
                        if waited.is_multiple_of(30) {
                            config.log(&format!("waiting for {address}: {e}"));
                        }
                        waited += 1;
                        tokio::time::sleep(Duration::from_secs(2)).await;
                        if stop.load(Ordering::Relaxed) {
                            return Ok(());
                        }
                    }
                    Err(e) => return Err(e).context(format!("cannot listen on {address}")),
                }
            }
        }
        let addresses = listeners
            .iter()
            .map(|l| l.local_addr())
            .collect::<std::io::Result<Vec<_>>>()?;
        for a in &addresses {
            config.log(&format!("listening on http://{a}/"));
        }
        ready(&addresses);
        let router = web::router(app);
        let mut servers = tokio::task::JoinSet::new();
        for listener in listeners {
            let service = router
                .clone()
                .into_make_service_with_connect_info::<SocketAddr>();
            servers.spawn(async move { axum::serve(listener, service).await });
        }
        loop {
            tokio::select! {
                Some(ended) = servers.join_next() => {
                    ended??;
                    bail!("the service stopped listening");
                }
                () = tokio::time::sleep(Duration::from_millis(250)) => {
                    if stop.load(Ordering::Relaxed) {
                        return Ok(());
                    }
                }
            }
        }
    });
    stop.store(true, Ordering::Relaxed);
    // Uploads and downloads still under way are cut off.
    runtime.shutdown_timeout(Duration::from_secs(3));
    let _ = worker.join();
    config.log("stopped");
    served
}

#[cfg(test)]
mod tests {
    use super::*;

    fn good() -> Config {
        Config {
            key: new_key().unwrap(),
            ..Config::default()
        }
    }

    #[test]
    fn a_configuration_may_not_open_the_service_wider() {
        good().check().unwrap();
        let bad = |change: fn(&mut Config)| {
            let mut c = good();
            change(&mut c);
            c.check().unwrap_err().to_string()
        };
        assert!(bad(|c| c.listen = vec!["0.0.0.0:80".parse().unwrap()]).contains("Tailscale"));
        assert!(bad(|c| c.listen.push("192.168.0.2:80".parse().unwrap())).contains("Tailscale"));
        bad(|c| c.listen.clear());
        bad(|c| c.key.clear());
        bad(|c| c.key = "password".into());
        bad(|c| c.scale = 3);
        bad(|c| c.max_clip_seconds = 0);
        bad(|c| c.model = Some("missing".into()));
        bad(|c| c.photo_model = Some("missing".into()));
    }

    #[test]
    fn only_listed_model_names_become_files() {
        let d = tempfile::tempdir().unwrap();
        for n in ["realesr_fp16.onnx", "fastdvdnet_fp16.onnx", "notes.txt"] {
            std::fs::write(d.path().join(n), b"x").unwrap();
        }
        let c = Config {
            models: Some(d.path().to_path_buf()),
            model: Some("realesr_fp16".into()),
            ..good()
        };
        assert_eq!(c.model_names(), ["realesr_fp16"]);
        let file = Some(Some(d.path().join("realesr_fp16.onnx")));
        assert_eq!(c.find_model(None), file);
        assert_eq!(c.find_model(Some("realesr_fp16.onnx")), file);
        for no in [
            "fastdvdnet_fp16",
            "notes",
            "../realesr_fp16",
            "/etc/passwd",
            "",
        ] {
            assert_eq!(c.find_model(Some(no)), None, "{no}");
        }
        // Pictures may have a usual model of their own; a named one still wins.
        std::fs::write(d.path().join("sharp.onnx"), b"x").unwrap();
        let two = Config {
            photo_model: Some("sharp".into()),
            ..c.clone()
        };
        assert_eq!(
            two.find_photo_model(None),
            Some(Some(d.path().join("sharp.onnx")))
        );
        assert_eq!(two.find_photo_model(Some("realesr_fp16")), file);
        assert_eq!(two.find_model(None), file, "clips keep the other");
        assert_eq!(c.find_photo_model(None), file);
        let none = good();
        assert_eq!(none.find_model(None), Some(None));
        assert_eq!(none.find_model(Some("realesr_fp16")), None);
    }

    #[test]
    fn a_file_with_unknown_settings_is_refused() {
        let d = tempfile::tempdir().unwrap();
        let f = d.path().join("c.json");
        std::fs::write(&f, "\u{feff}{\"scale\": 4}").unwrap();
        assert_eq!(Config::read(&f).unwrap().scale, 4);
        std::fs::write(&f, "{\"lisen\": []}").unwrap();
        assert!(Config::read(&f).is_err());
    }

    #[test]
    fn only_the_services_own_folders_are_cleared() {
        let d = tempfile::tempdir().unwrap();
        let ours = d.path().join("0123456789abcdef0123456789abcdef");
        let theirs = d.path().join("holiday photos");
        std::fs::create_dir(&ours).unwrap();
        std::fs::create_dir(&theirs).unwrap();
        clear_old_jobs(d.path());
        assert!(!ours.exists() && theirs.exists());
    }
}
