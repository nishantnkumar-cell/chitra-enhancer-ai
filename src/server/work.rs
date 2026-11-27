//! The one worker that does the enhancing, a job at a time, and the list of
//! clip jobs. The upscaling model stays loaded between photos.

use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use serde::Serialize;
use tokio::sync::oneshot;

use super::Config;
use crate::engine::{Engine, Lanczos, Rescaled};
use crate::ffmpeg::{self, Tools};
use crate::{job, photo};

/// How long a finished clip waits to be collected.
pub const KEPT_FOR: Duration = Duration::from_secs(60 * 60);
/// Finished clips kept at once; the oldest goes first.
pub const KEPT_AT_MOST: usize = 20;

pub struct PhotoRequest {
    pub bytes: Vec<u8>,
    pub scale: usize,
    pub model: Option<PathBuf>,
    pub distance: f32,
    pub reply: oneshot::Sender<Result<(Vec<u8>, photo::Summary)>>,
}

pub struct ClipRequest {
    pub id: String,
    pub input: PathBuf,
    pub scale: usize,
    pub model: Option<PathBuf>,
}

pub enum Work {
    Photo(PhotoRequest),
    Clip(ClipRequest),
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "state", rename_all = "lowercase")]
pub enum State {
    Queued,
    Running { done: u64, total: u64 },
    Done,
    Failed { message: String },
}

pub struct Clip {
    pub state: State,
    /// The folder holding this job's files.
    pub folder: PathBuf,
    finished: Option<Instant>,
}

/// The clip jobs the service knows about.
#[derive(Default)]
pub struct Clips(Mutex<HashMap<String, Clip>>);

impl Clips {
    fn lock(&self) -> MutexGuard<'_, HashMap<String, Clip>> {
        self.0.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn add(&self, id: &str, folder: PathBuf) {
        self.lock().insert(
            id.to_string(),
            Clip {
                state: State::Queued,
                folder,
                finished: None,
            },
        );
    }

    pub fn state(&self, id: &str) -> Option<State> {
        self.lock().get(id).map(|c| c.state.clone())
    }

    /// The finished video of a job that is done.
    pub fn result(&self, id: &str) -> Option<PathBuf> {
        self.lock()
            .get(id)
            .filter(|c| c.state == State::Done)
            .map(|c| c.folder.join(RESULT))
    }

    fn set(&self, id: &str, state: State) {
        if let Some(c) = self.lock().get_mut(id) {
            if matches!(state, State::Done | State::Failed { .. }) {
                c.finished = Some(Instant::now());
            }
            c.state = state;
        }
    }

    /// Forgets a job and removes its files. A job still waiting or running
    /// cannot be removed.
    pub fn remove(&self, id: &str) -> bool {
        let mut all = self.lock();
        if all.get(id).is_none_or(|c| c.finished.is_none()) {
            return false;
        }
        if let Some(c) = all.remove(id) {
            let _ = std::fs::remove_dir_all(c.folder);
        }
        true
    }

    /// Forgets a job that never started, with its files.
    pub fn discard(&self, id: &str) {
        if let Some(c) = self.lock().remove(id) {
            let _ = std::fs::remove_dir_all(c.folder);
        }
    }

    /// Drops results nobody collected in time, and the oldest beyond the limit.
    pub fn tidy(&self, now: Instant) {
        let mut done: Vec<(String, Instant)> = self
            .lock()
            .iter()
            .filter_map(|(id, c)| Some((id.clone(), c.finished?)))
            .collect();
        done.sort_by_key(|(_, at)| *at);
        let extra = done.len().saturating_sub(KEPT_AT_MOST);
        for (n, (id, at)) in done.into_iter().enumerate() {
            if n < extra || now.duration_since(at) >= KEPT_FOR {
                self.remove(&id);
            }
        }
    }
}

pub const RESULT: &str = "result.mp4";

/// An error worded for the person who sent the file: no folders of this
/// computer in it.
pub fn public_message(e: &anyhow::Error, config: &Config) -> String {
    let mut text = format!("{e:#}");
    for folder in config.private_folders() {
        let folder = folder.display().to_string();
        if !folder.is_empty() {
            text = text.replace(&folder, "…");
        }
    }
    text.chars().take(400).collect()
}

struct Engines {
    config: Arc<Config>,
    loaded: HashMap<(Option<PathBuf>, usize), Box<dyn Engine>>,
}

impl Engines {
    fn get(&mut self, model: Option<PathBuf>, scale: usize) -> Result<&mut Box<dyn Engine>> {
        let key = (model, scale);
        if !self.loaded.contains_key(&key) {
            // One at a time: the card's memory is not shared between models.
            self.loaded.clear();
            let engine: Box<dyn Engine> = match &key.0 {
                #[cfg(feature = "onnx")]
                Some(m) => Rescaled::wrap(
                    Box::new(crate::onnx::OnnxEngine::load(
                        m,
                        self.config.onnx_device()?,
                        self.config.vram_gb * 1024,
                        None,
                    )?),
                    scale,
                ),
                #[cfg(not(feature = "onnx"))]
                Some(_) => bail!("this build has no AI model support"),
                None => Rescaled::wrap(Box::new(Lanczos::new(scale)), scale),
            };
            self.config.log(&format!("loaded: {}", engine.describe()));
            self.loaded.insert(key.clone(), engine);
        }
        self.loaded.get_mut(&key).context("the model is not loaded")
    }
}

/// Takes jobs until every sender is gone or `stop` is set.
pub fn run(config: Arc<Config>, jobs: Receiver<Work>, clips: Arc<Clips>, stop: Arc<AtomicBool>) {
    let mut engines = Engines {
        config: config.clone(),
        loaded: HashMap::new(),
    };
    while !stop.load(Ordering::Relaxed) {
        let work = match jobs.recv_timeout(Duration::from_millis(500)) {
            Ok(w) => w,
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
        };
        match work {
            Work::Photo(request) => {
                let started = Instant::now();
                // A fault in one picture must not end the service.
                let made = catch_unwind(AssertUnwindSafe(|| enhance(&mut engines, &request)))
                    .unwrap_or_else(|_| {
                        engines.loaded.clear();
                        Err(anyhow!("the picture could not be processed"))
                    });
                match &made {
                    Ok((bytes, s)) => config.log(&format!(
                        "photo {}x{} -> {}x{}, {} KB in {:.1} s",
                        s.in_size.0,
                        s.in_size.1,
                        s.out_size.0,
                        s.out_size.1,
                        bytes.len() / 1024,
                        started.elapsed().as_secs_f64()
                    )),
                    Err(e) => {
                        engines.loaded.clear();
                        config.log(&format!("photo failed: {e:#}"));
                    }
                }
                let _ = request.reply.send(made);
            }
            Work::Clip(request) => {
                // The clip runs as its own program and needs the card's memory.
                engines.loaded.clear();
                clips.set(&request.id, State::Running { done: 0, total: 0 });
                let made =
                    catch_unwind(AssertUnwindSafe(|| clip(&config, &clips, &stop, &request)))
                        .unwrap_or_else(|_| Err(anyhow!("the clip could not be processed")));
                // The upload is not kept once it has been dealt with.
                let _ = std::fs::remove_file(&request.input);
                match made {
                    Ok(()) => {
                        config.log(&format!("clip {} done", &request.id[..8]));
                        clips.set(&request.id, State::Done);
                    }
                    Err(e) => {
                        config.log(&format!("clip {} failed: {e:#}", &request.id[..8]));
                        clips.set(
                            &request.id,
                            State::Failed {
                                message: public_message(&e, &config),
                            },
                        );
                    }
                }
            }
        }
    }
}

fn enhance(engines: &mut Engines, r: &PhotoRequest) -> Result<(Vec<u8>, photo::Summary)> {
    let folder = tempfile::tempdir_in(&engines.config.data)?;
    let options = photo::Options {
        // Only used to make sure the output is another file.
        input: folder.path().join("upload"),
        output: folder.path().join("result.jxl"),
        jxl_distance: r.distance,
        force: false,
    };
    let engine = engines.get(r.model.clone(), r.scale)?;
    let summary = photo::run(&options, &r.bytes, engine.as_mut())?;
    Ok((std::fs::read(&options.output)?, summary))
}

/// The enhancer itself, set up to run on the uploaded clip with these
/// arguments.
fn enhancer(config: &Config, args: &[String]) -> Result<Command> {
    let mut c = Command::new(match &config.program {
        Some(p) => p.clone(),
        None => std::env::current_exe()?,
    });
    c.args(args).stdin(Stdio::null());
    if let Some(f) = &config.ffmpeg {
        c.arg("--ffmpeg").arg(f);
    }
    if let Some(r) = &config.runtime {
        c.env("ORT_DYLIB_PATH", r);
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        c.creation_flags(0x0800_0000); // no console window
    }
    Ok(c)
}

/// Looks at the clip, takes the suggested settings and restores it.
fn clip(config: &Config, clips: &Clips, stop: &AtomicBool, r: &ClipRequest) -> Result<()> {
    let tools = Tools::locate(config.ffmpeg.as_deref())?;
    let info = ffmpeg::probe(&tools, &r.input).context("this video cannot be read")?;
    if !(info.duration > 0.0 && info.duration <= f64::from(config.max_clip_seconds)) {
        bail!(
            "the clip is {:.0} s long; the limit is {} s",
            info.duration,
            config.max_clip_seconds
        );
    }
    let mut settings = job::Settings {
        input: r.input.clone(),
        output: r.input.with_file_name(RESULT),
        device: config.device.clone(),
        vram_gb: config.vram_gb,
        ..job::Settings::default()
    };
    let looked = enhancer(config, &settings.analyze_args())?
        .stderr(Stdio::null())
        .output()?;
    let report = String::from_utf8_lossy(&looked.stdout)
        .lines()
        .find_map(job::parse_analysis)
        .context("this video cannot be analysed")?;
    settings.apply(&report.suggestion, None, None);
    settings.scale = r.scale;
    // The service's model, whatever the analysis found on its own: it looks in
    // the folders of the account it runs as.
    settings.model = r.model.clone().filter(|_| r.scale > 1);
    let mut child = enhancer(config, &settings.args())?
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("cannot start the enhancer")?;
    let (out, err) = (child.stdout.take(), child.stderr.take());
    std::thread::scope(|s| {
        // The last thing it said explains a failure.
        let last = s.spawn(move || {
            err.into_iter()
                .flat_map(|e| BufReader::new(e).split(b'\n'))
                .map_while(Result::ok)
                .map(|l| String::from_utf8_lossy(&l).trim().to_string())
                .filter(|l| !l.is_empty())
                .fold(String::new(), |kept, l| {
                    if l.starts_with("error") || !kept.starts_with("error") {
                        l
                    } else {
                        kept
                    }
                })
        });
        s.spawn(|| {
            for line in out
                .into_iter()
                .flat_map(|o| BufReader::new(o).lines())
                .map_while(Result::ok)
            {
                if let Some((done, total)) = job::parse_progress(&line) {
                    clips.set(&r.id, State::Running { done, total });
                }
            }
        });
        let status = loop {
            if stop.load(Ordering::Relaxed) {
                let _ = child.kill();
            }
            if let Some(status) = child.try_wait()? {
                break status;
            }
            std::thread::sleep(Duration::from_millis(200));
        };
        let said = last.join().unwrap_or_default();
        if !status.success() {
            bail!("{said}");
        }
        Ok(())
    })?;
    if !Path::new(&settings.output).is_file() {
        bail!("no result was made");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(n: usize) -> String {
        format!("{n:032x}")
    }

    #[test]
    fn uncollected_results_expire_and_only_finished_jobs_can_be_removed() {
        let d = tempfile::tempdir().unwrap();
        let clips = Clips::default();
        let folder = |n: usize| {
            let f = d.path().join(id(n));
            std::fs::create_dir(&f).unwrap();
            f
        };
        clips.add(&id(1), folder(1));
        assert!(!clips.remove(&id(1)), "still waiting");
        clips.tidy(Instant::now() + KEPT_FOR * 2);
        assert!(clips.state(&id(1)).is_some(), "never dropped while waiting");
        clips.set(&id(1), State::Done);
        assert_eq!(
            clips.result(&id(1)),
            Some(d.path().join(id(1)).join(RESULT))
        );
        clips.tidy(Instant::now());
        assert!(clips.state(&id(1)).is_some());
        clips.tidy(Instant::now() + KEPT_FOR * 2);
        assert!(clips.state(&id(1)).is_none());
        assert!(!d.path().join(id(1)).exists(), "its files go too");
        assert!(!clips.remove(&id(9)));

        for n in 10..10 + KEPT_AT_MOST + 3 {
            clips.add(&id(n), folder(n));
            clips.set(&id(n), State::Done);
        }
        clips.tidy(Instant::now());
        assert!(clips.state(&id(10)).is_none() && clips.state(&id(12)).is_none());
        assert!(clips.state(&id(13)).is_some());
    }
}
