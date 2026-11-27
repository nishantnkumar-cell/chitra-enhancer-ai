//! Sending pictures and clips to a service: `chitra-enhance remote`.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use ureq::Agent;

/// Where the service is and the key for it; kept on this computer.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Remote {
    /// "100.101.102.103:8765" or "name:8765".
    pub server: String,
    pub key: String,
}

impl Remote {
    pub fn read(file: &Path) -> Option<Remote> {
        serde_json::from_slice(&std::fs::read(file).ok()?).ok()
    }

    /// Writes the settings so that only this user can read the key.
    pub fn save(&self, file: &Path) -> Result<()> {
        if let Some(folder) = file.parent() {
            std::fs::create_dir_all(folder)?;
        }
        let mut options = std::fs::OpenOptions::new();
        options.write(true).create(true).truncate(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        options
            .open(file)?
            .write_all(&serde_json::to_vec_pretty(self)?)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(file, std::fs::Permissions::from_mode(0o600))?;
        }
        Ok(())
    }
}

/// What to ask for; `None` leaves the service's usual choice.
#[derive(Clone, Debug, Default)]
pub struct Wanted {
    pub scale: Option<usize>,
    pub model: Option<String>,
}

pub struct Client {
    agent: Agent,
    base: String,
    authorization: String,
}

#[derive(Deserialize)]
struct Refusal {
    error: String,
}

#[derive(Deserialize)]
struct Accepted {
    job: String,
}

#[derive(Deserialize)]
#[serde(tag = "state", rename_all = "lowercase")]
enum State {
    Queued,
    Running { done: u64, total: u64 },
    Done,
    Failed { message: String },
}

impl Client {
    pub fn new(remote: &Remote) -> Client {
        let agent: Agent = Agent::config_builder()
            // The service's own explanation is more use than the status code.
            .http_status_as_error(false)
            // The key goes only to the address it was given for.
            .max_redirects(0)
            .proxy(None)
            .timeout_connect(Some(Duration::from_secs(20)))
            .timeout_global(Some(Duration::from_secs(30 * 60)))
            .build()
            .into();
        let server = remote.server.trim_end_matches('/');
        let base = if server.starts_with("http://") || server.starts_with("https://") {
            server.to_string()
        } else {
            format!("http://{server}")
        };
        Client {
            agent,
            base,
            authorization: format!("Bearer {}", remote.key.trim()),
        }
    }

    fn url(&self, path: &str, wanted: Option<&Wanted>) -> String {
        let mut url = format!("{}{path}", self.base);
        if let Some(w) = wanted {
            let mut join = '?';
            if let Some(s) = w.scale {
                url.push_str(&format!("{join}scale={s}"));
                join = '&';
            }
            if let Some(m) = &w.model {
                // Model names are file names; anything unusual is escaped.
                let escaped: String = m
                    .bytes()
                    .map(|b| {
                        if b.is_ascii_alphanumeric() || b"-_.".contains(&b) {
                            char::from(b).to_string()
                        } else {
                            format!("%{b:02X}")
                        }
                    })
                    .collect();
                url.push_str(&format!("{join}model={escaped}"));
            }
        }
        url
    }

    /// The answer if the service accepted, or its reason for refusing.
    fn accept(
        &self,
        answer: Result<ureq::http::Response<ureq::Body>, ureq::Error>,
    ) -> Result<ureq::Body> {
        let answer =
            answer.with_context(|| format!("cannot reach the service at {}", self.base))?;
        let status = answer.status();
        let body = answer.into_body();
        if status.is_success() {
            return Ok(body);
        }
        let why = json::<Refusal>(body)
            .map(|r| r.error)
            .unwrap_or_else(|_| status.to_string());
        Err(anyhow!("the service refused: {why}"))
    }

    fn get(&self, path: &str) -> Result<ureq::Body> {
        self.accept(
            self.agent
                .get(self.url(path, None))
                .header("Authorization", &self.authorization)
                .call(),
        )
    }

    fn send(&self, path: &str, wanted: &Wanted, file: &Path) -> Result<ureq::Body> {
        let data =
            std::fs::File::open(file).with_context(|| format!("cannot read {}", file.display()))?;
        self.accept(
            self.agent
                .post(self.url(path, Some(wanted)))
                .header("Authorization", &self.authorization)
                .send(data),
        )
    }

    /// What the service offers, as it describes itself.
    pub fn about(&self) -> Result<serde_json::Value> {
        json(self.get("/v1/health")?)
    }

    /// Sends a picture and saves the JPEG XL that comes back.
    pub fn photo(&self, file: &Path, wanted: &Wanted, out: &Path) -> Result<()> {
        save(self.send("/v1/photo", wanted, file)?, out)
    }

    /// Sends a clip, waits for it, saves the result and has the service
    /// remove it. `progress` hears how far it is.
    pub fn clip(
        &self,
        file: &Path,
        wanted: &Wanted,
        out: &Path,
        mut progress: impl FnMut(u64, u64),
    ) -> Result<()> {
        let accepted: Accepted = json(self.send("/v1/clip", wanted, file)?)?;
        let job = format!("/v1/jobs/{}", accepted.job);
        loop {
            match json(self.get(&job)?)? {
                State::Done => break,
                State::Failed { message } => bail!("the service could not restore it: {message}"),
                State::Running { done, total } if total > 0 => progress(done, total),
                State::Running { .. } | State::Queued => {}
            }
            std::thread::sleep(Duration::from_millis(1000));
        }
        save(self.get(&format!("{job}/result"))?, out)?;
        // It would expire anyway, so a failure here loses nothing.
        let _ = self
            .agent
            .delete(self.url(&job, None))
            .header("Authorization", &self.authorization)
            .call();
        Ok(())
    }
}

fn json<T: serde::de::DeserializeOwned>(body: ureq::Body) -> Result<T> {
    let mut text = Vec::new();
    body.into_reader().take(1 << 20).read_to_end(&mut text)?;
    serde_json::from_slice(&text).context("the answer is not from this kind of service")
}

/// Writes beside the target first, so a failure never leaves half a file.
fn save(body: ureq::Body, out: &Path) -> Result<()> {
    let mut part = out.as_os_str().to_owned();
    part.push(".part");
    let part = PathBuf::from(part);
    let written = (|| -> Result<()> {
        let mut file = std::fs::File::create(&part)?;
        let mut from = body.into_reader().take(8 << 30);
        std::io::copy(&mut from, &mut file)?;
        file.flush()?;
        std::fs::rename(&part, out)?;
        Ok(())
    })();
    if written.is_err() {
        let _ = std::fs::remove_file(&part);
    }
    written.with_context(|| format!("cannot save {}", out.display()))
}
