//! The service, spoken to the way other computers will. No AI model: the plain
//! upscaler stands in, so this runs everywhere (the clip needs ffmpeg).
#![cfg(feature = "server")]

use std::net::SocketAddr;
use std::path::Path;
use std::process::Command;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::channel;
use std::thread::JoinHandle;
use std::time::Duration;

use chitra_enhance::server::client::{Client, Remote, Wanted};
use chitra_enhance::server::{self, Config};

struct Service {
    address: SocketAddr,
    key: String,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<anyhow::Result<()>>>,
    data: tempfile::TempDir,
}

impl Service {
    fn start(change: impl FnOnce(&mut Config)) -> Service {
        let data = tempfile::tempdir().unwrap();
        let key = server::new_key().unwrap();
        let mut config = Config {
            listen: vec!["127.0.0.1:0".parse().unwrap()],
            key: key.clone(),
            data: data.path().join("jobs"),
            program: Some(env!("CARGO_BIN_EXE_chitra-enhance").into()),
            log: Some(data.path().join("log.txt")),
            ..Config::default()
        };
        change(&mut config);
        let stop = Arc::new(AtomicBool::new(false));
        let (tell, told) = channel();
        let asked = stop.clone();
        let thread = std::thread::spawn(move || {
            server::run(config, asked, move |a| tell.send(a[0]).unwrap())
        });
        let address = told
            .recv_timeout(Duration::from_secs(20))
            .expect("the service starts");
        Service {
            address,
            key,
            stop,
            thread: Some(thread),
            data,
        }
    }

    fn client(&self, key: &str) -> Client {
        Client::new(&Remote {
            server: self.address.to_string(),
            key: key.to_string(),
        })
    }

    /// A raw request: status and text.
    fn raw(&self, method: &str, path: &str, key: Option<&str>, body: &[u8]) -> (u16, String) {
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .http_status_as_error(false)
            .proxy(None)
            .build()
            .into();
        let url = format!("http://{}{path}", self.address);
        let auth = key.map(|k| format!("Bearer {k}")).unwrap_or_default();
        let answer = match method {
            "GET" => agent.get(&url).header("Authorization", &auth).call(),
            "DELETE" => agent.delete(&url).header("Authorization", &auth).call(),
            _ => agent.post(&url).header("Authorization", &auth).send(body),
        }
        .expect("an answer");
        let status = answer.status().as_u16();
        let text = answer.into_body().read_to_string().unwrap_or_default();
        (status, text)
    }

    /// Nothing sent to the service is still on disk.
    fn nothing_left(&self) -> bool {
        std::fs::read_dir(self.data.path().join("jobs"))
            .unwrap()
            .next()
            .is_none()
    }
}

impl Drop for Service {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            let result = t.join().expect("no fault");
            if !std::thread::panicking() {
                result.expect("a clean stop");
            }
        }
    }
}

fn jpeg(path: &Path, w: u32, h: u32) {
    let picture = image::RgbImage::from_fn(w, h, |x, y| {
        image::Rgb([(x * 3) as u8, (y * 5) as u8, ((x + y) * 2) as u8])
    });
    picture.save(path).unwrap();
}

fn is_jxl(bytes: &[u8]) -> bool {
    bytes.starts_with(&[0xff, 0x0a]) || bytes.starts_with(b"\0\0\0\x0cJXL \r\n\x87\n")
}

#[test]
fn a_jpeg_comes_back_enlarged_as_jpeg_xl_and_only_with_the_key() {
    let s = Service::start(|_| {});
    let d = tempfile::tempdir().unwrap();
    let (input, out) = (d.path().join("in.jpg"), d.path().join("out.jxl"));
    jpeg(&input, 64, 48);
    let wanted = Wanted::default();
    s.client(&s.key).photo(&input, &wanted, &out).unwrap();
    assert!(is_jxl(&std::fs::read(&out).unwrap()));
    assert!(s.nothing_left());

    // Without the right key nothing is answered, whatever is asked.
    let wrong = server::new_key().unwrap();
    let other = d.path().join("no.jxl");
    let refused = s.client(&wrong).photo(&input, &wanted, &other);
    assert!(refused.unwrap_err().to_string().contains("key"));
    assert!(!other.exists());
    for (method, path) in [
        ("GET", "/v1/health"),
        ("POST", "/v1/photo"),
        ("POST", "/v1/clip"),
        ("GET", "/v1/jobs/0123456789abcdef0123456789abcdef"),
        ("DELETE", "/v1/jobs/0123456789abcdef0123456789abcdef"),
        ("GET", "/v1/jobs/0123456789abcdef0123456789abcdef/result"),
    ] {
        assert_eq!(s.raw(method, path, None, b"x").0, 401, "{method} {path}");
        assert_eq!(s.raw(method, path, Some(&wrong), b"x").0, 401, "{path}");
    }
    // The upload page needs none, and holds nothing private.
    let (status, page) = s.raw("GET", "/", None, b"");
    assert_eq!(status, 200);
    assert!(page.contains("Chitra Enhancer AI") && !page.contains(&s.key));
    let (status, about) = s.raw("GET", "/v1/health", Some(&s.key), b"");
    assert_eq!(status, 200, "{about}");
    assert!(about.contains("\"ok\":true"));
    // Keys never reach the log.
    let log = std::fs::read_to_string(s.data.path().join("log.txt")).unwrap();
    assert!(log.contains("photo 64x48 -> 128x96"), "{log}");
    assert!(!log.contains(&s.key) && !log.contains(&wrong));
}

#[test]
fn unsuitable_uploads_are_refused_and_the_service_carries_on() {
    let s = Service::start(|c| {
        c.max_photo_mb = 1;
        c.max_clip_mb = 1;
    });
    let key = Some(s.key.as_str());
    let jpeg_start: &[u8] = &[0xff, 0xd8, 0xff, 0xe0];
    for (path, body, status) in [
        ("/v1/photo", b"hello".to_vec(), 415),
        ("/v1/photo", jpeg_start.to_vec(), 422),
        ("/v1/photo", [jpeg_start, &vec![0; 2 << 20]].concat(), 413),
        ("/v1/photo?scale=3", jpeg_start.to_vec(), 400),
        ("/v1/photo?model=..%2Fsecret", jpeg_start.to_vec(), 400),
        ("/v1/photo?output=%2Ftmp%2Fx", jpeg_start.to_vec(), 400),
        ("/v1/photo?quality=99", jpeg_start.to_vec(), 400),
        (
            "/v1/clip",
            b"#EXTM3U\n#EXTINF:1,\nfile:///etc/passwd\n".to_vec(),
            415,
        ),
        (
            "/v1/clip",
            [&[0, 0, 1, 0xba][..], &vec![0; 2 << 20]].concat(),
            413,
        ),
    ] {
        let (got, text) = s.raw("POST", path, key, &body);
        assert_eq!(got, status, "{path}: {text}");
        assert!(
            !text.contains(s.data.path().to_str().unwrap()),
            "no folders in answers: {text}"
        );
    }
    assert!(s.nothing_left());
    for path in ["/v1/jobs/..", "/v1/jobs/0123456789abcdef0123456789abcdef"] {
        assert_eq!(s.raw("GET", path, key, b"").0, 404, "{path}");
    }
    // Still working.
    let d = tempfile::tempdir().unwrap();
    let (input, out) = (d.path().join("in.jpg"), d.path().join("out.jxl"));
    jpeg(&input, 32, 32);
    s.client(&s.key)
        .photo(
            &input,
            &Wanted {
                scale: Some(4),
                model: None,
            },
            &out,
        )
        .unwrap();
}

#[test]
fn a_short_clip_is_restored_collected_and_removed() {
    if Command::new("ffmpeg").arg("-version").output().is_err() {
        return;
    }
    let s = Service::start(|c| c.max_clip_seconds = 4);
    let d = tempfile::tempdir().unwrap();
    let make = |name: &str, seconds: u32| {
        let p = d.path().join(name);
        let ok = Command::new("ffmpeg")
            .args(["-v", "error", "-y", "-f", "lavfi", "-i"])
            .arg(format!("testsrc2=size=160x120:rate=25:duration={seconds}"))
            .args(["-c:v", "libx264", "-pix_fmt", "yuv420p"])
            .arg(&p)
            .status()
            .unwrap();
        assert!(ok.success());
        p
    };
    let client = s.client(&s.key);
    let (short, out) = (make("short.mp4", 2), d.path().join("out.mp4"));
    client
        .clip(&short, &Wanted::default(), &out, |_, _| {})
        .unwrap();
    let size = Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", "v:0"])
        .args(["-show_entries", "stream=width,height", "-of", "csv=p=0"])
        .arg(&out)
        .output()
        .unwrap();
    assert_eq!(String::from_utf8_lossy(&size.stdout).trim(), "320,240");
    assert!(s.nothing_left(), "upload and result are removed");

    let long = make("long.mp4", 6);
    let refused = client
        .clip(
            &long,
            &Wanted::default(),
            &d.path().join("no.mp4"),
            |_, _| {},
        )
        .unwrap_err()
        .to_string();
    assert!(refused.contains("limit is 4 s"), "{refused}");
    assert!(!refused.contains(s.data.path().to_str().unwrap()));
    assert!(s.nothing_left(), "a failed job is removed too");
}
