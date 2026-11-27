//! The requests the service answers.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::mpsc::{SyncSender, TrySendError};
use std::time::Duration;

use axum::body::Body;
use axum::extract::{ConnectInfo, Path, Query, Request, State};
use axum::http::{StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use http_body_util::BodyExt;
use serde::Deserialize;
use serde_json::json;
use tokio::io::AsyncWriteExt;
use tokio::sync::{Semaphore, oneshot};
use tokio_util::io::ReaderStream;

use super::Config;
use super::guard::{self, Key};
use super::work::{self, ClipRequest, Clips, PhotoRequest, Work};

/// A slow or stalled upload is given up after this long.
const UPLOAD_TIME: Duration = Duration::from_secs(10 * 60);
/// How much of a refused upload is still read, so that the sender gets the
/// reason rather than a dropped connection.
const READ_AFTER_REFUSING: u64 = 256 * 1024 * 1024;
/// Uploads received at once; each can take a lot of memory or disk.
const UPLOADS_AT_ONCE: usize = 4;

pub struct App {
    pub config: Arc<Config>,
    pub key: Key,
    pub queue: SyncSender<Work>,
    pub clips: Arc<Clips>,
    pub uploads: Semaphore,
}

impl App {
    pub fn new(
        config: Arc<Config>,
        key: Key,
        queue: SyncSender<Work>,
        clips: Arc<Clips>,
    ) -> Arc<App> {
        Arc::new(App {
            config,
            key,
            queue,
            clips,
            uploads: Semaphore::new(UPLOADS_AT_ONCE),
        })
    }
}

type Shared = State<Arc<App>>;

fn refuse(status: StatusCode, message: &str) -> Response {
    (status, Json(json!({ "error": message }))).into_response()
}

pub fn router(app: Arc<App>) -> Router {
    let api = Router::new()
        .route("/v1/health", get(health))
        // Both read what is sent themselves, each with its own limit.
        .route("/v1/photo", post(photo))
        .route("/v1/clip", post(clip))
        .route("/v1/jobs/{id}", get(status).delete(forget))
        .route("/v1/jobs/{id}/result", get(result))
        .route_layer(middleware::from_fn_with_state(app.clone(), key_needed));
    Router::new()
        // The upload page holds nothing private; every request it makes is checked.
        .route("/", get(page))
        .merge(api)
        .layer(middleware::from_fn(known_network))
        .with_state(app)
}

async fn known_network(
    ConnectInfo(from): ConnectInfo<SocketAddr>,
    request: Request,
    next: Next,
) -> Response {
    if !guard::allowed_address(from.ip()) {
        return refuse(StatusCode::FORBIDDEN, "not from this network");
    }
    next.run(request).await
}

async fn key_needed(State(app): Shared, request: Request, next: Next) -> Response {
    let given = request
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok());
    if !app.key.opens(given) {
        // Slows down anyone trying keys one after another.
        tokio::time::sleep(Duration::from_millis(500)).await;
        return refuse(StatusCode::UNAUTHORIZED, "the key is missing or wrong");
    }
    next.run(request).await
}

async fn page() -> Response {
    (
        [
            (
                header::CONTENT_SECURITY_POLICY,
                "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; img-src data:; form-action 'none'; frame-ancestors 'none'",
            ),
            (header::X_CONTENT_TYPE_OPTIONS, "nosniff"),
            (header::REFERRER_POLICY, "no-referrer"),
        ],
        Html(include_str!("page.html")),
    )
        .into_response()
}

async fn health(State(app): Shared) -> Response {
    Json(json!({
        "ok": true,
        "version": env!("CARGO_PKG_VERSION"),
        "models": app.config.model_names(),
        "max_photo_mb": app.config.max_photo_mb,
        "max_clip_mb": app.config.max_clip_mb,
        "max_clip_seconds": app.config.max_clip_seconds,
    }))
    .into_response()
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Settings {
    /// 1, 2 or 4 times larger.
    scale: Option<usize>,
    /// One of the names `health` lists.
    model: Option<String>,
    /// JPEG XL distance, photos only: 1.0 looks lossless.
    quality: Option<f32>,
}

impl Settings {
    fn check(&self, config: &Config) -> Result<(usize, Option<PathBuf>, f32), &'static str> {
        let scale = self.scale.unwrap_or(config.scale);
        if ![1, 2, 4].contains(&scale) {
            return Err("scale must be 1, 2 or 4");
        }
        let quality = self.quality.unwrap_or(1.0);
        if !(0.1..=3.0).contains(&quality) {
            return Err("quality must be 0.1 to 3.0 (smaller is better)");
        }
        let model = config
            .find_model(self.model.as_deref())
            .ok_or("there is no such model")?;
        Ok((scale, model, quality))
    }
}

/// Reads away what is left of an upload, up to a point, then refuses it.
async fn refuse_upload(mut body: Body, status: StatusCode, message: &str) -> Response {
    let _ = tokio::time::timeout(Duration::from_secs(60), async {
        let mut left = READ_AFTER_REFUSING;
        while left > 0 {
            match body.frame().await {
                Some(Ok(f)) => {
                    left = left.saturating_sub(f.data_ref().map_or(0, |d| d.len() as u64));
                }
                _ => break,
            }
        }
    })
    .await;
    refuse(status, message)
}

/// A picture, held in memory: no more than `limit` bytes of it.
async fn picture(body: &mut Body, limit: usize) -> Result<Vec<u8>, (StatusCode, &'static str)> {
    let mut all = Vec::new();
    while let Some(frame) = body.frame().await {
        let frame = frame.map_err(|_| (StatusCode::BAD_REQUEST, "the upload was cut off"))?;
        let Ok(data) = frame.into_data() else {
            continue;
        };
        if all.len() + data.len() > limit {
            return Err((StatusCode::PAYLOAD_TOO_LARGE, "the picture is too large"));
        }
        all.extend_from_slice(&data);
    }
    Ok(all)
}

fn queued(app: &App, work: Work) -> Result<(), Box<Response>> {
    app.queue.try_send(work).map_err(|e| {
        Box::new(match e {
            TrySendError::Full(_) => refuse(
                StatusCode::SERVICE_UNAVAILABLE,
                "too many jobs are waiting; try again shortly",
            ),
            TrySendError::Disconnected(_) => {
                refuse(StatusCode::SERVICE_UNAVAILABLE, "the service is stopping")
            }
        })
    })
}

async fn photo(State(app): Shared, Query(wanted): Query<Settings>, mut body: Body) -> Response {
    let (scale, model, distance) = match wanted.check(&app.config) {
        Ok(v) => v,
        Err(e) => return refuse_upload(body, StatusCode::BAD_REQUEST, e).await,
    };
    let Ok(_slot) = app.uploads.try_acquire() else {
        return refuse_upload(
            body,
            StatusCode::SERVICE_UNAVAILABLE,
            "busy; try again shortly",
        )
        .await;
    };
    let limit = app.config.max_photo_mb * 1024 * 1024;
    let bytes = match tokio::time::timeout(UPLOAD_TIME, picture(&mut body, limit)).await {
        Ok(Ok(b)) => b,
        Ok(Err((status, message))) => return refuse_upload(body, status, message).await,
        Err(_) => return refuse(StatusCode::REQUEST_TIMEOUT, "the upload took too long"),
    };
    if guard::photo_kind(&bytes).is_none() {
        return refuse(
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "send a JPEG, PNG, WebP or TIFF picture",
        );
    }
    let (reply, answer) = oneshot::channel();
    let work = Work::Photo(PhotoRequest {
        bytes,
        scale,
        model,
        distance,
        reply,
    });
    if let Err(r) = queued(&app, work) {
        return *r;
    }
    match answer.await {
        Ok(Ok((jxl, s))) => (
            [
                (header::CONTENT_TYPE, "image/jxl".to_string()),
                (
                    header::CONTENT_DISPOSITION,
                    "attachment; filename=\"photo-enhanced.jxl\"".to_string(),
                ),
                (
                    header::HeaderName::from_static("x-picture-size"),
                    format!("{}x{}", s.out_size.0, s.out_size.1),
                ),
            ],
            jxl,
        )
            .into_response(),
        Ok(Err(e)) => refuse(
            StatusCode::UNPROCESSABLE_ENTITY,
            &work::public_message(&e, &app.config),
        ),
        Err(_) => refuse(StatusCode::SERVICE_UNAVAILABLE, "the service is stopping"),
    }
}

/// Writes the upload to `to`, checking its kind and size on the way.
async fn receive(
    body: &mut Body,
    folder: &std::path::Path,
    limit: u64,
) -> Result<PathBuf, (StatusCode, &'static str)> {
    const FAILED: (StatusCode, &str) = (
        StatusCode::INTERNAL_SERVER_ERROR,
        "the upload could not be saved",
    );
    let part = folder.join("upload.part");
    let mut file = tokio::fs::File::create(&part).await.map_err(|_| FAILED)?;
    let (mut start, mut size) = (Vec::new(), 0u64);
    while let Some(frame) = body.frame().await {
        let frame = frame.map_err(|_| (StatusCode::BAD_REQUEST, "the upload was cut off"))?;
        let Ok(data) = frame.into_data() else {
            continue;
        };
        size += data.len() as u64;
        if size > limit {
            return Err((StatusCode::PAYLOAD_TOO_LARGE, "the clip is too large"));
        }
        if start.len() < 512 {
            start.extend_from_slice(&data[..data.len().min(512 - start.len())]);
        }
        file.write_all(&data).await.map_err(|_| FAILED)?;
    }
    file.flush().await.map_err(|_| FAILED)?;
    drop(file);
    let kind = guard::video_kind(&start).ok_or((
        StatusCode::UNSUPPORTED_MEDIA_TYPE,
        "send an MP4, MOV, MKV, AVI, MPEG or transport stream video",
    ))?;
    let input = folder.join(format!("upload.{kind}"));
    tokio::fs::rename(&part, &input).await.map_err(|_| FAILED)?;
    Ok(input)
}

async fn clip(State(app): Shared, Query(wanted): Query<Settings>, mut body: Body) -> Response {
    let (scale, model, _) = match wanted.check(&app.config) {
        Ok(v) => v,
        Err(e) => return refuse_upload(body, StatusCode::BAD_REQUEST, e).await,
    };
    let Ok(_slot) = app.uploads.try_acquire() else {
        return refuse_upload(
            body,
            StatusCode::SERVICE_UNAVAILABLE,
            "busy; try again shortly",
        )
        .await;
    };
    app.clips.tidy(std::time::Instant::now());
    let Ok(id) = guard::random_hex::<16>() else {
        return refuse(StatusCode::INTERNAL_SERVER_ERROR, "no job number");
    };
    let folder = app.config.data.join(&id);
    if tokio::fs::create_dir(&folder).await.is_err() {
        return refuse(
            StatusCode::INTERNAL_SERVER_ERROR,
            "the upload could not be saved",
        );
    }
    // Removes the folder unless the job is handed over.
    let cleanup = Cleanup(Some(folder.clone()));
    let limit = app.config.max_clip_mb as u64 * 1024 * 1024;
    let input = match tokio::time::timeout(UPLOAD_TIME, receive(&mut body, &folder, limit)).await {
        Ok(Ok(p)) => p,
        Ok(Err((status, message))) => return refuse_upload(body, status, message).await,
        Err(_) => return refuse(StatusCode::REQUEST_TIMEOUT, "the upload took too long"),
    };
    app.clips.add(&id, folder);
    let work = Work::Clip(ClipRequest {
        id: id.clone(),
        input,
        scale,
        model,
    });
    if let Err(r) = queued(&app, work) {
        app.clips.discard(&id);
        return *r;
    }
    cleanup.keep();
    (StatusCode::ACCEPTED, Json(json!({ "job": id }))).into_response()
}

struct Cleanup(Option<PathBuf>);

impl Cleanup {
    fn keep(mut self) {
        self.0 = None;
    }
}

impl Drop for Cleanup {
    fn drop(&mut self) {
        // Also when the sender goes away in the middle of an upload.
        if let Some(folder) = self.0.take() {
            let _ = std::fs::remove_dir_all(folder);
        }
    }
}

async fn status(State(app): Shared, Path(id): Path<String>) -> Response {
    match app.clips.state(&id) {
        Some(s) => Json(s).into_response(),
        None => refuse(StatusCode::NOT_FOUND, "there is no such job"),
    }
}

async fn forget(State(app): Shared, Path(id): Path<String>) -> Response {
    if app.clips.remove(&id) {
        StatusCode::NO_CONTENT.into_response()
    } else {
        refuse(
            StatusCode::NOT_FOUND,
            "there is no finished job with that number",
        )
    }
}

async fn result(State(app): Shared, Path(id): Path<String>) -> Response {
    let Some(path) = app.clips.result(&id) else {
        return refuse(
            StatusCode::NOT_FOUND,
            "there is no finished video for that job",
        );
    };
    let Ok(file) = tokio::fs::File::open(path).await else {
        return refuse(StatusCode::NOT_FOUND, "the video is no longer there");
    };
    let length = file.metadata().await.map(|m| m.len()).unwrap_or(0);
    (
        [
            (header::CONTENT_TYPE, "video/mp4".to_string()),
            (header::CONTENT_LENGTH, length.to_string()),
            (
                header::CONTENT_DISPOSITION,
                "attachment; filename=\"clip-enhanced.mp4\"".to_string(),
            ),
        ],
        Body::from_stream(ReaderStream::new(file)),
    )
        .into_response()
}
