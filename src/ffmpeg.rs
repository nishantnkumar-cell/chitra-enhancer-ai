//! ffmpeg and ffprobe as helper programs: probing a video and its frames,
//! and the command lines that decode and encode it. Frames travel through
//! pipes as raw RGB, so no picture files are written (an hour of DV is about
//! 100,000 frames).

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};
use serde_json::Value;

/// Where ffmpeg and ffprobe are.
#[derive(Clone, Debug)]
pub struct Tools {
    pub ffmpeg: PathBuf,
    pub ffprobe: PathBuf,
}

impl Tools {
    /// `ffmpeg` as given, or the one on the search path; `ffprobe` beside it
    /// when it is there, else on the search path.
    pub fn locate(ffmpeg: Option<&Path>) -> Result<Tools> {
        let ffmpeg = ffmpeg.map_or_else(|| PathBuf::from("ffmpeg"), Path::to_path_buf);
        let sibling = ffmpeg
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .map(|p| {
                p.join(if cfg!(windows) {
                    "ffprobe.exe"
                } else {
                    "ffprobe"
                })
            })
            .filter(|p| p.exists());
        let tools = Tools {
            ffprobe: sibling.unwrap_or_else(|| PathBuf::from("ffprobe")),
            ffmpeg,
        };
        for (name, p) in [("ffmpeg", &tools.ffmpeg), ("ffprobe", &tools.ffprobe)] {
            Command::new(p)
                .arg("-version")
                .output()
                .with_context(|| {
                    format!(
                        "cannot run {name} ({}): install FFmpeg (https://ffmpeg.org) or pass --ffmpeg PATH",
                        p.display()
                    )
                })?;
        }
        Ok(tools)
    }
}

/// What the first video stream and the file say about it.
#[derive(Clone, Debug, PartialEq)]
pub struct VideoInfo {
    pub width: u32,
    pub height: u32,
    pub fps_num: u32,
    pub fps_den: u32,
    /// Seconds.
    pub duration: f64,
    /// Tape-style interlaced fields (top or bottom field first).
    pub interlaced: bool,
    /// Pixel shape (DV is not square: 720x480 is shown at 4:3 or 16:9).
    pub sar: (u32, u32),
    pub has_audio: bool,
    pub frames: Option<u64>,
}

impl VideoInfo {
    pub fn fps(&self) -> f64 {
        f64::from(self.fps_num) / f64::from(self.fps_den.max(1))
    }

    /// The size of a frame once its pixels are made square, rounded to even.
    pub fn square_size(&self) -> (u32, u32) {
        let (n, d) = self.sar;
        if n == d || n == 0 || d == 0 {
            return (self.width, self.height);
        }
        let w = (f64::from(self.width) * f64::from(n) / f64::from(d) / 2.0).trunc() as u32 * 2;
        (w.max(2), self.height)
    }
}

fn ratio(s: &str) -> Option<(u32, u32)> {
    let (a, b) = s.split_once([':', '/'])?;
    Some((a.trim().parse().ok()?, b.trim().parse().ok()?))
}

/// Reads ffprobe's JSON (`-show_streams -show_format`).
pub fn parse_probe(json: &str) -> Result<VideoInfo> {
    let v: Value = serde_json::from_str(json).context("ffprobe gave no JSON")?;
    let streams = v["streams"].as_array().context("no streams in the file")?;
    let video = streams
        .iter()
        .find(|s| s["codec_type"] == "video")
        .context("the file has no video stream")?;
    let width = video["width"].as_u64().context("no width")? as u32;
    let height = video["height"].as_u64().context("no height")? as u32;
    let (fps_num, fps_den) = video["avg_frame_rate"]
        .as_str()
        .and_then(ratio)
        .filter(|&(n, d)| n > 0 && d > 0)
        .or_else(|| video["r_frame_rate"].as_str().and_then(ratio))
        .filter(|&(n, d)| n > 0 && d > 0)
        .context("no frame rate")?;
    let num = |x: &Value| x.as_str().and_then(|s| s.parse::<f64>().ok());
    let duration = num(&video["duration"])
        .or_else(|| num(&v["format"]["duration"]))
        .context("no duration")?;
    let interlaced = matches!(
        video["field_order"].as_str(),
        Some("tt" | "bb" | "tb" | "bt")
    );
    let sar = video["sample_aspect_ratio"]
        .as_str()
        .and_then(ratio)
        .filter(|&(n, d)| n > 0 && d > 0)
        .unwrap_or((1, 1));
    Ok(VideoInfo {
        width,
        height,
        fps_num,
        fps_den,
        duration,
        interlaced,
        sar,
        has_audio: streams.iter().any(|s| s["codec_type"] == "audio"),
        frames: video["nb_frames"].as_str().and_then(|s| s.parse().ok()),
    })
}

pub fn probe(tools: &Tools, input: &Path) -> Result<VideoInfo> {
    let out = Command::new(&tools.ffprobe)
        .args([
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
        ])
        .arg(input)
        .output()
        .context("cannot run ffprobe")?;
    if !out.status.success() {
        bail!(
            "ffprobe could not read {}: {}",
            input.display(),
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    parse_probe(&String::from_utf8_lossy(&out.stdout))
}

#[cfg(test)]
mod tests {
    use super::*;

    const DV: &str = r#"{"streams":[
      {"codec_type":"video","width":720,"height":480,"avg_frame_rate":"30000/1001",
       "r_frame_rate":"30000/1001","field_order":"bb","sample_aspect_ratio":"8:9",
       "duration":"61.061","nb_frames":"1830"},
      {"codec_type":"audio"}],"format":{"duration":"61.1"}}"#;

    #[test]
    fn reads_a_dv_camcorder_file() {
        let v = parse_probe(DV).unwrap();
        assert_eq!((v.width, v.height), (720, 480));
        assert_eq!((v.fps_num, v.fps_den), (30000, 1001));
        assert!(v.interlaced && v.has_audio);
        assert_eq!(v.sar, (8, 9));
        assert_eq!(v.frames, Some(1830));
        // 720 x 8/9 = 640: the picture is 4:3 once the pixels are square.
        assert_eq!(v.square_size(), (640, 480));
        assert!((v.fps() - 29.97).abs() < 0.01);
    }

    #[test]
    fn progressive_square_and_missing_values_have_defaults() {
        let v = parse_probe(
            r#"{"streams":[{"codec_type":"video","width":161,"height":99,
                "avg_frame_rate":"0/0","r_frame_rate":"25/1","field_order":"progressive",
                "sample_aspect_ratio":"N/A"}],"format":{"duration":"3.0"}}"#,
        )
        .unwrap();
        assert!(!v.interlaced && !v.has_audio);
        assert_eq!((v.fps_num, v.fps_den), (25, 1));
        assert_eq!(v.sar, (1, 1));
        assert_eq!(v.square_size(), (161, 99));
        assert_eq!(v.frames, None);
        assert!(parse_probe("{}").is_err());
        assert!(parse_probe(r#"{"streams":[{"codec_type":"audio"}]}"#).is_err());
    }
}
