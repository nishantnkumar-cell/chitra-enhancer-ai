//! What the GUI asks for, turned into a command line for the `chitra-enhance`
//! program, plus reading its progress lines. No window code lives here, so it is
//! tested everywhere.

use std::path::{Path, PathBuf};

#[derive(Clone, Debug, PartialEq)]
pub struct Settings {
    pub input: PathBuf,
    pub output: PathBuf,
    /// 1..=4
    pub scale: usize,
    pub model: Option<PathBuf>,
    /// For JPEGs that were heavily compressed.
    pub model_compressed: Option<PathBuf>,
    pub five_frame: bool,
    pub five_frame_model: Option<PathBuf>,
    /// Noise level (0..=100) for the five-frame model; None = measure it.
    pub sigma: Option<f32>,
    /// "off", "light", "medium", "strong" (used when five_frame is off).
    pub denoise: String,
    /// "auto", "on", "off"
    pub deinterlace: String,
    pub codec: String,
    pub grain: Option<u32>,
    pub device: String,
    pub vram_gb: usize,
    pub jxl_distance: f32,
    pub overwrite: bool,
    /// 1..=30 seconds tried by a preview.
    pub preview_secs: u32,
    /// Video quality, lower is better; None = the codec's default.
    pub crf: Option<u32>,
    /// Largest tile in pixels (128, 256, 512); None = chosen from the GPU memory.
    pub tile: Option<usize>,
    /// Seconds per resumable chunk; None = 30.
    pub chunk_secs: Option<u32>,
    /// Keep both fields of interlaced video (twice the frames).
    pub double_rate: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            input: PathBuf::new(),
            output: PathBuf::new(),
            scale: 2,
            model: None,
            model_compressed: None,
            five_frame: false,
            five_frame_model: None,
            sigma: None,
            denoise: "medium".into(),
            deinterlace: "auto".into(),
            codec: "h265".into(),
            grain: None,
            device: "auto".into(),
            vram_gb: 8,
            jxl_distance: 1.0,
            overwrite: false,
            preview_secs: 5,
            crf: None,
            tile: None,
            chunk_secs: None,
            double_rate: false,
        }
    }
}

pub fn is_photo(p: &Path) -> bool {
    matches!(
        p.extension()
            .map(|e| e.to_string_lossy().to_ascii_lowercase())
            .as_deref(),
        Some("jpg" | "jpeg" | "png" | "webp" | "bmp" | "gif" | "tif" | "tiff")
    )
}

/// `clip.avi` -> `clip-enhanced.mp4` next to it; `photo.jpg` -> `photo-enhanced.jxl`.
/// Never the input itself.
pub fn default_output(input: &Path) -> PathBuf {
    let stem = input.file_stem().unwrap_or_default().to_string_lossy();
    let ext = if is_photo(input) { "jxl" } else { "mp4" };
    input.with_file_name(format!("{stem}-enhanced.{ext}"))
}

/// Where a preview clip goes: `clip-preview.mp4` next to the input. It is not
/// the real output name, so a preview can never overwrite a finished result.
pub fn preview_output(input: &Path) -> PathBuf {
    let stem = input.file_stem().unwrap_or_default().to_string_lossy();
    input.with_file_name(format!("{stem}-preview.mp4"))
}

impl Settings {
    /// Why the job cannot start yet, if it cannot.
    pub fn problem(&self) -> Option<String> {
        if self.input.as_os_str().is_empty() {
            return Some("Choose a video or photo first.".into());
        }
        if !self.input.is_file() {
            return Some("The chosen file does not exist.".into());
        }
        if self.output.as_os_str().is_empty() {
            return Some("Choose where to save the result.".into());
        }
        if self.output == self.input {
            return Some("The result must not overwrite the original.".into());
        }
        if self.output.exists() && !self.overwrite && is_photo(&self.input) {
            return Some("The result file exists: tick \"Replace\" or pick another name.".into());
        }
        if self.five_frame && self.five_frame_model.is_none() {
            return Some("Pick the five-frame denoising model, or turn that option off.".into());
        }
        None
    }

    /// Arguments for a preview: the same settings on `preview_secs` seconds,
    /// written to the preview name so a finished result is never touched.
    pub fn preview_args(&self) -> Vec<String> {
        let mut s = self.clone();
        s.output = preview_output(&s.input);
        let mut a = s.args();
        a.extend([
            "--preview".into(),
            self.preview_secs.clamp(1, 30).to_string(),
            "--force".into(),
        ]);
        a
    }

    /// Arguments that only measure the noise (see `parse_noise`).
    pub fn measure_args(&self) -> Vec<String> {
        vec![
            "-i".into(),
            self.input.to_string_lossy().into_owned(),
            "-o".into(),
            preview_output(&self.input).to_string_lossy().into_owned(),
            "--deinterlace".into(),
            self.deinterlace.clone(),
            "--measure-noise".into(),
        ]
    }

    /// Arguments for `chitra-enhance` (without the program name).
    pub fn args(&self) -> Vec<String> {
        let s = |p: &Path| p.to_string_lossy().into_owned();
        let mut a: Vec<String> = vec![
            "-i".into(),
            s(&self.input),
            "-o".into(),
            s(&self.output),
            "--scale".into(),
            self.scale.to_string(),
            "--progress".into(),
        ];
        if let Some(m) = &self.model {
            a.extend(["--model".into(), s(m)]);
        }
        a.extend(["--device".into(), self.device.clone()]);
        a.extend(["--vram-budget".into(), self.vram_gb.to_string()]);
        if let Some(t) = self.tile {
            a.extend(["--tile-size".into(), t.to_string()]);
        }
        if is_photo(&self.input) {
            if let Some(m) = &self.model_compressed {
                a.extend(["--model-compressed".into(), s(m)]);
            }
            a.extend(["--jxl-distance".into(), format!("{:.2}", self.jxl_distance)]);
            if self.overwrite {
                a.push("--force".into());
            }
        } else {
            let denoise = if self.five_frame {
                "off"
            } else {
                &self.denoise
            };
            a.extend(["--denoise".into(), denoise.into()]);
            if self.five_frame
                && let Some(m) = &self.five_frame_model
            {
                a.extend(["--temporal-model".into(), s(m)]);
                if let Some(v) = self.sigma {
                    a.extend(["--temporal-sigma".into(), format!("{v:.0}")]);
                }
            }
            a.extend(["--deinterlace".into(), self.deinterlace.clone()]);
            a.extend(["--codec".into(), self.codec.clone()]);
            if let Some(c) = self.crf {
                a.extend(["--crf".into(), c.to_string()]);
            }
            if let Some(c) = self.chunk_secs {
                a.extend(["--chunk-seconds".into(), c.to_string()]);
            }
            if self.double_rate {
                a.push("--double-rate".into());
            }
            if let Some(g) = self.grain {
                a.extend(["--grain".into(), g.to_string()]);
            }
        }
        a
    }
}

/// `PROGRESS 120 900` -> (120, 900).
pub fn parse_progress(line: &str) -> Option<(u64, u64)> {
    let mut it = line.strip_prefix("PROGRESS ")?.split_whitespace();
    let done = it.next()?.parse().ok()?;
    let total = it.next()?.parse().ok()?;
    Some((done, total))
}

/// `NOISE 8.2 13` -> (measured 8.2, suggested 13); `NOISE none` -> None.
pub fn parse_noise(line: &str) -> Option<(f32, f32)> {
    let mut it = line.strip_prefix("NOISE ")?.split_whitespace();
    Some((it.next()?.parse().ok()?, it.next()?.parse().ok()?))
}

/// Where the models are usually kept: `<data dir>/chitra-enhance/models`.
pub fn models_dir() -> Option<PathBuf> {
    let base = if cfg!(windows) {
        std::env::var_os("LOCALAPPDATA").map(PathBuf::from)?
    } else if cfg!(target_os = "macos") {
        PathBuf::from(std::env::var_os("HOME")?).join("Library/Application Support")
    } else {
        std::env::var_os("XDG_DATA_HOME")
            .map(PathBuf::from)
            .or_else(|| Some(PathBuf::from(std::env::var_os("HOME")?).join(".local/share")))?
    };
    Some(base.join("chitra-enhance").join("models"))
}

/// The first `.onnx` in `dir` whose name starts with `prefix`, preferring FP16
/// on a GPU and FP32 on the CPU.
pub fn find_model(dir: &Path, prefix: &str, prefer_fp16: bool) -> Option<PathBuf> {
    let mut found: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            let n = p.file_name().unwrap_or_default().to_string_lossy();
            n.starts_with(prefix) && n.ends_with(".onnx")
        })
        .collect();
    found.sort();
    let want = if prefer_fp16 { "fp16" } else { "fp32" };
    found
        .iter()
        .find(|p| p.to_string_lossy().contains(want))
        .or(found.first())
        .cloned()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn video() -> Settings {
        Settings {
            input: "/v/old.avi".into(),
            output: "/v/old-enhanced.mp4".into(),
            ..Settings::default()
        }
    }

    #[test]
    fn outputs_sit_next_to_the_input_and_never_equal_it() {
        assert_eq!(
            default_output(Path::new("/v/old tape.avi")),
            Path::new("/v/old tape-enhanced.mp4")
        );
        assert_eq!(
            default_output(Path::new("/p/a.jpg")),
            Path::new("/p/a-enhanced.jxl")
        );
        assert_eq!(
            default_output(Path::new("/p/a.png")),
            Path::new("/p/a-enhanced.jxl")
        );
    }

    #[test]
    fn previews_have_their_own_name() {
        let p = preview_output(Path::new("/v/old tape.avi"));
        assert_eq!(p, Path::new("/v/old tape-preview.mp4"));
        assert_ne!(p, default_output(Path::new("/v/old tape.avi")));
    }

    #[test]
    fn a_video_job_becomes_the_expected_command_line() {
        let a = video().args();
        assert_eq!(
            a,
            [
                "-i",
                "/v/old.avi",
                "-o",
                "/v/old-enhanced.mp4",
                "--scale",
                "2",
                "--progress",
                "--device",
                "auto",
                "--vram-budget",
                "8",
                "--denoise",
                "medium",
                "--deinterlace",
                "auto",
                "--codec",
                "h265"
            ]
        );
    }

    #[test]
    fn five_frame_denoising_replaces_the_simple_filter() {
        let mut s = video();
        s.five_frame = true;
        s.five_frame_model = Some("/m/fastdvdnet_fp16.onnx".into());
        s.sigma = Some(18.0);
        s.grain = Some(4);
        let a = s.args();
        let at = |f: &str| a.iter().position(|x| x == f).map(|i| a[i + 1].as_str());
        assert_eq!(at("--denoise"), Some("off"));
        assert_eq!(at("--temporal-model"), Some("/m/fastdvdnet_fp16.onnx"));
        assert_eq!(at("--temporal-sigma"), Some("18"));
        assert_eq!(at("--grain"), Some("4"));
        s.sigma = None;
        assert!(
            !s.args().contains(&"--temporal-sigma".to_string()),
            "auto means no flag"
        );
    }

    #[test]
    fn photo_jobs_use_photo_options_only() {
        let s = Settings {
            input: "/p/a.jpg".into(),
            output: "/p/a-enhanced.jxl".into(),
            model: Some("/m/up.onnx".into()),
            model_compressed: Some("/m/fix.onnx".into()),
            overwrite: true,
            jxl_distance: 0.5,
            ..Settings::default()
        };
        let a = s.args();
        assert!(a.contains(&"--model-compressed".to_string()));
        assert!(a.contains(&"--force".to_string()));
        assert!(a.contains(&"0.50".to_string()));
        assert!(!a.contains(&"--denoise".to_string()));
        assert!(!a.contains(&"--codec".to_string()));
    }

    #[test]
    fn problems_are_explained_before_starting() {
        let d = tempfile::tempdir().unwrap();
        let f = d.path().join("c.avi");
        std::fs::write(&f, b"x").unwrap();
        let mut s = Settings::default();
        assert!(s.problem().unwrap().contains("Choose a video"));
        s.input = d.path().join("missing.avi");
        assert!(s.problem().unwrap().contains("does not exist"));
        s.input = f.clone();
        assert!(s.problem().unwrap().contains("where to save"));
        s.output = f.clone();
        assert!(s.problem().unwrap().contains("overwrite the original"));
        s.output = d.path().join("o.mp4");
        assert_eq!(s.problem(), None);
        s.five_frame = true;
        assert!(s.problem().unwrap().contains("five-frame"));
    }

    #[test]
    fn progress_lines_are_read_and_other_text_is_not() {
        assert_eq!(parse_progress("PROGRESS 120 900"), Some((120, 900)));
        assert_eq!(parse_progress("PROGRESS x 9"), None);
        assert_eq!(parse_progress("Done: 5 frames"), None);
    }

    #[test]
    fn every_automatic_choice_can_be_overridden_within_reason() {
        let mut s = video();
        s.crf = Some(18);
        s.tile = Some(256);
        s.chunk_secs = Some(60);
        s.double_rate = true;
        let a = s.args();
        let at = |f: &str| a.iter().position(|x| x == f).map(|i| a[i + 1].as_str());
        assert_eq!(at("--crf"), Some("18"));
        assert_eq!(at("--tile-size"), Some("256"));
        assert_eq!(at("--chunk-seconds"), Some("60"));
        assert!(a.contains(&"--double-rate".to_string()));
        // Left automatic, none of them is passed.
        let plain = video().args();
        for f in [
            "--crf",
            "--tile-size",
            "--chunk-seconds",
            "--double-rate",
            "--temporal-sigma",
        ] {
            assert!(!plain.contains(&f.to_string()), "{f}");
        }
    }

    #[test]
    fn previews_are_capped_at_30_seconds_and_never_use_the_real_output() {
        let mut s = video();
        s.preview_secs = 500;
        let a = s.preview_args();
        let at = |f: &str| a.iter().position(|x| x == f).map(|i| a[i + 1].as_str());
        assert_eq!(at("--preview"), Some("30"));
        assert_eq!(
            Path::new(at("-o").unwrap()),
            Path::new("/v/old-preview.mp4"),
            "compared as paths: Windows writes the separator differently"
        );
        assert!(a.contains(&"--force".to_string()));
        s.preview_secs = 0;
        assert!(s.preview_args().contains(&"1".to_string()));
    }

    #[test]
    fn the_noise_line_is_read() {
        assert_eq!(parse_noise("NOISE 8.2 13"), Some((8.2, 13.0)));
        assert_eq!(parse_noise("NOISE none"), None);
        assert_eq!(parse_noise("Done"), None);
    }

    #[test]
    fn models_are_found_by_prefix_and_precision() {
        let d = tempfile::tempdir().unwrap();
        for n in [
            "fastdvdnet_fp16.onnx",
            "fastdvdnet_fp32.onnx",
            "fastdvdnet.pth",
            "other.onnx",
        ] {
            std::fs::write(d.path().join(n), b"").unwrap();
        }
        let pick = |p: bool| find_model(d.path(), "fastdvdnet", p).unwrap();
        assert!(pick(true).ends_with("fastdvdnet_fp16.onnx"));
        assert!(pick(false).ends_with("fastdvdnet_fp32.onnx"));
        assert!(find_model(d.path(), "nothing", true).is_none());
    }
}
