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
    /// DVD only: a title number or "all"; None = the longest title.
    pub title: Option<String>,
    /// Pixels cut from left, right, top, bottom.
    pub crop: Option<[u32; 4]>,
    /// Colour-cast correction (blue, red).
    pub colour_shift: Option<(i32, i32)>,
    /// Lines the colour sits too low (blue, red).
    pub colour_align: Option<(i32, i32)>,
    /// 0..=100: softens bright outlines from camera sharpening.
    pub dehalo: u32,
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
            title: None,
            crop: None,
            colour_shift: None,
            colour_align: None,
            dehalo: 0,
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

pub fn is_video(p: &Path) -> bool {
    matches!(
        p.extension()
            .map(|e| e.to_string_lossy().to_ascii_lowercase())
            .as_deref(),
        Some(
            "mp4"
                | "m4v"
                | "mov"
                | "mkv"
                | "avi"
                | "mpg"
                | "mpeg"
                | "ts"
                | "m2ts"
                | "mts"
                | "wmv"
                | "flv"
                | "webm"
                | "3gp"
                | "dv"
                | "vob"
        )
    )
}

/// `clip.avi` -> `clip-enhanced.mp4` next to it; `photo.jpg` -> `photo-enhanced.jxl`.
/// Never the input itself.
pub fn default_output(input: &Path) -> PathBuf {
    if let Some(src) = crate::disc::source(input) {
        let stem = src.file_stem().unwrap_or_default().to_string_lossy();
        let name = format!(
            "{}-enhanced.mp4",
            if stem.is_empty() { "dvd".into() } else { stem }
        );
        // A disc image: next to it. A drive or disc folder is often read-only,
        // so the result goes to the home folder.
        return match std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")) {
            Some(home) if !src.is_file() => PathBuf::from(home).join(name),
            _ => src.with_file_name(name),
        };
    }
    if input.is_dir() {
        // A whole folder: results go into a folder next to it.
        return input.with_file_name(format!(
            "{}-enhanced",
            input.file_name().unwrap_or_default().to_string_lossy()
        ));
    }
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
        let disc = crate::disc::source(&self.input).is_some();
        if !self.input.is_file() && !self.input.is_dir() && !disc {
            return Some("The chosen file or folder does not exist.".into());
        }
        if self.output.as_os_str().is_empty() {
            return Some("Choose where to save the result.".into());
        }
        if self.output == self.input {
            return Some("The result must not overwrite the original.".into());
        }
        if self.output.is_file() && self.input.is_dir() && !disc {
            return Some("A folder needs a folder to save into, not a file.".into());
        }
        if self.output.is_file() && !self.overwrite && (disc || !self.input.is_dir()) {
            return Some("The result file exists: tick \"Replace\" or pick another name.".into());
        }
        if self.five_frame && self.five_frame_model.is_none() {
            return Some("Pick the five-frame denoising model, or turn that option off.".into());
        }
        None
    }

    /// Takes over what an analysis suggested. Everything stays editable.
    /// The film grain that will be added: the chosen amount, or else what the
    /// command line does on its own (3 after the simple denoise filter, which
    /// is not used with five-frame denoising; otherwise none).
    pub fn effective_grain(&self) -> u32 {
        self.grain
            .unwrap_or(if self.five_frame || self.denoise == "off" {
                0
            } else {
                3
            })
    }

    pub fn apply(
        &mut self,
        s: &crate::analyze::Suggestion,
        upscaler: Option<PathBuf>,
        denoiser: Option<PathBuf>,
    ) {
        self.scale = s.scale;
        self.model = if s.ai_upscaler { upscaler } else { None };
        self.five_frame = s.five_frame && denoiser.is_some();
        if denoiser.is_some() {
            self.five_frame_model = denoiser;
        }
        self.sigma = s.sigma;
        self.denoise = s.denoise.clone();
        self.deinterlace = if s.deinterlace { "on" } else { "off" }.into();
        self.double_rate = s.double_rate;
        self.grain = Some(s.grain);
        self.crf = Some(s.crf);
        self.crop = s.crop;
        self.colour_align = s.colour_align;
        self.dehalo = s.dehalo;
    }

    /// Arguments that analyse the input and print suggestions.
    pub fn analyze_args(&self) -> Vec<String> {
        let mut a = vec![
            "-i".into(),
            self.input.to_string_lossy().into_owned(),
            "--analyze".into(),
            "--progress".into(),
        ];
        if crate::disc::source(&self.input).is_some()
            && let Some(t) = &self.title
            && t != "all"
        {
            a.extend(["--title".into(), t.clone()]);
        }
        a
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
        // A folder gets both sets of options; each file uses the ones that fit it.
        let disc = crate::disc::source(&self.input).is_some();
        let folder = self.input.is_dir() && !disc;
        if disc && let Some(t) = &self.title {
            a.extend(["--title".into(), t.clone()]);
        }
        if is_photo(&self.input) || folder {
            if let Some(m) = &self.model_compressed {
                a.extend(["--model-compressed".into(), s(m)]);
            }
            a.extend(["--jxl-distance".into(), format!("{:.2}", self.jxl_distance)]);
        }
        if self.overwrite && !folder {
            a.push("--force".into());
        }
        if !is_photo(&self.input) {
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
            if let Some([l, r, t, b]) = self.crop {
                a.extend(["--crop".into(), format!("{l}:{r}:{t}:{b}")]);
            }
            if let Some((u, v)) = self.colour_shift {
                a.extend(["--colour-shift".into(), format!("{u}:{v}")]);
            }
            if let Some((u, v)) = self.colour_align {
                a.extend(["--colour-align".into(), format!("{u}:{v}")]);
            }
            if self.dehalo > 0 {
                a.extend(["--dehalo".into(), self.dehalo.to_string()]);
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

/// One argument, quoted so that a shell takes it literally. File names can hold
/// `;`, `$(...)`, backticks or quotes, and a printed command may be pasted.
/// Windows output is for PowerShell, elsewhere for POSIX shells and fish.
pub fn quote(arg: &str) -> String {
    let plain = !arg.is_empty()
        && !arg.starts_with('~')
        && arg.chars().all(|c| {
            c.is_ascii_alphanumeric() || "_-./:=+,@".contains(c) || (cfg!(windows) && c == '\\')
        });
    if plain {
        arg.to_string()
    } else if cfg!(windows) {
        format!("'{}'", arg.replace('\'', "''"))
    } else {
        format!("'{}'", arg.replace('\'', r"'\''"))
    }
}

/// The whole command for `args` (progress reporting left out).
pub fn command_line(args: &[String]) -> String {
    let words = args.iter().filter(|a| *a != "--progress").map(|a| quote(a));
    std::iter::once("chitra-enhance".to_string())
        .chain(words)
        .collect::<Vec<_>>()
        .join(" ")
}

/// `ANALYSIS {...}` -> the report.
pub fn parse_analysis(line: &str) -> Option<crate::analyze::Report> {
    serde_json::from_str(line.strip_prefix("ANALYSIS ")?).ok()
}

/// `TITLE 2 3600` -> (title 2, 3600 seconds).
pub fn parse_title(line: &str) -> Option<(u32, f64)> {
    let mut it = line.strip_prefix("TITLE ")?.split_whitespace();
    Some((it.next()?.parse().ok()?, it.next()?.parse().ok()?))
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
    fn folders_are_batches_with_both_option_sets() {
        let d = tempfile::tempdir().unwrap();
        let s = Settings {
            input: d.path().to_path_buf(),
            output: d.path().join("out"),
            jxl_distance: 0.5,
            ..Settings::default()
        };
        let a = s.args();
        assert!(a.contains(&"--jxl-distance".to_string()), "photo options");
        assert!(a.contains(&"--codec".to_string()), "video options");
        assert!(
            default_output(d.path())
                .to_string_lossy()
                .ends_with("-enhanced")
        );
        assert!(is_video(Path::new("a.MKV")) && is_video(Path::new("t.dv")));
        assert!(!is_video(Path::new("a.jpg")) && !is_video(Path::new("a.txt")));
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
    fn the_grain_shown_is_the_grain_added() {
        let mut s = Settings {
            denoise: "light".into(),
            ..Settings::default()
        };
        assert_eq!(s.effective_grain(), 3);
        s.denoise = "off".into();
        assert_eq!(s.effective_grain(), 0);
        s.denoise = "light".into();
        s.five_frame = true;
        assert_eq!(s.effective_grain(), 0);
        s.grain = Some(5);
        assert_eq!(s.effective_grain(), 5);
        s.five_frame = false;
        s.grain = Some(0);
        assert_eq!(s.effective_grain(), 0);
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
        s.input = d.path().to_path_buf();
        s.output = d.path().join("out");
        assert_eq!(s.problem(), None, "a folder is a valid input");
        s.output = PathBuf::new();
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
    fn a_dvd_gets_a_title_option_and_suggestions_can_be_applied() {
        let d = tempfile::tempdir().unwrap();
        let iso = d.path().join("wedding.iso");
        std::fs::write(&iso, b"x").unwrap();
        let mut s = Settings {
            input: iso.clone(),
            output: default_output(&iso),
            title: Some("2".into()),
            ..Settings::default()
        };
        assert_eq!(s.output, d.path().join("wedding-enhanced.mp4"));
        assert_eq!(s.problem(), None);
        let a = s.args();
        let at = |a: &[String], f: &str| a.iter().position(|x| x == f).map(|i| a[i + 1].clone());
        assert_eq!(at(&a, "--title").as_deref(), Some("2"));
        assert_eq!(at(&s.analyze_args(), "--title").as_deref(), Some("2"));
        let sug = crate::analyze::Suggestion {
            deinterlace: true,
            double_rate: true,
            denoise: "light".into(),
            five_frame: false,
            sigma: None,
            scale: 2,
            ai_upscaler: true,
            grain: 0,
            crf: 18,
            crop: Some([10, 6, 0, 0]),
            colour_align: Some((2, 3)),
            dehalo: 40,
        };
        s.apply(&sug, Some("/m/up.onnx".into()), Some("/m/dn.onnx".into()));
        let a = s.args();
        assert_eq!(at(&a, "--model").as_deref(), Some("/m/up.onnx"));
        assert_eq!(at(&a, "--deinterlace").as_deref(), Some("on"));
        assert_eq!(at(&a, "--denoise").as_deref(), Some("light"));
        assert_eq!(at(&a, "--grain").as_deref(), Some("0"));
        assert_eq!(at(&a, "--crf").as_deref(), Some("18"));
        assert_eq!(at(&a, "--crop").as_deref(), Some("10:6:0:0"));
        assert_eq!(at(&a, "--colour-align").as_deref(), Some("2:3"));
        assert_eq!(at(&a, "--dehalo").as_deref(), Some("40"));
        s.colour_shift = Some((-4, 2));
        assert_eq!(at(&s.args(), "--colour-shift").as_deref(), Some("-4:2"));
        assert!(a.contains(&"--double-rate".to_string()));
        assert!(!a.contains(&"--temporal-model".to_string()));
        assert_eq!(parse_title("TITLE 2 3600"), Some((2, 3600.0)));
        assert_eq!(parse_title("x"), None);
    }

    #[test]
    fn printed_commands_quote_anything_a_shell_would_act_on() {
        assert_eq!(quote("/v/old.avi"), "/v/old.avi");
        assert_eq!(quote("--crf"), "--crf");
        for nasty in [
            "a;b", "$(id)", "`id`", "a b", "x|y", "&", "~", "", "*.mp4", "a\nb",
        ] {
            let q = quote(nasty);
            assert!(q.starts_with('\'') && q.ends_with('\''), "{nasty:?} -> {q}");
        }
        let line = command_line(&["-i".into(), "a;b".into(), "--progress".into()]);
        assert_eq!(line, "chitra-enhance -i 'a;b'");
    }

    /// A real shell reads back exactly the arguments that went in.
    #[cfg(unix)]
    #[test]
    fn a_shell_reads_quoted_arguments_back_unchanged() {
        let args = [
            "it's",
            "a;echo INJECTED",
            "$(echo INJECTED)",
            "`id`",
            "two  spaces",
            "-x",
            "~root",
            "\"q\"",
            "back\\slash",
            "new\nline",
            "*",
        ];
        let words: Vec<String> = args.iter().map(|a| quote(a)).collect();
        let out = std::process::Command::new("sh")
            .arg("-c")
            .arg(format!("printf '%s\\0' {}", words.join(" ")))
            .output()
            .unwrap();
        let back: Vec<&str> = std::str::from_utf8(&out.stdout)
            .unwrap()
            .split_terminator('\0')
            .collect();
        assert_eq!(back, args);
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
