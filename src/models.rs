//! Fetching model files and finding ONNX Runtime.
//!
//! The model files are hosted on this project's `models-v1` release and every
//! download is checked against the SHA-256 built in below; a file that does not
//! match is deleted, never used. Nothing is downloaded unless asked for.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use sha2::{Digest, Sha256};

const BASE: &str =
    "https://github.com/nishantnkumar-cell/chitra-enhancer-ai/releases/download/models-v1";

pub struct Model {
    pub file: &'static str,
    pub sha256: &'static str,
    pub bytes: u64,
    pub what: &'static str,
}

pub const MODELS: &[Model] = &[
    Model {
        file: "realesr-general-x4v3_fp16.onnx",
        sha256: "157cb97194ea8304a7fb9594628cfd41de4be87f9958550765a935745db7899c",
        bytes: 2_444_614,
        what: "upscaler 4x (Real-ESRGAN compact), for GPUs",
    },
    Model {
        file: "realesr-general-x4v3_fp32.onnx",
        sha256: "6efee9a2f025d6e0bb58122aa44af00ba3d81260d076707add06559f371207a2",
        bytes: 4_866_365,
        what: "upscaler 4x (Real-ESRGAN compact), for the CPU",
    },
    Model {
        file: "fastdvdnet_fp16.onnx",
        sha256: "fe39dbb006ddd000851f2c1f239dab5e1fbebe695ba7edf64542bbb8eefc9175",
        bytes: 5_011_120,
        what: "five-frame video denoiser (FastDVDnet), for GPUs",
    },
    Model {
        file: "fastdvdnet_fp32.onnx",
        sha256: "f64b966f9219a486e800e19d7a17effcc4e8cff099475272137db9dc30a5d738",
        bytes: 9_948_246,
        what: "five-frame video denoiser (FastDVDnet), for the CPU",
    },
];

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

pub fn sha256_file(p: &Path) -> Result<String> {
    let mut f = std::fs::File::open(p)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 16];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(hex(&h.finalize()))
}

/// True if the file is there and is exactly the expected one.
pub fn is_installed(dir: &Path, m: &Model) -> bool {
    let p = dir.join(m.file);
    p.metadata().is_ok_and(|md| md.len() == m.bytes) && sha256_file(&p).is_ok_and(|h| h == m.sha256)
}

/// Downloads `m` into `dir` (resuming nothing: the files are small), checking
/// its hash before it gets its real name.
pub fn fetch(dir: &Path, m: &Model) -> Result<PathBuf> {
    std::fs::create_dir_all(dir)?;
    let dest = dir.join(m.file);
    let part = dir.join(format!("{}.part", m.file));
    let url = format!("{BASE}/{}", m.file);
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .https_only(true)
        .timeout_global(Some(std::time::Duration::from_secs(900)))
        .timeout_connect(Some(std::time::Duration::from_secs(20)))
        .build()
        .into();
    let resp = agent
        .get(&url)
        .call()
        .with_context(|| format!("cannot download {url}"))?;
    let mut body = resp.into_body().into_reader().take(m.bytes + 1);
    let mut out = std::fs::File::create(&part)?;
    let mut hash = Sha256::new();
    let mut total = 0u64;
    let mut buf = vec![0u8; 1 << 16];
    loop {
        let n = body.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hash.update(&buf[..n]);
        out.write_all(&buf[..n])?;
        total += n as u64;
    }
    drop(out);
    let got = hex(&hash.finalize());
    if total != m.bytes || got != m.sha256 {
        let _ = std::fs::remove_file(&part);
        bail!(
            "{} did not match the expected checksum (got {total} bytes, sha256 {got}); deleted",
            m.file
        );
    }
    std::fs::rename(&part, &dest)?;
    Ok(dest)
}

/// Picks the models named (a file name or a part of one), or all of them.
pub fn select(names: &[String]) -> Result<Vec<&'static Model>> {
    if names.is_empty() || names.iter().any(|n| n == "all") {
        return Ok(MODELS.iter().collect());
    }
    let mut out = Vec::new();
    for n in names {
        let hits: Vec<_> = MODELS
            .iter()
            .filter(|m| m.file.contains(n.as_str()))
            .collect();
        if hits.is_empty() {
            bail!("no model matches \"{n}\" (see `chitra-enhance models list`)");
        }
        out.extend(hits);
    }
    out.dedup_by_key(|m| m.file);
    Ok(out)
}

/// What a model file turned out to be.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ModelKind {
    /// One frame in, `scale` times larger out (scale 1: a restorer such as a
    /// denoiser or JPEG-artifact remover).
    Upscaler { scale: usize },
    /// Five frames plus a noise map in, the cleaned middle frame out.
    Denoiser5,
}

impl ModelKind {
    pub fn describe(self) -> String {
        match self {
            ModelKind::Upscaler { scale: 1 } => "single-frame restorer (same size)".into(),
            ModelKind::Upscaler { scale } => format!("upscaler {scale}x"),
            ModelKind::Denoiser5 => "five-frame video denoiser".into(),
        }
    }
}

/// A file name that is safe to create in the models folder: letters, digits,
/// `-`, `_`, `.`; ends in `.onnx`.
pub fn library_name(wanted: &str) -> Result<String> {
    let stem = wanted.strip_suffix(".onnx").unwrap_or(wanted);
    if stem.is_empty()
        || stem.len() > 80
        || stem.starts_with('.')
        || !stem
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
    {
        bail!("\"{wanted}\" is not a usable model name (letters, digits, - _ . only, up to 80)");
    }
    let name = format!("{stem}.onnx");
    if MODELS.iter().any(|m| m.file == name) {
        bail!("{name} is the name of a built-in model; choose another");
    }
    Ok(name)
}

/// Copies a checked model into the library under `name`. Refuses to replace an
/// existing file unless `force`.
pub fn add_to_library(dir: &Path, src: &Path, name: &str, force: bool) -> Result<PathBuf> {
    let name = library_name(name)?;
    std::fs::create_dir_all(dir)?;
    let dest = dir.join(&name);
    if dest.exists() && !force {
        bail!(
            "{} already exists; use --force to replace it",
            dest.display()
        );
    }
    let part = dir.join(format!("{name}.part"));
    std::fs::copy(src, &part).with_context(|| format!("cannot copy {}", src.display()))?;
    std::fs::rename(&part, &dest)?;
    Ok(dest)
}

/// Removes a model you added. The built-in ones are not removed this way.
pub fn remove_custom(dir: &Path, name: &str) -> Result<PathBuf> {
    let name = library_name(name)?;
    let p = dir.join(&name);
    if !p.is_file() {
        bail!("{name} is not in the library");
    }
    std::fs::remove_file(&p)?;
    Ok(p)
}

/// Every `.onnx` in the folder: the built-in ones that are installed and the
/// ones you added, sorted by name.
pub fn library(dir: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "onnx") && p.is_file())
        .collect();
    v.sort();
    v
}

/// A guess at whether a graphics processor will run the models, to pick the
/// FP16 (GPU) or FP32 (CPU) file. Windows and macOS almost always have a usable
/// one; on Linux only an NVIDIA card is used.
pub fn likely_gpu() -> bool {
    if cfg!(any(windows, target_os = "macos")) {
        true
    } else {
        Path::new("/dev/nvidia0").exists()
    }
}

/// Where ONNX Runtime usually lives. `ORT_DYLIB_PATH` always wins; this is the
/// fallback so most people never have to set it.
pub fn find_runtime() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("ORT_DYLIB_PATH") {
        return Some(PathBuf::from(p));
    }
    let lib = if cfg!(windows) {
        "onnxruntime.dll"
    } else if cfg!(target_os = "macos") {
        "libonnxruntime.dylib"
    } else {
        "libonnxruntime.so"
    };
    let mut dirs: Vec<PathBuf> = Vec::new();
    if let Ok(exe) = std::env::current_exe()
        && let Some(d) = exe.parent()
    {
        dirs.push(d.to_path_buf());
    }
    if let Some(m) = crate::job::models_dir().and_then(|m| m.parent().map(|p| p.join("runtime"))) {
        dirs.push(m.clone());
        dirs.push(m.join("lib"));
    }
    for d in [
        "/usr/lib",
        "/usr/lib64",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/local/lib",
        "/opt/onnxruntime/lib",
        "/opt/homebrew/lib",
        "/usr/local/opt/onnxruntime/lib",
    ] {
        dirs.push(d.into());
    }
    for d in dirs {
        let p = d.join(lib);
        if p.is_file() {
            return Some(p);
        }
        // Versioned names such as libonnxruntime.so.1.31.0.
        if let Ok(rd) = std::fs::read_dir(&d) {
            let mut v: Vec<PathBuf> = rd
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| {
                    p.file_name()
                        .is_some_and(|n| n.to_string_lossy().starts_with(lib))
                })
                .collect();
            v.sort();
            if let Some(p) = v.pop() {
                return Some(p);
            }
        }
    }
    None
}

/// NVIDIA's TensorRT RTX plugin: `CHITRA_TENSORRT_PLUGIN` names it, or it sits
/// beside the ONNX Runtime library.
pub fn find_tensorrt_plugin() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("CHITRA_TENSORRT_PLUGIN") {
        return Some(PathBuf::from(p)).filter(|p| p.is_file());
    }
    beside(&find_runtime()?)
}

fn beside(runtime: &Path) -> Option<PathBuf> {
    let name = if cfg!(windows) {
        "onnxruntime_providers_nv_tensorrt_rtx.dll"
    } else {
        "libonnxruntime_providers_nv_tensorrt_rtx.so"
    };
    // macOS has no such plugin.
    (!cfg!(target_os = "macos"))
        .then(|| runtime.with_file_name(name))
        .filter(|p| p.is_file())
}

pub const RUNTIME_HELP: &str = "ONNX Runtime was not found. Install it, or point ORT_DYLIB_PATH at it:\n  \
    Arch/CachyOS: sudo pacman -S onnxruntime-cpu (or onnxruntime-cuda for NVIDIA)\n  \
    Debian/Ubuntu or any Linux: pip install onnxruntime (or onnxruntime-gpu), then set ORT_DYLIB_PATH to\n    \
    .../site-packages/onnxruntime/capi/libonnxruntime.so.*\n  \
    Windows: the onnxruntime.dll from the Microsoft.ML.OnnxRuntime.DirectML NuGet package, next to chitra-enhance.exe\n  \
    macOS: brew install onnxruntime\n\
    Put the library in <data dir>/chitra-enhance/runtime/ and it is found automatically.";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_tensorrt_plugin_is_looked_for_beside_the_runtime() {
        let d = tempfile::tempdir().unwrap();
        let runtime = d.path().join("onnxruntime.dll");
        assert_eq!(beside(&runtime), None);
        let name = if cfg!(windows) {
            "onnxruntime_providers_nv_tensorrt_rtx.dll"
        } else {
            "libonnxruntime_providers_nv_tensorrt_rtx.so"
        };
        std::fs::write(d.path().join(name), b"x").unwrap();
        assert_eq!(beside(&runtime).is_some(), !cfg!(target_os = "macos"));
    }

    #[test]
    fn the_manifest_is_well_formed() {
        for m in MODELS {
            assert_eq!(m.sha256.len(), 64, "{}", m.file);
            assert!(m.sha256.chars().all(|c| c.is_ascii_hexdigit()));
            assert!(m.file.ends_with(".onnx") && m.bytes > 1_000_000);
        }
        let mut names: Vec<_> = MODELS.iter().map(|m| m.file).collect();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), MODELS.len());
    }

    #[test]
    fn names_select_models_and_unknown_names_are_refused() {
        assert_eq!(select(&[]).unwrap().len(), MODELS.len());
        assert_eq!(select(&["fastdvdnet".into()]).unwrap().len(), 2);
        let one = select(&["fastdvdnet_fp16".into()]).unwrap();
        assert_eq!(one.len(), 1);
        assert!(select(&["nope".into()]).is_err());
    }

    #[test]
    fn installed_means_exact_size_and_hash() {
        let d = tempfile::tempdir().unwrap();
        let m = &MODELS[0];
        assert!(!is_installed(d.path(), m));
        std::fs::write(d.path().join(m.file), vec![0u8; m.bytes as usize]).unwrap();
        assert!(!is_installed(d.path(), m), "right size, wrong content");
    }

    #[test]
    fn library_names_are_safe_and_never_shadow_built_in_models() {
        assert_eq!(library_name("my-model_v2").unwrap(), "my-model_v2.onnx");
        assert_eq!(library_name("x.onnx").unwrap(), "x.onnx");
        for bad in [
            "",
            ".hidden",
            "../evil",
            "a/b",
            "a b",
            "ü",
            "fastdvdnet_fp16",
        ] {
            assert!(library_name(bad).is_err(), "{bad:?}");
        }
        assert!(library_name(&"a".repeat(81)).is_err());
    }

    #[test]
    fn adding_copies_refuses_to_replace_and_removing_only_touches_the_library() {
        let d = tempfile::tempdir().unwrap();
        let (lib, src) = (d.path().join("lib"), d.path().join("mine.onnx"));
        std::fs::write(&src, b"model bytes").unwrap();
        let p = add_to_library(&lib, &src, "mine", false).unwrap();
        assert_eq!(std::fs::read(&p).unwrap(), b"model bytes");
        assert!(add_to_library(&lib, &src, "mine", false).is_err());
        std::fs::write(&src, b"newer").unwrap();
        add_to_library(&lib, &src, "mine", true).unwrap();
        assert_eq!(std::fs::read(&p).unwrap(), b"newer");
        assert_eq!(library(&lib), vec![p.clone()]);
        assert!(remove_custom(&lib, "nope").is_err());
        assert!(remove_custom(&lib, "../mine").is_err());
        remove_custom(&lib, "mine").unwrap();
        assert!(library(&lib).is_empty());
        assert!(src.is_file(), "the original is never touched");
    }

    #[test]
    fn hashing_matches_the_known_digest() {
        let d = tempfile::tempdir().unwrap();
        let p = d.path().join("abc");
        std::fs::write(&p, b"abc").unwrap();
        assert_eq!(
            sha256_file(&p).unwrap(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
