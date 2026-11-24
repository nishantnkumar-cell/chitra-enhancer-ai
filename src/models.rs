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
    let resp = ureq::get(&url)
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

pub const RUNTIME_HELP: &str = "ONNX Runtime was not found. Install it, or point ORT_DYLIB_PATH at it:\n  \
    Arch/CachyOS: sudo pacman -S onnxruntime (CPU) or onnxruntime-cuda\n  \
    Debian/Ubuntu or any Linux: pip install onnxruntime (or onnxruntime-gpu), then set ORT_DYLIB_PATH to\n    \
    .../site-packages/onnxruntime/capi/libonnxruntime.so.*\n  \
    Windows: the onnxruntime.dll from the Microsoft.ML.OnnxRuntime.DirectML NuGet package, next to chitra-enhance.exe\n  \
    macOS: brew install onnxruntime\n\
    Put the library in <data dir>/chitra-enhance/runtime/ and it is found automatically.";

#[cfg(test)]
mod tests {
    use super::*;

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
