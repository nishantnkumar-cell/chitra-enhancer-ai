//! DVD input: a disc image (`.iso`), a drive or mounted disc, or a copied
//! `VIDEO_TS` folder. ffmpeg's DVD-Video reader does the reading; a title is
//! copied out without re-encoding and then treated like any other video.
//!
//! Only discs **without copy protection** (home recordings) are accepted: the
//! raw sectors are checked first and a scrambled disc is refused, whatever
//! libraries happen to be installed.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};

use crate::ffmpeg::Tools;

const SECTOR: usize = 2048;
/// A title shorter than this is a menu loop or a disc oddity.
pub const MIN_TITLE_SECONDS: f64 = 3.0;

#[derive(Clone, Debug, PartialEq)]
pub struct Title {
    pub number: u32,
    pub seconds: f64,
}

fn child(dir: &Path, name: &str) -> Option<PathBuf> {
    std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok())
        .find_map(|e| {
            e.file_name()
                .to_string_lossy()
                .eq_ignore_ascii_case(name)
                .then(|| e.path())
        })
}

/// The disc root if `path` is (or is inside) a copied or mounted DVD folder.
fn dvd_folder(path: &Path) -> Option<PathBuf> {
    if !path.is_dir() {
        return None;
    }
    if let Some(ts) = child(path, "VIDEO_TS")
        && child(&ts, "VIDEO_TS.IFO").is_some()
    {
        return Some(path.to_path_buf());
    }
    let is_ts = path
        .file_name()
        .is_some_and(|n| n.to_string_lossy().eq_ignore_ascii_case("VIDEO_TS"));
    if is_ts && child(path, "VIDEO_TS.IFO").is_some() {
        return path.parent().map(Path::to_path_buf);
    }
    None
}

fn is_device(path: &Path) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileTypeExt;
        path.metadata()
            .is_ok_and(|m| m.file_type().is_block_device())
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        false
    }
}

/// What to hand ffmpeg, if `path` is a DVD in any of the accepted forms.
pub fn source(path: &Path) -> Option<PathBuf> {
    if let Some(root) = dvd_folder(path) {
        return Some(root);
    }
    let image = path.is_file()
        && path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("iso") || e.eq_ignore_ascii_case("img"));
    (image || is_device(path)).then(|| path.to_path_buf())
}

/// True if this 2048-byte sector is an MPEG-2 pack carrying scrambled data.
fn scrambled_sector(s: &[u8]) -> bool {
    if s.len() < SECTOR || s[0..4] != [0, 0, 1, 0xBA] || s[4] & 0xC0 != 0x40 {
        return false;
    }
    let mut p = 14 + usize::from(s[13] & 7);
    while p + 7 <= SECTOR && s[p..p + 3] == [0, 0, 1] {
        let id = s[p + 3];
        let len = usize::from(u16::from_be_bytes([s[p + 4], s[p + 5]]));
        // Video, audio and private streams carry a flags byte with the
        // scrambling bits; system headers and navigation packets do not.
        if matches!(id, 0xBD | 0xC0..=0xEF) && s[p + 6] & 0xC0 == 0x80 && s[p + 6] & 0x30 != 0 {
            return true;
        }
        p += 6 + len;
    }
    false
}

/// Looks at up to `samples` sectors spread evenly through a file or device.
fn scrambled_in(path: &Path, samples: u64) -> Result<bool> {
    let mut f = File::open(path).with_context(|| format!("cannot read {}", path.display()))?;
    let sectors = f.seek(SeekFrom::End(0))? / SECTOR as u64;
    let step = (sectors / samples.max(1)).max(1);
    let mut buf = [0u8; SECTOR];
    let mut at = 0;
    while at < sectors {
        f.seek(SeekFrom::Start(at * SECTOR as u64))?;
        if f.read_exact(&mut buf).is_err() {
            // Unreadable sectors are what a protected disc in a drive gives.
            bail!(
                "{} has sectors that cannot be read (damaged, or copy-protected)",
                path.display()
            );
        }
        if scrambled_sector(&buf) {
            return Ok(true);
        }
        at += step;
    }
    Ok(false)
}

/// Refuses copy-protected discs.
pub fn check_unprotected(src: &Path) -> Result<()> {
    let protected = match child(src, "VIDEO_TS") {
        Some(ts) if src.is_dir() => {
            let mut any = false;
            for e in std::fs::read_dir(ts)?.filter_map(|e| e.ok()) {
                let p = e.path();
                if p.extension().is_some_and(|x| x.eq_ignore_ascii_case("vob")) {
                    any |= scrambled_in(&p, 600)?;
                }
            }
            any
        }
        _ => scrambled_in(src, 6000)?,
    };
    if protected {
        bail!(
            "{} is copy-protected. Only discs without copy protection, such as your own recordings, are supported.",
            src.display()
        );
    }
    Ok(())
}

/// The input options that select one title; `-i` and the path follow.
fn dvd_input(cmd: &mut Command, title: u32) {
    cmd.args(["-f", "dvdvideo", "-title"])
        .arg(title.to_string());
}

/// Checks that this ffmpeg can read DVDs at all.
pub fn check_support(tools: &Tools) -> Result<()> {
    let out = Command::new(&tools.ffmpeg)
        .args(["-hide_banner", "-demuxers"])
        .output()
        .context("cannot run ffmpeg")?;
    if !String::from_utf8_lossy(&out.stdout).contains("dvdvideo") {
        bail!(
            "this ffmpeg cannot read DVDs: it needs the `dvdvideo` reader (ffmpeg 7 or newer built with libdvdread and libdvdnav)"
        );
    }
    Ok(())
}

/// The titles on the disc, in order, with their lengths.
pub fn titles(tools: &Tools, src: &Path) -> Result<Vec<Title>> {
    let mut found = Vec::new();
    let mut last_error = String::new();
    for number in 1..=99u32 {
        let mut cmd = Command::new(&tools.ffprobe);
        cmd.args(["-v", "error"]);
        dvd_input(&mut cmd, number);
        let out = cmd
            .args(["-show_entries", "format=duration", "-of", "csv=p=0"])
            .arg(src)
            .output()
            .context("cannot run ffprobe")?;
        if !out.status.success() {
            last_error = String::from_utf8_lossy(&out.stderr).trim().to_string();
            break;
        }
        if let Ok(seconds) = String::from_utf8_lossy(&out.stdout).trim().parse::<f64>()
            && seconds.is_finite()
        {
            found.push(Title { number, seconds });
        }
    }
    if found.is_empty() {
        bail!(
            "no titles could be read from {}: {last_error}",
            src.display()
        );
    }
    Ok(found)
}

/// The longest title: usually the recording people mean.
pub fn longest(titles: &[Title]) -> Option<&Title> {
    titles.iter().max_by(|a, b| a.seconds.total_cmp(&b.seconds))
}

/// Copies a title (or `range` = start and length in seconds of it) into `dest`
/// (Matroska) without re-encoding. Written under a temporary name first.
pub fn import(
    tools: &Tools,
    src: &Path,
    title: u32,
    range: Option<(f64, f64)>,
    dest: &Path,
) -> Result<()> {
    let part = dest.with_extension("part.mkv");
    let mut cmd = Command::new(&tools.ffmpeg);
    cmd.args(["-v", "error", "-nostdin", "-y"]);
    dvd_input(&mut cmd, title);
    if let Some((start, _)) = range {
        cmd.arg("-ss").arg(format!("{start:.3}"));
    }
    cmd.arg("-i").arg(src);
    if let Some((_, len)) = range {
        cmd.arg("-t").arg(format!("{len:.3}"));
    }
    let st = cmd
        .args(["-map", "0:v:0", "-map", "0:a?", "-c", "copy"])
        .arg(&part)
        .status()
        .context("cannot start ffmpeg")?;
    if !st.success() {
        let _ = std::fs::remove_file(&part);
        bail!("could not copy title {title} from the disc ({st})");
    }
    std::fs::rename(&part, dest)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A pack header followed by one video packet with the given flags byte.
    fn sector(flags: u8, stream: u8) -> Vec<u8> {
        let mut s = vec![0u8; SECTOR];
        s[0..4].copy_from_slice(&[0, 0, 1, 0xBA]);
        s[4] = 0x44;
        s[13] = 0xF8;
        s[14..18].copy_from_slice(&[0, 0, 1, stream]);
        s[18..20].copy_from_slice(&2028u16.to_be_bytes());
        s[20] = flags;
        s
    }

    #[test]
    fn scrambling_bits_are_found_in_video_and_audio_packets_only() {
        assert!(!scrambled_sector(&sector(0x80, 0xE0)), "clear video");
        assert!(scrambled_sector(&sector(0xB0, 0xE0)), "scrambled video");
        assert!(scrambled_sector(&sector(0x90, 0xBD)), "scrambled audio");
        assert!(!scrambled_sector(&sector(0xB0, 0xBF)), "navigation packet");
        assert!(!scrambled_sector(&vec![0xB0; SECTOR]), "not a pack at all");
        assert!(!scrambled_sector(&[0, 0, 1, 0xBA]), "short");
    }

    #[test]
    fn a_system_header_before_the_video_is_stepped_over() {
        let mut s = sector(0x80, 0xE0);
        s[14..20].copy_from_slice(&[0, 0, 1, 0xBB, 0, 12]);
        s[32..36].copy_from_slice(&[0, 0, 1, 0xE0]);
        s[36..38].copy_from_slice(&100u16.to_be_bytes());
        s[38] = 0xB0;
        assert!(scrambled_sector(&s));
    }

    #[test]
    fn a_scrambled_image_is_refused_and_a_clear_one_passes() {
        let d = tempfile::tempdir().unwrap();
        let clear = d.path().join("clear.iso");
        let bad = d.path().join("bad.iso");
        std::fs::write(&clear, sector(0x80, 0xE0).repeat(50)).unwrap();
        let mut b = sector(0x80, 0xE0).repeat(49);
        b.extend(sector(0xB0, 0xE0));
        std::fs::write(&bad, b).unwrap();
        assert!(check_unprotected(&clear).is_ok());
        let e = check_unprotected(&bad).unwrap_err().to_string();
        assert!(e.contains("copy-protected"), "{e}");
    }

    #[test]
    fn dvd_sources_are_recognised_by_shape() {
        let d = tempfile::tempdir().unwrap();
        let ts = d.path().join("disc/VIDEO_TS");
        std::fs::create_dir_all(&ts).unwrap();
        std::fs::write(ts.join("VIDEO_TS.IFO"), b"x").unwrap();
        let root = d.path().join("disc");
        assert_eq!(source(&root), Some(root.clone()));
        assert_eq!(
            source(&ts),
            Some(root.clone()),
            "the VIDEO_TS folder itself"
        );
        let iso = d.path().join("a.ISO");
        std::fs::write(&iso, b"x").unwrap();
        assert_eq!(source(&iso), Some(iso.clone()));
        let other = d.path().join("clip.mp4");
        std::fs::write(&other, b"x").unwrap();
        assert_eq!(source(&other), None);
        assert_eq!(
            source(d.path()),
            None,
            "an ordinary folder is a batch, not a disc"
        );
    }

    #[test]
    fn the_longest_title_is_the_default() {
        let t = [
            Title {
                number: 1,
                seconds: 8.0,
            },
            Title {
                number: 2,
                seconds: 3600.0,
            },
            Title {
                number: 3,
                seconds: 40.0,
            },
        ];
        assert_eq!(longest(&t).unwrap().number, 2);
        assert_eq!(longest(&[]), None);
    }
}
