//! Measuring how noisy a video is, so the five-frame denoiser can be told.
//!
//! The estimate is Immerkaer's: a Laplacian-style mask cancels smooth picture
//! content and leaves the noise, whose strength is then read from the mean
//! absolute response. It is done per 32x32 block and the quiet blocks (flat
//! walls, sky) are trusted, because detail and edges only ever add to the
//! response. Blocks that are pure black or white are skipped: clipped noise
//! reads as no noise. Result: Gaussian-equivalent sigma on a 0..255 scale.

use std::process::{Command, Stdio};

use anyhow::{Context, Result};

use crate::ffmpeg::VideoInfo;
use crate::video::Options;

const BLOCK: usize = 32;

/// Noise sigma (0..255 scale) of one grey picture, or None if no block is usable.
pub fn estimate_gray(w: usize, h: usize, px: &[u8]) -> Option<f32> {
    if px.len() < w * h {
        return None;
    }
    let mut blocks: Vec<f32> = Vec::new();
    for by in (0..h.saturating_sub(BLOCK - 1)).step_by(BLOCK) {
        for bx in (0..w.saturating_sub(BLOCK - 1)).step_by(BLOCK) {
            let mut sum = 0u64;
            let mut abs = 0u64;
            for y in by + 1..by + BLOCK - 1 {
                for x in bx + 1..bx + BLOCK - 1 {
                    let p = |dx: usize, dy: usize| px[(y + dy - 1) * w + x + dx - 1] as i32;
                    let r = p(0, 0) - 2 * p(1, 0) + p(2, 0) - 2 * p(0, 1) + 4 * p(1, 1)
                        - 2 * p(2, 1)
                        + p(0, 2)
                        - 2 * p(1, 2)
                        + p(2, 2);
                    abs += r.unsigned_abs() as u64;
                    sum += px[y * w + x] as u64;
                }
            }
            let n = ((BLOCK - 2) * (BLOCK - 2)) as f64;
            let mean = sum as f64 / n;
            if !(8.0..=247.0).contains(&mean) {
                continue;
            }
            blocks.push((std::f64::consts::FRAC_PI_2.sqrt() * abs as f64 / (6.0 * n)) as f32);
        }
    }
    if blocks.is_empty() {
        return None;
    }
    blocks.sort_by(f32::total_cmp);
    // A quarter of the way up: flat areas set the floor, detail raises the rest.
    Some(blocks[blocks.len() / 4])
}

/// Samples frames from across the video (decoded the way the pipeline will
/// see them, but without denoising) and returns their median noise sigma.
pub fn estimate_video(o: &Options, info: &VideoInfo) -> Result<Option<f32>> {
    let mut probe = o.clone();
    probe.denoise = crate::video::Denoise::Off;
    let filters = probe
        .decode_filters(info)
        .replace("format=rgb24", "format=gray");
    let (w, h) = probe.frame_dims(info);
    let samples = if info.duration < 8.0 { 3 } else { 8 };
    let mut found: Vec<f32> = Vec::new();
    for i in 0..samples {
        let t = info.duration * (i as f64 + 0.5) / samples as f64;
        let out = Command::new(&o.tools.ffmpeg)
            .args(["-v", "error", "-nostdin", "-ss", &format!("{t:.3}"), "-i"])
            .arg(&o.input)
            .args(["-an", "-sn", "-vf", &filters, "-frames:v", "3"])
            .args(["-f", "rawvideo", "-pix_fmt", "gray", "-"])
            .stderr(Stdio::null())
            .output()
            .context("cannot start ffmpeg to measure the noise")?;
        let n = w * h;
        // Use the last whole frame (deinterlacing needs a frame or two to settle).
        if out.stdout.len() >= n {
            let last = out.stdout.len() / n - 1;
            if let Some(s) = estimate_gray(w, h, &out.stdout[last * n..(last + 1) * n]) {
                found.push(s);
            }
        }
    }
    if found.is_empty() {
        return Ok(None);
    }
    found.sort_by(f32::total_cmp);
    Ok(Some(found[found.len() / 2]))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A picture with a smooth gradient plus Gaussian-ish noise of strength `sigma`.
    fn noisy(w: usize, h: usize, sigma: f32, seed: u64) -> Vec<u8> {
        let mut s = seed;
        let mut rnd = move || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32) / (1u64 << 31) as f32
        };
        (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f32, (i / w) as f32);
                // Sum of 12 uniforms minus 6 is close to a unit Gaussian.
                let g: f32 = (0..12).map(|_| rnd()).sum::<f32>() - 6.0;
                (60.0 + x * 0.3 + y * 0.2 + g * sigma)
                    .round()
                    .clamp(0.0, 255.0) as u8
            })
            .collect()
    }

    #[test]
    fn known_noise_levels_are_recovered() {
        for sigma in [4.0f32, 10.0, 20.0, 35.0] {
            let est = estimate_gray(256, 192, &noisy(256, 192, sigma, 7)).unwrap();
            assert!(
                (est - sigma).abs() < sigma * 0.15 + 0.5,
                "true {sigma}, estimated {est}"
            );
        }
    }

    #[test]
    fn a_clean_picture_reads_as_nearly_noiseless() {
        let est = estimate_gray(128, 128, &noisy(128, 128, 0.0, 1)).unwrap();
        assert!(est < 1.0, "{est}");
    }

    #[test]
    fn detail_does_not_fool_it_while_flat_areas_exist() {
        // Left half: hard checkerboard. Right half: flat with sigma 10 noise.
        let (w, h) = (256, 128);
        let mut px = noisy(w, h, 10.0, 3);
        for y in 0..h {
            for x in 0..w / 2 {
                px[y * w + x] = if (x / 2 + y / 2) % 2 == 0 { 40 } else { 200 };
            }
        }
        let est = estimate_gray(w, h, &px).unwrap();
        assert!((est - 10.0).abs() < 3.0, "{est}");
    }

    #[test]
    fn black_frames_and_tiny_pictures_give_no_answer() {
        assert_eq!(estimate_gray(128, 128, &vec![0u8; 128 * 128]), None);
        assert_eq!(estimate_gray(20, 20, &vec![90u8; 400]), None);
        assert_eq!(estimate_gray(128, 128, &[1, 2, 3]), None);
    }
}
