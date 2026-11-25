//! Looking at a few places in a video and suggesting settings, with reasons.
//!
//! Nothing here changes the video. Every suggestion is only a starting point:
//! each one maps to a setting the user can change.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::ffmpeg::{Tools, VideoInfo};
use crate::noise::estimate_gray;

/// Places looked at in a long video.
pub const SAMPLES: usize = 12;
/// Consecutive frames read at each place.
pub const FRAMES_PER_SAMPLE: usize = 12;

/// Frames above this size are sampled more sparingly, to bound memory.
const LARGE_FRAME: usize = 3840 * 2160;
/// Enough edges to know the typical one; more only costs memory and time.
const MOST_PROFILES: usize = 20_000;

/// One decoded frame (4:2:0).
pub struct Frame {
    pub y: Vec<u8>,
    pub u: Vec<u8>,
    pub v: Vec<u8>,
}

/// Colour lines tried each way when lining colour up (one is two picture lines).
const REACH: usize = 5;
/// Agreement between colour and brightness edges at each of those shifts.
pub type Curve = [f64; 2 * REACH + 1];

/// What one sample showed.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Measure {
    /// Noise in a single frame (0..255), from quiet areas.
    pub spatial: f32,
    /// Noise that changes between frames, brightness and colour.
    pub temporal: f32,
    pub temporal_colour: f32,
    /// Mean change between frames: camera and subject movement.
    pub motion: f32,
    /// Black edge widths in pixels: left, right, top, bottom.
    pub borders: [usize; 4],
    /// Average colour of whitish areas relative to neutral (blue, red), if any.
    pub cast: Option<(f32, f32)>,
    /// Share of pixels at or beyond video white.
    pub clipped: f32,
    /// How well the colour edges agree with the brightness edges when slid by
    /// each of -5..=5 colour lines (blue, red). Pooled over places later: one
    /// place alone is too weak to tell a shift.
    pub colour_fit: Option<[Curve; 2]>,
    /// How far brightness overshoots across sharp sideways edges (0.3 = 30%),
    /// if enough clean edges were found.
    pub overshoot: Option<f32>,
}

fn median(mut v: Vec<f32>) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(f32::total_cmp);
    v[v.len() / 2]
}

/// Frame-to-frame noise sigma from the quietest tiles (still background), so
/// that movement elsewhere does not count.
fn temporal_noise(a: &[u8], b: &[u8], w: usize, h: usize) -> Option<f32> {
    const T: usize = 16;
    let mut tiles = Vec::new();
    for ty in (T / 2..h.saturating_sub(T + T / 2)).step_by(T) {
        for tx in (T / 2..w.saturating_sub(T + T / 2)).step_by(T) {
            let (mut sum, mut sq, mut level) = (0.0f64, 0.0f64, 0u32);
            for y in ty..ty + T {
                for x in tx..tx + T {
                    let d = f64::from(b[y * w + x]) - f64::from(a[y * w + x]);
                    sum += d;
                    sq += d * d;
                    level += u32::from(a[y * w + x]);
                }
            }
            let n = (T * T) as f64;
            if !(8..=247).contains(&(level / (T * T) as u32)) {
                continue;
            }
            let var = (sq / n - (sum / n).powi(2)).max(0.0);
            tiles.push((var.sqrt() / std::f64::consts::SQRT_2) as f32);
        }
    }
    if tiles.is_empty() {
        return None;
    }
    tiles.sort_by(f32::total_cmp);
    Some(tiles[tiles.len() / 10])
}

/// Widths of the black strips at each edge of one picture.
fn black_edges(y: &[u8], w: usize, h: usize) -> [usize; 4] {
    const BLACK: u64 = 28;
    let col = |x: usize| (0..h).map(|r| u64::from(y[r * w + x])).sum::<u64>() / h as u64;
    let row = |r: usize| {
        y[r * w..(r + 1) * w]
            .iter()
            .map(|&v| u64::from(v))
            .sum::<u64>()
            / w as u64
    };
    let left = (0..w / 4).take_while(|&x| col(x) < BLACK).count();
    let right = (0..w / 4).take_while(|&x| col(w - 1 - x) < BLACK).count();
    let top = (0..h / 4).take_while(|&r| row(r) < BLACK).count();
    let bottom = (0..h / 4).take_while(|&r| row(h - 1 - r) < BLACK).count();
    [left, right, top, bottom]
}

/// Slides one colour plane's edges against the brightness edges and scores the
/// agreement at each shift (index `REACH` is no shift; higher means the colour
/// sits lower). Only places with a strong colour edge are used.
fn colour_curve(f: &Frame, w: usize, h: usize, blue: bool) -> Option<Curve> {
    let (cw, ch) = (w / 2, h / 2);
    let c = if blue { &f.u } else { &f.v };
    // Skip the borders, where black edges would dominate.
    let (x0, x1, y0, y1) = (cw / 16, cw - cw / 16, ch / 16, ch - ch / 16);
    if x1 <= x0 + 8 || y1 <= y0 + 2 * REACH + 8 {
        return None;
    }
    let luma = |x: usize, y: usize| {
        let (a, b) = (2 * y * w + 2 * x, (2 * y + 1) * w + 2 * x);
        let sum = u32::from(f.y[a]) + u32::from(f.y[a + 1]);
        (sum + u32::from(f.y[b]) + u32::from(f.y[b + 1])) as f32 / 4.0
    };
    let colour_step = |x: usize, y: usize| u32::from(c[(y + 1) * cw + x].abs_diff(c[y * cw + x]));
    let mut steps: Vec<u32> = (y0..y1 - 1)
        .flat_map(|y| (x0..x1).map(move |x| (x, y)))
        .map(|(x, y)| colour_step(x, y))
        .collect();
    steps.sort_unstable();
    let strong = steps[steps.len() * 98 / 100].max(4);
    let mut score = [0.0; 2 * REACH + 1];
    for (i, slot) in score.iter_mut().enumerate() {
        let (mut cross, mut cc, mut yy) = (0.0f64, 0.0f64, 0.0f64);
        for y in y0 + REACH..y1 - 1 - REACH {
            for x in x0..x1 {
                // The colour edge `i - REACH` lines below this brightness edge.
                let step = colour_step(x, y + i - REACH);
                let col = if step >= strong { f64::from(step) } else { 0.0 };
                let l = f64::from((luma(x, y + 1) - luma(x, y)).abs());
                cross += col * l;
                cc += col * col;
                yy += l * l;
            }
        }
        if cc > 0.0 && yy > 0.0 {
            *slot = cross / (cc * yy).sqrt();
        }
    }
    score.iter().any(|&v| v > 0.0).then_some(score)
}

fn mean_curve(curves: &[Curve]) -> Option<Curve> {
    (!curves.is_empty()).then(|| {
        std::array::from_fn(|i| curves.iter().map(|c| c[i]).sum::<f64>() / curves.len() as f64)
    })
}

/// Fewer places than this cannot tell a colour shift from chance.
const PLACES_FOR_COLOUR: usize = 6;

/// How many picture lines the colour sits low, from the curves of all places,
/// or 0 unless the evidence is clear: the pooled best shift improves agreement
/// by at least 3% and three places in four agree that it beats leaving it.
fn pooled_shift(places: &[Curve]) -> f64 {
    let Some(total) = mean_curve(places) else {
        return 0.0;
    };
    if places.len() < PLACES_FOR_COLOUR {
        return 0.0;
    }
    let best = (0..total.len())
        .max_by(|&a, &b| total[a].total_cmp(&total[b]))
        .unwrap_or(REACH);
    // A best score at the end of the range is not a real peak.
    if best == 0 || best == total.len() - 1 || best == REACH {
        return 0.0;
    }
    let votes = places.iter().filter(|c| c[best] > c[REACH]).count();
    if votes * 4 < places.len() * 3 || total[best] < total[REACH] * 1.03 {
        return 0.0;
    }
    let (l, m, r) = (total[best - 1], total[best], total[best + 1]);
    let curve = l - 2.0 * m + r;
    let fine = if curve < 0.0 {
        0.5 * (l - r) / curve
    } else {
        0.0
    };
    (best as f64 - REACH as f64 + fine) * 2.0
}

/// Positions kept before an edge's steepest step, and in all.
const BEFORE: usize = 9;
const PROFILE: usize = 20;

/// Collects clean, isolated sideways edges (flat on both sides, a big step),
/// each scaled so that it goes from 0 to 1.
fn edge_profiles(y: &[u8], w: usize, h: usize, out: &mut Vec<[f32; PROFILE]>) {
    if w < PROFILE + 16 {
        return;
    }
    for row in (0..h).step_by(2) {
        let line = &y[row * w..(row + 1) * w];
        let step = |x: usize| i32::from(line[x + 1]) - i32::from(line[x]);
        for x in BEFORE + 3..w - (PROFILE - BEFORE) - 4 {
            for sign in [1i32, -1] {
                let d = sign * step(x);
                if d <= 30 || (x - 4..=x + 4).any(|n| sign * step(n) > d) {
                    continue;
                }
                let seg: [f32; PROFILE] =
                    std::array::from_fn(|i| f32::from(line[x - BEFORE + i]) * sign as f32);
                let flat = |s: &[f32]| {
                    let mean = s.iter().sum::<f32>() / s.len() as f32;
                    let var = s.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / s.len() as f32;
                    (mean, var.sqrt())
                };
                let (lo, lsd) = flat(&seg[..4]);
                let (hi, hsd) = flat(&seg[PROFILE - 4..]);
                if lsd < 4.0 && hsd < 4.0 && hi - lo > 40.0 && out.len() < MOST_PROFILES {
                    out.push(seg.map(|v| (v - lo) / (hi - lo)));
                }
            }
        }
    }
}

/// The overshoot of the typical (median) edge, if there are enough of them.
fn overshoot_of(profiles: &[[f32; PROFILE]]) -> Option<f32> {
    if profiles.len() < 40 {
        return None;
    }
    let typical: [f32; PROFILE] =
        std::array::from_fn(|i| median(profiles.iter().map(|p| p[i]).collect()));
    Some((typical.iter().copied().fold(f32::MIN, f32::max) - 1.0).max(0.0))
}

/// Measures consecutive frames of one place. `w` x `h` is the brightness size;
/// the colour planes are half that each way.
// Filled in field by field: each measurement is its own few lines.
#[allow(clippy::field_reassign_with_default)]
pub fn measure(frames: &[Frame], w: usize, h: usize) -> Measure {
    let (cw, ch) = (w / 2, h / 2);
    let mut m = Measure::default();
    m.spatial = median(
        frames
            .iter()
            .filter_map(|f| estimate_gray(w, h, &f.y))
            .collect(),
    );
    let pairs = frames.windows(2);
    m.temporal = median(
        pairs
            .clone()
            .filter_map(|p| temporal_noise(&p[0].y, &p[1].y, w, h))
            .collect(),
    );
    m.temporal_colour = median(
        pairs
            .clone()
            .flat_map(|p| {
                [
                    temporal_noise(&p[0].u, &p[1].u, cw, ch),
                    temporal_noise(&p[0].v, &p[1].v, cw, ch),
                ]
            })
            .flatten()
            .collect(),
    );
    m.motion = median(
        pairs
            .map(|p| {
                let sum: u64 = p[0]
                    .y
                    .iter()
                    .zip(&p[1].y)
                    .map(|(&a, &b)| u64::from(a.abs_diff(b)))
                    .sum();
                sum as f32 / p[0].y.len() as f32
            })
            .collect(),
    );
    // An edge counts only if it is black in every frame looked at.
    m.borders = frames
        .iter()
        .map(|f| black_edges(&f.y, w, h))
        .reduce(|a, b| std::array::from_fn(|i| a[i].min(b[i])))
        .unwrap_or_default();
    let (mut du, mut dv, mut whites) = (0i64, 0i64, 0u64);
    let (mut clipped, mut total) = (0u64, 0u64);
    for f in frames {
        total += f.y.len() as u64;
        clipped += f.y.iter().filter(|&&v| v >= 235).count() as u64;
        for cy in 0..ch {
            for cx in 0..cw {
                let luma = f.y[cy * 2 * w + cx * 2];
                let (u, v) = (
                    i64::from(f.u[cy * cw + cx]) - 128,
                    i64::from(f.v[cy * cw + cx]) - 128,
                );
                if (150..=225).contains(&luma) && u.abs() < 22 && v.abs() < 22 {
                    du += u;
                    dv += v;
                    whites += 1;
                }
            }
        }
    }
    let fit = |blue: bool| {
        let v: Vec<Curve> = frames
            .iter()
            .step_by(3)
            .filter_map(|f| colour_curve(f, w, h, blue))
            .collect();
        mean_curve(&v)
    };
    m.colour_fit = fit(true).zip(fit(false)).map(|(b, r)| [b, r]);
    let mut profiles = Vec::new();
    for f in frames.iter().step_by(2) {
        edge_profiles(&f.y, w, h, &mut profiles);
    }
    m.overshoot = overshoot_of(&profiles);
    m.cast = (whites * 100 >= (frames.len() * cw * ch) as u64 && whites > 0)
        .then(|| (du as f32 / whites as f32, dv as f32 / whites as f32));
    m.clipped = if total > 0 {
        clipped as f32 / total as f32
    } else {
        0.0
    };
    m
}

/// Which AI model files are available.
#[derive(Clone, Copy, Debug, Default)]
pub struct Available {
    pub upscaler: bool,
    pub denoiser: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Suggestion {
    pub deinterlace: bool,
    pub double_rate: bool,
    /// "off", "light", "medium", "strong": the simple filter.
    pub denoise: String,
    pub five_frame: bool,
    /// Noise level for the five-frame model.
    pub sigma: Option<f32>,
    pub scale: usize,
    pub ai_upscaler: bool,
    pub grain: u32,
    pub crf: u32,
    /// Black edges to cut: left, right, top, bottom (even numbers).
    pub crop: Option<[u32; 4]>,
    /// Lines the colour sits too low (blue, red).
    pub colour_align: Option<(i32, i32)>,
    /// Strength for softening bright outlines, 0 = not needed.
    pub dehalo: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Report {
    pub width: u32,
    pub height: u32,
    pub fps: f64,
    pub seconds: f64,
    pub interlaced: bool,
    /// Typical noise level, 0..255.
    pub noise: f32,
    pub samples: usize,
    pub suggestion: Suggestion,
    /// Why each suggestion was made.
    pub reasons: Vec<String>,
    /// Things seen that the program cannot correct.
    pub notes: Vec<String>,
}

/// Turns measurements into suggested settings, with the reasons.
pub fn recommend(info: &VideoInfo, measures: &[Measure], have: Available) -> Report {
    let pick = |f: fn(&Measure) -> f32| median(measures.iter().map(f).collect());
    let noise = pick(|m| m.spatial).max(pick(|m| m.temporal));
    let colour = pick(|m| m.temporal_colour);
    let motion = pick(|m| m.motion);
    let mut reasons = Vec::new();
    let mut notes = Vec::new();

    let deinterlace = info.interlaced;
    let double_rate = deinterlace && motion > 4.0;
    if deinterlace {
        reasons.push("The video is interlaced (two half-pictures per frame): combine them.".into());
        if double_rate {
            reasons.push(
                "There is plenty of movement: keep both half-pictures for smoother motion (twice the frames, about twice the time)."
                    .into(),
            );
        }
    } else {
        reasons.push("The video is not interlaced.".into());
    }

    let sigma = (noise * 1.6).clamp(3.0, 60.0).round();
    let (denoise, five_frame, grain) = if noise < 2.5 && colour < 2.5 {
        reasons.push(format!(
            "Very little noise (about {noise:.1} of 255): only light cleaning, since strong denoising would smear detail."
        ));
        ("light", false, 0)
    } else if have.denoiser {
        reasons.push(format!(
            "Noticeable noise (about {noise:.1} of 255): the five-frame AI denoiser at level {sigma:.0}."
        ));
        ("off", true, if noise < 6.0 { 2 } else { 3 })
    } else {
        let level = if noise < 6.0 { "medium" } else { "strong" };
        reasons.push(format!(
            "Noise about {noise:.1} of 255: {level} cleaning (the five-frame AI denoiser would do better; fetch it with `models get fastdvdnet`)."
        ));
        (level, false, 2)
    };

    let scale = if info.height <= 720 { 2 } else { 1 };
    let ai_upscaler = scale > 1 && have.upscaler;
    reasons.push(if scale == 1 {
        "Already high definition: keep the size.".into()
    } else if ai_upscaler {
        format!("Standard definition ({}x{}): enlarge 2x with the AI upscaler, which draws crisper edges.", info.width, info.height)
    } else {
        "Standard definition: enlarge 2x (fetch the AI upscaler with `models get realesr` for crisper edges).".into()
    });

    let edges: [usize; 4] = std::array::from_fn(|i| {
        let mut v: Vec<usize> = measures.iter().map(|m| m.borders[i]).collect();
        v.sort_unstable();
        v.get(v.len() / 2).copied().unwrap_or(0)
    });
    // Rounded up to even, so no sliver of black is left.
    let crop = edges
        .iter()
        .any(|&e| e >= 4)
        .then(|| edges.map(|e| (e as u32).next_multiple_of(2)));
    if let Some([l, r, t, b]) = crop {
        reasons.push(format!(
            "Black edges (left {l}, right {r}, top {t}, bottom {b} pixels): cut them off."
        ));
    }
    let casts: Vec<(f32, f32)> = measures.iter().filter_map(|m| m.cast).collect();
    if !casts.is_empty() {
        let u = median(casts.iter().map(|c| c.0).collect());
        let v = median(casts.iter().map(|c| c.1).collect());
        if u.hypot(v) > 4.0 {
            let tint = match (u > 2.0, v > 2.0, u < -2.0, v < -2.0) {
                (true, true, _, _) => "pink or purple",
                (true, _, _, true) => "blue-green",
                (_, true, true, _) => "orange or yellow",
                (_, _, true, true) => "green",
                (true, ..) => "blue",
                (_, true, ..) => "red",
                (_, _, true, _) => "yellow",
                _ => "cyan",
            };
            let (su, sv) = (u.round() as i32, v.round() as i32);
            notes.push(format!(
                "Whitish areas lean {tint}: possibly a colour cast, though a sky counts as whitish too, so it is not applied. To try it: --colour-shift {su}:{sv} (in the window, \"Correct a colour cast\"), then check a preview."
            ));
        }
    }
    let lines = |channel: usize| {
        let places: Vec<Curve> = measures
            .iter()
            .filter_map(|m| m.colour_fit.map(|c| c[channel]))
            .collect();
        pooled_shift(&places)
    };
    let (blue, red) = (lines(0), lines(1));
    let colour_align = (blue.abs() >= 1.5 || red.abs() >= 1.5).then(|| {
        (
            (blue.round() as i32).clamp(-8, 8),
            (red.round() as i32).clamp(-8, 8),
        )
    });
    if let Some((u, v)) = colour_align {
        reasons.push(format!(
            "The colour is out of line with the picture (blue {u}, red {v} lines low, usual for tape): move it back."
        ));
    }
    let overshoots: Vec<f32> = measures.iter().filter_map(|m| m.overshoot).collect();
    let overshoot = median(overshoots.clone());
    let dehalo = if overshoots.len() >= 2 && overshoot >= 0.15 {
        40
    } else {
        0
    };
    if dehalo > 0 {
        reasons.push(format!(
            "Edges have bright outlines from the camera's sharpening (overshoot about {:.0}%): soften them (strength {dehalo}).",
            overshoot * 100.0
        ));
    }

    let clipped = pick(|m| m.clipped);
    if clipped > 0.05 {
        notes.push(format!(
            "About {:.0}% of the picture is blown out to white. That detail was never recorded and cannot be brought back.",
            clipped * 100.0
        ));
    }

    Report {
        width: info.width,
        height: info.height,
        fps: info.fps(),
        seconds: info.duration,
        interlaced: info.interlaced,
        noise,
        samples: measures.len(),
        suggestion: Suggestion {
            deinterlace,
            double_rate,
            denoise: denoise.into(),
            five_frame,
            sigma: five_frame.then_some(sigma),
            scale,
            ai_upscaler,
            grain,
            crf: 18,
            crop,
            colour_align,
            dehalo,
        },
        reasons,
        notes,
    }
}

/// Where to look in a video of this length: evenly spread, away from the ends.
pub fn positions(seconds: f64) -> Vec<f64> {
    let n = if seconds < 20.0 { 3 } else { SAMPLES };
    (0..n)
        .map(|i| seconds * (i as f64 + 0.5) / n as f64)
        .collect()
}

/// How many consecutive frames to read at each place.
fn frames_for(w: usize, h: usize) -> usize {
    if w * h > LARGE_FRAME {
        4
    } else {
        FRAMES_PER_SAMPLE
    }
}

/// Reads consecutive frames at `start` seconds, deinterlaced if asked.
fn read_sample(
    tools: &Tools,
    file: &Path,
    start: f64,
    interlaced: bool,
    w: usize,
    h: usize,
) -> Result<Vec<Frame>> {
    let filters = if interlaced {
        "bwdif=mode=send_frame:parity=auto:deint=all,format=yuv420p"
    } else {
        "format=yuv420p"
    };
    let crop = format!("{filters},crop={w}:{h}:0:0");
    let out = Command::new(&tools.ffmpeg)
        .args([
            "-v",
            "error",
            "-nostdin",
            "-ss",
            &format!("{start:.3}"),
            "-i",
        ])
        .arg(file)
        .args(["-an", "-sn", "-vf", &crop, "-frames:v"])
        .arg(frames_for(w, h).to_string())
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p", "-"])
        .stderr(Stdio::null())
        .output()
        .context("cannot start ffmpeg to analyse the video")?;
    let (ys, cs) = (w * h, w * h / 4);
    Ok(out
        .stdout
        .chunks_exact(ys + 2 * cs)
        .map(|c| Frame {
            y: c[..ys].to_vec(),
            u: c[ys..ys + cs].to_vec(),
            v: c[ys + cs..].to_vec(),
        })
        .collect())
}

/// Analyses samples given as (file, start second). For an ordinary video all
/// samples are in the same file; for a disc each was copied out on its own.
pub fn run(
    tools: &Tools,
    info: &VideoInfo,
    samples: &[(PathBuf, f64)],
    have: Available,
) -> Result<Report> {
    let (w, h) = ((info.width & !1) as usize, (info.height & !1) as usize);
    let mut measures = Vec::new();
    for (file, start) in samples {
        let frames = read_sample(tools, file, *start, info.interlaced, w, h)?;
        if frames.len() >= 2 {
            measures.push(measure(&frames, w, h));
        }
    }
    if measures.is_empty() {
        bail!("no frames could be read to analyse");
    }
    Ok(recommend(info, &measures, have))
}

#[cfg(test)]
mod tests {
    use super::*;

    const W: usize = 192;
    const H: usize = 128;

    fn frames(noise: u8, shift: usize, seed: u64) -> Vec<Frame> {
        let mut s = seed;
        let mut rnd = move || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 33) as u32
        };
        (0..6)
            .map(|f| {
                let y = (0..W * H)
                    .map(|i| {
                        let x = (i % W + f * shift) % W;
                        let base = 60 + (x / 2) as i32;
                        let n = if noise == 0 {
                            0
                        } else {
                            (rnd() % (2 * u32::from(noise) + 1)) as i32 - i32::from(noise)
                        };
                        (base + n).clamp(0, 255) as u8
                    })
                    .collect();
                Frame {
                    y,
                    u: vec![128; W * H / 4],
                    v: vec![128; W * H / 4],
                }
            })
            .collect()
    }

    fn info(w: u32, h: u32, interlaced: bool) -> VideoInfo {
        VideoInfo {
            width: w,
            height: h,
            fps_num: 25,
            fps_den: 1,
            duration: 60.0,
            interlaced,
            sar: (1, 1),
            has_audio: true,
            frames: None,
        }
    }

    const ALL: Available = Available {
        upscaler: true,
        denoiser: true,
    };

    #[test]
    fn noise_between_frames_is_measured() {
        let clean = measure(&frames(0, 0, 1), W, H);
        assert!(clean.temporal < 0.1 && clean.motion < 0.1, "{clean:?}");
        // Uniform noise of +-10 has a standard deviation of about 6.
        let noisy = measure(&frames(10, 0, 2), W, H);
        assert!((4.0..7.5).contains(&noisy.temporal), "{noisy:?}");
        let moving = measure(&frames(0, 9, 3), W, H);
        assert!(moving.motion > 3.0, "{moving:?}");
    }

    #[test]
    fn black_edges_and_colour_casts_are_seen() {
        let mut f = frames(0, 0, 4);
        for fr in &mut f {
            for y in 0..H {
                fr.y[y * W..y * W + 10].fill(16);
            }
            for y in 20..100 {
                fr.y[y * W + 40..y * W + 150].fill(190);
            }
            fr.u.fill(138);
            fr.v.fill(139);
        }
        let m = measure(&f, W, H);
        assert_eq!(m.borders[0], 10);
        assert_eq!(m.borders[1], 0);
        let (u, v) = m.cast.expect("white areas");
        assert!((u - 10.0).abs() < 0.5 && (v - 11.0).abs() < 0.5, "{u} {v}");
        let r = recommend(&info(720, 576, false), &[m], ALL);
        assert!(
            r.notes.iter().any(|n| n.contains("lean pink or purple")),
            "{:?}",
            r.notes
        );
        assert_eq!(r.suggestion.crop, Some([10, 0, 0, 0]));
        assert!(
            r.notes.iter().any(|n| n.contains("--colour-shift 10:11")),
            "{:?}",
            r.notes
        );
    }

    #[test]
    fn a_quiet_interlaced_tape_gets_gentle_settings() {
        let m = Measure {
            spatial: 0.4,
            temporal: 1.3,
            temporal_colour: 1.2,
            motion: 9.0,
            ..Measure::default()
        };
        let s = recommend(&info(720, 576, true), &[m], ALL).suggestion;
        assert!(s.deinterlace && s.double_rate && !s.five_frame && s.ai_upscaler);
        assert_eq!(
            (s.denoise.as_str(), s.scale, s.grain, s.sigma),
            ("light", 2, 0, None)
        );
    }

    #[test]
    fn a_noisy_tape_gets_the_five_frame_denoiser_and_hd_is_not_enlarged() {
        let m = Measure {
            spatial: 7.0,
            temporal: 8.0,
            motion: 1.0,
            ..Measure::default()
        };
        let s = recommend(&info(1920, 1080, false), std::slice::from_ref(&m), ALL).suggestion;
        assert!(s.five_frame && !s.double_rate && !s.deinterlace);
        assert_eq!(
            (s.sigma, s.scale, s.ai_upscaler, s.grain),
            (Some(13.0), 1, false, 3)
        );
        // Without the model files it falls back to what is built in, and says so.
        let r = recommend(&info(720, 480, false), &[m], Available::default());
        assert!(!r.suggestion.five_frame && !r.suggestion.ai_upscaler);
        assert_eq!(r.suggestion.denoise, "strong");
        assert!(r.reasons.iter().any(|x| x.contains("models get")));
    }

    /// A picture of horizontal bands; the colour bands sit `low` lines lower.
    fn bands(low: usize) -> Frame {
        // Irregular band heights, so only one alignment fits.
        let edge = |y: usize| {
            [0, 17, 29, 52, 60, 83, 101, 118]
                .iter()
                .filter(|&&e| y >= e)
                .count()
                % 2
                == 1
        };
        let y = (0..W * H)
            .map(|i| if edge(i / W) { 190 } else { 60 })
            .collect();
        let plane: Vec<u8> = (0..W * H / 4)
            .map(|i| {
                // A colour row covers two picture lines: average them.
                let row = (i / (W / 2)) * 2;
                let at = |r: usize| {
                    if edge(r.saturating_sub(low)) {
                        170
                    } else {
                        110
                    }
                };
                ((at(row) + at(row + 1)) / 2) as u8
            })
            .collect();
        Frame {
            y,
            u: plane.clone(),
            v: plane,
        }
    }

    #[test]
    fn colour_that_sits_low_is_measured_over_many_places() {
        for low in [0usize, 2, 4, 6] {
            let c = colour_curve(&bands(low), W, H, true).expect("edges");
            let got = pooled_shift(&[c; 8]);
            assert!((got - low as f64).abs() < 1.0, "true {low}, measured {got}");
        }
    }

    #[test]
    fn colour_is_moved_only_when_enough_places_agree() {
        let curve = |low| colour_curve(&bands(low), W, H, true).unwrap();
        let place = |low| Measure {
            colour_fit: Some([curve(low), curve(low)]),
            ..Measure::default()
        };
        let suggest = |places: &[Measure]| {
            recommend(&info(720, 576, true), places, ALL)
                .suggestion
                .colour_align
        };
        let mostly: Vec<Measure> = (0..8).map(|i| place(if i < 7 { 4 } else { 0 })).collect();
        assert_eq!(suggest(&mostly), Some((4, 4)));
        // Half the places say otherwise: leave the colour alone.
        let split: Vec<Measure> = (0..8).map(|i| place(if i < 4 { 4 } else { 0 })).collect();
        assert_eq!(suggest(&split), None);
        // Too few places to trust, however much they agree.
        assert_eq!(suggest(&[place(4), place(4), place(4)]), None);
        let flat: Vec<Measure> = (0..8).map(|_| place(0)).collect();
        assert_eq!(suggest(&flat), None);
    }

    #[test]
    fn sharpening_outlines_are_measured_and_a_clean_edge_needs_nothing() {
        let picture = |profile: &[u8]| -> Vec<u8> {
            let mut line = [60u8; W];
            line[80..80 + profile.len()].copy_from_slice(profile);
            line[80 + profile.len()..].fill(180);
            line.repeat(H)
        };
        let measured = |p: &[u8]| {
            let mut v = Vec::new();
            edge_profiles(&picture(p), W, H, &mut v);
            overshoot_of(&v).expect("enough edges")
        };
        let clean = measured(&[80, 150]);
        let ringing = measured(&[45, 120, 215, 220, 195]);
        assert!(clean < 0.05, "{clean}");
        assert!((ringing - 0.33).abs() < 0.05, "{ringing}");
        let with = |o: f32| Measure {
            overshoot: Some(o),
            ..Measure::default()
        };
        let places = [with(ringing), with(0.3)];
        assert_eq!(
            recommend(&info(720, 576, true), &places, ALL)
                .suggestion
                .dehalo,
            40
        );
        let places = [with(clean), with(0.05)];
        assert_eq!(
            recommend(&info(720, 576, true), &places, ALL)
                .suggestion
                .dehalo,
            0
        );
    }

    #[test]
    fn large_video_and_busy_pictures_use_bounded_memory() {
        assert_eq!(frames_for(720, 576), FRAMES_PER_SAMPLE);
        assert_eq!(frames_for(7680, 4320), 4);
        let mut line = [60u8; W];
        line[80..].fill(180);
        let picture = line.repeat(H);
        let mut v = vec![[0.0; PROFILE]; MOST_PROFILES];
        edge_profiles(&picture, W, H, &mut v);
        assert_eq!(v.len(), MOST_PROFILES);
    }

    #[test]
    fn tiny_pictures_are_measured_without_failing() {
        for (w, h) in [(16, 16), (18, 64), (64, 18), (40, 40)] {
            let f = || Frame {
                y: (0..w * h).map(|i| (i * 7) as u8).collect(),
                u: (0..w * h / 4).map(|i| (i * 13) as u8).collect(),
                v: (0..w * h / 4).map(|i| (i * 5) as u8).collect(),
            };
            let m = measure(&[f(), f(), f()], w, h);
            let _ = recommend(&info(720, 576, false), &[m], ALL);
        }
    }

    #[test]
    fn samples_are_spread_away_from_the_ends() {
        let p = positions(600.0);
        assert_eq!(p.len(), SAMPLES);
        assert!(p[0] > 0.0 && p[SAMPLES - 1] < 600.0 && p.windows(2).all(|w| w[1] > w[0]));
        assert_eq!(positions(8.0).len(), 3);
    }
}
