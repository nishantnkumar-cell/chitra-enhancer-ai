//! Overlapping tiles with seamless blending.
//!
//! A model that needs a bounded amount of memory is run on one tile at a time.
//! The image is padded (by reflection) to a multiple of 16, cut into
//! overlapping tiles, each tile goes through the model, and the results are put
//! back with a **linear feather** across the overlaps: every output pixel is the
//! weighted mean of the tiles that cover it, with weights that rise from near 0
//! at a tile's inner border to 1, and are 1 at the image's own edge. Because the
//! weights are normalised, a model that changes nothing gives back exactly the
//! input (to f32 rounding), so the tiling itself leaves no seams.

use std::borrow::Cow;

use anyhow::{Result, bail};
use rayon::prelude::*;

/// An image as interleaved `f32` samples (height, width, channels).
#[derive(Clone, Debug, PartialEq)]
pub struct Plane {
    pub w: usize,
    pub h: usize,
    pub c: usize,
    pub data: Vec<f32>,
}

impl Plane {
    pub fn new(w: usize, h: usize, c: usize) -> Plane {
        Plane {
            w,
            h,
            c,
            data: vec![0.0; w * h * c],
        }
    }

    fn at(&self, x: usize, y: usize) -> &[f32] {
        let i = (y * self.w + x) * self.c;
        &self.data[i..i + self.c]
    }
}

/// Index into `0..n` for a position that may be past the end, mirroring at the
/// edge without repeating the edge sample (`... 2 1 0 1 2 ... n-2 n-1 n-2 ...`).
pub(crate) fn reflect(i: usize, n: usize) -> usize {
    if n == 1 {
        return 0;
    }
    let period = 2 * (n - 1);
    let m = i % period;
    if m < n { m } else { period - m }
}

/// The smallest multiple of `m` that is at least `n`.
pub(crate) fn round_up(n: usize, m: usize) -> usize {
    n.div_ceil(m) * m
}

/// `src` padded on the right and below to a multiple of `multiple`, mirroring
/// the picture into the padding.
pub fn pad_reflect(src: &Plane, multiple: usize) -> Plane {
    let (pw, ph) = (round_up(src.w, multiple), round_up(src.h, multiple));
    if (pw, ph) == (src.w, src.h) {
        return src.clone();
    }
    let mut out = Plane::new(pw, ph, src.c);
    out.data
        .par_chunks_mut(pw * src.c)
        .enumerate()
        .for_each(|(y, row)| {
            let sy = reflect(y, src.h);
            for (x, px) in row.chunks_exact_mut(src.c).enumerate() {
                px.copy_from_slice(src.at(reflect(x, src.w), sy));
            }
        });
    out
}

/// Tile start positions along a line of `len` samples, `tile` long with
/// `overlap` shared between neighbours; the last tile is moved back so it ends
/// exactly at the edge.
fn starts(len: usize, tile: usize, overlap: usize) -> Vec<usize> {
    if len <= tile {
        return vec![0];
    }
    let stride = tile - overlap;
    let mut v = Vec::new();
    let mut x = 0;
    while x + tile < len {
        v.push(x);
        x += stride;
    }
    v.push(len - tile);
    v
}

/// The weight of output sample `i` of a tile `len` long, with a ramp of `ramp`
/// samples on the sides that border another tile.
fn weight(i: usize, len: usize, ramp: usize, left: bool, right: bool) -> f32 {
    let mut w = 1.0f32;
    let r = (ramp + 1) as f32;
    if left {
        w = w.min((i + 1) as f32 / r);
    }
    if right {
        w = w.min((len - i) as f32 / r);
    }
    w
}

/// Runs `model` over `src` in tiles and returns the picture `scale` times
/// larger. `model` gets one tile and must return it enlarged by `scale`.
/// `tile` is the tile side and `overlap` the width shared by neighbouring tiles.
pub fn upscale_tiled<F>(
    src: &Plane,
    scale: usize,
    tile: usize,
    overlap: usize,
    model: F,
) -> Result<Plane>
where
    F: FnMut(&Plane) -> Result<Plane>,
{
    map_tiled(src, scale, src.c, tile, overlap, model)
}

/// Like `upscale_tiled`, but the model may return a different number of
/// channels than it was given (`out_c`), e.g. five stacked frames in, one out.
pub fn map_tiled<F>(
    src: &Plane,
    scale: usize,
    out_c: usize,
    tile: usize,
    overlap: usize,
    mut model: F,
) -> Result<Plane>
where
    F: FnMut(&Plane) -> Result<Plane>,
{
    if scale == 0 || out_c == 0 || tile < 16 || overlap * 2 >= tile {
        bail!("bad tiling: scale {scale}, tile {tile}, overlap {overlap}");
    }
    let padded = if src.w.is_multiple_of(16) && src.h.is_multiple_of(16) {
        Cow::Borrowed(src)
    } else {
        Cow::Owned(pad_reflect(src, 16))
    };
    let (pw, ph, c) = (padded.w, padded.h, padded.c);
    let (tw, th) = (tile.min(pw), tile.min(ph));
    let xs = starts(pw, tw, overlap);
    let ys = starts(ph, th, overlap);
    let (fw, fh) = (src.w * scale, src.h * scale);
    let checked = |out: Plane| -> Result<Plane> {
        if (out.w, out.h, out.c) != (tw * scale, th * scale, out_c) {
            bail!(
                "the model returned {}x{}x{} for a {tw}x{th}x{c} tile at scale {scale} \
                 (expected {out_c} channels)",
                out.w,
                out.h,
                out.c
            );
        }
        Ok(out)
    };

    // One tile covers the whole picture: nothing to cut out or blend.
    if xs.len() == 1 && ys.len() == 1 {
        let out = checked(model(&padded)?)?;
        return Ok(if (out.w, out.h) == (fw, fh) {
            out
        } else {
            crop(&out, fw, fh)
        });
    }

    let (ow, oh) = (pw * scale, ph * scale);
    let mut acc = vec![0.0f32; ow * oh * out_c];
    let mut wsum = vec![0.0f32; ow * oh];
    let ramp = overlap * scale;

    for &y0 in &ys {
        for &x0 in &xs {
            let mut t = Plane::new(tw, th, c);
            for y in 0..th {
                let s = ((y0 + y) * pw + x0) * c;
                t.data[y * tw * c..(y + 1) * tw * c].copy_from_slice(&padded.data[s..s + tw * c]);
            }
            let out = checked(model(&t)?)?;
            let (left, right) = (x0 > 0, x0 + tw < pw);
            let (top, bottom) = (y0 > 0, y0 + th < ph);
            let across: Vec<f32> = (0..out.w)
                .map(|x| weight(x, out.w, ramp, left, right))
                .collect();
            acc.par_chunks_mut(ow * out_c)
                .zip(wsum.par_chunks_mut(ow))
                .skip(y0 * scale)
                .take(out.h)
                .enumerate()
                .for_each(|(y, (acc, wsum))| {
                    let wy = weight(y, out.h, ramp, top, bottom);
                    let row = &out.data[y * out.w * out_c..(y + 1) * out.w * out_c];
                    let from = x0 * scale;
                    for (((px, a), sum), wx) in row
                        .chunks_exact(out_c)
                        .zip(acc[from * out_c..].chunks_exact_mut(out_c))
                        .zip(&mut wsum[from..])
                        .zip(&across)
                    {
                        let wgt = wy * wx;
                        for (a, v) in a.iter_mut().zip(px) {
                            *a += v * wgt;
                        }
                        *sum += wgt;
                    }
                });
        }
    }

    // Normalise and crop away the padding.
    let mut result = Plane::new(fw, fh, out_c);
    result
        .data
        .par_chunks_mut(fw * out_c)
        .enumerate()
        .for_each(|(y, row)| {
            for (x, px) in row.chunks_exact_mut(out_c).enumerate() {
                let g = y * ow + x;
                let inv = 1.0 / wsum[g];
                for (k, v) in px.iter_mut().enumerate() {
                    *v = acc[g * out_c + k] * inv;
                }
            }
        });
    Ok(result)
}

/// The top left `w` x `h` of a picture at least that large.
fn crop(src: &Plane, w: usize, h: usize) -> Plane {
    let mut out = Plane::new(w, h, src.c);
    out.data
        .par_chunks_mut(w * src.c)
        .zip(src.data.par_chunks(src.w * src.c))
        .for_each(|(out, row)| out.copy_from_slice(&row[..out.len()]));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A picture with detail everywhere, so a seam or a shift would show.
    fn picture(w: usize, h: usize) -> Plane {
        let mut p = Plane::new(w, h, 3);
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) * 3;
                p.data[i] = (x * 7 + y * 3) as f32 % 251.0 / 251.0;
                p.data[i + 1] = (x * x + y) as f32 % 241.0 / 241.0;
                p.data[i + 2] = ((x ^ y) * 5) as f32 % 239.0 / 239.0;
            }
        }
        p
    }

    /// The "model" that enlarges by repeating each pixel.
    fn nearest(scale: usize) -> impl FnMut(&Plane) -> Result<Plane> {
        move |t: &Plane| {
            let mut o = Plane::new(t.w * scale, t.h * scale, t.c);
            for y in 0..o.h {
                for x in 0..o.w {
                    let s = t.at(x / scale, y / scale).to_vec();
                    let d = (y * o.w + x) * t.c;
                    o.data[d..d + t.c].copy_from_slice(&s);
                }
            }
            Ok(o)
        }
    }

    fn max_diff(a: &Plane, b: &Plane) -> f32 {
        assert_eq!((a.w, a.h, a.c), (b.w, b.h, b.c));
        a.data
            .iter()
            .zip(&b.data)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    #[test]
    fn reflection_mirrors_without_repeating_the_edge() {
        let idx: Vec<usize> = (0..9).map(|i| reflect(i, 4)).collect();
        assert_eq!(idx, [0, 1, 2, 3, 2, 1, 0, 1, 2]);
        assert_eq!(reflect(5, 1), 0);
    }

    #[test]
    fn padding_reaches_a_multiple_of_16_and_keeps_the_picture() {
        for (w, h) in [(1, 1), (15, 17), (16, 16), (33, 5), (100, 70)] {
            let p = picture(w, h);
            let q = pad_reflect(&p, 16);
            assert_eq!((q.w % 16, q.h % 16), (0, 0), "{w}x{h}");
            assert!(q.w >= w && q.h >= h && q.w - w < 16 && q.h - h < 16);
            for y in 0..h {
                for x in 0..w {
                    assert_eq!(q.at(x, y), p.at(x, y));
                }
            }
        }
    }

    #[test]
    fn a_model_may_take_five_frames_and_return_one() {
        // 15 channels in; the model keeps the middle frame (channels 6..9).
        let mut src = Plane::new(50, 37, 15);
        for (i, v) in src.data.iter_mut().enumerate() {
            *v = ((i * 7) % 101) as f32 / 101.0;
        }
        let out = map_tiled(&src, 1, 3, 32, 8, |t| {
            let mut o = Plane::new(t.w, t.h, 3);
            for i in 0..t.w * t.h {
                o.data[i * 3..i * 3 + 3].copy_from_slice(&t.data[i * 15 + 6..i * 15 + 9]);
            }
            Ok(o)
        })
        .unwrap();
        assert_eq!((out.w, out.h, out.c), (50, 37, 3));
        for i in 0..50 * 37 {
            for k in 0..3 {
                assert!((out.data[i * 3 + k] - src.data[i * 15 + 6 + k]).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn tiles_cover_the_line_and_end_at_the_edge() {
        for (len, tile, ov) in [
            (100, 32, 8),
            (64, 32, 8),
            (33, 32, 8),
            (500, 512, 32),
            (96, 64, 16),
        ] {
            let s = starts(len, tile.min(len), ov);
            assert_eq!(s[0], 0);
            assert_eq!(s.last().copied().unwrap() + tile.min(len), len);
            for pair in s.windows(2) {
                assert!(pair[1] > pair[0] && pair[1] - pair[0] <= tile - ov);
            }
        }
    }

    #[test]
    fn a_model_that_changes_nothing_gives_back_the_input_with_no_seams() {
        for (w, h, tile, ov) in [
            (100, 70, 32, 8),
            (64, 64, 32, 8),
            (37, 91, 48, 16),
            (200, 33, 64, 16),
        ] {
            let src = picture(w, h);
            let out = upscale_tiled(&src, 1, tile, ov, |t| Ok(t.clone())).unwrap();
            assert!(max_diff(&out, &src) < 1e-5, "{w}x{h} tile {tile}");
        }
    }

    #[test]
    fn upscaled_tiles_join_into_exactly_the_upscaled_picture() {
        for scale in [2, 3, 4] {
            let src = picture(70, 45);
            let whole = nearest(scale)(&src).unwrap();
            let tiled = upscale_tiled(&src, scale, 32, 8, nearest(scale)).unwrap();
            assert_eq!((tiled.w, tiled.h), (70 * scale, 45 * scale));
            assert!(max_diff(&tiled, &whole) < 1e-5, "scale {scale}");
        }
    }

    #[test]
    fn a_flat_picture_stays_flat_across_every_overlap() {
        let mut src = Plane::new(90, 90, 3);
        src.data.fill(0.5);
        // Even a model that scales by a constant must not show the tile grid.
        let out = upscale_tiled(&src, 2, 32, 8, |t| {
            let mut o = nearest(2)(t)?;
            o.data.iter_mut().for_each(|v| *v *= 0.8);
            Ok(o)
        })
        .unwrap();
        let (lo, hi) = out
            .data
            .iter()
            .fold((f32::MAX, f32::MIN), |(l, h), &v| (l.min(v), h.max(v)));
        assert!(hi - lo < 1e-6, "range {lo}..{hi}");
    }

    #[test]
    fn a_picture_smaller_than_a_tile_is_one_tile_and_bad_settings_are_refused() {
        let src = picture(20, 10);
        let mut calls = 0;
        let out = upscale_tiled(&src, 2, 512, 32, |t| {
            calls += 1;
            nearest(2)(t)
        })
        .unwrap();
        assert_eq!(calls, 1);
        assert_eq!((out.w, out.h), (40, 20));
        assert!(upscale_tiled(&src, 2, 32, 16, nearest(2)).is_err());
        assert!(upscale_tiled(&src, 0, 32, 8, nearest(2)).is_err());
        // A model that returns the wrong size is reported, not blended.
        assert!(upscale_tiled(&src, 2, 32, 8, |t| Ok(t.clone())).is_err());
    }
}
