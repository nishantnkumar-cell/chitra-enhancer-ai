//! The parts of running a model that need no ONNX Runtime: how a frame is laid
//! out as a tensor, half-precision conversion, and how big a tile may be.

use rayon::prelude::*;

use crate::tiler::{Plane, reflect};

/// Tile sizes to try, largest first. After an out-of-memory error the next
/// smaller one is used. A tile that holds the whole picture is much the
/// fastest: nothing is processed twice and nothing has to be blended.
pub const TILE_LADDER: [usize; 5] = [1024, 768, 512, 256, 128];

/// The overlap that goes with a tile size.
pub fn overlap_for(tile: usize) -> usize {
    match tile {
        t if t >= 512 => 32,
        t if t >= 256 => 24,
        _ => 16,
    }
}

/// The next smaller tile after `tile`, or `None` at the bottom.
pub fn smaller_tile(tile: usize) -> Option<usize> {
    TILE_LADDER.iter().copied().find(|&t| t < tile)
}

/// The largest tile of the ladder whose `bytes_per_pixel` fit in `vram_mb`
/// megabytes with a fifth to spare; the smallest if none does.
fn largest_tile(vram_mb: usize, bytes_per_pixel: f64) -> usize {
    let budget = vram_mb as f64 * 1024.0 * 1024.0 * 0.8;
    TILE_LADDER
        .into_iter()
        .find(|&t| (t * t) as f64 * bytes_per_pixel <= budget)
        .unwrap_or(TILE_LADDER[TILE_LADDER.len() - 1])
}

/// A first guess for the largest tile that fits `vram_mb` megabytes for a model
/// that makes `scale`x larger pictures in `bytes` per sample (2 for FP16).
/// A convolutional network keeps a few dozen feature maps alive; 48 channels at
/// full resolution plus the enlarged output is a deliberately cautious model.
pub fn tile_for_budget(vram_mb: usize, scale: usize, bytes: usize) -> usize {
    let per_pixel = (48 * 6 + 3 * scale * scale * 2) * bytes;
    largest_tile(vram_mb, per_pixel as f64)
}

/// The same for the five-frame denoiser, which needs far more: about 1250
/// samples a pixel, measured on DirectML. It leaves a quarter of the memory to
/// the upscaler.
pub fn denoiser_tile_for_budget(vram_mb: usize, bytes: usize) -> usize {
    largest_tile(vram_mb / 4 * 3, (1250 * bytes) as f64)
}

/// Samples handed to one thread at a time by the conversions below.
pub const BLOCK: usize = 1 << 14;

/// The Plane (interleaved HWC) as planar CHW, the layout models expect.
pub fn to_chw(p: &Plane) -> Vec<f32> {
    let (w, h, c) = (p.w, p.h, p.c);
    let mut out = vec![0.0f32; w * h * c];
    if out.is_empty() {
        return out;
    }
    // One row of one channel at a time: row `i` is row `i % h` of channel `i / h`.
    out.par_chunks_mut(w).enumerate().for_each(|(i, row)| {
        let (k, y) = (i / h, i % h);
        let src = &p.data[y * w * c..(y + 1) * w * c];
        for (o, px) in row.iter_mut().zip(src.chunks_exact(c)) {
            *o = px[k];
        }
    });
    out
}

/// Planar CHW back to an interleaved Plane; values are clamped to 0..=1,
/// since a model may overshoot a little.
pub fn from_chw(data: &[f32], w: usize, h: usize, c: usize) -> Plane {
    assert_eq!(data.len(), w * h * c);
    let mut p = Plane::new(w, h, c);
    if data.is_empty() {
        return p;
    }
    p.data
        .par_chunks_mut(w * c)
        .enumerate()
        .for_each(|(y, row)| {
            for k in 0..c {
                let src = &data[(k * h + y) * w..][..w];
                for (px, v) in row.chunks_exact_mut(c).zip(src) {
                    px[k] = v.clamp(0.0, 1.0);
                }
            }
        });
    p
}

/// A whole 8-bit RGB picture straight into a model's planar input, padded to
/// `pw` x `ph` by mirroring. `values[v]` is what the model is given for a
/// sample of `v`.
pub fn chw_from_rgb8<T: Copy + Default + Send + Sync>(
    rgb: &[u8],
    (w, h): (usize, usize),
    (pw, ph): (usize, usize),
    values: &[T; 256],
) -> Vec<T> {
    assert_eq!(rgb.len(), w * h * 3);
    let mut out = vec![T::default(); pw * ph * 3];
    if out.is_empty() || rgb.is_empty() {
        return out;
    }
    out.par_chunks_mut(pw).enumerate().for_each(|(i, row)| {
        let (k, y) = (i / ph, i % ph);
        let sy = reflect(y, h);
        let src = &rgb[sy * w * 3..(sy + 1) * w * 3];
        let (inside, beyond) = row.split_at_mut(w);
        for (o, px) in inside.iter_mut().zip(src.as_chunks::<3>().0) {
            *o = values[usize::from(px[k])];
        }
        for (x, o) in beyond.iter_mut().enumerate() {
            *o = values[usize::from(src[reflect(w + x, w) * 3 + k])];
        }
    });
    out
}

/// A model's planar output (`pw` samples wide, `ph` high per channel) straight
/// to the 8-bit RGB picture in its top left `w` x `h`.
pub fn rgb8_from_chw<T: Copy + Sync>(
    data: &[T],
    (pw, ph): (usize, usize),
    (w, h): (usize, usize),
    to_byte: impl Fn(T) -> u8 + Sync,
) -> Vec<u8> {
    assert_eq!(data.len(), pw * ph * 3);
    assert!(w <= pw && h <= ph);
    let mut out = vec![0u8; w * h * 3];
    if out.is_empty() {
        return out;
    }
    let (r, rest) = data.split_at(pw * ph);
    let (g, b) = rest.split_at(pw * ph);
    out.par_chunks_mut(w * 3).enumerate().for_each(|(y, row)| {
        let line = y * pw..y * pw + w;
        for (((px, &r), &g), &b) in row
            .as_chunks_mut::<3>()
            .0
            .iter_mut()
            .zip(&r[line.clone()])
            .zip(&g[line.clone()])
            .zip(&b[line])
        {
            *px = [to_byte(r), to_byte(g), to_byte(b)];
        }
    });
    out
}

/// A model's sample (0..=1, which it may overshoot) as 8 bits.
pub fn byte(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

/// 8-bit RGB (interleaved) to a Plane in 0..=1. The models are trained on
/// gamma-encoded sRGB, so no linearisation is done.
pub fn plane_from_rgb8(rgb: &[u8], w: usize, h: usize) -> Plane {
    let mut data = vec![0.0f32; rgb.len()];
    data.par_chunks_mut(BLOCK)
        .zip(rgb.par_chunks(BLOCK))
        .for_each(|(out, src)| {
            for (o, &v) in out.iter_mut().zip(src) {
                *o = f32::from(v) / 255.0;
            }
        });
    Plane { w, h, c: 3, data }
}

pub fn rgb8_from_plane(p: &Plane) -> Vec<u8> {
    let mut out = vec![0u8; p.data.len()];
    out.par_chunks_mut(BLOCK)
        .zip(p.data.par_chunks(BLOCK))
        .for_each(|(out, src)| {
            for (o, v) in out.iter_mut().zip(src) {
                *o = (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            }
        });
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tiler::{pad_reflect, round_up};

    #[test]
    fn the_tile_ladder_steps_down_to_the_bottom() {
        assert_eq!(smaller_tile(1024), Some(768));
        assert_eq!(smaller_tile(768), Some(512));
        assert_eq!(smaller_tile(512), Some(256));
        assert_eq!(smaller_tile(256), Some(128));
        assert_eq!(smaller_tile(128), None);
        assert_eq!(smaller_tile(300), Some(256));
        assert_eq!(
            (overlap_for(512), overlap_for(256), overlap_for(128)),
            (32, 24, 16)
        );
        // Every rung leaves a usable tile (overlap well under half).
        assert!(TILE_LADDER.iter().all(|&t| overlap_for(t) * 2 < t));
    }

    #[test]
    fn the_budget_picks_smaller_tiles_for_less_memory() {
        let big = tile_for_budget(8 * 1024, 2, 2);
        let small = tile_for_budget(512, 2, 2);
        let tiny = tile_for_budget(16, 4, 4);
        assert!(big >= small && small >= tiny);
        assert_eq!(big, 1024);
        assert_eq!(tile_for_budget(1024, 2, 4), 768);
        assert_eq!(tiny, 128);
    }

    #[test]
    fn the_denoiser_gets_a_tile_that_fits_beside_the_upscaler() {
        // A standard definition frame (768 x 576) stays whole from 8 GB up.
        assert_eq!(denoiser_tile_for_budget(8 * 1024, 4), 768);
        assert_eq!(denoiser_tile_for_budget(8 * 1024, 2), 1024);
        assert_eq!(denoiser_tile_for_budget(4 * 1024, 4), 512);
        assert_eq!(denoiser_tile_for_budget(256, 4), 128);
    }

    #[test]
    fn chw_round_trips_and_clamps() {
        let mut p = Plane::new(5, 3, 3);
        for (i, v) in p.data.iter_mut().enumerate() {
            *v = (i % 97) as f32 / 97.0;
        }
        let back = from_chw(&to_chw(&p), 5, 3, 3);
        assert_eq!(back, p);
        let over = from_chw(&[-0.5, 1.5, 0.25], 1, 1, 3);
        assert_eq!(over.data, [0.0, 1.0, 0.25]);
    }

    #[test]
    fn the_direct_route_gives_exactly_what_the_step_by_step_one_does() {
        // A picture that needs padding in both directions, and a tiny one whose
        // mirror image has to repeat.
        for (w, h) in [(37, 19), (3, 2), (32, 16), (1, 1)] {
            let rgb: Vec<u8> = (0..w * h * 3).map(|i| (i * 31 % 256) as u8).collect();
            let (pw, ph) = (round_up(w, 16), round_up(h, 16));
            let values: [f32; 256] = std::array::from_fn(|v| v as f32 / 255.0);
            let direct = chw_from_rgb8(&rgb, (w, h), (pw, ph), &values);
            let stepped = to_chw(&pad_reflect(&plane_from_rgb8(&rgb, w, h), 16));
            assert_eq!(direct, stepped, "in {w}x{h}");
            // Back out, with some samples outside 0..=1.
            let out: Vec<f32> = direct.iter().map(|v| v * 1.2 - 0.1).collect();
            let direct = rgb8_from_chw(&out, (pw, ph), (w, h), byte);
            let whole = from_chw(&out, pw, ph, 3);
            let stepped: Vec<u8> = rgb8_from_plane(&whole)
                .chunks(pw * 3)
                .take(h)
                .flat_map(|row| row[..w * 3].to_vec())
                .collect();
            assert_eq!(direct, stepped, "out {w}x{h}");
        }
    }

    #[test]
    fn eight_bit_round_trip_is_exact() {
        let rgb: Vec<u8> = (0..=255).collect();
        let p = plane_from_rgb8(&rgb[..255], 85, 1);
        assert_eq!(rgb8_from_plane(&p), &rgb[..255]);
    }
}
