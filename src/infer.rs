//! The parts of running a model that need no ONNX Runtime: how a frame is laid
//! out as a tensor, half-precision conversion, and how big a tile may be.

use crate::tiler::Plane;

/// Tile sizes to try, largest first. After an out-of-memory error the next
/// smaller one is used (the spec: 512, then 256, then 128).
pub const TILE_LADDER: [usize; 3] = [512, 256, 128];

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

/// A first guess for the largest tile that fits `vram_mb` megabytes for a model
/// that makes `scale`x larger pictures in `bytes` per sample (2 for FP16).
/// A convolutional network keeps a few dozen feature maps alive; 48 channels at
/// full resolution plus the enlarged output is a deliberately cautious model.
pub fn tile_for_budget(vram_mb: usize, scale: usize, bytes: usize) -> usize {
    let budget = vram_mb as f64 * 1024.0 * 1024.0 * 0.8;
    for &t in &TILE_LADDER {
        let px = (t * t) as f64;
        let need = px * (48.0 * 6.0 + 3.0 * (scale * scale) as f64 * 2.0) * bytes as f64;
        if need <= budget {
            return t;
        }
    }
    *TILE_LADDER.last().unwrap()
}

/// The Plane (interleaved HWC) as planar CHW, the layout models expect.
pub fn to_chw(p: &Plane) -> Vec<f32> {
    let (w, h, c) = (p.w, p.h, p.c);
    let mut out = vec![0.0f32; w * h * c];
    for y in 0..h {
        for x in 0..w {
            for k in 0..c {
                out[k * w * h + y * w + x] = p.data[(y * w + x) * c + k];
            }
        }
    }
    out
}

/// Planar CHW back to an interleaved Plane; values are clamped to 0..=1,
/// since a model may overshoot a little.
pub fn from_chw(data: &[f32], w: usize, h: usize, c: usize) -> Plane {
    assert_eq!(data.len(), w * h * c);
    let mut p = Plane::new(w, h, c);
    for y in 0..h {
        for x in 0..w {
            for k in 0..c {
                p.data[(y * w + x) * c + k] = data[k * w * h + y * w + x].clamp(0.0, 1.0);
            }
        }
    }
    p
}

/// 8-bit RGB (interleaved) to a Plane in 0..=1. The models are trained on
/// gamma-encoded sRGB, so no linearisation is done.
pub fn plane_from_rgb8(rgb: &[u8], w: usize, h: usize) -> Plane {
    Plane {
        w,
        h,
        c: 3,
        data: rgb.iter().map(|&v| f32::from(v) / 255.0).collect(),
    }
}

pub fn rgb8_from_plane(p: &Plane) -> Vec<u8> {
    p.data
        .iter()
        .map(|&v| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_tile_ladder_steps_down_to_the_bottom() {
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
        assert_eq!(big, 512);
        assert_eq!(tiny, 128);
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
    fn eight_bit_round_trip_is_exact() {
        let rgb: Vec<u8> = (0..=255).collect();
        let p = plane_from_rgb8(&rgb[..255], 85, 1);
        assert_eq!(rgb8_from_plane(&p), &rgb[..255]);
    }
}
