//! The part that makes a frame better and larger. A frame goes in as RGB bytes
//! and comes out `scale` times bigger. The plain engine is a Lanczos resize
//! (no AI); the ONNX engine (feature `onnx`) runs a model on the GPU.

use anyhow::{Context, Result};
use fast_image_resize::images::Image;
use fast_image_resize::{FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer};

pub trait Engine {
    /// How many times larger the frames come out.
    fn scale(&self) -> usize;
    /// One frame of `w` x `h` RGB bytes in, `w*scale` x `h*scale` out.
    fn process(&mut self, rgb: &[u8], w: usize, h: usize) -> Result<Vec<u8>>;
    /// A line for the log.
    fn describe(&self) -> String;
}

/// A Lanczos3 resize: the baseline, and the choice when there is no model.
pub struct Lanczos {
    scale: usize,
    resizer: Resizer,
}

impl Lanczos {
    pub fn new(scale: usize) -> Lanczos {
        Lanczos {
            scale: scale.max(1),
            resizer: Resizer::new(),
        }
    }
}

impl Engine for Lanczos {
    fn scale(&self) -> usize {
        self.scale
    }

    fn process(&mut self, rgb: &[u8], w: usize, h: usize) -> Result<Vec<u8>> {
        if self.scale == 1 {
            return Ok(rgb.to_vec());
        }
        let src = Image::from_vec_u8(w as u32, h as u32, rgb.to_vec(), PixelType::U8x3)
            .context("a frame of the wrong size")?;
        let mut dst = Image::new(
            (w * self.scale) as u32,
            (h * self.scale) as u32,
            PixelType::U8x3,
        );
        self.resizer
            .resize(
                &src,
                &mut dst,
                &ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Lanczos3)),
            )
            .context("resizing a frame")?;
        Ok(dst.into_vec())
    }

    fn describe(&self) -> String {
        format!("Lanczos {}x (no AI model)", self.scale)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lanczos_scales_and_keeps_flat_colour() {
        let mut e = Lanczos::new(2);
        let frame = vec![100u8; 16 * 8 * 3];
        let out = e.process(&frame, 16, 8).unwrap();
        assert_eq!(out.len(), 32 * 16 * 3);
        assert!(out.iter().all(|&v| (99..=101).contains(&v)));
        let same = Lanczos::new(1).process(&frame, 16, 8).unwrap();
        assert_eq!(same, frame);
    }
}
