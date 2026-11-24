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

/// Resizes `rgb` (w x h, 8-bit RGB) to `nw` x `nh` with Lanczos3.
pub fn resize_rgb(
    resizer: &mut Resizer,
    rgb: &[u8],
    w: usize,
    h: usize,
    nw: usize,
    nh: usize,
) -> Result<Vec<u8>> {
    let src = Image::from_vec_u8(w as u32, h as u32, rgb.to_vec(), PixelType::U8x3)
        .context("a frame of the wrong size")?;
    let mut dst = Image::new(nw as u32, nh as u32, PixelType::U8x3);
    resizer
        .resize(
            &src,
            &mut dst,
            &ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Lanczos3)),
        )
        .context("resizing a frame")?;
    Ok(dst.into_vec())
}

/// An engine whose result is resized to `target` times the input, for a model
/// that enlarges by another factor (a 4x model used for a 2x result is
/// usually sharper than a model trained for 2x).
pub struct Rescaled {
    inner: Box<dyn Engine>,
    target: usize,
    resizer: Resizer,
}

impl Rescaled {
    pub fn wrap(inner: Box<dyn Engine>, target: usize) -> Box<dyn Engine> {
        if inner.scale() == target {
            return inner;
        }
        Box::new(Rescaled {
            inner,
            target,
            resizer: Resizer::new(),
        })
    }
}

impl Engine for Rescaled {
    fn scale(&self) -> usize {
        self.target
    }

    fn process(&mut self, rgb: &[u8], w: usize, h: usize) -> Result<Vec<u8>> {
        let s = self.inner.scale();
        let big = self.inner.process(rgb, w, h)?;
        resize_rgb(
            &mut self.resizer,
            &big,
            w * s,
            h * s,
            w * self.target,
            h * self.target,
        )
    }

    fn describe(&self) -> String {
        format!("{}, resized to {}x", self.inner.describe(), self.target)
    }
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

    #[test]
    fn a_4x_engine_can_give_a_2x_result() {
        let mut e = Rescaled::wrap(Box::new(Lanczos::new(4)), 2);
        assert_eq!(e.scale(), 2);
        let frame = vec![90u8; 16 * 8 * 3];
        let out = e.process(&frame, 16, 8).unwrap();
        assert_eq!(out.len(), 32 * 16 * 3);
        assert!(out.iter().all(|&v| (89..=91).contains(&v)));
        // Same scale: no wrapper, no extra work.
        assert_eq!(
            Rescaled::wrap(Box::new(Lanczos::new(2)), 2).describe(),
            "Lanczos 2x (no AI model)"
        );
    }
}
