//! Multi-frame (temporal) denoising. A model such as FastDVDnet looks at five
//! consecutive frames to clean the middle one, so noise that changes from frame
//! to frame is averaged away while real detail, which stays put, is kept.
//!
//! `Window` feeds the model: it holds the frames around the one being cleaned
//! and repeats the first or last frame at the ends of a clip.

use std::collections::VecDeque;

use anyhow::Result;

use crate::tiler::Plane;

/// Frames on each side of the one being cleaned.
pub const REACH: usize = 2;
/// Frames the model sees at once.
pub const FRAMES: usize = 2 * REACH + 1;

pub trait Temporal {
    /// Cleans `w[REACH]` using its neighbours; all five are the same size.
    fn denoise(&mut self, w: &[&Plane; FRAMES]) -> Result<Plane>;
    fn describe(&self) -> String;
}

/// A sliding window over a stream of frames.
#[derive(Default)]
pub struct Window {
    frames: VecDeque<Plane>,
    /// Index (in the whole stream) of the first frame still held.
    base: usize,
    total: usize,
    centre: usize,
}

impl Window {
    pub fn new() -> Window {
        Window::default()
    }

    pub fn push(&mut self, p: Plane) {
        self.frames.push_back(p);
        self.total += 1;
    }

    /// The frames around the next one to clean, or None if more frames are
    /// needed first. When `last` is set no more will come, so the end repeats.
    pub fn next(&self, last: bool) -> Option<[&Plane; FRAMES]> {
        if self.centre >= self.total || (!last && self.centre + REACH >= self.total) {
            return None;
        }
        let at = |d: usize| {
            let want = (self.centre + d).saturating_sub(REACH).min(self.total - 1);
            &self.frames[want - self.base]
        };
        Some([at(0), at(1), at(2), at(3), at(4)])
    }

    /// Moves on to the next frame and lets go of those no longer needed.
    pub fn advance(&mut self) {
        self.centre += 1;
        while self.base + REACH < self.centre && self.frames.len() > 1 {
            self.frames.pop_front();
            self.base += 1;
        }
    }
}

/// The five frames as one picture with 15 channels (frame 0's RGB first).
pub fn stack(w: &[&Plane; FRAMES]) -> Plane {
    let (pw, ph) = (w[0].w, w[0].h);
    let mut out = Plane::new(pw, ph, 3 * FRAMES);
    for i in 0..pw * ph {
        for (f, p) in w.iter().enumerate() {
            out.data[i * 3 * FRAMES + f * 3..i * 3 * FRAMES + f * 3 + 3]
                .copy_from_slice(&p.data[i * 3..i * 3 + 3]);
        }
    }
    out
}

/// Stand-in used by tests: returns the middle frame untouched.
pub struct Identity;

impl Temporal for Identity {
    fn denoise(&mut self, w: &[&Plane; FRAMES]) -> Result<Plane> {
        Ok(w[REACH].clone())
    }
    fn describe(&self) -> String {
        "identity".into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(v: f32) -> Plane {
        let mut p = Plane::new(2, 1, 3);
        p.data.fill(v);
        p
    }

    /// Runs `n` frames (value = index / 100) through the window.
    fn windows(n: usize) -> Vec<[f32; FRAMES]> {
        let mut win = Window::new();
        let mut out = Vec::new();
        for i in 0..n {
            win.push(frame(i as f32 / 100.0));
            while let Some(w) = win.next(false) {
                out.push(w.map(|p| p.data[0]));
                win.advance();
            }
        }
        while let Some(w) = win.next(true) {
            out.push(w.map(|p| p.data[0]));
            win.advance();
        }
        out
    }

    #[test]
    fn every_frame_comes_out_once_in_order_with_the_ends_repeated() {
        let w = windows(6);
        assert_eq!(w.len(), 6);
        for (i, f) in w.iter().enumerate() {
            assert_eq!(f[REACH], i as f32 / 100.0, "the middle is frame {i}");
        }
        let at = |v: usize| v as f32 / 100.0;
        assert_eq!(w[0], [at(0), at(0), at(0), at(1), at(2)]);
        assert_eq!(w[3], [at(1), at(2), at(3), at(4), at(5)]);
        assert_eq!(w[5], [at(3), at(4), at(5), at(5), at(5)]);
    }

    #[test]
    fn clips_shorter_than_the_window_still_work() {
        assert_eq!(windows(1), vec![[0.0; FRAMES]]);
        assert_eq!(windows(2).len(), 2);
        assert!(windows(0).is_empty());
    }

    #[test]
    fn stacking_keeps_each_frame_in_its_own_channels() {
        let fs: Vec<Plane> = (0..5).map(|i| frame(i as f32)).collect();
        let r: [&Plane; FRAMES] = std::array::from_fn(|i| &fs[i]);
        let s = stack(&r);
        assert_eq!((s.w, s.h, s.c), (2, 1, 15));
        assert_eq!(
            &s.data[..15],
            &[0., 0., 0., 1., 1., 1., 2., 2., 2., 3., 3., 3., 4., 4., 4.]
        );
    }
}
