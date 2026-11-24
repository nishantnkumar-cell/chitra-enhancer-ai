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

/// Two consecutive frames this far apart (mean absolute difference, 0..1) are
/// different shots. Sensor noise is well below 0.05 and a fast pan below 0.12.
const CUT: f32 = 0.2;

/// Mean absolute difference between two pictures of the same size, from a
/// sample of their pixels (every fourth in each direction).
pub fn difference(a: &Plane, b: &Plane) -> f32 {
    let (mut sum, mut n) = (0.0f64, 0u64);
    for y in (0..a.h).step_by(4) {
        for x in (0..a.w).step_by(4) {
            let i = (y * a.w + x) * a.c;
            for k in 0..a.c {
                sum += (a.data[i + k] - b.data[i + k]).abs() as f64;
                n += 1;
            }
        }
    }
    if n == 0 { 0.0 } else { (sum / n as f64) as f32 }
}

/// A sliding window over a stream of frames. A cut between shots ends the
/// neighbourhood: frames from the other shot are replaced by the nearest frame
/// of the same shot, so the denoiser never blends two scenes.
#[derive(Default)]
pub struct Window {
    frames: VecDeque<Plane>,
    /// Shot number of each held frame.
    shots: VecDeque<usize>,
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
        let shot = match (self.frames.back(), self.shots.back()) {
            (Some(prev), Some(&sh)) if prev.w == p.w && difference(prev, &p) > CUT => sh + 1,
            (_, Some(&sh)) => sh,
            _ => 0,
        };
        self.frames.push_back(p);
        self.shots.push_back(shot);
        self.total += 1;
    }

    /// The frames around the next one to clean, or None if more frames are
    /// needed first. When `last` is set no more will come, so the end repeats.
    pub fn next(&self, last: bool) -> Option<[&Plane; FRAMES]> {
        if self.centre >= self.total || (!last && self.centre + REACH >= self.total) {
            return None;
        }
        let shot = self.shots[self.centre - self.base];
        let at = |d: usize| {
            let mut want = (self.centre + d).saturating_sub(REACH).min(self.total - 1);
            // Step back towards the centre until we are in the same shot.
            while self.shots[want - self.base] != shot {
                want = if want > self.centre {
                    want - 1
                } else {
                    want + 1
                };
            }
            &self.frames[want - self.base]
        };
        Some([at(0), at(1), at(2), at(3), at(4)])
    }

    /// Moves on to the next frame and lets go of those no longer needed.
    pub fn advance(&mut self) {
        self.centre += 1;
        while self.base + REACH < self.centre && self.frames.len() > 1 {
            self.frames.pop_front();
            self.shots.pop_front();
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
    fn a_cut_between_shots_is_never_blended_across() {
        // Four frames of a dark shot (0.1), then four of a bright one (0.9).
        let mut win = Window::new();
        for v in [0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9] {
            win.push(frame(v));
        }
        let mut seen = Vec::new();
        while let Some(w) = win.next(true) {
            seen.push(w.map(|p| p.data[0]));
            win.advance();
        }
        assert_eq!(seen.len(), 8);
        for (i, w) in seen.iter().enumerate() {
            let want = if i < 4 { 0.1 } else { 0.9 };
            assert!(w.iter().all(|&v| v == want), "frame {i} saw {w:?}");
        }
    }

    #[test]
    fn noise_and_gentle_motion_are_not_cuts() {
        let mut win = Window::new();
        for i in 0..8 {
            win.push(frame(0.4 + 0.02 * (i % 3) as f32 + 0.01 * i as f32));
        }
        let shots: Vec<usize> = win.shots.iter().copied().collect();
        assert!(shots.iter().all(|&s| s == 0), "{shots:?}");
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
