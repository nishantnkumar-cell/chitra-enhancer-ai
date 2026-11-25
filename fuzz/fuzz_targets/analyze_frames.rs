#![no_main]
use chitra_enhance::analyze::{self, Frame};
use libfuzzer_sys::fuzz_target;

// Decoded pictures of any size and content: the measurements index into them in
// many ways. First two bytes pick an even width and height (16..=270).
fuzz_target!(|data: &[u8]| {
    if data.len() < 3 {
        return;
    }
    let (w, h) = (16 + (data[0] as usize & !1), 16 + (data[1] as usize & !1));
    let body = &data[2..];
    let plane = |n: usize, k: usize| -> Vec<u8> {
        (0..n).map(|i| body[(i * k) % body.len()]).collect()
    };
    let frames: Vec<Frame> = (1..4)
        .map(|k| Frame {
            y: plane(w * h, k),
            u: plane(w * h / 4, k + 3),
            v: plane(w * h / 4, k + 5),
        })
        .collect();
    let _ = analyze::measure(&frames, w, h);
});
