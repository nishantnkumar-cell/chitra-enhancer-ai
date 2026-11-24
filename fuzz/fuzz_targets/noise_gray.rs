#![no_main]
use libfuzzer_sys::fuzz_target;

// First two bytes pick a width and height; the rest is the picture.
fuzz_target!(|data: &[u8]| {
    if data.len() < 2 {
        return;
    }
    let (w, h) = (data[0] as usize + 1, data[1] as usize + 1);
    let _ = chitra_enhance::noise::estimate_gray(w, h, &data[2..]);
});
