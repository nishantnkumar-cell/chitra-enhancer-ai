#![no_main]
use libfuzzer_sys::fuzz_target;

// Lines the GUI reads from the command-line program, and ffprobe's JSON.
fuzz_target!(|data: &[u8]| {
    let s = String::from_utf8_lossy(data);
    for line in s.lines() {
        let _ = chitra_enhance::job::parse_progress(line);
        let _ = chitra_enhance::job::parse_noise(line);
        let _ = chitra_enhance::job::parse_title(line);
        let _ = chitra_enhance::job::parse_analysis(line);
        let _ = chitra_enhance::job::quote(line);
    }
    let _ = chitra_enhance::ffmpeg::parse_probe(&s);
});
