#![no_main]
use libfuzzer_sys::fuzz_target;

// The inspector reads quantisation tables, subsampling, EXIF and ICC markers
// from untrusted files: it must never panic, whatever the bytes.
fuzz_target!(|data: &[u8]| {
    if let Some(info) = chitra_enhance::inspector::inspect(data) {
        let _ = (info.tier(), info.estimated_quality());
    }
});
