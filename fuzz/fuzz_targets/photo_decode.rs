#![no_main]
use libfuzzer_sys::fuzz_target;

// Decoding a photo (any supported format) must fail cleanly, never panic or
// allocate without limit.
fuzz_target!(|data: &[u8]| {
    let _ = chitra_enhance::photo::decode(data);
});
