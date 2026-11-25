#![no_main]
use libfuzzer_sys::fuzz_target;

// Decoding a photo (any supported format) must fail cleanly, never panic or
// allocate without limit.
fuzz_target!(|data: &[u8]| {
    // A small size limit: memory for a large picture is bounded and expected,
    // and would only stop the fuzzer.
    let _ = chitra_enhance::photo::decode_with_limit(data, 4_000_000);
});
