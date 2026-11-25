#![no_main]
use libfuzzer_sys::fuzz_target;

// Sectors come straight off an untrusted disc: the copy-protection check walks
// packet lengths in them and must never read out of bounds.
fuzz_target!(|data: &[u8]| {
    let _ = chitra_enhance::disc::scrambled_sector(data);
    // Also with a pack header put in front, so the walk is actually reached.
    let mut s = vec![0u8; 2048];
    s[..5].copy_from_slice(&[0, 0, 1, 0xBA, 0x44]);
    let n = data.len().min(2048 - 13);
    s[13..13 + n].copy_from_slice(&data[..n]);
    let _ = chitra_enhance::disc::scrambled_sector(&s);
});
