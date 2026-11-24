//! Looking inside a JPEG without decoding it: size, chroma subsampling, the
//! quantisation tables (which say how hard it was compressed), and whether it
//! carries EXIF and an ICC colour profile. Every read is bounds-checked.

/// The standard luminance table (JPEG Annex K) in natural (row by row) order.
const STD_LUMA: [u16; 64] = [
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113,
    92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
];

/// Position in the zigzag order of each natural-order index.
const ZIGZAG: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tier {
    /// High quality: little to repair, so upscale without touching the detail.
    Clean,
    /// Visibly compressed (blocking, ringing, smeared chroma): clean up as well.
    Compressed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Subsampling {
    S444,
    S422,
    S420,
    Other,
}

#[derive(Clone, Debug, PartialEq)]
pub struct JpegInfo {
    pub width: u32,
    pub height: u32,
    pub subsampling: Subsampling,
    /// The IJG quality (1-100) the luminance table matches, if it matches one.
    pub quality: Option<u8>,
    /// Mean of the luminance table; small means gentle compression.
    pub mean_quant: f32,
    pub has_exif: bool,
    pub has_icc: bool,
    pub progressive: bool,
}

impl JpegInfo {
    pub fn tier(&self) -> Tier {
        let q = self.estimated_quality();
        let subsampled = matches!(self.subsampling, Subsampling::S420 | Subsampling::S422);
        if q < 85 || (subsampled && q < 92) {
            Tier::Compressed
        } else {
            Tier::Clean
        }
    }

    /// The quality that matches the table, else one guessed from its mean.
    pub fn estimated_quality(&self) -> u8 {
        if let Some(q) = self.quality {
            return q;
        }
        // Custom tables (cameras, editors): judge by how coarse they are.
        match self.mean_quant {
            m if m <= 2.5 => 97,
            m if m <= 4.0 => 92,
            m if m <= 7.0 => 85,
            m if m <= 12.0 => 75,
            m if m <= 20.0 => 60,
            _ => 40,
        }
    }
}

/// The table the IJG library makes for `quality`, in natural order.
fn ijg_table(quality: u32) -> [u16; 64] {
    let q = quality.clamp(1, 100);
    let scale = if q < 50 { 5000 / q } else { 200 - q * 2 };
    let mut t = [0u16; 64];
    for (o, &s) in t.iter_mut().zip(STD_LUMA.iter()) {
        *o = ((u32::from(s) * scale + 50) / 100).clamp(1, 255) as u16;
    }
    t
}

/// The IJG quality whose table best equals `natural`, when the fit is close.
fn match_quality(natural: &[u16; 64]) -> Option<u8> {
    let mut best = (u32::MAX, 0u32);
    for q in 1..=100u32 {
        let t = ijg_table(q);
        let err: u32 = t
            .iter()
            .zip(natural)
            .map(|(&a, &b)| u32::from(a.abs_diff(b)))
            .sum();
        if err < best.0 {
            best = (err, q);
        }
    }
    // Allow a small rounding difference (an average of 0.25 per entry).
    (best.0 <= 16).then_some(best.1 as u8)
}

fn be16(d: &[u8], at: usize) -> Option<usize> {
    Some(usize::from(*d.get(at)?) << 8 | usize::from(*d.get(at + 1)?))
}

/// Reads the headers of a JPEG file; `None` if it is not a JPEG or is cut short
/// before its frame header.
pub fn inspect(data: &[u8]) -> Option<JpegInfo> {
    if data.get(..2)? != [0xFF, 0xD8] {
        return None;
    }
    let mut pos = 2;
    let mut luma: Option<[u16; 64]> = None;
    let (mut w, mut h, mut sub, mut progressive) = (0u32, 0u32, Subsampling::Other, false);
    let (mut has_exif, mut has_icc, mut frame) = (false, false, false);
    while pos + 4 <= data.len() {
        if data[pos] != 0xFF {
            pos += 1;
            continue;
        }
        let marker = data[pos + 1];
        if marker == 0xFF {
            pos += 1;
            continue;
        }
        if marker == 0xD8 || marker == 0x01 || (0xD0..=0xD7).contains(&marker) || marker == 0x00 {
            pos += 2;
            continue;
        }
        if marker == 0xD9 || marker == 0xDA {
            break;
        }
        let len = be16(data, pos + 2)?;
        if len < 2 {
            return None;
        }
        let seg = data.get(pos + 4..pos + 2 + len)?;
        match marker {
            0xDB => {
                // One or more tables: a byte (precision, id), then 64 values.
                let mut s = seg;
                while let Some((&pq_tq, rest)) = s.split_first() {
                    let wide = pq_tq >> 4 != 0;
                    let id = pq_tq & 15;
                    let n = if wide { 128 } else { 64 };
                    let vals = rest.get(..n)?;
                    if id == 0 && luma.is_none() {
                        let mut nat = [0u16; 64];
                        for i in 0..64 {
                            let v = if wide {
                                be16(vals, i * 2)? as u16
                            } else {
                                u16::from(vals[i])
                            };
                            nat[ZIGZAG[i]] = v;
                        }
                        luma = Some(nat);
                    }
                    s = &rest[n..];
                }
            }
            0xC0..=0xC2 => {
                progressive = marker == 0xC2;
                h = be16(seg, 1)? as u32;
                w = be16(seg, 3)? as u32;
                let n = usize::from(*seg.get(5)?);
                let comp = |i: usize| seg.get(6 + i * 3 + 1).copied();
                if n == 3 {
                    let (y, cb, cr) = (comp(0)?, comp(1)?, comp(2)?);
                    sub = match (y, cb == 0x11 && cr == 0x11) {
                        (0x11, true) => Subsampling::S444,
                        (0x21, true) => Subsampling::S422,
                        (0x22, true) => Subsampling::S420,
                        _ => Subsampling::Other,
                    };
                } else if n == 1 {
                    sub = Subsampling::S444;
                }
                frame = true;
            }
            0xE1 => has_exif |= seg.starts_with(b"Exif\0\0"),
            0xE2 => has_icc |= seg.starts_with(b"ICC_PROFILE\0"),
            _ => {}
        }
        pos += 2 + len;
    }
    if !frame || w == 0 || h == 0 {
        return None;
    }
    let table = luma?;
    let mean = table.iter().map(|&v| f32::from(v)).sum::<f32>() / 64.0;
    Some(JpegInfo {
        width: w,
        height: h,
        subsampling: sub,
        quality: match_quality(&table),
        mean_quant: mean,
        has_exif,
        has_icc,
        progressive,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::codecs::jpeg::JpegEncoder;
    use image::{ExtendedColorType, ImageEncoder};

    fn jpeg(q: u8, w: u32, h: u32) -> Vec<u8> {
        let px: Vec<u8> = (0..w * h * 3)
            .map(|i| ((i * 7 + i / 3 * 13) % 251) as u8)
            .collect();
        let mut out = Vec::new();
        JpegEncoder::new_with_quality(&mut out, q)
            .write_image(&px, w, h, ExtendedColorType::Rgb8)
            .unwrap();
        out
    }

    #[test]
    fn the_ijg_tables_are_the_known_ones() {
        assert_eq!(ijg_table(50), STD_LUMA);
        assert_eq!(ijg_table(100), [1u16; 64]);
        assert_eq!(ijg_table(90)[0], 3);
        assert_eq!(ijg_table(10)[0], 80);
    }

    #[test]
    fn quality_is_recovered_from_real_jpegs() {
        for q in [20, 50, 75, 85, 90, 95, 98] {
            let info = inspect(&jpeg(q, 40, 24)).unwrap();
            assert_eq!((info.width, info.height), (40, 24));
            let got = info.quality.unwrap_or(0);
            assert!(got.abs_diff(q) <= 1, "asked {q}, got {got}");
        }
    }

    #[test]
    fn tiers_follow_quality_and_chroma() {
        assert_eq!(inspect(&jpeg(95, 16, 16)).unwrap().tier(), Tier::Clean);
        assert_eq!(inspect(&jpeg(60, 16, 16)).unwrap().tier(), Tier::Compressed);
        let mut i = inspect(&jpeg(95, 16, 16)).unwrap();
        i.subsampling = Subsampling::S420;
        i.quality = Some(90);
        assert_eq!(
            i.tier(),
            Tier::Compressed,
            "4:2:0 at 90 still shows smeared colour"
        );
        i.quality = None;
        i.mean_quant = 3.0;
        assert_eq!(i.estimated_quality(), 92);
    }

    #[test]
    fn exif_and_icc_markers_are_found() {
        let base = jpeg(90, 8, 8);
        let mut d = base[..2].to_vec();
        let exif = b"Exif\0\0MM\0*";
        d.extend([0xFF, 0xE1, 0, (exif.len() + 2) as u8]);
        d.extend(exif);
        let icc = b"ICC_PROFILE\0\x01\x01xx";
        d.extend([0xFF, 0xE2, 0, (icc.len() + 2) as u8]);
        d.extend(icc);
        d.extend(&base[2..]);
        let i = inspect(&d).unwrap();
        assert!(i.has_exif && i.has_icc);
        let plain = inspect(&base).unwrap();
        assert!(!plain.has_icc);
    }

    #[test]
    fn junk_and_truncated_files_are_refused_not_panicked_on() {
        assert!(inspect(b"").is_none());
        assert!(inspect(b"\x89PNG\r\n").is_none());
        let good = jpeg(80, 32, 32);
        for cut in 0..good.len().min(700) {
            let _ = inspect(&good[..cut]);
        }
        assert!(inspect(&good[..20]).is_none());
        let mut bad = good.clone();
        bad[4] = 0xFF;
        bad[5] = 0xFF;
        let _ = inspect(&bad);
    }
}
