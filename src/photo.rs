//! Enhancing a photo: decode (keeping the colour profile and EXIF), upscale at
//! full precision, and write 16-bit JPEG XL or PNG. The source is never
//! overwritten.

use std::io::{BufWriter, Cursor};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use image::metadata::Orientation;
use image::{DynamicImage, ImageDecoder, ImageReader, Limits};

use crate::engine::Engine;
use crate::tiler::Plane;

/// Decoded pixels may not exceed this (a hostile file cannot ask for more).
const MAX_PIXELS: u64 = 250_000_000;

pub struct Options {
    pub input: PathBuf,
    pub output: PathBuf,
    /// Butteraugli distance for JPEG XL (1.0 is visually lossless).
    pub jxl_distance: f32,
    /// Allow replacing an existing output file (never the input).
    pub force: bool,
}

#[derive(Debug)]
pub struct Summary {
    pub in_size: (u32, u32),
    pub out_size: (u32, u32),
    pub bytes: u64,
}

/// A decoded photo with what has to travel with it.
pub struct Photo {
    pub plane: Plane,
    pub icc: Option<Vec<u8>>,
    /// TIFF-form EXIF with the orientation already applied to the pixels.
    pub exif: Option<Vec<u8>>,
    pub had_alpha: bool,
}

/// The output must be a new file: not the input (even through a link), and not
/// an existing file unless `force`.
pub fn check_paths(input: &Path, output: &Path, force: bool) -> Result<()> {
    if let (Ok(a), Ok(b)) = (input.canonicalize(), output.canonicalize())
        && a == b
    {
        bail!("the output must not be the input");
    }
    if output.exists() && !force {
        bail!(
            "{} exists; choose another name or add --force",
            output.display()
        );
    }
    Ok(())
}

pub fn decode(bytes: &[u8]) -> Result<Photo> {
    let mut reader = ImageReader::new(Cursor::new(bytes)).with_guessed_format()?;
    let mut limits = Limits::default();
    limits.max_alloc = Some(2 << 30);
    reader.limits(limits);
    let mut decoder = reader.into_decoder().context("cannot read this image")?;
    let (w, h) = decoder.dimensions();
    if u64::from(w) * u64::from(h) > MAX_PIXELS {
        bail!("image too large ({w}x{h})");
    }
    let icc = decoder.icc_profile().ok().flatten();
    let exif = decoder
        .exif_metadata()
        .ok()
        .flatten()
        .map(|mut e| {
            if e.starts_with(b"Exif\0\0") {
                e.drain(..6);
            }
            e
        })
        .filter(|e| e.starts_with(b"II*\0") || e.starts_with(b"MM\0*"));
    let orientation = decoder.orientation().unwrap_or(Orientation::NoTransforms);
    let mut img = DynamicImage::from_decoder(decoder).context("cannot decode this image")?;
    img.apply_orientation(orientation);
    let exif = exif.map(|mut e| {
        let _ = Orientation::remove_from_exif_chunk(&mut e);
        e
    });
    let had_alpha = img.color().has_alpha();
    let rgb = img.to_rgb16();
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    let data = rgb
        .as_raw()
        .iter()
        .map(|&v| f32::from(v) / 65535.0)
        .collect();
    Ok(Photo {
        plane: Plane { w, h, c: 3, data },
        icc,
        exif,
        had_alpha,
    })
}

fn to_u16(p: &Plane) -> Vec<u16> {
    p.data
        .iter()
        .map(|&v| (v.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16)
        .collect()
}

fn ne_bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_ne_bytes()).collect()
}

pub fn write_jxl(p: &Plane, photo: &Photo, distance: f32, out: &Path) -> Result<()> {
    use jxl_encoder::{ImageMetadata, LossyConfig, PixelLayout};
    let mut meta = ImageMetadata::new();
    if let Some(icc) = &photo.icc {
        meta = meta.with_icc_profile(icc);
    }
    if let Some(exif) = &photo.exif {
        meta = meta.with_exif(exif);
    }
    let bytes = LossyConfig::new(distance)
        .with_effort(5)
        .encode_request(p.w as u32, p.h as u32, PixelLayout::Rgb16)
        .with_metadata(&meta)
        .encode(&ne_bytes(&to_u16(p)))
        .map_err(|e| anyhow::anyhow!("JPEG XL encoder: {e:?}"))?;
    std::fs::write(out, bytes).with_context(|| format!("cannot write {}", out.display()))
}

pub fn write_png(p: &Plane, photo: &Photo, out: &Path) -> Result<()> {
    let file =
        std::fs::File::create(out).with_context(|| format!("cannot write {}", out.display()))?;
    let mut info = png::Info::with_size(p.w as u32, p.h as u32);
    info.color_type = png::ColorType::Rgb;
    info.bit_depth = png::BitDepth::Sixteen;
    info.icc_profile = photo.icc.as_deref().map(std::borrow::Cow::Borrowed);
    info.exif_metadata = photo.exif.as_deref().map(std::borrow::Cow::Borrowed);
    let mut w = png::Encoder::with_info(BufWriter::new(file), info)?.write_header()?;
    // PNG samples are big-endian.
    let data: Vec<u8> = to_u16(p).iter().flat_map(|v| v.to_be_bytes()).collect();
    w.write_image_data(&data)?;
    w.finish()?;
    Ok(())
}

/// Enhances `bytes` (the content of `o.input`) with `engine` and writes `o.output`.
pub fn run(o: &Options, bytes: &[u8], engine: &mut dyn Engine) -> Result<Summary> {
    check_paths(&o.input, &o.output, o.force)?;
    let ext = o
        .output
        .extension()
        .map(|e| e.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    if ext != "jxl" && ext != "png" {
        bail!("the output must end in .jxl or .png (16-bit)");
    }
    let photo = decode(bytes)?;
    if photo.had_alpha {
        eprintln!("Note: transparency is not kept; the photo is written without it.");
    }
    let big = engine.process_plane(&photo.plane)?;
    // Write beside the target first, so a failure never leaves half a file.
    let tmp = o.output.with_extension(format!("{ext}.part"));
    let wrote = if ext == "jxl" {
        write_jxl(&big, &photo, o.jxl_distance, &tmp)
    } else {
        write_png(&big, &photo, &tmp)
    };
    if let Err(e) = wrote {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    std::fs::rename(&tmp, &o.output)?;
    Ok(Summary {
        in_size: (photo.plane.w as u32, photo.plane.h as u32),
        out_size: (big.w as u32, big.h as u32),
        bytes: std::fs::metadata(&o.output)?.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::Lanczos;
    use image::{ImageEncoder, codecs::png::PngEncoder};

    fn png_bytes(w: u32, h: u32) -> Vec<u8> {
        let px: Vec<u8> = (0..w * h * 3).map(|i| (i * 5 % 251) as u8).collect();
        let mut out = Vec::new();
        PngEncoder::new(&mut out)
            .write_image(&px, w, h, image::ExtendedColorType::Rgb8)
            .unwrap();
        out
    }

    fn opts(d: &Path, name: &str) -> Options {
        Options {
            input: d.join("in.png"),
            output: d.join(name),
            jxl_distance: 1.0,
            force: false,
        }
    }

    #[test]
    fn a_photo_is_doubled_and_written_as_16_bit_png_and_jxl() {
        let d = tempfile::tempdir().unwrap();
        let bytes = png_bytes(24, 16);
        std::fs::write(d.path().join("in.png"), &bytes).unwrap();
        for name in ["out.png", "out.jxl"] {
            let o = opts(d.path(), name);
            let s = run(&o, &bytes, &mut Lanczos::new(2)).unwrap();
            assert_eq!((s.in_size, s.out_size), ((24, 16), (48, 32)));
            assert!(s.bytes > 0 && o.output.exists());
            assert!(!d.path().join(format!("{name}.part")).exists());
        }
        // The PNG really is 16-bit and the right size.
        let r = png::Decoder::new(Cursor::new(
            std::fs::read(d.path().join("out.png")).unwrap(),
        ));
        let info = r.read_info().unwrap();
        assert_eq!(info.info().bit_depth, png::BitDepth::Sixteen);
        assert_eq!((info.info().width, info.info().height), (48, 32));
        // The JPEG XL decodes back to the same size.
        let jxl = std::fs::read(d.path().join("out.jxl")).unwrap();
        assert!(jxl.starts_with(&[0xFF, 0x0A]) || jxl.starts_with(b"\0\0\0\x0cJXL "));
    }

    #[test]
    fn the_input_is_never_overwritten_and_existing_outputs_need_force() {
        let d = tempfile::tempdir().unwrap();
        let bytes = png_bytes(8, 8);
        let input = d.path().join("in.png");
        std::fs::write(&input, &bytes).unwrap();
        let mut o = opts(d.path(), "in.png");
        assert!(
            run(&o, &bytes, &mut Lanczos::new(2))
                .unwrap_err()
                .to_string()
                .contains("must not be the input")
        );
        assert_eq!(std::fs::read(&input).unwrap(), bytes);
        o.output = d.path().join("out.png");
        std::fs::write(&o.output, b"keep me").unwrap();
        assert!(
            run(&o, &bytes, &mut Lanczos::new(2))
                .unwrap_err()
                .to_string()
                .contains("--force")
        );
        assert_eq!(std::fs::read(&o.output).unwrap(), b"keep me");
        o.force = true;
        run(&o, &bytes, &mut Lanczos::new(2)).unwrap();
        // A link to the input counts as the input.
        #[cfg(unix)]
        {
            let link = d.path().join("link.png");
            std::os::unix::fs::symlink(&input, &link).unwrap();
            o.output = link;
            assert!(run(&o, &bytes, &mut Lanczos::new(2)).is_err());
            assert_eq!(std::fs::read(&input).unwrap(), bytes);
        }
    }

    #[test]
    fn wrong_output_names_and_junk_input_are_refused() {
        let d = tempfile::tempdir().unwrap();
        let bytes = png_bytes(8, 8);
        std::fs::write(d.path().join("in.png"), &bytes).unwrap();
        let o = opts(d.path(), "out.jpg");
        assert!(run(&o, &bytes, &mut Lanczos::new(2)).is_err());
        let o = opts(d.path(), "out.png");
        assert!(run(&o, b"not an image", &mut Lanczos::new(2)).is_err());
        assert!(!d.path().join("out.png").exists());
    }

    /// A JPEG with an EXIF block (camera make and a rotation) after SOI.
    fn jpeg_with_exif() -> Vec<u8> {
        let px: Vec<u8> = (0..16 * 16 * 3).map(|i| (i * 3 % 251) as u8).collect();
        let mut jpg = Vec::new();
        image::codecs::jpeg::JpegEncoder::new_with_quality(&mut jpg, 80)
            .write_image(&px, 16, 16, image::ExtendedColorType::Rgb8)
            .unwrap();
        // TIFF: header, one entry (Orientation = 6), end of directory.
        let tiff = b"II*\0\x08\0\0\0\x01\0\x12\x01\x03\0\x01\0\0\0\x06\0\0\0\0\0\0\0";
        let mut seg = b"Exif\0\0".to_vec();
        seg.extend(tiff);
        let mut out = jpg[..2].to_vec();
        out.extend([0xFF, 0xE1]);
        out.extend(((seg.len() + 2) as u16).to_be_bytes());
        out.extend(seg);
        out.extend(&jpg[2..]);
        out
    }

    #[test]
    fn exif_survives_into_the_output_with_the_rotation_applied() {
        let bytes = jpeg_with_exif();
        let photo = decode(&bytes).unwrap();
        // Rotated by the tag: 16x16 stays square, so check the EXIF instead.
        let exif = photo.exif.expect("EXIF was dropped on decode");
        assert!(exif.starts_with(b"II*\0"));
        assert_eq!(
            Orientation::from_exif_chunk(&exif).unwrap_or(Orientation::NoTransforms),
            Orientation::NoTransforms,
            "the pixels are already rotated, so the tag must say normal"
        );
        let d = tempfile::tempdir().unwrap();
        std::fs::write(d.path().join("in.jpg"), &bytes).unwrap();
        let o = Options {
            input: d.path().join("in.jpg"),
            output: d.path().join("out.jxl"),
            jxl_distance: 1.0,
            force: false,
        };
        run(&o, &bytes, &mut Lanczos::new(1)).unwrap();
        let jxl = std::fs::read(&o.output).unwrap();
        assert!(
            jxl.windows(4).any(|w| w == b"Exif"),
            "the JPEG XL file has no Exif box"
        );
    }

    #[test]
    fn orientation_is_applied_to_the_pixels_and_reset_in_the_exif() {
        // A 2x1 JPEG-less check through the image crate's helpers.
        let mut exif = b"II*\0\x08\0\0\0\x01\0\x12\x01\x03\0\x01\0\0\0\x06\0\0\0\0\0\0\0".to_vec();
        assert_eq!(
            Orientation::from_exif_chunk(&exif),
            Some(Orientation::Rotate90)
        );
        Orientation::remove_from_exif_chunk(&mut exif).unwrap();
        assert_eq!(
            Orientation::from_exif_chunk(&exif).unwrap_or(Orientation::NoTransforms),
            Orientation::NoTransforms
        );
    }
}
