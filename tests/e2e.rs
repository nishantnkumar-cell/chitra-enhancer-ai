//! Needs ffmpeg; skipped quietly when it is missing.
use std::path::Path;
use std::process::Command;

fn have_ffmpeg() -> bool {
    Command::new("ffmpeg").arg("-version").output().is_ok()
}

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_chitra-enhance"))
}

fn make_clip(p: &Path) {
    let st = Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x120:rate=25:duration=6",
        ])
        .args([
            "-f",
            "lavfi",
            "-i",
            "sine=duration=6",
            "-vf",
            "noise=alls=20:allf=t,interlace",
        ])
        .args([
            "-c:v",
            "libx264",
            "-flags",
            "+ilme+ildct",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
        ])
        .arg(p)
        .status()
        .unwrap();
    assert!(st.success());
}

fn frames(p: &Path) -> (String, String) {
    let o = Command::new("ffprobe")
        .args(["-v", "error", "-count_frames", "-select_streams", "v:0"])
        .args([
            "-show_entries",
            "stream=width,height,nb_read_frames",
            "-of",
            "csv=p=0",
        ])
        .arg(p)
        .output()
        .unwrap();
    let s = String::from_utf8_lossy(&o.stdout).trim().to_string();
    let a = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "csv=p=0",
        ])
        .arg(p)
        .output()
        .unwrap();
    (s, String::from_utf8_lossy(&a.stdout).trim().to_string())
}

#[test]
fn stops_resumes_and_finishes_with_audio() {
    if !have_ffmpeg() {
        eprintln!("ffmpeg missing: skipped");
        return;
    }
    let d = tempfile::tempdir().unwrap();
    let (i, o) = (d.path().join("in.mp4"), d.path().join("out.mp4"));
    make_clip(&i);
    let run = |extra: &[&str]| {
        bin()
            .args(["-i", i.to_str().unwrap(), "-o", o.to_str().unwrap()])
            .args([
                "--chunk-seconds",
                "2",
                "--codec",
                "h264",
                "--preset",
                "ultrafast",
            ])
            .args(if extra.contains(&"--denoise") {
                &[][..]
            } else {
                &["--denoise", "light"][..]
            })
            .args(extra)
            .output()
            .unwrap()
    };
    let first = run(&["--stop-after-chunks", "1"]);
    assert_eq!(
        first.status.code(),
        Some(130),
        "{}",
        String::from_utf8_lossy(&first.stderr)
    );
    assert!(!o.exists());
    let work = d.path().join("out.mp4.work");
    assert!(work.join("chunk_00000.mkv").exists());
    assert!(!work.join("chunk_00001.mkv").exists());

    // Other settings must not be mixed into the unfinished run.
    let other = run(&["--denoise", "strong"]);
    assert!(!other.status.success());
    assert!(String::from_utf8_lossy(&other.stderr).contains("--restart"));

    let second = run(&[]);
    assert!(
        second.status.success(),
        "{}",
        String::from_utf8_lossy(&second.stderr)
    );
    assert!(String::from_utf8_lossy(&second.stderr).contains("(1 resumed)"));
    let (video, audio) = frames(&o);
    assert_eq!(
        video, "320,240,75",
        "320x240 (2x), 6 s at 12.5 fps (interlace halves the rate)"
    );
    assert_eq!(audio, "aac");
    assert!(!work.exists(), "work folder is removed when done");
}

#[test]
fn refuses_the_input_as_output_and_a_missing_file() {
    let d = tempfile::tempdir().unwrap();
    let f = d.path().join("a.mp4");
    std::fs::write(&f, b"x").unwrap();
    let r = bin()
        .args(["-i", f.to_str().unwrap(), "-o", f.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(!r.status.success());
    let r = bin()
        .args(["-i", "/nonexistent.mp4", "-o", "/tmp/x.mp4"])
        .output()
        .unwrap();
    assert!(!r.status.success());
}

/// Runs a real ONNX model when `ORT_DYLIB_PATH` and `CHITRA_TEST_MODEL` (a 2x
/// model) are set and the binary was built with `--features onnx`.
#[cfg(feature = "onnx")]
#[test]
fn an_onnx_model_doubles_the_frame_size() {
    let (Some(_), Some(model)) = (
        std::env::var_os("ORT_DYLIB_PATH"),
        std::env::var_os("CHITRA_TEST_MODEL"),
    ) else {
        eprintln!("no ONNX Runtime/model configured: skipped");
        return;
    };
    if !have_ffmpeg() {
        return;
    }
    let d = tempfile::tempdir().unwrap();
    let (i, o) = (d.path().join("in.mp4"), d.path().join("out.mkv"));
    make_clip(&i);
    let r = bin()
        .args(["-i", i.to_str().unwrap(), "-o", o.to_str().unwrap()])
        .args(["--model", model.to_str().unwrap(), "--device", "cpu"])
        .args([
            "--codec",
            "h264",
            "--preset",
            "ultrafast",
            "--denoise",
            "off",
        ])
        .output()
        .unwrap();
    assert!(r.status.success(), "{}", String::from_utf8_lossy(&r.stderr));
    assert_eq!(frames(&o).0, "320,240,75");
}

/// A five-frame denoiser sees the whole clip: every frame comes out once, in
/// every chunk, including the last frames of a chunk that trail the decoder.
#[test]
fn a_temporal_stage_keeps_every_frame_of_every_chunk() {
    use chitra_enhance::{engine::Lanczos, ffmpeg, temporal::Identity, video};
    if !have_ffmpeg() {
        return;
    }
    let d = tempfile::tempdir().unwrap();
    let (i, o) = (d.path().join("in.mp4"), d.path().join("out.mkv"));
    make_clip(&i);
    let tools = ffmpeg::Tools::locate(None).unwrap();
    let info = ffmpeg::probe(&tools, &i).unwrap();
    let opts = video::Options {
        input: i,
        output: o.clone(),
        tools,
        denoise: video::Denoise::Off,
        deinterlace: video::Deinterlace::Off,
        double_rate: false,
        codec: video::Codec::H264,
        crf: Some(20),
        preset: Some("ultrafast".into()),
        chunk_secs: 2.0,
        restart: false,
        keep_work: false,
        grain: 0,
        stop_after_chunks: None,
    };
    let mut engine = Lanczos::new(1);
    let mut temporal = Identity;
    let s = video::run(&opts, &mut engine, Some(&mut temporal), &info).unwrap();
    assert_eq!(s.chunks, 3);
    assert_eq!(s.frames, 75);
    assert_eq!(frames(&o).0, "160,120,75");
}

/// A real five-frame model (FastDVDnet), when `CHITRA_TEST_TEMPORAL_MODEL` is set.
#[cfg(feature = "onnx")]
#[test]
fn a_five_frame_onnx_model_denoises_without_changing_the_size() {
    let (Some(_), Some(model)) = (
        std::env::var_os("ORT_DYLIB_PATH"),
        std::env::var_os("CHITRA_TEST_TEMPORAL_MODEL"),
    ) else {
        eprintln!("no ONNX Runtime/temporal model configured: skipped");
        return;
    };
    if !have_ffmpeg() {
        return;
    }
    let d = tempfile::tempdir().unwrap();
    let (i, o) = (d.path().join("in.mp4"), d.path().join("out.mkv"));
    make_clip(&i);
    let r = bin()
        .args(["-i", i.to_str().unwrap(), "-o", o.to_str().unwrap()])
        .args(["--temporal-model", model.to_str().unwrap()])
        .args(["--device", "cpu", "--scale", "1", "--denoise", "off"])
        .args(["--codec", "h264", "--preset", "ultrafast", "--grain", "0"])
        .output()
        .unwrap();
    assert!(r.status.success(), "{}", String::from_utf8_lossy(&r.stderr));
    assert_eq!(frames(&o).0, "160,120,75");
}

#[test]
fn a_preview_makes_a_short_clip_and_a_before_after_picture_without_touching_real_output() {
    if !have_ffmpeg() {
        return;
    }
    let d = tempfile::tempdir().unwrap();
    let (i, o) = (d.path().join("in.mp4"), d.path().join("prev.mp4"));
    make_clip(&i);
    let run = |extra: &[&str]| {
        bin()
            .args(["-i", i.to_str().unwrap(), "-o", o.to_str().unwrap()])
            .args(["--preview", "2", "--denoise", "off", "--codec", "h264"])
            .args(["--preset", "ultrafast", "--scale", "2"])
            .args(extra)
            .output()
            .unwrap()
    };
    let r = run(&[]);
    assert!(r.status.success(), "{}", String::from_utf8_lossy(&r.stderr));
    // 2 s of the 12.5 fps (interlaced-halved) test clip, doubled in size.
    let (w, h, n) = {
        let s = frames(&o).0;
        let v: Vec<&str> = s.split(',').collect();
        (
            v[0].to_string(),
            v[1].to_string(),
            v[2].parse::<u32>().unwrap(),
        )
    };
    assert_eq!((w.as_str(), h.as_str()), ("320", "240"));
    assert!((24..=26).contains(&n), "{n} frames");
    let still = image::open(d.path().join("prev.compare.png")).unwrap();
    assert_eq!(
        (still.width(), still.height()),
        (640, 240),
        "two pictures side by side"
    );
    // A second preview must not silently replace an existing file.
    let again = run(&[]);
    assert!(!again.status.success());
    assert!(String::from_utf8_lossy(&again.stderr).contains("--force"));
    assert!(run(&["--force"]).status.success());
}

#[test]
fn noise_can_be_measured_and_previews_are_capped_at_30_seconds() {
    if !have_ffmpeg() {
        return;
    }
    let d = tempfile::tempdir().unwrap();
    let (i, o) = (d.path().join("in.mp4"), d.path().join("out.mp4"));
    make_clip(&i);
    let r = bin()
        .args([
            "-i",
            i.to_str().unwrap(),
            "-o",
            o.to_str().unwrap(),
            "--measure-noise",
        ])
        .output()
        .unwrap();
    assert!(r.status.success(), "{}", String::from_utf8_lossy(&r.stderr));
    let out = String::from_utf8_lossy(&r.stdout);
    let (measured, suggested) = chitra_enhance::job::parse_noise(out.trim()).expect(&out);
    assert!((0.0..40.0).contains(&measured), "{measured}");
    let want = (measured * 1.6).clamp(3.0, 60.0);
    assert!((suggested - want).abs() < 1.0, "{suggested} vs {want}");
    assert!(!o.exists(), "measuring writes nothing");
    let r = bin()
        .args([
            "-i",
            i.to_str().unwrap(),
            "-o",
            o.to_str().unwrap(),
            "--preview",
            "31",
        ])
        .output()
        .unwrap();
    assert!(!r.status.success());
    assert!(String::from_utf8_lossy(&r.stderr).contains("1 to 30"));
}

/// An upscaler and a five-frame denoiser in the same run (ONNX Runtime allows
/// one environment per process, so they must share it). Needs ORT_DYLIB_PATH,
/// CHITRA_TEST_MODEL and CHITRA_TEST_TEMPORAL_MODEL.
#[cfg(feature = "onnx")]
#[test]
fn an_upscaler_and_a_denoiser_work_together() {
    let (Some(_), Some(up), Some(dn)) = (
        std::env::var_os("ORT_DYLIB_PATH"),
        std::env::var_os("CHITRA_TEST_MODEL"),
        std::env::var_os("CHITRA_TEST_TEMPORAL_MODEL"),
    ) else {
        eprintln!("models not configured: skipped");
        return;
    };
    if !have_ffmpeg() {
        return;
    }
    let d = tempfile::tempdir().unwrap();
    let (i, o) = (d.path().join("in.mp4"), d.path().join("out.mkv"));
    make_clip(&i);
    let r = bin()
        .args(["-i", i.to_str().unwrap(), "-o", o.to_str().unwrap()])
        .args(["--model", up.to_str().unwrap()])
        .args(["--temporal-model", dn.to_str().unwrap()])
        .args(["--device", "cpu", "--denoise", "off", "--grain", "0"])
        .args(["--codec", "h264", "--preset", "ultrafast", "--scale", "2"])
        .output()
        .unwrap();
    assert!(r.status.success(), "{}", String::from_utf8_lossy(&r.stderr));
    assert_eq!(frames(&o).0, "320,240,75");
}

#[test]
fn the_models_command_lists_and_refuses_unknown_names_without_touching_the_network() {
    let d = tempfile::tempdir().unwrap();
    let run = |args: &[&str]| {
        bin()
            .arg("models")
            .args(args)
            // Keep the models folder inside the temp dir on every OS.
            .env("XDG_DATA_HOME", d.path())
            .env("LOCALAPPDATA", d.path())
            .env("HOME", d.path())
            .output()
            .unwrap()
    };
    let r = run(&["list"]);
    assert!(r.status.success());
    let out = String::from_utf8_lossy(&r.stdout);
    assert!(
        out.contains("fastdvdnet_fp16.onnx") && out.contains("not installed"),
        "{out}"
    );
    let r = run(&["get", "no-such-model"]);
    assert!(!r.status.success());
    assert!(String::from_utf8_lossy(&r.stderr).contains("no model matches"));
    assert!(!run(&["frobnicate"]).status.success());
}

#[test]
fn a_folder_is_a_batch_that_skips_finished_files_and_survives_a_bad_one() {
    if !have_ffmpeg() {
        return;
    }
    let d = tempfile::tempdir().unwrap();
    let (inp, out) = (d.path().join("tapes"), d.path().join("done"));
    std::fs::create_dir(&inp).unwrap();
    make_clip(&inp.join("a.mp4"));
    image::RgbImage::from_fn(30, 20, |x, y| {
        image::Rgb([(x * 8) as u8, (y * 12) as u8, 90])
    })
    .save(inp.join("b.png"))
    .unwrap();
    std::fs::write(inp.join("notes.txt"), "not media").unwrap();
    std::fs::write(inp.join("broken.avi"), "not a video").unwrap();
    let run = || {
        bin()
            .args(["-i", inp.to_str().unwrap(), "-o", out.to_str().unwrap()])
            .args(["--scale", "1", "--denoise", "off", "--codec", "h264"])
            .args(["--preset", "ultrafast", "--grain", "0"])
            .output()
            .unwrap()
    };
    let r = run();
    let err = String::from_utf8_lossy(&r.stderr);
    assert!(
        !r.status.success(),
        "the broken file makes the batch fail overall"
    );
    assert!(
        err.contains("FAILED") && err.contains("broken.avi"),
        "{err}"
    );
    assert!(err.contains("2 done") || err.contains("2 done,"), "{err}");
    assert_eq!(frames(&out.join("a-enhanced.mp4")).0, "160,120,75");
    assert!(out.join("b-enhanced.jxl").is_file());
    assert!(!out.join("notes-enhanced.mp4").exists());
    // Without the bad file, a second run only skips.
    std::fs::remove_file(inp.join("broken.avi")).unwrap();
    let r = run();
    let err = String::from_utf8_lossy(&r.stderr);
    assert!(r.status.success(), "{err}");
    assert_eq!(err.matches("already done").count(), 2, "{err}");
    // The output folder may not be the input folder.
    let same = bin()
        .args(["-i", inp.to_str().unwrap(), "-o", inp.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(!same.status.success());
    assert!(String::from_utf8_lossy(&same.stderr).contains("must not be the input folder"));
}

#[test]
fn photo_mode_writes_16_bit_png_and_refuses_to_overwrite() {
    let d = tempfile::tempdir().unwrap();
    let i = d.path().join("p.png");
    let img = image::RgbImage::from_fn(30, 20, |x, y| {
        image::Rgb([(x * 8) as u8, (y * 12) as u8, 90])
    });
    img.save(&i).unwrap();
    let o = d.path().join("big.png");
    let run = |extra: &[&str]| {
        bin()
            .args([
                "-i",
                i.to_str().unwrap(),
                "-o",
                o.to_str().unwrap(),
                "--scale",
                "2",
            ])
            .args(extra)
            .output()
            .unwrap()
    };
    let r = run(&[]);
    assert!(r.status.success(), "{}", String::from_utf8_lossy(&r.stderr));
    assert!(String::from_utf8_lossy(&r.stderr).contains("30x20 -> 60x40"));
    let again = run(&[]);
    assert!(!again.status.success());
    assert!(String::from_utf8_lossy(&again.stderr).contains("--force"));
    assert!(run(&["--force"]).status.success());
    let jxl = d.path().join("big.jxl");
    let r = bin()
        .args([
            "-i",
            i.to_str().unwrap(),
            "-o",
            jxl.to_str().unwrap(),
            "--scale",
            "2",
            "--jxl-distance",
            "0.5",
        ])
        .output()
        .unwrap();
    assert!(r.status.success(), "{}", String::from_utf8_lossy(&r.stderr));
    assert!(jxl.metadata().unwrap().len() > 0);
}
