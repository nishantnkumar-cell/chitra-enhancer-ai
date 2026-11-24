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
