//! The desktop window. It collects settings, runs the `chitra-enhance` program
//! (which sits next to this one) as a child process and shows its progress.
//! Stopping kills the child; finished chunks are kept, so Start carries on.

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{Receiver, channel};

use chitra_enhance::analyze::Report;
use chitra_enhance::job::{self, Settings};
use eframe::egui::{self, Color32, RichText};

const TEAL: Color32 = Color32::from_rgb(0x3f, 0xd0, 0xb5);

enum Msg {
    Progress(u64, u64),
    Line(String),
}

/// Answers from the command-line program to a question (nothing is changed).
enum Answer {
    Titles(Result<Vec<(u32, f64)>, String>),
    Analysis(Result<Box<Report>, String>),
}

struct Running {
    child: Child,
    rx: Receiver<Msg>,
}

#[derive(PartialEq)]
enum Phase {
    Idle,
    Running,
    Stopped,
    Done,
    Failed(String),
}

struct App {
    s: Settings,
    ort: String,
    log: Vec<String>,
    progress: (u64, u64),
    run: Option<Running>,
    phase: Phase,
    restart: bool,
    started: Option<std::time::Instant>,
    /// The running job is a preview (its before/after picture is shown at the end).
    is_preview: bool,
    preview_clip: Option<PathBuf>,
    preview_tex: Option<egui::TextureHandle>,
    /// Waiting for a noise measurement from the command-line program.
    measure_rx: Option<Receiver<Option<(f32, f32)>>>,
    /// (measured, suggested) noise level of the chosen video.
    measured: Option<(f32, f32)>,
    measure_note: String,
    query_rx: Option<Receiver<Answer>>,
    /// Titles of the chosen DVD.
    titles: Vec<(u32, f64)>,
    report: Option<Box<Report>>,
    query_note: String,
    /// Every .onnx in the models folder.
    library: Vec<PathBuf>,
    /// A file or disc given on the command line, opened on the first frame.
    open_first: Option<PathBuf>,
    /// For development: save a picture of the window here, then close.
    screenshot: Option<PathBuf>,
    frames: u32,
}

fn cli_path() -> PathBuf {
    let name = if cfg!(windows) {
        "chitra-enhance.exe"
    } else {
        "chitra-enhance"
    };
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.join(name)))
        .filter(|p| p.is_file())
        .unwrap_or_else(|| PathBuf::from(name))
}

fn pick_file(title: &str, exts: &[&str]) -> Option<PathBuf> {
    let mut d = rfd::FileDialog::new().set_title(title);
    if !exts.is_empty() {
        d = d.add_filter("files", exts);
    }
    d.pick_file()
}

/// A model choice: pick from the library, or any other .onnx file.
fn model_row(ui: &mut egui::Ui, label: &str, value: &mut Option<PathBuf>, library: &[PathBuf]) {
    ui.horizontal(|ui| {
        ui.label(label);
        let shown = value
            .as_deref()
            .and_then(Path::file_name)
            .map_or("none".to_string(), |n| n.to_string_lossy().into_owned());
        egui::ComboBox::from_id_salt(label)
            .selected_text(shown)
            .width(170.0)
            .show_ui(ui, |ui| {
                ui.selectable_value(value, None, "none");
                for p in library {
                    let name = p
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .into_owned();
                    ui.selectable_value(value, Some(p.clone()), name);
                }
            });
        if ui.small_button("Other file…").clicked()
            && let Some(p) = pick_file(label, &["onnx"])
        {
            *value = Some(p);
        }
    });
}

impl App {
    fn new() -> App {
        let mut s = Settings::default();
        if let Some(dir) = job::models_dir() {
            let fp16 = chitra_enhance::models::likely_gpu();
            s.model = job::find_model(&dir, "realesr", fp16);
            s.five_frame_model = job::find_model(&dir, "fastdvdnet", fp16);
        }
        App {
            s,
            ort: chitra_enhance::models::find_runtime()
                .map(|p| p.display().to_string())
                .unwrap_or_default(),
            log: vec![],
            progress: (0, 0),
            run: None,
            phase: Phase::Idle,
            restart: false,
            started: None,
            is_preview: false,
            preview_clip: None,
            preview_tex: None,
            measure_rx: None,
            measured: None,
            measure_note: String::new(),
            query_rx: None,
            titles: vec![],
            report: None,
            query_note: String::new(),
            open_first: std::env::args_os().nth(1).map(PathBuf::from),
            screenshot: std::env::var_os("CHITRA_GUI_SCREENSHOT").map(PathBuf::from),
            frames: 0,
            library: job::models_dir()
                .map(|d| chitra_enhance::models::library(&d))
                .unwrap_or_default(),
        }
    }

    fn start(&mut self, ctx: &egui::Context, preview: bool) {
        let mut args = if preview {
            self.s.preview_args()
        } else {
            self.s.args()
        };
        if !preview && self.restart {
            args.push("--restart".into());
        }
        self.is_preview = preview;
        self.preview_tex = None;
        self.preview_clip = preview.then(|| job::preview_output(&self.s.input));
        self.spawn(ctx, args);
    }

    /// Runs the command-line program with `args`, showing its output in the log.
    fn spawn(&mut self, ctx: &egui::Context, args: Vec<String>) {
        let mut cmd = Command::new(cli_path());
        cmd.args(&args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .stdin(Stdio::null());
        if !self.ort.is_empty() {
            cmd.env("ORT_DYLIB_PATH", &self.ort);
        }
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            cmd.creation_flags(0x0800_0000); // no console window
        }
        self.log.clear();
        self.progress = (0, 0);
        match cmd.spawn() {
            Ok(mut child) => {
                let (tx, rx) = channel();
                let out = child.stdout.take().unwrap();
                let err = child.stderr.take().unwrap();
                let (t1, c1) = (tx.clone(), ctx.clone());
                std::thread::spawn(move || {
                    for l in BufReader::new(out).lines().map_while(Result::ok) {
                        let m = job::parse_progress(&l)
                            .map_or(Msg::Line(l), |(d, t)| Msg::Progress(d, t));
                        if t1.send(m).is_err() {
                            break;
                        }
                        c1.request_repaint();
                    }
                });
                let c2 = ctx.clone();
                std::thread::spawn(move || {
                    for l in BufReader::new(err).split(b'\n').map_while(Result::ok) {
                        let l = String::from_utf8_lossy(&l).trim().to_string();
                        if !l.is_empty() && tx.send(Msg::Line(l)).is_err() {
                            break;
                        }
                        c2.request_repaint();
                    }
                });
                self.run = Some(Running { child, rx });
                self.phase = Phase::Running;
                self.started = Some(std::time::Instant::now());
            }
            Err(e) => {
                self.phase = Phase::Failed(format!("cannot start {}: {e}", cli_path().display()));
            }
        }
    }

    /// Looks for the model files in the usual folder.
    fn find_models(&mut self) {
        if let Some(dir) = job::models_dir() {
            let fp16 = chitra_enhance::models::likely_gpu();
            self.s.model = self
                .s
                .model
                .take()
                .or_else(|| job::find_model(&dir, "realesr", fp16));
            self.s.five_frame_model = self
                .s
                .five_frame_model
                .take()
                .or_else(|| job::find_model(&dir, "fastdvdnet", fp16));
        }
    }

    /// Asks the command-line program to measure the noise of the chosen video.
    fn measure(&mut self, ctx: &egui::Context) {
        let (tx, rx) = channel();
        let args = self.s.measure_args();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let mut cmd = Command::new(cli_path());
            cmd.args(args).stdin(Stdio::null()).stderr(Stdio::null());
            #[cfg(windows)]
            {
                use std::os::windows::process::CommandExt;
                cmd.creation_flags(0x0800_0000);
            }
            let found = cmd.output().ok().and_then(|o| {
                String::from_utf8_lossy(&o.stdout)
                    .lines()
                    .find_map(job::parse_noise)
            });
            let _ = tx.send(found);
            ctx.request_repaint();
        });
        self.measure_rx = Some(rx);
        self.measure_note = "measuring…".into();
    }

    /// A new input: forget everything learnt about the old one.
    fn choose(&mut self, ctx: &egui::Context, p: PathBuf) {
        self.s.output = job::default_output(&p);
        self.s.input = p;
        self.s.title = None;
        self.phase = Phase::Idle;
        self.titles.clear();
        self.report = None;
        self.query_rx = None;
        self.query_note.clear();
        self.measured = None;
        self.measure_note.clear();
        self.preview_tex = None;
        if chitra_enhance::disc::source(&self.s.input).is_some() {
            self.ask(ctx, true);
        }
    }

    /// Lists the DVD's titles (`titles`) or analyses the input, in the background.
    fn ask(&mut self, ctx: &egui::Context, titles: bool) {
        let args = if titles {
            vec![
                "-i".into(),
                self.s.input.display().to_string(),
                "--list-titles".into(),
                "--progress".into(),
            ]
        } else {
            self.s.analyze_args()
        };
        let (tx, rx) = channel();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let mut cmd = Command::new(cli_path());
            cmd.args(args).stdin(Stdio::null());
            #[cfg(windows)]
            {
                use std::os::windows::process::CommandExt;
                cmd.creation_flags(0x0800_0000);
            }
            let done = match cmd.output() {
                Ok(o) if o.status.success() => Ok(String::from_utf8_lossy(&o.stdout).into_owned()),
                Ok(o) => Err(String::from_utf8_lossy(&o.stderr)
                    .lines()
                    .rev()
                    .find(|l| !l.trim().is_empty())
                    .unwrap_or("it failed")
                    .to_string()),
                Err(e) => Err(e.to_string()),
            };
            let answer = if titles {
                Answer::Titles(done.map(|t| t.lines().filter_map(job::parse_title).collect()))
            } else {
                Answer::Analysis(done.and_then(|t| {
                    t.lines()
                        .find_map(job::parse_analysis)
                        .map(Box::new)
                        .ok_or_else(|| "no analysis was returned".to_string())
                }))
            };
            let _ = tx.send(answer);
            ctx.request_repaint();
        });
        self.query_rx = Some(rx);
        self.query_note = if titles {
            "reading the disc…".into()
        } else {
            "looking at a few places in the video…".into()
        };
    }

    fn take_answer(&mut self) {
        let Some(rx) = &self.query_rx else { return };
        let Ok(answer) = rx.try_recv() else { return };
        self.query_rx = None;
        self.query_note.clear();
        match answer {
            Answer::Titles(Ok(t)) => self.titles = t,
            Answer::Analysis(Ok(r)) => {
                let dir = job::models_dir();
                let fp16 = chitra_enhance::models::likely_gpu();
                let find = |p: &str| dir.as_deref().and_then(|d| job::find_model(d, p, fp16));
                self.s
                    .apply(&r.suggestion, find("realesr"), find("fastdvdnet"));
                self.report = Some(r);
                self.query_note = "suggested settings filled in: change anything you like".into();
            }
            Answer::Titles(Err(e)) | Answer::Analysis(Err(e)) => self.query_note = e,
        }
    }

    /// Development aid (`CHITRA_GUI_SCREENSHOT=file.png`): once any analysis
    /// has finished, saves what the window shows and closes it.
    fn take_screenshot(&mut self, ctx: &egui::Context) {
        if self.screenshot.is_none() {
            return;
        }
        ctx.request_repaint();
        self.take_answer();
        if self.query_rx.is_some() {
            return;
        }
        self.frames += 1;
        if self.frames != 20 {
            return;
        }
        let Some(path) = self.screenshot.clone() else {
            return;
        };
        let close = ctx.clone();
        ctx.request_screenshot(move |img: std::sync::Arc<egui::ColorImage>| {
            let (w, h) = (img.size[0] as u32, img.size[1] as u32);
            let bytes: Vec<u8> = img.pixels.iter().flat_map(|p| p.to_array()).collect();
            if let Err(e) = image::save_buffer(&path, &bytes, w, h, image::ColorType::Rgba8) {
                eprintln!("could not save the picture: {e}");
            }
            close.send_viewport_cmd(egui::ViewportCommand::Close);
            close.request_repaint();
        });
    }

    fn stop(&mut self) {
        if let Some(r) = &mut self.run {
            let _ = r.child.kill();
            let _ = r.child.wait();
        }
        self.run = None;
        self.phase = Phase::Stopped;
    }

    fn poll(&mut self) {
        let Some(r) = &mut self.run else { return };
        while let Ok(m) = r.rx.try_recv() {
            match m {
                Msg::Progress(d, t) => self.progress = (d, t),
                Msg::Line(l) => {
                    self.log.push(l);
                    if self.log.len() > 500 {
                        self.log.remove(0);
                    }
                }
            }
        }
        if let Ok(Some(status)) = r.child.try_wait() {
            // Let the readers finish the last lines.
            std::thread::sleep(std::time::Duration::from_millis(100));
            while let Ok(m) = r.rx.try_recv() {
                if let Msg::Line(l) = m {
                    self.log.push(l);
                }
            }
            self.phase = if status.success() {
                Phase::Done
            } else {
                Phase::Failed(
                    self.log
                        .iter()
                        .rev()
                        .find(|l| l.starts_with("Error") || l.contains("error"))
                        .or(self.log.last())
                        .cloned()
                        .unwrap_or_else(|| format!("the job ended with {status}")),
                )
            };
            self.run = None;
            self.find_models();
            if let Some(d) = job::models_dir() {
                self.library = chitra_enhance::models::library(&d);
            }
        }
    }
}

/// Opens a file with the system's default program.
fn open_file(p: &Path) {
    let (prog, args): (&str, Vec<&std::ffi::OsStr>) = if cfg!(windows) {
        (
            "cmd",
            vec!["/C".as_ref(), "start".as_ref(), "".as_ref(), p.as_os_str()],
        )
    } else if cfg!(target_os = "macos") {
        ("open", vec![p.as_os_str()])
    } else {
        ("xdg-open", vec![p.as_os_str()])
    };
    let _ = Command::new(prog).args(args).spawn();
}

fn setup_style(ctx: &egui::Context) {
    let mut v = egui::Visuals::dark();
    v.panel_fill = Color32::from_rgb(0x17, 0x19, 0x1c);
    v.window_fill = Color32::from_rgb(0x1e, 0x21, 0x25);
    v.selection.bg_fill = TEAL.gamma_multiply(0.45);
    v.hyperlink_color = TEAL;
    ctx.set_visuals(v);
}

impl eframe::App for App {
    fn logic(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
        if let Some(p) = self.open_first.take() {
            let p = std::path::absolute(&p).unwrap_or(p);
            let p_input = p.clone();
            self.choose(ctx, p);
            // (A disc is busy listing its titles; analysing would replace that.)
            if self.screenshot.is_some() && !job::is_photo(&p_input) && self.query_rx.is_none() {
                self.ask(ctx, false);
            }
        }
        self.take_screenshot(ctx);
        self.poll();
        if self.run.is_some() {
            ctx.request_repaint_after(std::time::Duration::from_millis(250));
        }
        if self.phase == Phase::Done && self.is_preview && self.preview_tex.is_none() {
            if let Some(clip) = &self.preview_clip
                && let Ok(img) = image::open(clip.with_extension("compare.png"))
            {
                let rgba = img.to_rgba8();
                let size = [rgba.width() as usize, rgba.height() as usize];
                let color = egui::ColorImage::from_rgba_unmultiplied(size, rgba.as_raw());
                self.preview_tex =
                    Some(ctx.load_texture("preview", color, egui::TextureOptions::LINEAR));
            }
            self.is_preview = false;
        }
        if let Some(rx) = &self.measure_rx
            && let Ok(r) = rx.try_recv()
        {
            self.measure_rx = None;
            self.measured = r;
            self.measure_note = match r {
                Some((m, u)) => format!("measured about {m:.0}, suggested {u:.0}"),
                None => "could not measure this video".into(),
            };
            if let (Some((_, u)), Some(_)) = (r, self.s.sigma) {
                self.s.sigma = Some(u); // already tweaking by hand: start from the suggestion
            }
        }
    }

    fn ui(&mut self, root: &mut egui::Ui, _: &mut eframe::Frame) {
        let ctx = root.ctx().clone();
        let ctx = &ctx;
        self.take_answer();
        let running = self.phase == Phase::Running;
        let photo = job::is_photo(&self.s.input);

        egui::Panel::left("settings")
            .min_size(300.0)
            .show(root, |ui| {
                ui.add_enabled_ui(!running, |ui| {
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        ui.heading("Settings");
                        ui.add_space(6.0);
                        if ui
                            .button("Add your own model…")
                            .on_hover_text(
                                "An ONNX model: it is checked, then kept in your models folder",
                            )
                            .clicked()
                            && let Some(p) = pick_file("Choose an ONNX model", &["onnx"])
                        {
                            self.spawn(
                                ctx,
                                vec!["models".into(), "add".into(), p.display().to_string()],
                            );
                        }
                        if self.s.model.is_none() || self.s.five_frame_model.is_none() {
                            ui.horizontal(|ui| {
                                if ui
                                    .button("Download models")
                                    .on_hover_text(
                                        "The AI upscaler and the five-frame denoiser (about 20 MB), checked against known checksums",
                                    )
                                    .clicked()
                                {
                                    self.spawn(ctx, vec!["models".into(), "get".into(), "all".into()]);
                                }
                            });
                        }
                        ui.horizontal(|ui| {
                            ui.label("Enlarge ×");
                            for n in 1..=4 {
                                ui.selectable_value(&mut self.s.scale, n, n.to_string());
                            }
                        });
                        model_row(ui, "AI upscaler", &mut self.s.model, &self.library);
                        if photo {
                            model_row(
                                ui,
                                "For compressed JPEGs",
                                &mut self.s.model_compressed,
                                &self.library,
                            );
                            ui.add(
                                egui::Slider::new(&mut self.s.jxl_distance, 0.1..=3.0)
                                    .text("JXL distance (lower = bigger file)"),
                            );
                        } else {
                            ui.separator();
                            ui.checkbox(
                                &mut self.s.five_frame,
                                "Five-frame AI denoising (best for camcorder noise)",
                            );
                            if self.s.five_frame {
                                model_row(
                                    ui,
                                    "Denoise model",
                                    &mut self.s.five_frame_model,
                                    &self.library,
                                );
                                let mut auto = self.s.sigma.is_none();
                                ui.horizontal(|ui| {
                                    ui.checkbox(&mut auto, "Measure noise level automatically");
                                });
                                if auto {
                                    self.s.sigma = None;
                                } else if self.s.sigma.is_none() {
                                    self.s.sigma = Some(self.measured.map_or(15.0, |(_, u)| u));
                                }
                                if let Some(v) = &mut self.s.sigma {
                                    ui.add(
                                        egui::Slider::new(v, 3.0..=60.0)
                                            .text("Noise level")
                                            .suffix(" (5 faint · 20 noisy · 40 very)"),
                                    );
                                }
                                ui.horizontal(|ui| {
                                    let can = !self.s.input.as_os_str().is_empty()
                                        && self.measure_rx.is_none();
                                    if ui
                                        .add_enabled(can, egui::Button::new("Measure now"))
                                        .on_hover_text("Look at frames across the video")
                                        .clicked()
                                    {
                                        self.measure(ctx);
                                    }
                                    ui.label(RichText::new(&self.measure_note).weak());
                                });
                            } else {
                                egui::ComboBox::from_label("Simple denoise")
                                    .selected_text(self.s.denoise.clone())
                                    .show_ui(ui, |ui| {
                                        for d in ["off", "light", "medium", "strong"] {
                                            ui.selectable_value(&mut self.s.denoise, d.into(), d);
                                        }
                                    });
                            }
                            egui::ComboBox::from_label("Interlaced tape")
                                .selected_text(self.s.deinterlace.clone())
                                .show_ui(ui, |ui| {
                                    for d in ["auto", "on", "off"] {
                                        ui.selectable_value(&mut self.s.deinterlace, d.into(), d);
                                    }
                                });
                            if self.s.deinterlace != "off" {
                                ui.checkbox(
                                    &mut self.s.double_rate,
                                    "Keep both half-pictures (smoother, 2x time)",
                                );
                            }
                            egui::ComboBox::from_label("Video format")
                                .selected_text(self.s.codec.clone())
                                .show_ui(ui, |ui| {
                                    for d in ["h265", "h264", "av1", "prores"] {
                                        ui.selectable_value(&mut self.s.codec, d.into(), d);
                                    }
                                });
                            let mut g = self.s.grain.map(|g| g as f32);
                            let mut on = g.is_some();
                            ui.horizontal(|ui| {
                                ui.checkbox(&mut on, "Film grain");
                                if on {
                                    let v = g.get_or_insert(3.0);
                                    ui.add(egui::Slider::new(v, 0.0..=30.0).integer());
                                }
                            });
                            self.s.grain = on.then(|| g.unwrap_or(3.0) as u32);
                            let mut crop = self.s.crop.is_some();
                            ui.checkbox(&mut crop, "Cut black edges");
                            if crop {
                                let c = self.s.crop.get_or_insert([0; 4]);
                                ui.horizontal(|ui| {
                                    for (v, name) in c.iter_mut().zip(["L", "R", "T", "B"]) {
                                        ui.label(name);
                                        ui.add(
                                            egui::DragValue::new(v).range(0..=400).speed(2),
                                        );
                                        *v -= *v % 2;
                                    }
                                });
                            } else {
                                self.s.crop = None;
                            }
                            let mut tint = self.s.colour_shift.is_some();
                            ui.checkbox(&mut tint, "Correct a colour cast");
                            if tint {
                                let (u, v) = self.s.colour_shift.get_or_insert((0, 0));
                                ui.add(egui::Slider::new(u, -40..=40).text("less blue"));
                                ui.add(egui::Slider::new(v, -40..=40).text("less red"));
                            } else {
                                self.s.colour_shift = None;
                            }
                            ui.collapsing("Advanced", |ui| {
                                let mut crf = self.s.crf;
                                ui.horizontal(|ui| {
                                    let mut on = crf.is_some();
                                    ui.checkbox(&mut on, "Video quality");
                                    if on {
                                        let v = crf.get_or_insert(20);
                                        ui.add(
                                            egui::Slider::new(v, 10..=35).text("lower = better"),
                                        );
                                    }
                                    self.s.crf = on.then(|| crf.unwrap_or(20));
                                });
                                let mut chunk = self.s.chunk_secs;
                                ui.horizontal(|ui| {
                                    let mut on = chunk.is_some();
                                    ui.checkbox(&mut on, "Save progress every");
                                    if on {
                                        let v = chunk.get_or_insert(30);
                                        ui.add(egui::Slider::new(v, 10..=120).suffix(" s"));
                                    }
                                    self.s.chunk_secs = on.then(|| chunk.unwrap_or(30));
                                });
                            });
                            ui.checkbox(&mut self.restart, "Start over (ignore unfinished work)");
                        }
                        ui.separator();
                        ui.checkbox(&mut self.s.overwrite, "Replace the result file if it exists");
                        ui.label(RichText::new("Hardware").strong());
                        egui::ComboBox::from_label("Run on")
                            .selected_text(self.s.device.clone())
                            .show_ui(ui, |ui| {
                                for d in ["auto", "cuda", "directml", "coreml", "cpu"] {
                                    ui.selectable_value(&mut self.s.device, d.into(), d);
                                }
                            });
                        ui.add(
                            egui::Slider::new(&mut self.s.vram_gb, 2..=24).text("GPU memory GB"),
                        );
                        ui.horizontal(|ui| {
                            ui.label("ONNX Runtime library");
                        });
                        ui.text_edit_singleline(&mut self.ort);
                    });
                });
            });

        egui::CentralPanel::default().show(root, |ui| {
            ui.add_enabled_ui(!running, |ui| {
                ui.horizontal(|ui| {
                    if ui.button("Choose a video or photo…").clicked()
                        && let Some(p) = pick_file("Choose a file", &[])
                    {
                        self.choose(ctx, p);
                    }
                    if ui
                        .button("DVD image…")
                        .on_hover_text("An .iso of a disc without copy protection")
                        .clicked()
                        && let Some(p) = pick_file("Choose a DVD image", &["iso", "img"])
                    {
                        self.choose(ctx, p);
                    }
                    if ui
                        .button("Folder or disc…")
                        .on_hover_text(
                            "A DVD drive or a folder with VIDEO_TS, or a folder of photos and videos",
                        )
                        .clicked()
                        && let Some(p) = rfd::FileDialog::new()
                            .set_title("Choose a folder or disc")
                            .pick_folder()
                    {
                        self.choose(ctx, p);
                    }
                });
                if !self.s.input.as_os_str().is_empty() {
                    let full = self.s.input.display().to_string();
                    ui.add(egui::Label::new(RichText::new(&full).monospace()).truncate())
                        .on_hover_text(full);
                }
                ui.horizontal(|ui| {
                    ui.label("Save as");
                    let mut text = self.s.output.display().to_string();
                    if ui
                        .add(egui::TextEdit::singleline(&mut text).desired_width(f32::INFINITY))
                        .changed()
                    {
                        self.s.output = PathBuf::from(text);
                    }
                });
                let disc = chitra_enhance::disc::source(&self.s.input).is_some();
                if disc && !self.titles.is_empty() {
                    let name = |n: &Option<String>| match n.as_deref() {
                        None => "the longest title".to_string(),
                        Some("all") => "all titles (into a folder)".to_string(),
                        Some(n) => format!("title {n}"),
                    };
                    let before = self.s.title.clone();
                    egui::ComboBox::from_label("DVD title")
                        .selected_text(name(&self.s.title))
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut self.s.title, None, "the longest title");
                            for (n, secs) in &self.titles {
                                ui.selectable_value(
                                    &mut self.s.title,
                                    Some(n.to_string()),
                                    format!(
                                        "title {n}  ({}:{:02})",
                                        *secs as u64 / 60,
                                        *secs as u64 % 60
                                    ),
                                );
                            }
                            ui.selectable_value(
                                &mut self.s.title,
                                Some("all".to_string()),
                                "all titles (into a folder)",
                            );
                        });
                    if self.s.title != before {
                        let all = self.s.title.as_deref() == Some("all");
                        let plain = job::default_output(&self.s.input);
                        self.s.output = if all {
                            plain.with_extension("")
                        } else {
                            plain
                        };
                        self.report = None;
                    }
                }
                if !self.s.input.as_os_str().is_empty()
                    && !job::is_photo(&self.s.input)
                    && (disc || self.s.input.is_file())
                {
                    ui.horizontal(|ui| {
                        if ui
                            .add_enabled(
                                self.query_rx.is_none(),
                                egui::Button::new("Analyse and suggest settings"),
                            )
                            .on_hover_text("Looks at a few places; changes nothing in the video")
                            .clicked()
                        {
                            self.ask(ctx, false);
                        }
                        ui.label(RichText::new(&self.query_note).weak());
                    });
                }
            });
            ui.add_space(8.0);

            let (d, t) = self.progress;
            let frac = if t > 0 { d as f32 / t as f32 } else { 0.0 };
            let bar = egui::ProgressBar::new(frac).fill(TEAL);
            let status = match &self.phase {
                        Phase::Running if t > 0 => {
                            let secs = self.started.map_or(0.0, |s| s.elapsed().as_secs_f32());
                            let eta = if d > 0 {
                                secs * (t - d) as f32 / d as f32
                            } else {
                                0.0
                            };
                            format!("{d} / {t} frames — about {:.0} min left", eta / 60.0)
                        }
                        Phase::Running => "starting…".into(),
                        Phase::Done => "Finished".into(),
                        Phase::Stopped => "Stopped — Start continues where it left off".into(),
                        Phase::Failed(_) => "Failed".into(),
                        Phase::Idle => "Ready".into(),
            };
            if t > 0 {
                ui.add(bar.text(format!("{:.0}%  {status}", frac * 100.0)));
            } else {
                ui.label(status);
            }
            if let Phase::Failed(m) = &self.phase {
                ui.colored_label(Color32::from_rgb(0xff, 0x7a, 0x70), m);
            }
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                let problem = self.s.problem();
                if running {
                    if ui.button("Stop").clicked() {
                        self.stop();
                    }
                } else {
                    let label = if self.phase == Phase::Stopped {
                        "Continue"
                    } else {
                        "Start"
                    };
                    let b = ui.add_enabled(
                        problem.is_none(),
                        egui::Button::new(RichText::new(label).strong().color(Color32::BLACK))
                            .fill(TEAL)
                            .min_size(egui::vec2(90.0, 0.0)),
                    );
                    if b.clicked() {
                        self.start(ctx, false);
                    }
                    if !photo {
                        let label = format!("Preview {} s", self.s.preview_secs);
                        if ui
                            .add_enabled(problem.is_none(), egui::Button::new(label))
                            .on_hover_text("Try these settings on a few seconds from the middle")
                            .clicked()
                        {
                            self.start(ctx, true);
                        }
                        ui.add(
                            egui::Slider::new(&mut self.s.preview_secs, 1..=30)
                                .suffix(" s")
                                .show_value(false),
                        );
                    }
                    if let Some(p) = &problem
                        && !self.s.input.as_os_str().is_empty()
                    {
                        ui.label(RichText::new(p).weak());
                    }
                }
            });
            ui.add_space(6.0);
            ui.collapsing("Command line", |ui| {
                ui.label(
                    RichText::new(format!("chitra-enhance {}", self.s.args().join(" ")))
                        .monospace()
                        .small(),
                );
            });
            if let Some(r) = &self.report {
                egui::CollapsingHeader::new("What the analysis found")
                    .default_open(true)
                    .show(ui, |ui| {
                        for line in &r.reasons {
                            ui.label(format!("• {line}"));
                        }
                        for line in &r.notes {
                            ui.label(RichText::new(format!("• {line}")).weak());
                        }
                    });
            }
            if let Some(tex) = &self.preview_tex {
                ui.label(RichText::new("Preview: original (left) and result (right)").strong());
                let avail = ui.available_width();
                let ratio = tex.size()[1] as f32 / tex.size()[0] as f32;
                ui.image((tex.id(), egui::vec2(avail, avail * ratio)));
                if let Some(c) = &self.preview_clip
                    && ui.button("Play the preview clip").clicked()
                {
                    open_file(c);
                }
            }
            if !self.log.is_empty() {
                ui.label(RichText::new("Log").strong());
            }
            egui::ScrollArea::vertical()
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    for l in &self.log {
                        ui.label(RichText::new(l).monospace().small());
                    }
                });
        });
    }

    fn on_exit(&mut self, _: Option<&eframe::glow::Context>) {
        self.stop();
    }
}

fn main() -> eframe::Result {
    let opts = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([980.0, 640.0])
            .with_min_inner_size([760.0, 480.0]),
        ..Default::default()
    };
    eframe::run_native(
        "Chitra Enhancer AI",
        opts,
        Box::new(|cc| {
            setup_style(&cc.egui_ctx);
            Ok(Box::new(App::new()))
        }),
    )
}
