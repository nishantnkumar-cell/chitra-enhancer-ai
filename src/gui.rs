//! The desktop window. It collects settings, runs the `chitra-enhance` program
//! (which sits next to this one) as a child process and shows its progress.
//! Stopping kills the child; finished chunks are kept, so Start carries on.

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{Receiver, channel};

use chitra_enhance::job::{self, Settings};
use eframe::egui::{self, Color32, RichText};

const TEAL: Color32 = Color32::from_rgb(0x3f, 0xd0, 0xb5);

enum Msg {
    Progress(u64, u64),
    Line(String),
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
    /// Every .onnx in the models folder.
    library: Vec<PathBuf>,
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
            s.model = job::find_model(&dir, "realesr", true);
            s.five_frame_model = job::find_model(&dir, "fastdvdnet", true);
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
            self.s.model = self
                .s
                .model
                .take()
                .or_else(|| job::find_model(&dir, "realesr", true));
            self.s.five_frame_model = self
                .s
                .five_frame_model
                .take()
                .or_else(|| job::find_model(&dir, "fastdvdnet", true));
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
                                    ui.add(egui::Slider::new(v, 0.0..=30.0));
                                }
                            });
                            self.s.grain = on.then(|| g.unwrap_or(3.0) as u32);
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
                                ui.checkbox(
                                    &mut self.s.double_rate,
                                    "Interlaced: keep both fields (smoother, 2× frames)",
                                );
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
                        self.s.output = job::default_output(&p);
                        self.s.input = p;
                        self.phase = Phase::Idle;
                    }
                    if ui
                        .button("Choose a folder…")
                        .on_hover_text("Every photo and video in it, one after the other")
                        .clicked()
                        && let Some(p) = rfd::FileDialog::new()
                            .set_title("Choose a folder")
                            .pick_folder()
                    {
                        self.s.output = job::default_output(&p);
                        self.s.input = p;
                        self.phase = Phase::Idle;
                    }
                    ui.label(RichText::new(self.s.input.display().to_string()).monospace());
                });
                ui.horizontal(|ui| {
                    ui.label("Save as");
                    let mut text = self.s.output.display().to_string();
                    if ui.text_edit_singleline(&mut text).changed() {
                        self.s.output = PathBuf::from(text);
                    }
                });
            });
            ui.add_space(8.0);

            let (d, t) = self.progress;
            let frac = if t > 0 { d as f32 / t as f32 } else { 0.0 };
            ui.add(
                egui::ProgressBar::new(frac)
                    .fill(TEAL)
                    .show_percentage()
                    .text(match &self.phase {
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
                    }),
            );
            if let Phase::Failed(m) = &self.phase {
                ui.colored_label(Color32::from_rgb(0xff, 0x7a, 0x70), m);
            }
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                let problem = self.s.problem();
                if running {
                    if ui.button("■ Stop").clicked() {
                        self.stop();
                    }
                } else {
                    let label = if self.phase == Phase::Stopped {
                        "▶ Continue"
                    } else {
                        "▶ Start"
                    };
                    let b = ui.add_enabled(problem.is_none(), egui::Button::new(label));
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
            ui.label(RichText::new("Log").strong());
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
