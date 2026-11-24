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

fn path_row(ui: &mut egui::Ui, label: &str, value: &mut Option<PathBuf>, exts: &[&str]) {
    ui.horizontal(|ui| {
        ui.label(label);
        let shown = value
            .as_deref()
            .and_then(Path::file_name)
            .map_or("none".to_string(), |n| n.to_string_lossy().into_owned());
        ui.label(RichText::new(shown).monospace());
        if ui.button("Choose…").clicked()
            && let Some(p) = pick_file(label, exts)
        {
            *value = Some(p);
        }
        if value.is_some() && ui.small_button("✕").clicked() {
            *value = None;
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
            ort: std::env::var("ORT_DYLIB_PATH").unwrap_or_default(),
            log: vec![],
            progress: (0, 0),
            run: None,
            phase: Phase::Idle,
            restart: false,
            started: None,
        }
    }

    fn start(&mut self, ctx: &egui::Context) {
        let mut args = self.s.args();
        if self.restart {
            args.push("--restart".into());
        }
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
        }
    }
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
    fn update(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
        self.poll();
        if self.run.is_some() {
            ctx.request_repaint_after(std::time::Duration::from_millis(250));
        }
        let running = self.phase == Phase::Running;
        let photo = job::is_photo(&self.s.input);

        egui::SidePanel::left("settings")
            .min_width(300.0)
            .show(ctx, |ui| {
                ui.add_enabled_ui(!running, |ui| {
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        ui.heading("Settings");
                        ui.add_space(6.0);
                        ui.horizontal(|ui| {
                            ui.label("Enlarge ×");
                            for n in 1..=4 {
                                ui.selectable_value(&mut self.s.scale, n, n.to_string());
                            }
                        });
                        path_row(ui, "AI upscaler", &mut self.s.model, &["onnx"]);
                        if photo {
                            path_row(
                                ui,
                                "For compressed JPEGs",
                                &mut self.s.model_compressed,
                                &["onnx"],
                            );
                            ui.add(
                                egui::Slider::new(&mut self.s.jxl_distance, 0.1..=3.0)
                                    .text("JXL distance (lower = bigger file)"),
                            );
                            ui.checkbox(&mut self.s.overwrite, "Replace the result if it exists");
                        } else {
                            ui.separator();
                            ui.checkbox(
                                &mut self.s.five_frame,
                                "Five-frame AI denoising (best for camcorder noise)",
                            );
                            if self.s.five_frame {
                                path_row(
                                    ui,
                                    "Denoise model",
                                    &mut self.s.five_frame_model,
                                    &["onnx"],
                                );
                                ui.add(
                                    egui::Slider::new(&mut self.s.sigma, 0.0..=60.0)
                                        .text("Noise level"),
                                );
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
                            ui.checkbox(&mut self.restart, "Start over (ignore unfinished work)");
                        }
                        ui.separator();
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

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.add_enabled_ui(!running, |ui| {
                ui.horizontal(|ui| {
                    if ui.button("Choose a video or photo…").clicked()
                        && let Some(p) = pick_file("Choose a file", &[])
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
                        self.start(ctx);
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
