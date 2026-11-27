//! Running an ONNX upscaling model on the GPU (CUDA, DirectML, Core ML) or the
//! CPU through ONNX Runtime. The library is loaded at run time: set
//! `ORT_DYLIB_PATH` to `libonnxruntime.so` / `onnxruntime.dll` / `.dylib`.
//!
//! With NVIDIA's TensorRT RTX plugin beside the runtime, RTX cards use it.
//!
//! A model is loaded once and kept for the whole run. The tile size starts from
//! the memory budget; when the GPU runs out of memory the session is rebuilt
//! (a failed CUDA allocation can leave it unusable) and the tile size steps
//! down 1024 -> 768 -> 512 -> 256 -> 128.

use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock, Mutex, OnceLock};

use anyhow::{Context, Result, anyhow, bail};
use clap::ValueEnum;
use half::f16;
use half::slice::HalfFloatSliceExt;
use ort::environment::Environment;
use ort::ep::{self, ExecutionProviderDispatch};
use ort::session::Session;
use ort::session::builder::SessionBuilder;
use ort::value::{Tensor, TensorElementType, ValueType};
use rayon::prelude::*;

use crate::engine::Engine;
use crate::infer::{
    BLOCK, byte, chw_from_rgb8, denoiser_tile_for_budget, from_chw, overlap_for, plane_from_rgb8,
    rgb8_from_chw, rgb8_from_plane, smaller_tile, tile_for_budget, to_chw,
};
use crate::shrink;
use crate::temporal::{FRAMES, Temporal, stack};
use crate::tiler::{Plane, map_tiled, round_up, upscale_tiled};

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum Device {
    /// The best this computer offers: TensorRT if its plugin is there, then
    /// CUDA, DirectML or Core ML, else CPU.
    Auto,
    /// NVIDIA's TensorRT RTX plugin (RTX 30 series or newer), kept beside
    /// ONNX Runtime.
    Tensorrt,
    Cuda,
    Directml,
    Coreml,
    Cpu,
}

impl Device {
    fn providers(self) -> Vec<(&'static str, ExecutionProviderDispatch)> {
        let cuda = ("CUDA", ep::CUDA::default().build());
        let dml = ("DirectML", ep::DirectML::default().build());
        let coreml = ("Core ML", ep::CoreML::default().build());
        let cpu = ("CPU", ep::CPU::default().build());
        match self {
            // The plugin is not one of these; it is tried before them.
            Device::Tensorrt => vec![],
            Device::Cuda => vec![cuda, cpu],
            Device::Directml => vec![dml, cpu],
            Device::Coreml => vec![coreml, cpu],
            Device::Cpu => vec![cpu],
            Device::Auto => {
                if cfg!(target_os = "windows") {
                    vec![cuda, dml, cpu]
                } else if cfg!(target_os = "macos") {
                    vec![coreml, cpu]
                } else {
                    vec![cuda, cpu]
                }
            }
        }
    }
}

/// A model that is loaded and ready, with what is needed to load it again.
struct Loaded {
    env: Arc<Environment>,
    source: Source,
    device: Device,
    session: Session,
    /// The processor it runs on.
    provider: String,
    /// The picture size it was last prepared for.
    sized: Option<(usize, usize)>,
    /// The names of the model's free height and width.
    sides: [String; 2],
}

/// Where a model is read from.
#[derive(Clone)]
enum Source {
    File(PathBuf),
    /// A model changed in memory (see `shrink`).
    Memory(Arc<[u8]>),
}

impl Loaded {
    /// `first` is the picture size used until `prepare` says otherwise, for a
    /// processor that must know one from the start.
    fn new(path: &Path, device: Device, first: usize) -> Result<Loaded> {
        let env = environment()?;
        let source = Source::File(path.to_path_buf());
        let (session, provider) = build_session(&env, &source, device, first, &[])?;
        // The usual names; the model may say others (when it has not been
        // given sizes already).
        let mut sides = ["h".to_string(), "w".to_string()];
        if let Some(ValueType::Tensor {
            dimension_symbols, ..
        }) = session.inputs().first().map(|i| i.dtype())
        {
            for (side, index) in sides.iter_mut().zip([2, 3]) {
                if let Some(name) = dimension_symbols.get(index).filter(|n| !n.is_empty()) {
                    side.clone_from(name);
                }
            }
        }
        Ok(Loaded {
            env,
            source,
            device,
            session,
            provider,
            sized: None,
            sides,
        })
    }

    /// A fresh session, as after the card ran out of memory.
    fn reload(&mut self) -> Result<()> {
        let first = 64;
        (self.session, self.provider) =
            build_session(&self.env, &self.source, self.device, first, &[])?;
        self.sized = None;
        Ok(())
    }

    /// DirectML runs much faster (about 1.5x with half precision) when told the
    /// picture size in advance, and TensorRT has to be, so the model is loaded
    /// again for each new size. A video has one size.
    fn prepare(&mut self, w: usize, h: usize) -> Result<()> {
        let must = self.provider == TENSORRT;
        if !(must || self.provider == "DirectML") || self.sized == Some((w, h)) {
            return Ok(());
        }
        let sizes = [(self.sides[0].clone(), h), (self.sides[1].clone(), w)];
        // The same processor again, not whatever Auto would pick this time.
        let device = if must { Device::Tensorrt } else { self.device };
        match build_session(&self.env, &self.source, device, w.max(h), &sizes) {
            Ok((session, provider)) => (self.session, self.provider) = (session, provider),
            // DirectML can carry on unprepared; TensorRT cannot.
            Err(e) if must => return Err(e),
            Err(_) => {}
        }
        self.sized = Some((w, h));
        Ok(())
    }
}

/// What each 8-bit sample is as a half-precision input.
static HALF_IN: LazyLock<[f16; 256]> =
    LazyLock::new(|| std::array::from_fn(|v| f16::from_f32(v as f32 / 255.0)));
static FULL_IN: LazyLock<[f32; 256]> = LazyLock::new(|| std::array::from_fn(|v| v as f32 / 255.0));
/// Every half-precision value as an 8-bit sample, by its bits.
static HALF_OUT: LazyLock<Vec<u8>> = LazyLock::new(|| {
    (0..=u16::MAX)
        .map(|bits| byte(f16::from_bits(bits).to_f32()))
        .collect()
});

pub struct OnnxEngine {
    model: Loaded,
    /// The model file's name, for the log.
    file: String,
    half: bool,
    scale: usize,
    tile: usize,
    in_name: String,
    out_name: String,
    /// The model's own scale, when its result is shrunk inside it.
    shrunk_from: Option<usize>,
}

/// What ONNX Runtime calls the TensorRT RTX plugin.
const PLUGIN: &str = "NvTensorRTRTXExecutionProvider";
const TENSORRT: &str = "TensorRT";

/// Makes the plugin known to ONNX Runtime, once.
fn register_plugin(env: &Environment) -> Result<()> {
    static DONE: OnceLock<std::result::Result<(), String>> = OnceLock::new();
    DONE.get_or_init(|| {
        let file = crate::models::find_tensorrt_plugin()
            .ok_or("its library is not beside ONNX Runtime (see the user guide)")?;
        env.register_ep_library(PLUGIN, file)
            .map_err(|e| e.to_string())
    })
    .clone()
    .map_err(|e| anyhow!("the TensorRT plugin cannot be used: {e}"))
}

fn session_options(env: &Environment) -> ort::Result<SessionBuilder> {
    // Idle model threads sleep instead of spinning: the stages of the video
    // pipeline run at the same time and share the processor.
    Session::builder(env)?
        .with_config_entry("session.intra_op.allow_spinning", "0")
        .map_err(|e| e.into())
}

fn finish(
    mut b: SessionBuilder,
    source: &Source,
    sizes: &[(String, usize)],
) -> ort::Result<Session> {
    for (dim, size) in sizes {
        b = b
            .with_dimension_override(dim, *size as i64)
            .map_err(|e| -> ort::Error { e.into() })?;
    }
    match source {
        Source::File(path) => b.commit_from_file(path),
        Source::Memory(bytes) => b.commit_from_memory(bytes),
    }
}

/// The model on the TensorRT plugin, entirely on the card: letting parts fall
/// to the processor made it far slower in testing.
fn build_on_plugin(
    env: &Environment,
    source: &Source,
    first: usize,
    sizes: &[(String, usize)],
) -> Result<Session> {
    register_plugin(env)?;
    let cards: Vec<_> = env
        .devices()
        .filter(|d| d.ep().is_ok_and(|name| name == PLUGIN))
        .collect();
    if cards.is_empty() {
        bail!("the TensorRT plugin found no graphics card it can use");
    }
    let b = session_options(env)?
        .with_devices(cards, None)
        .map_err(|e| -> ort::Error { e.into() })?
        .with_config_entry("session.disable_cpu_ep_fallback", "1")
        .map_err(|e| -> ort::Error { e.into() })?;
    // It cannot build for sizes that are left open.
    let open = [("h".to_string(), first), ("w".to_string(), first)];
    let sizes = if sizes.is_empty() { &open[..] } else { sizes };
    Ok(finish(b, source, sizes)?)
}

/// Loads a model on the first processor of `device` that takes it, and says
/// which one. `sizes` fixes the model's free dimensions (by name) in advance.
fn build_session(
    env: &Environment,
    source: &Source,
    device: Device,
    first: usize,
    sizes: &[(String, usize)],
) -> Result<(Session, String)> {
    let mut last = None;
    let plugin = match device {
        Device::Tensorrt => true,
        Device::Auto => crate::models::find_tensorrt_plugin().is_some(),
        _ => false,
    };
    if plugin {
        match build_on_plugin(env, source, first, sizes) {
            Ok(s) => return Ok((s, TENSORRT.to_string())),
            // Asked for by name: say why rather than use something else.
            Err(e) if device == Device::Tensorrt => return Err(e),
            Err(e) => eprintln!("TensorRT is not used: {e:#}"),
        }
    }
    for (name, p) in device.providers() {
        let built = (|| -> ort::Result<Session> {
            let mut b = session_options(env)?
                .with_execution_providers([p.error_on_failure()])
                .map_err(|e| -> ort::Error { e.into() })?;
            if name == "DirectML" {
                // DirectML does not support ONNX Runtime's memory planning.
                b = b
                    .with_memory_pattern(false)
                    .map_err(|e| -> ort::Error { e.into() })?;
            }
            finish(b, source, sizes)
        })();
        match built {
            Ok(s) => return Ok((s, name.to_string())),
            Err(e) => last = Some(format!("{name}: {e}")),
        }
    }
    let file = match source {
        Source::File(path) => path.display().to_string(),
        Source::Memory(_) => "the changed model".to_string(),
    };
    Err(anyhow!(
        "no execution provider could load {file}: {}",
        last.unwrap_or_default()
    ))
}

fn is_out_of_memory(e: &ort::Error) -> bool {
    let m = e.to_string().to_ascii_lowercase();
    m.contains("out of memory")
        || m.contains("not enough memory")
        || m.contains("8007000e")
        || m.contains("failed to allocate")
        || m.contains("bad_alloc")
        || m.contains("oom")
}

fn ran_out(e: &anyhow::Error) -> bool {
    e.downcast_ref::<ort::Error>().is_some_and(is_out_of_memory)
}

/// Half precision for a model that wants it, converted on every core.
fn to_half(v: &[f32]) -> Vec<f16> {
    let mut out = vec![f16::ZERO; v.len()];
    out.par_chunks_mut(BLOCK)
        .zip(v.par_chunks(BLOCK))
        .for_each(|(o, i)| o.convert_from_f32_slice(i));
    out
}

fn from_half(v: &[f16]) -> Vec<f32> {
    let mut out = vec![0.0f32; v.len()];
    out.par_chunks_mut(BLOCK)
        .zip(v.par_chunks(BLOCK))
        .for_each(|(o, i)| i.convert_to_f32_slice(o));
    out
}

impl OnnxEngine {
    /// Loads `model` and works out its scale by running one small tile.
    pub fn load(
        model: &Path,
        device: Device,
        vram_mb: usize,
        tile: Option<usize>,
    ) -> Result<OnnxEngine> {
        let loaded = Loaded::new(model, device, 16)?;
        let session = &loaded.session;
        let input = session.inputs().first().context("the model has no input")?;
        let half = match input.dtype() {
            ValueType::Tensor {
                ty: TensorElementType::Float16,
                ..
            } => true,
            ValueType::Tensor {
                ty: TensorElementType::Float32,
                ..
            } => false,
            other => bail!("unsupported model input type {other:?}"),
        };
        let in_name = input.name().to_string();
        let out_name = session
            .outputs()
            .first()
            .context("the model has no output")?
            .name()
            .to_string();
        let mut e = OnnxEngine {
            model: loaded,
            file: model
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned(),
            half,
            scale: 1,
            tile: tile.map_or_else(
                || tile_for_budget(vram_mb, 2, if half { 2 } else { 4 }),
                |t| t.clamp(64, 1024),
            ),
            in_name,
            out_name,
            shrunk_from: None,
        };
        e.scale = e.probe()?;
        Ok(e)
    }

    /// The model's scale, found by running one small tile.
    fn probe(&mut self) -> Result<usize> {
        let out = self.run_tile(&Plane::new(16, 16, 3))?;
        if out.w % 16 != 0 || out.w == 0 {
            bail!(
                "the model's output ({}x{}) is not a whole multiple of its input",
                out.w,
                out.h
            );
        }
        Ok(out.w / 16)
    }

    /// For a result `target` times the input from a model that makes twice
    /// that: shrinks inside the model, so the card does it and a quarter of
    /// the data comes back (about a quarter faster in testing). Does nothing
    /// on the processor, or if the model cannot be changed this way.
    pub fn shrink_to(&mut self, target: usize) {
        if self.model.provider != "CPU" {
            self.shrink_anywhere(target);
        }
    }

    /// `shrink_to` on any processor; returns whether the model now shrinks.
    pub fn shrink_anywhere(&mut self, target: usize) -> bool {
        if self.shrunk_from.is_some() || self.scale != target * 2 {
            return false;
        }
        let Source::File(path) = &self.model.source else {
            return false;
        };
        let Some(changed) = std::fs::read(path)
            .ok()
            .and_then(|bytes| shrink::halve_output(&bytes))
        else {
            return false;
        };
        let before = (self.model.source.clone(), self.out_name.clone());
        self.model.source = Source::Memory(changed.into());
        let works = self.use_current_source().is_ok_and(|scale| scale == target);
        if works {
            self.shrunk_from = Some(self.scale);
            self.scale = target;
        } else {
            (self.model.source, self.out_name) = before;
            // It worked a moment ago; a failure shows at the first frame.
            let _ = self.use_current_source();
        }
        works
    }

    fn use_current_source(&mut self) -> Result<usize> {
        // Sized for the probe below.
        (self.model.session, self.model.provider) = build_session(
            &self.model.env,
            &self.model.source,
            self.model.device,
            16,
            &[],
        )?;
        self.model.sized = None;
        self.out_name = self
            .model
            .session
            .outputs()
            .first()
            .context("the model has no output")?
            .name()
            .to_string();
        self.probe()
    }

    fn run_tile(&mut self, t: &Plane) -> Result<Plane> {
        // (16 x 16 is the probe made while loading.)
        if t.w > 16 {
            self.model.prepare(t.w, t.h)?;
        }
        let shape = [1i64, t.c as i64, t.h as i64, t.w as i64];
        let chw = to_chw(t);
        if self.half {
            let input = Tensor::from_array((shape, to_half(&chw)))?;
            let outs = self
                .model
                .session
                .run(ort::inputs![self.in_name.as_str() => input])?;
            let (s, d) = outs[self.out_name.as_str()].try_extract_tensor::<f16>()?;
            let (h, w) = (s[2] as usize, s[3] as usize);
            Ok(from_chw(&from_half(d), w, h, t.c))
        } else {
            let input = Tensor::from_array((shape, chw))?;
            let outs = self
                .model
                .session
                .run(ort::inputs![self.in_name.as_str() => input])?;
            let (s, d) = outs[self.out_name.as_str()].try_extract_tensor::<f32>()?;
            let (h, w) = (s[2] as usize, s[3] as usize);
            Ok(from_chw(d, w, h, t.c))
        }
    }

    /// A frame that fits in one tile, from 8-bit straight to 8-bit: the same
    /// pictures as the tiled route, without its larger copies in between.
    fn run_whole(&mut self, rgb: &[u8], w: usize, h: usize) -> Result<Vec<u8>> {
        let (pw, ph) = (round_up(w, 16), round_up(h, 16));
        self.model.prepare(pw, ph)?;
        let s = self.scale;
        let (big, kept) = ((pw * s, ph * s), (w * s, h * s));
        let shape = [1i64, 3, ph as i64, pw as i64];
        let expected = [1i64, 3, big.1 as i64, big.0 as i64];
        let wrong = |got: &[i64]| anyhow!("the model returned {got:?}, expected {expected:?}");
        if self.half {
            let data = chw_from_rgb8(rgb, (w, h), (pw, ph), &HALF_IN);
            let input = Tensor::from_array((shape, data))?;
            let outs = self
                .model
                .session
                .run(ort::inputs![self.in_name.as_str() => input])?;
            let (got, d) = outs[self.out_name.as_str()].try_extract_tensor::<f16>()?;
            if got[..] != expected {
                return Err(wrong(got));
            }
            let table = HALF_OUT.as_slice();
            Ok(rgb8_from_chw(d, big, kept, |v| {
                table[usize::from(v.to_bits())]
            }))
        } else {
            let data = chw_from_rgb8(rgb, (w, h), (pw, ph), &FULL_IN);
            let input = Tensor::from_array((shape, data))?;
            let outs = self
                .model
                .session
                .run(ort::inputs![self.in_name.as_str() => input])?;
            let (got, d) = outs[self.out_name.as_str()].try_extract_tensor::<f32>()?;
            if got[..] != expected {
                return Err(wrong(got));
            }
            Ok(rgb8_from_chw(d, big, kept, byte))
        }
    }

    /// The next smaller tile and a fresh session, after the card ran out of
    /// memory with this one.
    fn shrink(&mut self, e: anyhow::Error) -> Result<()> {
        let next = smaller_tile(self.tile)
            .ok_or_else(|| e.context("out of GPU memory even with 128-pixel tiles"))?;
        eprintln!("\nOut of GPU memory: reloading the model with {next}-pixel tiles");
        self.model.reload()?;
        self.tile = next;
        Ok(())
    }

    /// Runs the model over a whole frame, stepping the tile size down (with a
    /// fresh session) if the GPU runs out of memory.
    fn upscale_plane(&mut self, src: &Plane) -> Result<Plane> {
        loop {
            let tile = self.tile;
            let scale = self.scale;
            let mut oom = false;
            let r = upscale_tiled(src, scale, tile, overlap_for(tile), |t| {
                self.run_tile(t).inspect_err(|e| oom = ran_out(e))
            });
            match r {
                Ok(p) => return Ok(p),
                Err(e) if oom => self.shrink(e)?,
                Err(e) => return Err(e),
            }
        }
    }
}

impl Engine for OnnxEngine {
    fn scale(&self) -> usize {
        self.scale
    }

    fn process(&mut self, rgb: &[u8], w: usize, h: usize) -> Result<Vec<u8>> {
        while round_up(w, 16) <= self.tile && round_up(h, 16) <= self.tile {
            match self.run_whole(rgb, w, h) {
                Err(e) if ran_out(&e) => self.shrink(e)?,
                done => return done,
            }
        }
        let out = self.upscale_plane(&plane_from_rgb8(rgb, w, h))?;
        Ok(rgb8_from_plane(&out))
    }

    fn describe(&self) -> String {
        let scale = match self.shrunk_from {
            Some(own) => format!("{own}x shrunk to {}x on the card", self.scale),
            None => format!("{}x", self.scale),
        };
        format!(
            "{} on {} ({}, {scale}, tiles up to {})",
            self.file,
            self.model.provider,
            if self.half { "FP16" } else { "FP32" },
            self.tile
        )
    }

    fn process_plane(&mut self, src: &Plane) -> Result<Plane> {
        self.upscale_plane(src)
    }
}

/// ONNX Runtime allows one environment per process, so every model shares it.
fn environment() -> Result<Arc<Environment>> {
    static ENV: Mutex<Option<Arc<Environment>>> = Mutex::new(None);
    let mut slot = ENV.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(e) = slot.as_ref() {
        return Ok(Arc::clone(e));
    }
    let lib = crate::models::find_runtime().with_context(|| {
        format!(
            "ONNX Runtime is not set up.\n{}",
            crate::models::RUNTIME_HELP
        )
    })?;
    let env = Arc::new(
        ort::init_from(lib)
            .map_err(|e| anyhow!("cannot load ONNX Runtime: {e}"))?
            // Microsoft's Windows builds report usage unless told not to.
            // Nothing about the user's work leaves this computer.
            .with_telemetry(false)
            .build()?,
    );
    *slot = Some(Arc::clone(&env));
    Ok(env)
}

/// A five-frame video denoiser (FastDVDnet's interface): inputs `frames`
/// [1,15,H,W] and `sigma` [1,1,H,W], output [1,3,H,W].
pub struct OnnxDenoiser {
    model: Loaded,
    half: bool,
    tile: usize,
    sigma: f32,
    label: String,
}

impl OnnxDenoiser {
    /// `sigma` is the noise level on a 0..255 scale (about 5 = faint, 20 = clearly
    /// noisy, 40 = very noisy).
    pub fn load(
        model: &Path,
        device: Device,
        vram_mb: usize,
        tile: Option<usize>,
        sigma: f32,
    ) -> Result<OnnxDenoiser> {
        let loaded = Loaded::new(model, device, 64)?;
        let (session, provider) = (&loaded.session, loaded.provider.clone());
        let names: Vec<&str> = session.inputs().iter().map(|i| i.name()).collect();
        if names != ["frames", "sigma"] {
            bail!(
                "{} is not a five-frame denoiser: it has inputs {names:?}, expected [\"frames\", \"sigma\"]",
                model.display()
            );
        }
        let half = matches!(
            session.inputs()[0].dtype(),
            ValueType::Tensor {
                ty: TensorElementType::Float16,
                ..
            }
        );
        let tile = tile.map_or_else(
            || denoiser_tile_for_budget(vram_mb, if half { 2 } else { 4 }),
            |t| t.clamp(64, 1024),
        );
        Ok(OnnxDenoiser {
            model: loaded,
            half,
            tile,
            sigma,
            label: format!(
                "{} on {} ({}, sigma {sigma}, tiles up to {tile})",
                model.file_name().unwrap_or_default().to_string_lossy(),
                provider,
                if half { "FP16" } else { "FP32" },
            ),
        })
    }

    fn run_tile(&mut self, t: &Plane) -> Result<Plane> {
        self.model.prepare(t.w, t.h)?;
        let (w, h) = (t.w as i64, t.h as i64);
        let chw = to_chw(t);
        let level = self.sigma / 255.0;
        let (fs, ss) = ([1i64, (3 * FRAMES) as i64, h, w], [1i64, 1, h, w]);
        if self.half {
            let n = vec![f16::from_f32(level); t.w * t.h];
            let outs = self.model.session.run(ort::inputs![
                "frames" => Tensor::from_array((fs, to_half(&chw)))?,
                "sigma" => Tensor::from_array((ss, n))?
            ])?;
            let (_, d) = outs[0].try_extract_tensor::<f16>()?;
            Ok(from_chw(&from_half(d), t.w, t.h, 3))
        } else {
            let n = vec![level; t.w * t.h];
            let outs = self.model.session.run(ort::inputs![
                "frames" => Tensor::from_array((fs, chw))?,
                "sigma" => Tensor::from_array((ss, n))?
            ])?;
            let (_, d) = outs[0].try_extract_tensor::<f32>()?;
            Ok(from_chw(d, t.w, t.h, 3))
        }
    }
}

impl Temporal for OnnxDenoiser {
    fn denoise(&mut self, w: &[&Plane; FRAMES]) -> Result<Plane> {
        let stacked = stack(w);
        loop {
            let tile = self.tile;
            let mut oom = false;
            let r = map_tiled(&stacked, 1, 3, tile, overlap_for(tile), |t| {
                self.run_tile(t).inspect_err(|e| oom = ran_out(e))
            });
            match r {
                Ok(p) => return Ok(p),
                Err(e) if oom => {
                    let next = smaller_tile(tile)
                        .ok_or_else(|| e.context("out of GPU memory even with 128-pixel tiles"))?;
                    eprintln!(
                        "\nOut of GPU memory: reloading the denoiser with {next}-pixel tiles"
                    );
                    self.model.reload()?;
                    self.tile = next;
                }
                Err(e) => return Err(e),
            }
        }
    }

    fn describe(&self) -> String {
        self.label.clone()
    }
}

/// Loads a model on the CPU and works out what it is, or says why it cannot be
/// used. A denoiser is also run once on a small picture to confirm its output.
pub fn inspect(path: &Path) -> Result<crate::models::ModelKind> {
    use crate::models::ModelKind;
    let env = environment()?;
    let source = Source::File(path.to_path_buf());
    let (session, _) = build_session(&env, &source, Device::Cpu, 64, &[])?;
    let names: Vec<String> = session
        .inputs()
        .iter()
        .map(|i| i.name().to_string())
        .collect();
    drop(session);
    drop(env);
    if names == ["frames", "sigma"] {
        let mut d = OnnxDenoiser::load(path, Device::Cpu, 2048, Some(64), 15.0)?;
        let mut f = Plane::new(64, 48, 3);
        f.data.fill(0.5);
        let five = [&f, &f, &f, &f, &f];
        let out = d.denoise(&five)?;
        if (out.w, out.h, out.c) != (64, 48, 3) {
            bail!(
                "the five-frame model returned {}x{}x{} for a 64x48 picture",
                out.w,
                out.h,
                out.c
            );
        }
        return Ok(ModelKind::Denoiser5);
    }
    if names.len() != 1 {
        bail!(
            "unsupported model: it has {} inputs ({}); expected one image input, or `frames` + `sigma`",
            names.len(),
            names.join(", ")
        );
    }
    let e = OnnxEngine::load(path, Device::Cpu, 2048, Some(64))?;
    Ok(ModelKind::Upscaler { scale: e.scale() })
}
