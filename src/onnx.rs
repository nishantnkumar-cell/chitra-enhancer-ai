//! Running an ONNX upscaling model on the GPU (CUDA, DirectML, Core ML) or the
//! CPU through ONNX Runtime. The library is loaded at run time: set
//! `ORT_DYLIB_PATH` to `libonnxruntime.so` / `onnxruntime.dll` / `.dylib`.
//!
//! A model is loaded once and kept for the whole run. The tile size starts from
//! the memory budget; when the GPU runs out of memory the session is rebuilt
//! (a failed CUDA allocation can leave it unusable) and the tile size steps
//! down 512 -> 256 -> 128.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, anyhow, bail};
use clap::ValueEnum;
use half::f16;
use half::slice::HalfFloatSliceExt;
use ort::environment::Environment;
use ort::ep::{self, ExecutionProviderDispatch};
use ort::session::Session;
use ort::value::{Tensor, TensorElementType, ValueType};
use rayon::prelude::*;

use crate::engine::Engine;
use crate::infer::{
    BLOCK, from_chw, overlap_for, plane_from_rgb8, rgb8_from_plane, smaller_tile, tile_for_budget,
    to_chw,
};
use crate::temporal::{FRAMES, Temporal, stack};
use crate::tiler::{Plane, map_tiled, upscale_tiled};

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum Device {
    /// The best this computer offers: CUDA, DirectML or Core ML, else CPU.
    Auto,
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

pub struct OnnxEngine {
    env: Arc<Environment>,
    path: PathBuf,
    device: Device,
    session: Session,
    half: bool,
    scale: usize,
    tile: usize,
    in_name: String,
    out_name: String,
    label: String,
}

fn build_session(env: &Environment, path: &Path, device: Device) -> Result<(Session, String)> {
    let mut last = None;
    for (name, p) in device.providers() {
        let built = (|| -> ort::Result<Session> {
            let mut b = Session::builder(env)?
                .with_execution_providers([p.error_on_failure()])
                .map_err(|e| -> ort::Error { e.into() })?;
            // Idle model threads sleep instead of spinning: the stages of the
            // video pipeline run at the same time and share the processor.
            b = b
                .with_config_entry("session.intra_op.allow_spinning", "0")
                .map_err(|e| -> ort::Error { e.into() })?;
            if name == "DirectML" {
                // DirectML does not support ONNX Runtime's memory planning.
                b = b
                    .with_memory_pattern(false)
                    .map_err(|e| -> ort::Error { e.into() })?;
            }
            b.commit_from_file(path)
        })();
        match built {
            Ok(s) => return Ok((s, name.to_string())),
            Err(e) => last = Some(format!("{name}: {e}")),
        }
    }
    Err(anyhow!(
        "no execution provider could load {}: {}",
        path.display(),
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
        let env = environment()?;
        let (session, provider) = build_session(&env, model, device)?;
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
            env,
            path: model.to_path_buf(),
            device,
            session,
            half,
            scale: 1,
            tile: tile.map_or_else(
                || tile_for_budget(vram_mb, 2, if half { 2 } else { 4 }),
                |t| t.clamp(64, 1024),
            ),
            in_name,
            out_name,
            label: String::new(),
        };
        let probe = e.run_tile(&Plane::new(16, 16, 3))?;
        if probe.w % 16 != 0 || probe.w == 0 {
            bail!(
                "the model's output ({}x{}) is not a whole multiple of its input",
                probe.w,
                probe.h
            );
        }
        e.scale = probe.w / 16;
        e.label = format!(
            "{} on {} ({}, {}x, tiles up to {})",
            model.file_name().unwrap_or_default().to_string_lossy(),
            provider,
            if half { "FP16" } else { "FP32" },
            e.scale,
            e.tile
        );
        Ok(e)
    }

    fn run_tile(&mut self, t: &Plane) -> Result<Plane> {
        let shape = [1i64, t.c as i64, t.h as i64, t.w as i64];
        let chw = to_chw(t);
        if self.half {
            let input = Tensor::from_array((shape, to_half(&chw)))?;
            let outs = self
                .session
                .run(ort::inputs![self.in_name.as_str() => input])?;
            let (s, d) = outs[self.out_name.as_str()].try_extract_tensor::<f16>()?;
            let (h, w) = (s[2] as usize, s[3] as usize);
            Ok(from_chw(&from_half(d), w, h, t.c))
        } else {
            let input = Tensor::from_array((shape, chw))?;
            let outs = self
                .session
                .run(ort::inputs![self.in_name.as_str() => input])?;
            let (s, d) = outs[self.out_name.as_str()].try_extract_tensor::<f32>()?;
            let (h, w) = (s[2] as usize, s[3] as usize);
            Ok(from_chw(d, w, h, t.c))
        }
    }

    /// Runs the model over a whole frame, stepping the tile size down (with a
    /// fresh session) if the GPU runs out of memory.
    fn upscale_plane(&mut self, src: &Plane) -> Result<Plane> {
        loop {
            let tile = self.tile;
            let scale = self.scale;
            let mut oom = false;
            let r = upscale_tiled(src, scale, tile, overlap_for(tile), |t| {
                self.run_tile(t).inspect_err(|e| {
                    oom = e.downcast_ref::<ort::Error>().is_some_and(is_out_of_memory);
                })
            });
            match r {
                Ok(p) => return Ok(p),
                Err(e) if oom => {
                    let next = smaller_tile(tile)
                        .ok_or_else(|| e.context("out of GPU memory even with 128-pixel tiles"))?;
                    eprintln!("\nOut of GPU memory: reloading the model with {next}-pixel tiles");
                    let (s, _) = build_session(&self.env, &self.path, self.device)?;
                    self.session = s;
                    self.tile = next;
                }
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
        let out = self.upscale_plane(&plane_from_rgb8(rgb, w, h))?;
        Ok(rgb8_from_plane(&out))
    }

    fn describe(&self) -> String {
        self.label.clone()
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
            .build()?,
    );
    *slot = Some(Arc::clone(&env));
    Ok(env)
}

/// A five-frame video denoiser (FastDVDnet's interface): inputs `frames`
/// [1,15,H,W] and `sigma` [1,1,H,W], output [1,3,H,W].
pub struct OnnxDenoiser {
    env: Arc<Environment>,
    path: PathBuf,
    device: Device,
    session: Session,
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
        let env = environment()?;
        let (session, provider) = build_session(&env, model, device)?;
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
        // Five frames in and two U-Net stages: roughly 4x the memory of an upscaler tile.
        let tile = tile.map_or_else(
            || tile_for_budget(vram_mb / 3, 1, if half { 2 } else { 4 }),
            |t| t.clamp(64, 1024),
        );
        Ok(OnnxDenoiser {
            env,
            path: model.to_path_buf(),
            device,
            session,
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
        let (w, h) = (t.w as i64, t.h as i64);
        let chw = to_chw(t);
        let level = self.sigma / 255.0;
        let (fs, ss) = ([1i64, (3 * FRAMES) as i64, h, w], [1i64, 1, h, w]);
        if self.half {
            let n = vec![f16::from_f32(level); t.w * t.h];
            let outs = self.session.run(ort::inputs![
                "frames" => Tensor::from_array((fs, to_half(&chw)))?,
                "sigma" => Tensor::from_array((ss, n))?
            ])?;
            let (_, d) = outs[0].try_extract_tensor::<f16>()?;
            Ok(from_chw(&from_half(d), t.w, t.h, 3))
        } else {
            let n = vec![level; t.w * t.h];
            let outs = self.session.run(ort::inputs![
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
                self.run_tile(t).inspect_err(|e| {
                    oom = e.downcast_ref::<ort::Error>().is_some_and(is_out_of_memory);
                })
            });
            match r {
                Ok(p) => return Ok(p),
                Err(e) if oom => {
                    let next = smaller_tile(tile)
                        .ok_or_else(|| e.context("out of GPU memory even with 128-pixel tiles"))?;
                    eprintln!(
                        "\nOut of GPU memory: reloading the denoiser with {next}-pixel tiles"
                    );
                    let (s, _) = build_session(&self.env, &self.path, self.device)?;
                    self.session = s;
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
    let (session, _) = build_session(&env, path, Device::Cpu)?;
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
