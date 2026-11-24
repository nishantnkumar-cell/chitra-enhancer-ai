//! Running an ONNX upscaling model on the GPU (CUDA, DirectML, Core ML) or the
//! CPU through ONNX Runtime. The library is loaded at run time: set
//! `ORT_DYLIB_PATH` to `libonnxruntime.so` / `onnxruntime.dll` / `.dylib`.
//!
//! A model is loaded once and kept for the whole run. The tile size starts from
//! the memory budget; when the GPU runs out of memory the session is rebuilt
//! (a failed CUDA allocation can leave it unusable) and the tile size steps
//! down 512 -> 256 -> 128.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use clap::ValueEnum;
use half::f16;
use ort::environment::Environment;
use ort::ep::{self, ExecutionProviderDispatch};
use ort::session::Session;
use ort::value::{Tensor, TensorElementType, ValueType};

use crate::engine::Engine;
use crate::infer::{
    from_chw, overlap_for, plane_from_rgb8, rgb8_from_plane, smaller_tile, tile_for_budget, to_chw,
};
use crate::tiler::{Plane, upscale_tiled};

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
    env: Environment,
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
            Ok(b.commit_from_file(path)?)
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
        || m.contains("failed to allocate")
        || m.contains("bad_alloc")
        || m.contains("oom")
}

impl OnnxEngine {
    /// Loads `model` and works out its scale by running one small tile.
    pub fn load(model: &Path, device: Device, vram_mb: usize) -> Result<OnnxEngine> {
        let lib = std::env::var_os("ORT_DYLIB_PATH").context(
            "ONNX Runtime is not set up: set ORT_DYLIB_PATH to the ONNX Runtime library \
             (libonnxruntime.so, onnxruntime.dll or libonnxruntime.dylib)",
        )?;
        let env = ort::init_from(lib)
            .map_err(|e| anyhow!("cannot load ONNX Runtime: {e}"))?
            .build()?;
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
            tile: tile_for_budget(vram_mb, 2, if half { 2 } else { 4 }),
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
        let (out_w, out_h);
        let flat: Vec<f32> = if self.half {
            let data: Vec<f16> = chw.iter().map(|&v| f16::from_f32(v)).collect();
            let input = Tensor::from_array((shape, data))?;
            let outs = self
                .session
                .run(ort::inputs![self.in_name.as_str() => input])?;
            let (s, d) = outs[self.out_name.as_str()].try_extract_tensor::<f16>()?;
            (out_h, out_w) = (s[2] as usize, s[3] as usize);
            d.iter().map(|v| v.to_f32()).collect()
        } else {
            let input = Tensor::from_array((shape, chw))?;
            let outs = self
                .session
                .run(ort::inputs![self.in_name.as_str() => input])?;
            let (s, d) = outs[self.out_name.as_str()].try_extract_tensor::<f32>()?;
            (out_h, out_w) = (s[2] as usize, s[3] as usize);
            d.to_vec()
        };
        Ok(from_chw(&flat, out_w, out_h, t.c))
    }

    /// Runs the model over a whole frame, stepping the tile size down (with a
    /// fresh session) if the GPU runs out of memory.
    fn upscale_plane(&mut self, src: &Plane) -> Result<Plane> {
        loop {
            let tile = self.tile;
            let scale = self.scale;
            let mut oom = false;
            let r = upscale_tiled(src, scale, tile, overlap_for(tile), |t| {
                self.run_tile(t).map_err(|e| {
                    oom = e.downcast_ref::<ort::Error>().is_some_and(is_out_of_memory);
                    e
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
}
