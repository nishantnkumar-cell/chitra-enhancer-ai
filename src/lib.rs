//! Chitra Enhancer AI: restore and upscale old, noisy video (and photos).

pub mod analyze;
pub mod disc;
pub mod engine;
pub mod ffmpeg;
pub mod infer;
pub mod inspector;
pub mod job;
pub mod legal;
pub mod models;
pub mod noise;
#[cfg(feature = "onnx")]
pub mod onnx;
pub mod options;
pub mod photo;
pub mod shrink;
pub mod temporal;
pub mod tiler;
pub mod video;
