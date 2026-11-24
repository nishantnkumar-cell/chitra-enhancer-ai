# Chitra Enhancer AI

Restore and enlarge old, noisy video (camcorder, tape) on your own computer.
Everything runs locally; nothing is uploaded.

```
chitra-enhance -i tape.avi -o restored.mp4 --model realesr-2x-fp16.onnx
```

## What it does to a video

1. **Deinterlaces** tape-style fields (`bwdif`) when the file says it is interlaced.
2. **Removes noise** over space and time (`hqdn3d`; `--denoise light|medium|strong|off`).
3. Makes pixels **square** (DV 720x480 becomes 640x480).
4. **Enlarges** every frame with an AI model (ONNX) or, without one, a Lanczos resize.
5. Encodes (H.265 default; `--codec h264|av1|prores`) and copies the **audio**.

Work is done in **chunks** (`--chunk-seconds`, default 30). Press Ctrl-C at any time;
run the same command again to carry on from the first unfinished chunk. Changed
settings or input are refused unless you add `--restart`.

## Needs

* FFmpeg and ffprobe on the PATH (or `--ffmpeg PATH`).
* For AI models: build with `cargo build --release --features onnx` and point
  `ORT_DYLIB_PATH` at the ONNX Runtime library (`libonnxruntime.so`,
  `onnxruntime.dll`, `libonnxruntime.dylib`). `--device auto|cuda|directml|coreml|cpu`.
* Models are FP16 or FP32 ONNX, input RGB 0..1 (NCHW). Tile size follows
  `--vram-budget` (GB) and steps down 512 -> 256 -> 128 if the GPU runs out of
  memory. Run tests with a model: `CHITRA_TEST_MODEL=x.onnx cargo test --features onnx`.

## Getting a model

`scripts/export_realesr_compact.py` turns the official Real-ESRGAN "compact"
weights (BSD-3-Clause; `realesr-general-x4v3.pth` or `realesr-animevideov3.pth`
from the xinntao/Real-ESRGAN releases) into FP32 and FP16 ONNX files; see the
header of the script for the one `uv run` command. A 4x model can give a 2x
result (`--scale 2`), which is usually sharper than a native 2x model.
Weights are read with `weights_only=True` so a bad file cannot run code. Check
the download's SHA-256 against the project's release notes first.

On a CPU the FP32 file is the faster one; on a GPU use FP16.

## Status

Video pipeline, resume, tiler, ONNX engine, grain and 1-4x scaling work, tested
with a real Real-ESRGAN model. Not yet: photo mode (JPEG tiering, JXL output),
video-aware (multi-frame) denoise models.
