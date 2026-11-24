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

## Status

Video pipeline, resume, tiler and ONNX engine work. Not yet: photo mode
(JPEG tiering, JXL output), video-aware denoise models, grain re-adding.
