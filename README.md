# Chitra Enhancer AI

Restore and enlarge old, noisy video (camcorder, tape) on your own computer.
Everything runs locally; nothing is uploaded.

```
chitra-enhance -i tape.avi -o restored.mp4 --model realesr-2x-fp16.onnx
```

## What it does to a video

1. **Deinterlaces** tape-style fields (`bwdif`) when the file says it is interlaced.
2. **Removes noise** over space and time (`hqdn3d`; `--denoise light|medium|strong|off`), or,
   far better on camcorder grain, with a five-frame AI denoiser (`--temporal-model`, below).
3. Makes pixels **square** (DV 720x480 becomes 640x480).
4. **Enlarges** every frame with an AI model (ONNX) or, without one, a Lanczos resize.
5. Encodes (H.265 default; `--codec h264|av1|prores`) and copies the **audio**.

Work is done in **chunks** (`--chunk-seconds`, default 30). Press Ctrl-C at any time;
run the same command again to carry on from the first unfinished chunk. Changed
settings or input are refused unless you add `--restart`.

### Five-frame denoising (recommended for old camcorder footage)

`--temporal-model fastdvdnet_fp16.onnx --temporal-sigma 15 --denoise off` shows the
model each frame together with the two before and two after it, so noise that
changes from frame to frame is averaged out and the picture that stays put is
kept. `--temporal-sigma` is the noise level on a 0-255 scale (5 faint, 15-20 clearly
noisy, 40 very noisy); too high softens detail. Left out, it is **measured** from
samples across the video (8 frames, the quietest quarter of each) and scaled by
1.6, which tuned best on three test clips of known noise. The model is FastDVDnet (MIT);
`scripts/export_fastdvdnet.py` makes the ONNX files from its `model.pth`.
On a test clip (natural picture panning, ffmpeg noise strength 25) the result
scored 36.3 dB against the clean original, against 33.7 dB for `--denoise medium`
and 33.1 dB with no denoising. At a cut between shots the window never reaches
across: frames from the other shot are replaced by the nearest of the same one. At the start and end of each chunk
the edge frame is repeated, which is invisible in practice.

## You have the final say

Everything the program chooses by itself can be set by hand, within sensible
ranges: noise level (`--temporal-sigma 3..60`, or *Measure now* in the window to see
the measurement and then adjust it), grain (`--grain 0..30`), deinterlacing
(`--deinterlace`), tile size (`--tile-size`), video quality (`--crf`), chunk length
(`--chunk-seconds`), output name, device and GPU memory. `--measure-noise` prints
what it measured and suggests. Previews can be 1 to 30 seconds (`--preview`).

## Trying settings first

`--preview 5` processes only 5 seconds from the middle of the video with your real
settings and writes the short clip plus a before/after still (`<name>.compare.png`),
so you can judge noise level, scale and model in seconds. The window has a
**Preview 5 s** button that shows the still. A preview never overwrites a
finished result (the window uses `<name>-preview.mp4`; the command line asks for
`--force`).

## The window (GUI)

`cargo build --release --features gui,onnx` makes two programs, `chitra-enhance`
(command line) and `chitra-enhance-gui` (a window; keep them in the same folder).
Pick a video or photo, tick *Five-frame AI denoising* for camcorder noise, press
Start. Stop keeps every finished chunk, and Continue picks up where it left off.
The window finds models in `~/.local/share/chitra-enhance/models`
(`%LOCALAPPDATA%\chitra-enhance\models` on Windows) and shows the exact command
line it runs, so anything done in the window can be repeated in a script.

## Photos

```
chitra-enhance -i old.jpg -o old-2x.jxl --model realesr-2x.onnx
```

The mode is chosen from the input (JPEG, PNG, WebP, BMP, GIF, TIFF). Output is
a **16-bit** JPEG XL (`--jxl-distance 1.0` is visually lossless) or 16-bit PNG,
with the colour profile and EXIF kept. The photo is rotated by its EXIF tag and
the tag reset, so it is not rotated twice. The source is never overwritten and
an existing output needs `--force`.

For JPEGs the quality is read from the file's quantisation tables. **Clean**
files (quality about 85 and up) are only enlarged; **compressed** files (below
that, or 4:2:0 chroma below 92) use `--model-compressed` if you give one, e.g. a
JPEG-artifact remover. Big photos are tiled (`--tile-size`, `--vram-budget`).
Transparency is not kept.

## Needs

* FFmpeg and ffprobe on the PATH (or `--ffmpeg PATH`).
* For AI models: build with `cargo build --release --features onnx` and point
  `ORT_DYLIB_PATH` at the ONNX Runtime library (`libonnxruntime.so`,
  `onnxruntime.dll`, `libonnxruntime.dylib`). `--device auto|cuda|directml|coreml|cpu`.
* Models are FP16 or FP32 ONNX, input RGB 0..1 (NCHW). Tile size follows
  `--vram-budget` (GB) and steps down 512 -> 256 -> 128 if the GPU runs out of
  memory. Run tests with a model: `CHITRA_TEST_MODEL=x.onnx cargo test --features onnx`.

## Getting the models

`chitra-enhance models list` shows the models (an AI upscaler and the five-frame
denoiser, FP16 for GPUs and FP32 for CPUs, about 20 MB in all);
`chitra-enhance models get all` downloads them into the models folder
(`~/.local/share/chitra-enhance/models`, `%LOCALAPPDATA%\chitra-enhance\models`
on Windows). Each file is checked against a SHA-256 built into the program and
deleted if it does not match; nothing downloads unless you ask. The window has a
**Download models** button. The files are format conversions of Real-ESRGAN's
"compact" model (BSD-3-Clause) and FastDVDnet (MIT); their licences are in
`docs/MODEL_LICENCES.md`. `scripts/export_*.py` rebuild them from the original
weights if you prefer to verify them yourself. On a CPU the FP32 file is faster;
on a GPU use FP16. A 4x model can give a 2x result (`--scale 2`), usually sharper
than a native 2x model.

ONNX Runtime itself is looked for automatically (next to the program, in
`<data dir>/chitra-enhance/runtime`, and the usual system folders);
`chitra-enhance models runtime` says what it found or how to install it, and
`ORT_DYLIB_PATH` overrides the search.

## Status

Video pipeline, resume, tiler, ONNX engine, grain, 1-4x scaling and photo mode
(JPEG quality tiers, 16-bit JXL/PNG) work, tested with a real Real-ESRGAN model.
Five-frame video denoising (FastDVDnet) and CI on Linux, Windows and macOS are in.
The GUI works but has had only a smoke test (it starts and runs), no visual review yet.

## Licence

AGPL-3.0-or-later (see `LICENSE`), the same as Chitra, because the JPEG XL
writer (`jxl-encoder`) is AGPL or commercial. FFmpeg and ONNX Runtime are
separate programs/libraries that you install yourself.
