# Release notes

## 0.1.0: first release

Restores old camcorder, tape and DVD footage, and photos, on your own computer.
Nothing is uploaded.

- Deinterlacing, noise removal (including a five-frame AI denoiser), an AI
  upscaler, tape repairs (colour line-up, edge outlines), cropping.
- DVD images, drives and folders (discs without copy protection only).
- **Three options to choose from:** your own footage restored three ways, side by
  side, with the time each would take.
- Suggested settings from samples of your footage; previews with a time estimate.
- Stop and carry on later; whole folders; photos to 16-bit JPEG XL or PNG.
- A window and a command line.

### Before you start

- Install **FFmpeg** and **ONNX Runtime** (1.22 or newer) yourself; neither is in
  the download.
- Fetch the AI models with `chitra-enhance models get all`.
- See `USER_GUIDE.md`.

### What has been tested

- On Linux by hand, on the processor only.
- On Windows 11 from the command line with an NVIDIA RTX 4070 Ti, standard
  definition, both AI models: about 23 frames per second with DirectML, about 28
  with NVIDIA's TensorRT RTX plugin (an optional extra; see the guide).
- **Not tried on real hardware:** the Windows window, macOS, CUDA, and AMD or Intel
  graphics.
- Without a graphics card the AI upscaler is very slow: about 3 hours for a
  32-second standard-definition clip on a laptop-class processor.

### Licence

AGPL-3.0-or-later, with no warranty. Notices for the libraries and fonts built
in are in `THIRD_PARTY_LICENCES.txt`; the complete source, libraries included,
is the `source` file in the same release.
