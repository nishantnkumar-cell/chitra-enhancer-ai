# Security

## Reporting a problem

Please report security issues privately through GitHub's *Report a vulnerability*
button on the Security tab of this repository, not in a public issue. Say what
you found, how to reproduce it and which version. You will get an answer within
a few days.

## What the program trusts (and does not)

Chitra Enhancer AI is a local tool: it reads files you point it at and writes
files you ask for. It treats these as **untrusted** and bounds them:

- **Photos and videos.** Picture size, frame rate, length and pixel shape are
  checked before anything is allocated from them (`src/ffmpeg.rs`, `src/photo.rs`);
  decoding is done by the `image` crate and by ffmpeg, which are fuzzed upstream,
  and the JPEG header reader and decoder wrapper are fuzzed here (`fuzz/`).
  Photos declaring more than 120 million pixels are refused before decoding; enlarged
  video frames are capped at 100 million pixels (photos 120 million).
- **DVD images and drives.** Raw sectors are walked with bounds checks (fuzzed)
  to refuse copy-protected discs; the check samples sectors, so it is a guard
  against ordinary protected discs, not a proof about a crafted one. The disc
  itself is parsed by ffmpeg with libdvdread. Titles are limited to 99 and their
  lengths bounded.
- **Printed commands.** File names are quoted for the shell (PowerShell on
  Windows), so pasting a suggested command cannot run anything hidden in a name.
- **ffmpeg / ffprobe.** Started as separate programs with argument lists (never a
  shell); paths are made absolute so a name starting with `-` is not an option.
- **The network service (`serve`).** Off unless started. It refuses to listen
  anywhere but this computer and Tailscale addresses and answers only such
  addresses; every request needs a key (at least 24 characters, compared in
  constant time). Uploads are limited in size and number, recognised by their
  first bytes (play lists and scripts that could make ffmpeg read other files
  are refused), stored under names the service chooses, and removed once dealt
  with. Error answers do not name folders. Anyone with the key can use the
  computer's processor and graphics card and, as with any file you open,
  exercises the picture and video decoders: share it only with your own devices.
  The Windows installer runs it as an unprivileged service account.
- **No usage reporting.** ONNX Runtime's reporting, which Microsoft's Windows
  builds switch on by default, is turned off. The program connects to the network
  only to download models and to talk to a service you name, when asked.
- **Downloaded model files.** Fetched over HTTPS only, size-limited, and checked
  against a SHA-256 built into the program; a mismatch is deleted.
- **Your own models and ONNX Runtime.** A model is a program for the inference
  engine, and ONNX Runtime (a native library, found next to the program, in the
  data folder or in system folders, or via `ORT_DYLIB_PATH`) is code that runs
  inside this process. Only add models and runtimes from sources you trust.

Existing results are never overwritten without `--force`, and the original is
never touched (a result may not be the input, even through a symlink).

## Dependencies

CI runs `cargo deny` (advisories, licences, bans, sources) on every change and
weekly, Dependabot proposes updates, and GitHub Actions are pinned to commit
hashes. Release bundles come with SHA-256 files; they are not code-signed.
