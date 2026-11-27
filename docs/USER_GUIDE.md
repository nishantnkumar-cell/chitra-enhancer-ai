# User guide: from tape or DVD to a better video

Chitra Enhancer AI works entirely on your own computer. It never changes your
original files and never uploads anything. The only download is the AI models,
and only when you ask for them.

## 1. Install (Arch, CachyOS and similar)

```
sudo pacman -S --needed git rust ffmpeg onnxruntime-cpu
git clone https://github.com/nishantnkumar-cell/chitra-enhancer-ai
cd chitra-enhancer-ai
scripts/install-linux.sh
~/.local/bin/chitra-enhance models get all
~/.local/bin/chitra-enhance models runtime
```

The last command should say *ONNX Runtime found*. **Chitra Enhancer AI** is now in
your application menu. If your shell cannot find `chitra-enhance`, add
`~/.local/bin` to your `PATH` or type the full path.

On other systems you need ffmpeg, a current Rust and ONNX Runtime;
`chitra-enhance models runtime` says how to get the last one.

## 2. Get your footage onto the computer

| You have | What to give the program |
|---|---|
| A file captured from a camcorder or tape | the file itself, whatever its name |
| A DVD in a drive | the drive's folder (the one holding `VIDEO_TS`) |
| A copy of a DVD | the `.iso` image, or the copied folder |
| A folder of old clips and photos | the folder |

Only DVDs **without copy protection** are accepted, such as discs made by a
camcorder or burnt at home.

Keep the capture as close to the original as you can. Converting it to another
format first only loses detail.

## 3. The easy way: the window

1. Choose your video or DVD.
2. Press **Find the 3 best options**. The program restores a few seconds in three
   ways on your computer, shows them side by side, and says how long the whole
   video would take with each. Play the samples, then press **Use these
   settings** on the one you like and go to step 5. The *suggested* mark comes
   from measurements of your video; it cannot judge which looks best, so trust
   your eyes. The slider beside **Preview** sets the sample length (1 to 30 s).

Or set things up yourself:

2. Press **Analyse and suggest settings**. The program looks at a few places and
   fills in the settings, saying why.
3. Press **Preview 5 s**. You get a before and after picture and a short clip. The
   log also says how long the whole video would take.
4. Change anything you like and preview again.
5. Press **Start**. **Stop** keeps what is finished; **Continue** carries on.

The result is saved next to the original as `name-enhanced.mp4`. For a DVD drive
or disc folder it goes to your home folder.

## 4. The same from the command line

```
chitra-enhance -i tape.mp4 --analyze
```

This prints suggested settings and a ready-made command. Add `--preview 5` to that
command to try it first, then run it without.

To compare three ways of restoring it:

```
chitra-enhance -i tape.mp4 --options
```

The samples go into a `tape-options` folder (`-o` picks another), and the command
for each choice is printed. `--preview 10` makes the samples 10 seconds long.

For a DVD:

```
chitra-enhance -i wedding.iso --list-titles
chitra-enhance -i wedding.iso --analyze
chitra-enhance -i wedding.iso -o wedding.mp4            # the longest title
chitra-enhance -i wedding.iso -o wedding/ --title all   # every title, into a folder
```

## 5. What the settings do

| Setting | Command | When to change it |
|---|---|---|
| Interlaced tape | `--deinterlace auto/on/off` | Leave on *auto*. Force *on* if you still see comb-like teeth on moving edges. |
| Keep both half-pictures | `--double-rate` | Smoother movement for handheld footage. Takes about twice as long. |
| Simple denoise | `--denoise off/light/medium/strong` | Raise it only if the picture is visibly grainy. Too much looks painted. |
| Five-frame AI denoising | `--temporal-model`, `--temporal-sigma` | For heavy grain. *Measure now* sets the noise level; lower it if detail disappears. |
| AI upscaler | `--model`, `--scale` | Sharper edges when enlarging. Far slower without a graphics card. |
| Film grain | `--grain 0-30` | A little (about 3) stops cleaned video looking plastic. |
| Cut black edges | `--crop L:R:T:B` | Removes the black strips at the sides. |
| Soften edge outlines | `--dehalo 0-100` | For bright rims around dark objects. About 40; higher turns soft. |
| Line up colour | `--colour-align BLUE:RED` | For colour spilling above or below objects, common on tape. |
| Correct a colour cast | `--colour-shift BLUE:RED` | If whites look tinted. Not applied automatically: check with a preview. |
| Video quality | `--crf` | Lower is better and bigger. 18 is very good. |
| Video format | `--codec h265/h264/av1/prores` | *h264* plays on the oldest devices. |

## 6. How long it takes

It depends mostly on whether the AI runs on a graphics card. Always use the
estimate from the three options or a preview.

Measured on an NVIDIA RTX 4070 Ti under Windows (DirectML), standard-definition
interlaced video enlarged 2x, in frames per second:

| What is used | Frames per second | 30 minutes of tape |
|---|---|---|
| AI upscaler and five-frame AI denoiser, half precision (FP16) | 17 | about 45 minutes |
| The same, full precision (FP32) | 9 | about 1 hour 25 minutes |
| AI upscaler with the simple denoise filter, FP16 | 26 | about 30 minutes |

The downloaded FP16 models are chosen on Windows by themselves; their pictures
differ from FP32 by too little to see. Larger sizes are slower (1280 x 720 with
both models: about 4 frames per second). The program uses large tiles when they
fit; set **GPU memory** (`--vram-budget`) to your card's memory if it is under 8 GB.

Without a graphics card it is far slower. As one measured example, on a laptop-class processor (Ryzen 7
4700U) with **no** graphics acceleration, a 32-second standard-definition clip took:

- about 4 minutes without AI, keeping both half-pictures;
- about 3 hours with the AI upscaler.

At that speed an hour of tape with AI would take around two weeks. Without a
supported graphics card, use the settings without the AI upscaler for long tapes,
and keep AI for the parts you care most about.

Graphics acceleration is built for NVIDIA cards, Windows (DirectML) and macOS. Of
these only DirectML on an NVIDIA card has been tried on real hardware. AMD and
Intel graphics on Linux are not used yet.

## 7. Stopping, resuming and disk space

- Press **Stop** or Ctrl-C at any time. Run the same job again to carry on from
  the last finished section (30 seconds by default).
- If you changed the settings, the program refuses to mix them with the old work.
  Tick **Start over** (`--restart`) to begin again.
- An existing result is never replaced unless you tick **Replace** (`--force`).
- A DVD title is first copied to your disk, which can need several GB. The copy is
  removed when that title is finished.

## 8. Photos

Give the program a photo instead of a video. It writes a 16-bit `.jxl` or `.png`
and keeps the colour profile and camera information. Heavily compressed JPEGs can
use a second model for cleaning (`--model-compressed`).

## 9. When something goes wrong

| Message | What to do |
|---|---|
| ONNX Runtime was not found | Install it (section 1), then run `chitra-enhance models runtime`. |
| this build has no AI model support | Build with `--features onnx`; the install script does this. |
| already exists; use --force | Pick another name, or allow replacing it. |
| holds an unfinished run with other settings | Use **Start over** (`--restart`), or restore the earlier settings. |
| is copy-protected | The disc is protected and cannot be used. |
| this ffmpeg cannot read DVDs | Install ffmpeg 7 or newer built with DVD support (the Arch package has it). |
| Out of GPU memory | The program retries with smaller tiles by itself; you can also lower **GPU memory**. |

To report a problem, see `SECURITY.md` for security issues, otherwise open an issue
on GitHub. Please do not attach private footage.

## 10. Licence

Free software (AGPL-3.0-or-later) with no warranty. **About** in the window, or
`chitra-enhance licences`, shows the notice. Your pictures and videos remain
yours.

