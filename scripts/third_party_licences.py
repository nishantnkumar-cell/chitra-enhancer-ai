#!/usr/bin/env python3
"""Write THIRD_PARTY_LICENCES.txt: the notices that go with every download.

    cargo install cargo-about --version 0.9.3 --locked --features cli
    python3 scripts/third_party_licences.py THIRD_PARTY_LICENCES.txt

cargo-about collects the licences of the Rust libraries. This adds what it does
not: the notices of the fonts built into the window, of the libjxl code the
JPEG XL writer derives from, of the Rust standard library, and any NOTICE
files. It fails rather than leave one out.
"""

import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
FEATURES = "gui,onnx"
# Font name -> the notice file epaint_default_fonts ships for it.
FONTS = {
    "Hack": "Hack-Regular.txt",
    "emoji-icon-font": "emoji-icon-font-mit-license.txt",
    "egui icons": "egui-icons.txt",
    "Noto Emoji": "OFL.txt",
    "Ubuntu": "UFL.txt",
}
BSD_3_LIBJXL = """\
Copyright (c) the JPEG XL Project Authors.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""


def run(*args):
    # Only stdout is captured, so a failing command's own message is shown.
    done = subprocess.run(args, cwd=ROOT, stdout=subprocess.PIPE, text=True)
    if done.returncode != 0:
        sys.exit(f"failed ({done.returncode}): {' '.join(args)}")
    return done.stdout


def section(title, text):
    rule = "-" * 80
    return f"\n{rule}\n{title}\n{rule}\n\n{text.rstrip()}\n"


def main(out):
    out = pathlib.Path(out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    run("cargo", "about", "generate", "--features", FEATURES, "--fail", "--locked",
        "-o", str(out), "packaging/third-party.hbs")
    meta = json.loads(run("cargo", "metadata", "--format-version", "1", "--locked",
                          "--features", FEATURES))
    packages = meta["packages"]
    folder = {p["name"]: pathlib.Path(p["manifest_path"]).parent for p in packages}
    used = {n["id"] for n in meta["resolve"]["nodes"]}
    extra = []

    fonts = folder["epaint_default_fonts"] / "fonts"
    for name, file in FONTS.items():
        extra.append(section(f"Font built into the window: {name}",
                             (fonts / file).read_text(encoding="utf-8")))

    if "jxl-encoder" not in folder:
        sys.exit("jxl-encoder is gone: revisit the libjxl notice")
    extra.append(section(
        "libjxl (BSD-3-Clause): parts of jxl-encoder derive from it\n"
        "https://github.com/libjxl/libjxl", BSD_3_LIBJXL))

    extra.append(section(
        "Rust standard library (MIT or Apache-2.0)\nhttps://github.com/rust-lang/rust",
        "Copyright (c) The Rust Project Contributors. Used under the MIT licence, "
        "whose text appears above."))

    for p in sorted(packages, key=lambda p: p["name"]):
        if p["id"] not in used or not p.get("source"):
            continue
        for notice in sorted(pathlib.Path(p["manifest_path"]).parent.glob("NOTICE*")):
            extra.append(section(f"NOTICE of {p['name']} {p['version']}",
                                 notice.read_text(encoding="utf-8", errors="replace")))

    with out.open("a", encoding="utf-8", newline="\n") as f:
        f.write("".join(extra))
    print(f"wrote {out} ({out.stat().st_size} bytes, {len(extra)} added notices)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
