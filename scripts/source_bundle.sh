#!/bin/sh
# The complete source of a build, libraries included, as the licence requires
# to be offered beside the programs. Builds without a network:
#     tar xf NAME.tar.xz && cd NAME && cargo build --release --features gui,onnx --offline
# usage: scripts/source_bundle.sh NAME OUTDIR
set -eu
[ $# = 2 ] || { echo "usage: $0 NAME OUTDIR" >&2; exit 2; }
name=$1
mkdir -p "$2"
out=$(cd "$2" && pwd)
cd "$(dirname "$0")/.."
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
git archive --format=tar --prefix="$name/" HEAD | tar -xf - -C "$work"
mkdir -p "$work/$name/.cargo"
# Not --quiet: that also drops the settings it prints for using the folder.
(cd "$work/$name" && cargo vendor --locked vendor >.cargo/config.toml)
grep -q 'directory = "vendor"' "$work/$name/.cargo/config.toml" || { echo "vendor settings missing" >&2; exit 1; }
tar -C "$work" -cJf "$out/$name.tar.xz" "$name"
(cd "$out" && sha256sum "$name.tar.xz" >"$name.tar.xz.sha256")
ls -l "$out/$name.tar.xz"
