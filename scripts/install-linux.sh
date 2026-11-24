#!/bin/sh
# Builds Chitra Enhancer AI and installs it for the current user (no root):
# programs in ~/.local/bin and a menu entry.
#   PREFIX=/some/dir   install elsewhere     SKIP_BUILD=1   use an existing build
set -eu
PREFIX="${PREFIX:-$HOME/.local}"
BIN="$PREFIX/bin"
APPS="$PREFIX/share/applications"
ENTRY="$APPS/chitra-enhancer-ai.desktop"
cd "$(dirname "$0")/.."

[ -n "${SKIP_BUILD:-}" ] || cargo build --release --features gui,onnx
mkdir -p "$BIN" "$APPS"
install -m 755 target/release/chitra-enhance target/release/chitra-enhance-gui "$BIN/"
cat > "$ENTRY" <<DESKTOP
[Desktop Entry]
Type=Application
Name=Chitra Enhancer AI
Comment=Restore and enlarge old video, DVDs and photos
Exec="$BIN/chitra-enhance-gui" %f
Icon=video-x-generic
Terminal=false
Categories=AudioVideo;Video;
MimeType=video/mp4;video/x-msvideo;video/x-matroska;video/mpeg;video/dv;video/quicktime;application/x-cd-image;image/jpeg;image/png;
DESKTOP
echo "Installed to $BIN. Next:"
echo "  $BIN/chitra-enhance models get all     (the AI models)"
echo "  $BIN/chitra-enhance models runtime     (checks for ONNX Runtime)"
echo "ffmpeg must be installed. Start it from the menu: Chitra Enhancer AI."
echo "To uninstall, delete the two programs in $BIN and $ENTRY"
