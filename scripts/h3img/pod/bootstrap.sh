#!/bin/bash
# A fresh pod for the H3-Image work: the Vulkan userspace, Rust, a venv with
# the picture tools, and the public container from Hugging Face (checked by
# its sha256). Safe to run again: every step skips what is already there.
#   bootstrap.sh            everything
#   bootstrap.sh nomodel    without the 24.6 GB download
set -u
root=${H3I_ROOT:-/root/h3i}
repo=infosave/MiniMax-H3-Image-cmf
file=minimax-h3-image-q4tp.cmf
sha=f4af395ada66213cd401a455fcc7fd7252d2d4c4813f8d1437ff58a89740f309
mkdir -p "$root"/{bin,cmf,out,logs,scripts,prompts,srcrs}
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends vulkan-tools libglvnd0 libegl1 libgl1 libglx0 \
  build-essential pkg-config python3-venv ca-certificates > /dev/null
[ -x "$HOME/.cargo/bin/cargo" ] || curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal > /dev/null
[ -x "$root/venv/bin/python" ] || python3 -m venv "$root/venv"
"$root/venv/bin/pip" -q install numpy pillow huggingface_hub hf_xet
echo "vulkan: $(XDG_RUNTIME_DIR=/tmp vulkaninfo --summary 2>&1 | grep -m1 deviceName)"
echo "cargo: $("$HOME/.cargo/bin/cargo" --version)"
echo BOOTSTRAP_TOOLS_DONE
[ "${1:-}" = nomodel ] && exit 0
if [ ! -f "$root/cmf/$file.ok" ]; then
  "$root/venv/bin/hf" download "$repo" "$file" --local-dir "$root/cmf" > /dev/null
  got=$(sha256sum "$root/cmf/$file" | cut -d' ' -f1)
  [ "$got" = "$sha" ] && touch "$root/cmf/$file.ok" || { echo "sha256 mismatch: $got"; exit 1; }
fi
echo "model: $(stat -c %s "$root/cmf/$file") bytes, sha256 ok"
echo BOOTSTRAP_DONE
