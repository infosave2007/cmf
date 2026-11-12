#!/bin/bash
# The closing sequence for Qwen-Image-Edit-2509 on CUDA: the release build,
# the 1024 x 1024 edit at 8 steps twice and within an 8 GB ceiling, at the
# card's 40 steps once, then clippy; after it, the probe pass over the
# popular models starts.
#   qie_final.sh [TAGPREFIX]
set -u
root=${H3I_ROOT:-/root/h3i}
p=${1:-qf}
cd "$root" || exit 1
py=$root/venv/bin/python
psnr() { $py - "$1" "$2" <<'PY'
import sys
import numpy as np
from PIL import Image
a = np.asarray(Image.open(sys.argv[1]).convert("RGB"), dtype=float)
b = np.asarray(Image.open(sys.argv[2]).convert("RGB"), dtype=float)
mse = ((a - b) ** 2).mean()
print("identical" if mse == 0 else "psnr %.2f dB" % (10 * np.log10(255 ** 2 / mse)))
PY
}
stages() { tr '\r' '\n' < out/$1.log | grep -E "^(text encoder|reference VAE|denoiser: (0|8|40)/|decode VAE)|steps in" | tr '\n' ' '; echo; }
bash scripts/build.sh release > logs/${p}_build_release.log 2>&1 || { echo BUILD_RELEASE_FAILED; exit 1; }
export BIN=$root/bin/cortiq
bash scripts/qie.sh ${p}_cuda cuda 1024 8 | head -1; stages ${p}_cuda
bash scripts/qie.sh ${p}_cuda2 cuda 1024 8 | head -1
echo "cuda twice: $(psnr out/${p}_cuda.png out/${p}_cuda2.png)"
echo "cuda vs vulkan (0.8.79): $(psnr out/${p}_cuda.png out/qb_vk.png)"
VRAM_MB=8000 bash scripts/qie.sh ${p}_cuda_8000 cuda 1024 8 | head -1; stages ${p}_cuda_8000
echo "cuda 8000 vs none: $(psnr out/${p}_cuda_8000.png out/${p}_cuda.png)"
bash scripts/qie.sh ${p}_cuda_40 cuda 1024 40 | head -1; stages ${p}_cuda_40
unset BIN
. "$HOME/.cargo/env"
( cd srcrs && cargo clippy --locked --workspace --all-targets --features cuda 2>&1 | grep -cE "^error"; echo "clippy cuda rc=${PIPESTATUS[0]}"; cargo clippy --locked --workspace --all-targets 2>&1 | grep -cE "^error"; echo "clippy rc=${PIPESTATUS[0]}" )
echo QIE_FINAL_DONE
rm -f cmf/qwen-image-edit-2509-q4tp.cmf
bash scripts/probe.sh > logs/probe_run.log 2>&1
