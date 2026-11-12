#!/bin/bash
# The 0.8.82 closing runs on the release build: Qwen-Image-2.1 on CUDA at
# 512 and 1024 (40 steps) and within an 8 GB ceiling, then the Uncensored
# file on CUDA and over Vulkan at 1024; clippy; the files deleted.
set -u
root=${H3I_ROOT:-/root/h3i}
p=${1:-r82}
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
sumline() { tr '\r' '\n' < out/$1.log | grep -E "steps in|declined" | sort -u | tr '\n' ' ' | cut -c1-400; echo; }
bash scripts/build.sh release > logs/${p}_build_release.log 2>&1 || { echo BUILD_RELEASE_FAILED; tail -30 logs/${p}_build_release.log; exit 1; }
export BIN=$root/bin/cortiq
for s in 512 1024; do bash scripts/qi21.sh ${p}_cuda_$s cuda $s 40 | head -1; sumline ${p}_cuda_$s; done
bash scripts/qi21.sh ${p}_cuda_1024b cuda 1024 40 | head -1
echo "cuda twice: $(psnr out/${p}_cuda_1024.png out/${p}_cuda_1024b.png)"
echo "cuda vs vulkan 1024: $(psnr out/${p}_cuda_1024.png out/q21b_vk_1024.png)"
VRAM_MB=6000 bash scripts/qi21.sh ${p}_cuda_1024_6000 cuda 1024 40 | head -1; sumline ${p}_cuda_1024_6000
rm -f cmf/qwen-image-2.1.cmf
u=cmf/qwen-image-2.1-uncensored.cmf
venv/bin/hf download infosave/Qwen-Image-2.1-Uncensored-CMF qwen-image-2.1-uncensored.cmf --local-dir cmf > logs/${p}_dl.log 2>&1
for b in cuda vulkan; do MODEL=$u bash scripts/qi21.sh ${p}_unc_$b $b 1024 40 | head -1; sumline ${p}_unc_$b; done
rm -f $u
. "$HOME/.cargo/env"
( cd srcrs && cargo clippy --locked --workspace --all-targets --features cuda 2>&1 | grep -cE "^error"; echo "clippy cuda rc=${PIPESTATUS[0]}"; cargo clippy --locked --workspace --all-targets 2>&1 | grep -cE "^error"; echo "clippy rc=${PIPESTATUS[0]}" )
echo REL82_DONE
