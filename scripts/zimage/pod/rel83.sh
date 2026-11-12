#!/bin/bash
# The 0.8.83 closing runs on the release build: Z-Image-Turbo on CUDA at 512
# and 1024 (8 steps); the base Z-Image (28 steps, CFG 4) on CUDA and over
# Vulkan at 512 and 1024; clippy; the files deleted.
set -u
root=${H3I_ROOT:-/root/h3i}
p=${1:-r83}
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
for s in 512 1024; do bash scripts/zi.sh ${p}_t_cuda_$s cuda $s 8 | head -1; sumline ${p}_t_cuda_$s; done
echo "turbo cuda vs vulkan 1024: $(psnr out/${p}_t_cuda_1024.png out/zib_vk_1024.png)"
rm -f cmf/z-image-turbo.cmf
b=cmf/z-image.cmf
venv/bin/hf download infosave/Z-Image-cmf z-image.cmf --local-dir cmf > logs/${p}_dl.log 2>&1
for s in 512 1024; do
  for be in cuda vulkan; do MODEL=$b bash scripts/zi.sh ${p}_b_${be}_$s $be $s 28 | head -1; sumline ${p}_b_${be}_$s; done
  echo "base cuda vs vulkan $s: $(psnr out/${p}_b_cuda_$s.png out/${p}_b_vulkan_$s.png)"
done
rm -f $b
. "$HOME/.cargo/env"
( cd srcrs && cargo clippy --locked --workspace --all-targets --features cuda 2>&1 | grep -cE "^error"; echo "clippy cuda rc=${PIPESTATUS[0]}"; cargo clippy --locked --workspace --all-targets 2>&1 | grep -cE "^error"; echo "clippy rc=${PIPESTATUS[0]}" )
echo REL83_DONE
