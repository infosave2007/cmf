#!/bin/bash
# Qwen-Image-2.1 on its CUDA arm (fast build): 512 and 1024 at 40 steps,
# 1024 within an 8 GB ceiling; each picture against the Vulkan one.
#   qi21_cuda.sh [TAGPREFIX]
set -u
root=${H3I_ROOT:-/root/h3i}
p=${1:-q21c}
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
bash scripts/build.sh fast > logs/${p}_build.log 2>&1 || { echo BUILD_FAILED; tail -40 logs/${p}_build.log; exit 1; }
CMF_CUDA_DEBUG=1 bash scripts/qi21.sh ${p}_512 cuda 512 40
echo "cuda vs vulkan 512: $(psnr out/${p}_512.png out/q21b_vk_512.png)"
bash scripts/qi21.sh ${p}_1024 cuda 1024 40
echo "cuda vs vulkan 1024: $(psnr out/${p}_1024.png out/q21b_vk_1024.png)"
VRAM_MB=8000 bash scripts/qi21.sh ${p}_1024_8000 cuda 1024 40
echo "8000 vs none: $(psnr out/${p}_1024_8000.png out/${p}_1024.png)"
echo Q21_CUDA_DONE
