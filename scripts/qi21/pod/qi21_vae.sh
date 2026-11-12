#!/bin/bash
# Qwen-Image-2.1 on CUDA with the VAE decode chain (fast build): 512 and 1024
# at 40 steps; 512 with the per-convolution decoder for the comparison.
set -u
root=${H3I_ROOT:-/root/h3i}
p=${1:-q21v}
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
bash scripts/qi21.sh ${p}_512 cuda 512 40 | head -1; tr '\r' '\n' < out/${p}_512.log | grep -E "steps in|declined" | sort -u
echo "vae chain vs per-conv (0.8.81 build): $(psnr out/${p}_512.png out/q21c_512.png)"
bash scripts/qi21.sh ${p}_1024 cuda 1024 40 | head -1; tr '\r' '\n' < out/${p}_1024.log | grep -E "steps in|declined" | sort -u
echo "1024 vae chain vs per-conv: $(psnr out/${p}_1024.png out/q21c_1024.png)"
echo Q21_VAE_DONE
