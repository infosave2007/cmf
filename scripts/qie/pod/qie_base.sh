#!/bin/bash
# Qwen-Image-Edit-2509 before and after the CUDA chain: an edit of a
# 1024 x 1024 picture at STEPS steps (CFG 4) over Vulkan and on CUDA with the
# 0.8.79 binary, then on CUDA with the chain; pictures compared.
#   qie_base.sh [TAGPREFIX] [STEPS]
set -u
root=${H3I_ROOT:-/root/h3i}
p=${1:-qb}; steps=${2:-8}
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
BIN=$root/bin/cortiq bash scripts/qie.sh ${p}_vk vulkan 1024 $steps
BIN=$root/bin/cortiq bash scripts/qie.sh ${p}_cuda_old cuda 1024 $steps
CMF_CUDA_DEBUG=1 bash scripts/qie.sh ${p}_cuda cuda 1024 $steps
grep -i "declin\|qwen image chain" out/${p}_cuda.log | head -4 | cut -c1-200
echo "cuda chain vs vulkan:  $(psnr out/${p}_cuda.png out/${p}_vk.png)"
echo "cuda chain vs cuda old: $(psnr out/${p}_cuda.png out/${p}_cuda_old.png)"
echo QIE_BASE_DONE
