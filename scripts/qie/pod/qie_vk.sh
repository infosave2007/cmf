#!/bin/bash
# Qwen-Image-Edit-2509 over Vulkan on the H3 parts' chain: the fast build, the
# 1024 x 1024 edit at 8 steps against the 0.8.79 Vulkan picture and the CUDA
# one, then the original chain (CMF_QWEN_WGPU_H3=0) for the same picture.
#   qie_vk.sh [TAGPREFIX]
set -u
root=${H3I_ROOT:-/root/h3i}
p=${1:-qv}
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
stages() { tr '\r' '\n' < out/$1.log | grep -E "^(text encoder|reference VAE|denoiser: (0|8|40)/|decode VAE)|steps in|declined|stood down" | sort -u | tr '\n' ' ' | cut -c1-900; echo; }
bash scripts/build.sh fast > logs/${p}_build.log 2>&1 || { echo BUILD_FAILED; tail -30 logs/${p}_build.log; exit 1; }
bash scripts/qie.sh ${p}_vk vulkan 1024 8 | head -1; stages ${p}_vk
echo "new vulkan vs 0.8.79 vulkan: $(psnr out/${p}_vk.png out/qb_vk.png)"
echo "new vulkan vs cuda: $(psnr out/${p}_vk.png out/qf_cuda.png)"
${QV_FULL:-false} && { VRAM_MB=8000 bash scripts/qie.sh ${p}_vk_8000 vulkan 1024 8 | head -1; stages ${p}_vk_8000; echo "8000 vs none: $(psnr out/${p}_vk_8000.png out/${p}_vk.png)"; }
echo QIE_VK_DONE
