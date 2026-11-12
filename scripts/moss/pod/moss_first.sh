#!/bin/bash
# The first MOSS-Image pass: the Qwen-Image-Edit Vulkan check (qie_vk.sh,
# which also builds the fast binary), then MOSS-Image-SFT q4tp at 512 x 512,
# 8 steps, per operation and as one chain, on CUDA and over Vulkan.
#   moss_first.sh [TAGPREFIX]
set -u
root=${H3I_ROOT:-/root/h3i}
p=${1:-mf}
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
bash scripts/qie_vk.sh qv > logs/qv.log 2>&1
cat logs/qv.log
f=cmf/moss-image-sft-q4tp.cmf
[ -f "$f" ] || venv/bin/hf download infosave/MOSS-Image-SFT-cmf moss-image-sft-q4tp.cmf --local-dir cmf > logs/${p}_dl.log 2>&1
ls -la "$f" || { echo MOSS_DOWNLOAD_FAILED; exit 1; }
CMF_MOSS_CHAIN=0 bash scripts/moss.sh ${p}_cuda_base cuda 512 8 | head -1
bash scripts/moss.sh ${p}_cuda cuda 512 8 | head -3
echo "cuda chain vs per-op: $(psnr out/${p}_cuda.png out/${p}_cuda_base.png)"
bash scripts/moss.sh ${p}_vk vulkan 512 8 | head -3
echo "vulkan chain vs cuda chain: $(psnr out/${p}_vk.png out/${p}_cuda.png)"
CMF_MOSS_CHAIN=0 bash scripts/moss.sh ${p}_vk_base vulkan 512 8 | head -1
echo "vulkan chain vs vulkan per-op: $(psnr out/${p}_vk.png out/${p}_vk_base.png)"
for t in ${p}_cuda_base ${p}_cuda ${p}_vk ${p}_vk_base; do
  echo "$t: $(tr '\r' '\n' < out/$t.log | grep -iE "declined|stood down|median|steps in" | sort -u | tr '\n' ' ' | cut -c1-600)"
done
echo MOSS_FIRST_DONE
