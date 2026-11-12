#!/bin/bash
# Z-Image-Turbo on its CUDA arm (fast build): 512 and 1024 at 8 steps, each
# against the Vulkan picture; 1024 twice.
set -u
root=${H3I_ROOT:-/root/h3i}
p=${1:-zic}
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
for s in 512 1024; do
  CMF_CUDA_DEBUG=1 bash scripts/zi.sh ${p}_$s cuda $s 8
  echo "cuda vs vulkan $s: $(psnr out/${p}_$s.png out/zib_vk_$s.png)"
done
bash scripts/zi.sh ${p}_1024b cuda 1024 8 | head -1
echo "cuda twice: $(psnr out/${p}_1024.png out/${p}_1024b.png)"
echo ZI_CUDA_DONE
