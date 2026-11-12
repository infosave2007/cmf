#!/bin/bash
# The closing sequence for Lumina-Image-2.0 on a pod: the release build, the
# card's reference picture (512 x 512, 30 steps, CFG 4) on CUDA twice and
# over Vulkan, 1024 x 1024 on both, then the unit tests and clippy.
#   lum_final.sh [TAGPREFIX]
set -u
root=${H3I_ROOT:-/root/h3i}
p=${1:-lmf}
cd "$root" || exit 1
py=$root/venv/bin/python
bash scripts/build.sh release > logs/${p}_build_release.log 2>&1 || { echo BUILD_RELEASE_FAILED; exit 1; }
export BIN=$root/bin/cortiq
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
bash scripts/lum.sh ${p}_cuda cuda | head -1
bash scripts/lum.sh ${p}_cuda2 cuda | head -1
echo "cuda twice: $(psnr out/${p}_cuda.ppm out/${p}_cuda2.ppm)"
bash scripts/lum.sh ${p}_vk vulkan | head -1
echo "cuda vs vulkan: $(psnr out/${p}_cuda.ppm out/${p}_vk.ppm)"
CMF_LUMINA_CUDA=0 bash scripts/lum.sh ${p}_cuda_perop cuda | head -1
echo "cuda block vs per-op: $(psnr out/${p}_cuda.ppm out/${p}_cuda_perop.ppm)"
bash scripts/lum.sh ${p}_cuda_1024 cuda 1024 | head -1
bash scripts/lum.sh ${p}_vk_1024 vulkan 1024 | head -1
for t in ${p}_cuda ${p}_vk ${p}_cuda_1024 ${p}_vk_1024; do
  echo "$t: $(tr '\r' '\n' < out/$t.log | grep -E '^step 1/|^step 30/' | tr '\n' ' ')"
done
unset BIN
. "$HOME/.cargo/env"
cd srcrs
cargo clippy --locked --workspace --all-targets --features cuda 2>&1 | grep -cE "^error"; echo "clippy cuda rc=${PIPESTATUS[0]}"
cargo clippy --locked --workspace --all-targets 2>&1 | grep -cE "^error"; echo "clippy rc=${PIPESTATUS[0]}"
echo LUM_FINAL_DONE
