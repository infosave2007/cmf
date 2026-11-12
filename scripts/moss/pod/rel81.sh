#!/bin/bash
# The 0.8.81 closing runs on the release build: MOSS-Image over Vulkan with
# the Flux2 decoder on the VAE chain (its widths 384/192/96), then
# Lumina-Image 2.0 on CUDA and over Vulkan with the decode chain against the
# per-convolution decoder. Deletes the model files at the end.
#   rel81.sh [TAGPREFIX]
set -u
root=${H3I_ROOT:-/root/h3i}
p=${1:-r81}
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
sumline() { tr '\r' '\n' < out/$1.log | grep -E "MOSS image|stages:|declined|stood down" | sort -u | tr '\n' ' ' | cut -c1-500; echo; }
bash scripts/build.sh release > logs/${p}_build_release.log 2>&1 || { echo BUILD_RELEASE_FAILED; tail -30 logs/${p}_build_release.log; exit 1; }
export BIN=$root/bin/cortiq
m=cmf/moss-keep.cmf
for s in 512 1024; do
  MODEL=$m bash scripts/moss.sh ${p}_moss_vk_$s vulkan $s 8 | head -1; sumline ${p}_moss_vk_$s
done
echo "moss vulkan 512, vae chain vs per-op decoder: $(psnr out/${p}_moss_vk_512.png out/mz_vulkan_512.png)"
echo "moss vulkan 1024, vae chain vs per-op decoder: $(psnr out/${p}_moss_vk_1024.png out/mz_vulkan_1024.png)"
rm -f $m
l=cmf/lumina-q4tp.cmf
[ -f "$l" ] || venv/bin/hf download infosave/Lumina-Image-2.0cmf lumina-q4tp.cmf --local-dir cmf > logs/${p}_dl.log 2>&1
for b in cuda vulkan; do
  MODEL=$l bash scripts/lum.sh ${p}_lum_$b $b 512 30 | head -1; sumline ${p}_lum_$b
done
CMF_FLUX_VAE_CUDA=0 MODEL=$l bash scripts/lum.sh ${p}_lum_cuda_vaeop cuda 512 30 | head -1; sumline ${p}_lum_cuda_vaeop
CMF_ZI_VAE=0 MODEL=$l bash scripts/lum.sh ${p}_lum_vk_vaeop vulkan 512 30 | head -1; sumline ${p}_lum_vk_vaeop
echo "lumina cuda, vae chain vs per-op: $(psnr out/${p}_lum_cuda.ppm out/${p}_lum_cuda_vaeop.ppm)"
echo "lumina vulkan, vae chain vs per-op: $(psnr out/${p}_lum_vulkan.ppm out/${p}_lum_vk_vaeop.ppm)"
for b in cuda vulkan; do
  MODEL=$l bash scripts/lum.sh ${p}_lum_${b}_1024 $b 1024 30 | head -1; sumline ${p}_lum_${b}_1024
done
rm -f $l
echo REL81_DONE
