#!/bin/bash
# The closing MOSS-Image pass on the release build: first the Qwen-Image-Edit
# Vulkan edit again on this binary (cmf/qie-keep.cmf), then MOSS-Image-SFT
# q4tp and q8_2f on CUDA and over Vulkan — 512 x 512 and 1024 x 1024 at
# 8 steps, an 8 GB ceiling, the card's 50-step recipe — and the Flux VAE
# chain against the per-operation decoder.
#   moss_final.sh [TAGPREFIX]
set -u
root=${H3I_ROOT:-/root/h3i}
p=${1:-mz}
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
sumline() { tr '\r' '\n' < out/$1.log | grep -E "MOSS image|declined|stood down" | sort -u | tr '\n' ' ' | cut -c1-500; echo; }
bash scripts/build.sh release > logs/${p}_build_release.log 2>&1 || { echo BUILD_RELEASE_FAILED; tail -30 logs/${p}_build_release.log; exit 1; }
export BIN=$root/bin/cortiq
MODEL=cmf/qie-keep.cmf bash scripts/qie.sh ${p}_qie_vk vulkan 1024 8 | head -1
echo "qie vulkan release vs fast: $(psnr out/${p}_qie_vk.png out/qv_vk.png)"
rm -f cmf/qie-keep.cmf
q4=cmf/moss-image-sft-q4tp.cmf
for b in cuda vulkan; do
  MODEL=$q4 bash scripts/moss.sh ${p}_${b}_512 $b 512 8 | head -1; sumline ${p}_${b}_512
done
echo "512 cuda vs vulkan: $(psnr out/${p}_cuda_512.png out/${p}_vulkan_512.png)"
echo "512 cuda release (vae chain) vs fast (vae per-op): $(psnr out/${p}_cuda_512.png out/mf_cuda.png)"
CMF_FLUX_VAE_CUDA=0 MODEL=$q4 bash scripts/moss.sh ${p}_cuda_512_vaeop cuda 512 8 | head -1; sumline ${p}_cuda_512_vaeop
echo "vae chain vs per-op: $(psnr out/${p}_cuda_512.png out/${p}_cuda_512_vaeop.png)"
for b in cuda vulkan; do
  MODEL=$q4 bash scripts/moss.sh ${p}_${b}_1024 $b 1024 8 | head -1; sumline ${p}_${b}_1024
done
VRAM_MB=8000 MODEL=$q4 bash scripts/moss.sh ${p}_cuda_1024_8000 cuda 1024 8 | head -1; sumline ${p}_cuda_1024_8000
echo "8000 vs none: $(psnr out/${p}_cuda_1024_8000.png out/${p}_cuda_1024.png)"
MODEL=$q4 bash scripts/moss.sh ${p}_cuda_1024_50 cuda 1024 50 | head -1; sumline ${p}_cuda_1024_50
q8=cmf/moss-image-sft-q8_2f.cmf
[ -f "$q8" ] || venv/bin/hf download infosave/MOSS-Image-SFT-cmf moss-image-sft-q8_2f.cmf --local-dir cmf > logs/${p}_dl8.log 2>&1
for b in cuda vulkan; do
  MODEL=$q8 bash scripts/moss.sh ${p}_${b}_512_q8 $b 512 8 | head -1; sumline ${p}_${b}_512_q8
done
echo "q8 cuda vs q4 cuda: $(psnr out/${p}_cuda_512_q8.png out/${p}_cuda_512.png)"
MODEL=$q8 bash scripts/moss.sh ${p}_cuda_1024_q8 cuda 1024 8 | head -1; sumline ${p}_cuda_1024_q8
echo MOSS_FINAL_DONE
