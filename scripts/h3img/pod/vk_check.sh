#!/bin/bash
# Vulkan against CUDA on the same card: text to image at 1024 x 1024 and
# 2048 x 2048, and an edit of the 1024 x 1024 picture, one line each with
# the phases and the PSNR between the two backends' pictures.
#   vk_check.sh [TAGPREFIX]    needs out/base_cuda_{1024,2048}.png (baseline.sh)
set -u
root=${H3I_ROOT:-/root/h3i}
p=${1:-vkc}
cd "$root" || exit 1
line() { # tag
  echo "$1 $(grep -E "image\(s\), " out/$1.log | sed -E "s/.*in ([0-9.]+)s \(text ([0-9.]+)s.*denoise ([0-9.]+)s, median step ([0-9.]+)s.*vae ([0-9.]+)s.*/total \1 text \2 denoise \3 step \4 vae \5/") peak $(sort -n out/$1.vram_trace | tail -1) MiB"
  grep -E "declined|panicked|rror" out/$1.log | grep -v "tile batches" | head -3
}
MIN_AREA=1024 bash scripts/gen.sh ${p}_t2i_1024 vulkan 1024 1024 > /dev/null; line ${p}_t2i_1024
"$root/venv/bin/python" scripts/cmp_png.py out/base_cuda_1024.png out/${p}_t2i_1024.png
bash scripts/gen.sh ${p}_t2i_2048 vulkan 2048 2048 > /dev/null; line ${p}_t2i_2048
"$root/venv/bin/python" scripts/cmp_png.py out/base_cuda_2048.png out/${p}_t2i_2048.png
export IMAGE=$root/out/base_cuda_1024.png PROMPT=$root/prompts/edit_min_cafe.txt
[ -f out/base_edit_cuda.png ] || { MIN_AREA=1024 bash scripts/gen.sh base_edit_cuda cuda 1024 1024 > /dev/null; line base_edit_cuda; }
MIN_AREA=1024 bash scripts/gen.sh ${p}_edit_1024 vulkan 1024 1024 > /dev/null; line ${p}_edit_1024
"$root/venv/bin/python" scripts/cmp_png.py out/base_edit_cuda.png out/${p}_edit_1024.png
echo VK_CHECK_DONE
