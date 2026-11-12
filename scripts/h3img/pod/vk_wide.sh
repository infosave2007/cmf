#!/bin/bash
# The larger runs on both backends of one card: 4096 x 4096 text to image
# and an edit of a 1024 x 1024 picture at the default 2048 x 2048 canvas.
#   vk_wide.sh [TAGPREFIX]    needs out/base_cuda_1024.png (baseline.sh)
set -u
root=${H3I_ROOT:-/root/h3i}
p=${1:-wide}
cd "$root" || exit 1
line() {
  echo "$1 $(grep -E "image\(s\), " out/$1.log | sed -E "s/.*in ([0-9.]+)s \(text ([0-9.]+)s.*denoise ([0-9.]+)s, median step ([0-9.]+)s.*vae ([0-9.]+)s.*/total \1 text \2 denoise \3 step \4 vae \5/") peak $(sort -n out/$1.vram_trace | tail -1) MiB"
  grep -E "references:|declined|panicked" out/$1.log | grep -v "tile batches" | cut -c1-160 | head -3
}
for b in cuda vulkan; do
  bash scripts/gen.sh ${p}_${b}_4096 $b 4096 4096 > /dev/null; line ${p}_${b}_4096
done
"$root/venv/bin/python" scripts/cmp_png.py out/${p}_cuda_4096.png out/${p}_vulkan_4096.png
export IMAGE=$root/out/base_cuda_1024.png PROMPT=$root/prompts/edit_min_cafe.txt
for b in cuda vulkan; do
  bash scripts/gen.sh ${p}_${b}_edit2048 $b 2048 2048 > /dev/null; line ${p}_${b}_edit2048
done
"$root/venv/bin/python" scripts/cmp_png.py out/${p}_cuda_edit2048.png out/${p}_vulkan_edit2048.png
echo VK_WIDE_DONE
