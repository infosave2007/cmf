#!/bin/bash
# Vulkan under a ceiling on the card's memory (CMF_GPU_VRAM_MB): 2048 x 2048
# text to image without one and under each ceiling given, one line each
# with the phases, the card's peak and the PSNR against the run without.
#   vk_ceil.sh [TAGPREFIX] [CEILING_MB ...]     default ceilings: 12000 8000 6000
set -u
root=${H3I_ROOT:-/root/h3i}
p=${1:-vkceil}; shift || true
[ $# -eq 0 ] && set -- 12000 8000 6000
cd "$root" || exit 1
line() {
  echo "$1 $(grep -E "image\(s\), " out/$1.log | sed -E "s/.*in ([0-9.]+)s \(text ([0-9.]+)s.*denoise ([0-9.]+)s, median step ([0-9.]+)s.*vae ([0-9.]+)s.*/total \1 text \2 denoise \3 step \4 vae \5/") peak $(sort -n out/$1.vram_trace | tail -1) MiB"
  grep -E "declined|panicked|kept on the card" out/$1.log | grep -v "tile batches" | cut -c1-200 | head -4
}
CMF_H3_WGPU_PROF=${PROF:-0} bash scripts/gen.sh ${p}_none vulkan 2048 2048 > /dev/null; line ${p}_none
for mb in "$@"; do
  VRAM_MB=$mb CMF_H3_WGPU_PROF=${PROF:-0} bash scripts/gen.sh ${p}_$mb vulkan 2048 2048 > /dev/null; line ${p}_$mb
  "$root/venv/bin/python" scripts/cmp_png.py out/${p}_none.png out/${p}_$mb.png
done
echo VK_CEIL_DONE
