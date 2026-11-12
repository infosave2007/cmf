#!/bin/bash
# The card's baseline before a change: text to image on CUDA and on Vulkan at
# 1024 x 1024 and 2048 x 2048, one line each with the phases. The first CUDA
# run is a warm-up (the file's pages, the kernels' compile) and is repeated.
#   baseline.sh [TAGPREFIX]     env BIN as gen.sh, SKIP="cuda" or "vulkan"
set -u
root=${H3I_ROOT:-/root/h3i}
p=${1:-base}
cd "$root" || exit 1
one() { # tag backend size [env...]
  local tag=$1 backend=$2 size=$3; shift 3
  env "$@" bash scripts/gen.sh "$tag" "$backend" "$size" "$size" > /dev/null
  echo "$tag $(grep -E "image\(s\), " out/$tag.log | sed -E "s/.*in ([0-9.]+)s \(text ([0-9.]+)s.*denoise ([0-9.]+)s, median step ([0-9.]+)s.*vae ([0-9.]+)s.*/total \1 text \2 denoise \3 step \4 vae \5/") peak $(sort -n out/$tag.vram_trace | tail -1) MiB"
}
case " ${SKIP:-} " in *" cuda "*) ;; *)
  one ${p}_cuda_warm cuda 1024 MIN_AREA=1024
  one ${p}_cuda_1024 cuda 1024 MIN_AREA=1024
  one ${p}_cuda_2048 cuda 2048
esac
case " ${SKIP:-} " in *" vulkan "*) ;; *)
  one ${p}_vk_1024 vulkan 1024 MIN_AREA=1024
  one ${p}_vk_2048 vulkan 2048
  [ -f out/${p}_cuda_1024.png ] && "$root/venv/bin/python" scripts/cmp_png.py out/${p}_cuda_1024.png out/${p}_vk_1024.png
esac
echo BASELINE_DONE
