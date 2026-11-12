#!/bin/bash
# The measurement suite of the MiniMax-H3-Image page: text-to-image at three
# sizes, the memory ceilings, an edit, and Vulkan beside CUDA. Three runs of
# everything short, one of the 16-megapixel ones. Tags are
# PREFIX_<backend>_<what>_<size>_c<ceiling or 0>_r<run>; table.py reads them.
#   suite.sh PREFIX [cuda|vulkan|all]
set -u
root=${H3I_ROOT:-/root/h3i}
pre=${1:-s}
what=${2:-all}
cd "$root" || exit 1
run() { # backend kind size cap runs
  local b=$1 kind=$2 size=$3 cap=$4 n=$5
  # a size below the working range is measured on its own canvas
  [ "$size" -lt 2048 ] && export MIN_AREA=1024 || unset MIN_AREA
  for r in $(seq 1 "$n"); do
    local tag=${pre}_${b}_${kind}_${size}_c${cap}_r${r}
    if [ "$kind" = edit ]; then
      VRAM_MB=$([ "$cap" != 0 ] && echo "$cap") PROMPT=prompts/edit_cafe.txt IMAGE=$root/out/cafe2048.png \
        bash scripts/gen.sh "$tag" "$b" "$size" "$size" | head -1
    else
      VRAM_MB=$([ "$cap" != 0 ] && echo "$cap") bash scripts/gen.sh "$tag" "$b" "$size" "$size" | head -1
    fi
  done
}
if [ "$what" = cuda ] || [ "$what" = all ]; then
  run cuda t2i 1024 0 3
  run cuda t2i 2048 0 3
  for cap in 8000 6000 4000; do run cuda t2i 2048 $cap 3; done
  run cuda edit 2048 0 2
  run cuda edit 2048 6000 1
  run cuda t2i 4096 0 1
  run cuda t2i 4096 6000 1
fi
if [ "$what" = vulkan ] || [ "$what" = all ]; then
  run vulkan t2i 1024 0 3
  run vulkan t2i 2048 0 1
fi
echo SUITE_DONE
