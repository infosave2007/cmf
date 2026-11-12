#!/bin/bash
# The release build, then the reference clip (prompt from the cache) on both
# backends without a ceiling and within 8 and 6 GB, the card's memory traced,
# and the larger clip and the two-stage render on both backends.
#   ltx_ceil.sh [TAGPREFIX]
set -u
root=${H3I_ROOT:-/root/h3i}
p=${1:-lc3}
cd "$root" || exit 1
py=$root/venv/bin/python
bash scripts/build.sh release > logs/${p}_build_release.log 2>&1 || { echo BUILD_RELEASE_FAILED; exit 1; }
export BIN=$root/bin/cortiq
for be in cuda vulkan; do
  bash scripts/ltx.sh ${p}_${be} $be
  for mb in 8000 6000; do
    VRAM_MB=$mb bash scripts/ltx.sh ${p}_${be}_$mb $be
    echo "$be $mb vs none: $($py scripts/cmp_y4m.py out/${p}_${be}_$mb.y4m out/${p}_${be}.y4m)"
  done
  bash scripts/ltx.sh ${p}_${be}_wide $be 768 512 49
  EXTRA=--two-stage bash scripts/ltx.sh ${p}_${be}_hq $be 768 512 49
done
echo "vulkan vs cuda: $($py scripts/cmp_y4m.py out/${p}_vulkan.y4m out/${p}_cuda.y4m)"
echo "cuda vs before: $($py scripts/cmp_y4m.py out/${p}_cuda.y4m out/lf2_cuda.y4m)"
echo LTX_CEIL_DONE
