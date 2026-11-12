#!/bin/bash
# A clip from a first frame (--first-frame): the VAE encoder of the keyframe
# and the vision tower of the prompt on both backends. The keyframe is frame
# 0 of the reference clip REF rendered before.
#   turbo_first.sh [TAGPREFIX] [REF]
set -u
root=${H3I_ROOT:-/root/h3i}
p=${1:-tff}; ref=${2:-tfin_cuda}
cd "$root" || exit 1
export BIN=${BIN:-$root/bin/cortiq}
first=$root/out/$ref.frames/frame_0000.ppm
[ -f "$first" ] || { echo "no keyframe $first"; exit 1; }
FIRST=$first bash scripts/anim.sh ${p}_cuda cuda
FIRST=$first bash scripts/anim.sh ${p}_vk vulkan
rm -rf out/${p}_*.frames
echo TURBO_FIRST_DONE
