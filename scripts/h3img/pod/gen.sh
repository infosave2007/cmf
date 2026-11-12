#!/bin/bash
# One image from the packed container, with the card's memory traced.
#   gen.sh TAG BACKEND WIDTH HEIGHT [SEED]
#     BACKEND  cuda | vulkan | cpu
#   env: PROMPT (file, default prompts/lighthouse.txt), IMAGE (a reference
#        picture: the run is an edit of it), MODEL, VRAM_MB (the ceiling,
#        CMF_GPU_VRAM_MB), STEPS, MIN_AREA (--min-area: 1024 runs a size
#        below 2048 x 2048 on its own canvas), BIN, plus any CMF_* knob.
# Leaves out/TAG.{png,log,vram_trace} and one summary line in out/TAG.sum.
set -u
root=${H3I_ROOT:-/root/h3i}
tag=$1; backend=$2; w=$3; h=$4; seed=${5:-42}
model=${MODEL:-$root/cmf/minimax-h3-image-q4tp.cmf}
prompt=${PROMPT:-$root/scripts/../prompts/lighthouse.txt}
bin=${BIN:-$root/bin/cortiq-fast}
out=$root/out/$tag
mkdir -p "$root/out"
export XDG_RUNTIME_DIR=/tmp
case $backend in
  cuda) export CMF_GPU=cuda ;;
  vulkan) export CMF_GPU=1 ;;
  cpu) export CMF_GPU=0 ;;
  *) echo "backend: cuda | vulkan | cpu"; exit 2 ;;
esac
[ -n "${VRAM_MB:-}" ] && export CMF_GPU_VRAM_MB=$VRAM_MB
( while :; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 0.25; done ) > "$out.vram_trace" 2>/dev/null &
tr=$!
t0=$(date +%s.%N)
"$bin" imagine "$model" --prompt "@$prompt" ${IMAGE:+--image "$IMAGE"} --width "$w" --height "$h" --seed "$seed" ${STEPS:+--steps $STEPS} ${MIN_AREA:+--min-area $MIN_AREA} --out "$out.png" > "$out.log" 2>&1
rc=$?
t1=$(date +%s.%N)
kill $tr 2>/dev/null
peak=$(sort -n "$out.vram_trace" | tail -1)
rss=$(grep -m1 VmHWM "$out.log" 2>/dev/null | awk '{print $2}')
echo "tag=$tag backend=$backend size=${w}x${h} seed=$seed rc=$rc wall_s=$(awk "BEGIN{printf \"%.1f\", $t1-$t0}") peak_vram_mib=$peak ${VRAM_MB:+ceiling_mib=$VRAM_MB}" | tee "$out.sum"
grep -E "image\(s\)|error|Error|panicked|declined" "$out.log" | tail -5
