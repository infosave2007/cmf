#!/bin/bash
# One clip from a MiniMax-H3-Turbo container, with the card's memory traced
# and the stage times of the run.
#   anim.sh TAG BACKEND [WIDTH HEIGHT FRAMES STEPS]
#     BACKEND  cuda | vulkan | cpu
#   env: MODEL (default cmf/mmh3-turbo-q4tp.cmf), PROMPT (text), VRAM_MB
#        (the ceiling, CMF_GPU_VRAM_MB), FIRST (a P6 PPM: --first-frame),
#        BIN, EXTRA (more flags), plus any CMF_* knob.
# Leaves out/TAG.{avi,wav,log,vram_trace}, out/TAG.frames/ and one summary
# line in out/TAG.sum.
set -u
root=${H3I_ROOT:-/root/h3i}
tag=$1; backend=$2; w=${3:-512}; h=${4:-288}; frames=${5:-39}; steps=${6:-4}
model=${MODEL:-$root/cmf/mmh3-turbo-q4tp.cmf}
prompt=${PROMPT:-A corgi in a chef hat flipping a pancake, sizzling sounds and a cheerful bark.}
bin=${BIN:-$root/bin/cortiq-fast}
out=$root/out/$tag
mkdir -p "$root/out" "$out.frames"
export XDG_RUNTIME_DIR=/tmp RUST_LOG=${RUST_LOG:-info}
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
"$bin" animate "$model" --prompt "$prompt" --width "$w" --height "$h" --frames "$frames" --steps "$steps" --seed 42 \
  ${FIRST:+--first-frame "$FIRST"} ${EXTRA:-} --frames-dir "$out.frames" --out "$out.avi" > "$out.log" 2>&1
rc=$?
t1=$(date +%s.%N)
kill $tr 2>/dev/null
peak=$(sort -n "$out.vram_trace" | tail -1)
echo "tag=$tag backend=$backend size=${w}x${h}x${frames} steps=$steps rc=$rc wall_s=$(awk "BEGIN{printf \"%.1f\", $t1-$t0}") peak_vram_mib=$peak ${VRAM_MB:+ceiling_mib=$VRAM_MB}" | tee "$out.sum"
grep -E "text encode|denoise|video vae|audio vae|upscale|total|error|Error|panicked|declined" "$out.log" | tail -14
