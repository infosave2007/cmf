#!/bin/bash
# One LTX-2.5 clip from the packed container, with the card's memory traced
# and the stage lines of the run.
#   ltx.sh TAG BACKEND [WIDTH HEIGHT FRAMES]
#     BACKEND  cuda | vulkan | cpu
#   env: MODEL (default cmf/ltx25-q4tp.cmf), PROMPT (text), VRAM_MB (the
#        ceiling, CMF_GPU_VRAM_MB), BIN, EXTRA (more flags), any CMF_* knob.
# Leaves out/TAG.{y4m,wav,log,vram_trace} and one summary line in out/TAG.sum.
# EXTRA=--no-context-cache makes the run encode the prompt again.
set -u
root=${H3I_ROOT:-/root/h3i}
tag=$1; backend=$2; w=${3:-384}; h=${4:-256}; frames=${5:-49}
model=${MODEL:-$root/cmf/ltx25-q4tp.cmf}
prompt=${PROMPT:-A corgi in a chef hat flips a pancake in a sunlit kitchen. Warm morning light, static camera.}
bin=${BIN:-$root/bin/cortiq-fast}
out=$root/out/$tag
mkdir -p "$root/out"
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
"$bin" ltx-video --model "$model" --prompt "$prompt" --height "$h" --width "$w" --frames "$frames" --fps 24 --seed 42 \
  --out-audio "$out.wav" ${EXTRA:-} --out "$out.y4m" > "$out.log" 2>&1
rc=$?
t1=$(date +%s.%N)
kill $tr 2>/dev/null
peak=$(sort -n "$out.vram_trace" | tail -1)
echo "tag=$tag backend=$backend size=${w}x${h}x${frames} rc=$rc wall_s=$(awk "BEGIN{printf \"%.1f\", $t1-$t0}") peak_vram_mib=$peak ${VRAM_MB:+ceiling_mib=$VRAM_MB}" | tee "$out.sum"
tr '\r' '\n' < "$out.log" | grep -v "^$" | grep -iE "prompt:|stage|denoised|decoded|stereo|upscal|total|error|panicked|declined" | tail -24 | cut -c1-200
