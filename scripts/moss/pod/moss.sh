#!/bin/bash
# One MOSS-Image picture, with the card's memory traced, under gpu.lock.
#   moss.sh TAG BACKEND [SIZE STEPS]
#     BACKEND  cuda | vulkan | cpu
#   env: MODEL (default cmf/moss-image-sft-q4tp.cmf), PROMPT, BIN, EXTRA, CMF_*.
# Leaves out/TAG.{png,log,vram_trace} and a summary line in out/TAG.sum.
set -u
root=${H3I_ROOT:-/root/h3i}
tag=$1; backend=$2; size=${3:-512}; steps=${4:-8}
model=${MODEL:-$root/cmf/moss-image-sft-q4tp.cmf}
prompt=${PROMPT:-best quality, highres, RAW editorial wildlife photograph of an adult red fox standing on a moss-covered stone beside a clear forest stream at dawn, detailed natural fur, amber eyes, soft mist}
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
exec 9>"$root/gpu.lock"; flock 9
( while :; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 0.25; done ) > "$out.vram_trace" 2>/dev/null &
tr=$!
t0=$(date +%s.%N)
"$bin" imagine "$model" --prompt "$prompt" --width "$size" --height "$size" --steps "$steps" --cfg 3 --shift 5 --seed 42 \
  ${EXTRA:-} --out "$out.png" > "$out.log" 2>&1
rc=$?
t1=$(date +%s.%N)
kill $tr 2>/dev/null
peak=$(sort -n "$out.vram_trace" | tail -1)
echo "tag=$tag backend=$backend size=${size} steps=$steps rc=$rc wall_s=$(awk "BEGIN{printf \"%.1f\", $t1-$t0}") peak_vram_mib=$peak ${VRAM_MB:+ceiling_mib=$VRAM_MB}" | tee "$out.sum"
tr '\r' '\n' < "$out.log" | grep -v "^$" | grep -iE "step|text|vae|decod|total|declined|stood|error|panicked|median" | tail -8 | cut -c1-220
