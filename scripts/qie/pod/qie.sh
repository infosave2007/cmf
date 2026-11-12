#!/bin/bash
# One Qwen-Image-Edit-2509 edit, with the card's memory traced.
#   qie.sh TAG BACKEND [SIZE STEPS]
#     BACKEND  cuda | vulkan | cpu
#   env: MODEL (default cmf/qwen-image-edit-2509-q4tp.cmf), IMAGE (the
#        picture to edit), PROMPT, BIN, EXTRA, CMF_*.
# Leaves out/TAG.{png,log,vram_trace} and a summary line in out/TAG.sum.
set -u
root=${H3I_ROOT:-/root/h3i}
tag=$1; backend=$2; size=${3:-1024}; steps=${4:-40}
model=${MODEL:-$root/cmf/qwen-image-edit-2509-q4tp.cmf}
image=${IMAGE:-$root/ref/edit_in.png}
prompt=${PROMPT:-Add a blue knitted scarf.}
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
# the card is shared with other runs: the clock starts once it is ours
exec 9>"$root/gpu.lock"; flock 9
( while :; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 0.25; done ) > "$out.vram_trace" 2>/dev/null &
tr=$!
t0=$(date +%s.%N)
"$bin" imagine "$model" --image "$image" --prompt "$prompt" --height "$size" --width "$size" --steps "$steps" --cfg 4 --seed 7 \
  --reference-size "$size" ${EXTRA:-} --out "$out.png" > "$out.log" 2>&1
rc=$?
t1=$(date +%s.%N)
kill $tr 2>/dev/null
peak=$(sort -n "$out.vram_trace" | tail -1)
echo "tag=$tag backend=$backend size=${size} steps=$steps rc=$rc wall_s=$(awk "BEGIN{printf \"%.1f\", $t1-$t0}") peak_vram_mib=$peak ${VRAM_MB:+ceiling_mib=$VRAM_MB}" | tee "$out.sum"
tr '\r' '\n' < "$out.log" | grep -v "^$" | grep -v INFO | grep -iE "encode|step 1/|step $steps/|vae|decod|total|image|error|panicked|declined|s/step" | tail -8 | cut -c1-200
