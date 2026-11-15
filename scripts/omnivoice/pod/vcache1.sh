#!/bin/bash
# Vulkan card slots over system-memory expert banks (`moe_cache`) and the
# batch prefill on the same placement: Ornith-122B q4tp within 24 / 32 GB
# and the whole card, slots on (default) and off (CMF_WGPU_EXPERT_CACHE=0,
# the banks read in place), decode after 512; then the perplexity check on
# 24 GB and the whole card (CPU 3.049 on prompts/rail.txt, 256 tokens).
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
mkdir -p logs/vcache1
bash scripts/build.sh fast > logs/vcache1/build.log 2>&1
grep -q BUILD_FAST_DONE logs/vcache1/build.log || { echo VCACHE1_BUILD_FAILED; grep -E '^error' -A8 logs/vcache1/build.log | head -40; exit 1; }
B=bin/cortiq-fast
M=cmf/big/ornith-1.5-122b-q4tp.cmf
run() { # tag capMB(0 = the card) env...
  local tag=$1 cap=$2; shift 2
  local l=logs/vcache1/${tag}_$cap.log x="CMF_GPU_VRAM_MB=$cap"
  [ $cap = 0 ] && x="CMF_X=1"
  flock gpu.lock env CMF_GPU=1 $x "$@" RUST_LOG=info timeout 2400 $B bench $M --tokens 64 --ctx 512 --ignore-eos --json > $l 2>&1
  echo "$tag ${cap}MB: exit $? | $(grep -hoE '"(decode_tok_s_steady|ttft_s)": [0-9.e+-]+' $l | tr '\n' ' ')| $(sed -E 's/\x1b\[[0-9;]*m//g' $l | grep -oE 'routed experts: [^|]*|expert cache: [^|]*|batch graph declined[^|]*|batched prefill [A-Za-z]+' | sort -u | tr '\n' ';' | cut -c1-400)"
}
for cap in 32768 24576 0; do
  run on $cap
  run off $cap CMF_WGPU_EXPERT_CACHE=0
done
for cap in 24576 0; do
  x="CMF_GPU_VRAM_MB=$cap"; [ $cap = 0 ] && x="CMF_X=1"
  flock gpu.lock env CMF_GPU=1 $x timeout 3600 $B ppl $M --file prompts/rail.txt --tokens 256 > logs/vcache1/ppl_$cap.log 2>&1
  echo "ppl ${cap}MB (CPU 3.049): exit $? | $(grep -hiE 'ppl|perplexity' logs/vcache1/ppl_$cap.log | tail -1 | cut -c1-200)"
done
echo VCACHE1_DONE
