#!/bin/bash
# Vulkan expert slots on Ornith q2tp-v2 (48 GB: its banks fit the pod's RAM
# where q4tp's do not, so the slots get the card's room at 32 GB too):
# 32 GB and the whole card, slots on / off, decode after 512. After vcache1.sh.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
mkdir -p logs/vcache2
end=$((SECONDS + 14400))
until grep -qE "VCACHE1_DONE|VCACHE1_BUILD_FAILED" logs/vcache1.out 2>/dev/null || [ $SECONDS -gt $end ]; do sleep 30; done
grep -q VCACHE1_BUILD_FAILED logs/vcache1.out && { echo VCACHE2_SKIPPED; exit 1; }
B=bin/cortiq-fast
M=cmf/big/ornith-1.5-122b-q2tp-v2.cmf
run() { # tag capMB(0 = the card) env...
  local tag=$1 cap=$2; shift 2
  local l=logs/vcache2/${tag}_$cap.log x="CMF_GPU_VRAM_MB=$cap"
  [ $cap = 0 ] && x="CMF_X=1"
  flock gpu.lock env CMF_GPU=1 $x "$@" RUST_LOG=info timeout 2400 $B bench $M --tokens 64 --ctx 512 --ignore-eos --json > $l 2>&1
  echo "$tag ${cap}MB: exit $? | $(grep -hoE '"(decode_tok_s_steady|ttft_s)": [0-9.e+-]+' $l | tr '\n' ' ')| $(sed -E 's/\x1b\[[0-9;]*m//g' $l | grep -oE 'routed experts: [^|]*|expert cache: [^|]*|batch graph declined[^|]*|batched prefill [A-Za-z]+' | sort -u | tr '\n' ';' | cut -c1-400)"
}
for cap in 32768 0; do
  run on $cap
  run off $cap CMF_WGPU_EXPERT_CACHE=0
done
echo VCACHE2_DONE
