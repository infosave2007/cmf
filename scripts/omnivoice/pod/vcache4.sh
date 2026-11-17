#!/bin/bash
# Vulkan expert slots, second pass: the fill skips its hits, the copy is an
# indirect dispatch sized by the misses, and a spill of an eighth or less
# stays read in place. Ornith q2tp-v2 (whole card: 28.4 in place / 25.6 with
# slots in vcache2; 32 GB: 20.7 / 25.2) and Ornith q4tp (whole card 30.0,
# 32 GB 16.9 in vcache1), decode after 512. After vcache3.sh.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
mkdir -p logs/vcache4
end=$((SECONDS + 14400))
until grep -qE "VCACHE3_DONE|VCACHE3_BUILD_FAILED" logs/vcache3.out 2>/dev/null || [ $SECONDS -gt $end ]; do sleep 30; done
bash scripts/build.sh fast > logs/vcache4/build.log 2>&1
grep -q BUILD_FAST_DONE logs/vcache4/build.log || { echo VCACHE4_BUILD_FAILED; grep -E '^error' -A8 logs/vcache4/build.log | head -40; exit 1; }
B=bin/cortiq-fast
run() { # tag file capMB(0 = the card) env...
  local tag=$1 f=$2 cap=$3; shift 3
  local l=logs/vcache4/${tag}_$cap.log x="CMF_GPU_VRAM_MB=$cap"
  [ $cap = 0 ] && x="CMF_X=1"
  flock gpu.lock env CMF_GPU=1 $x "$@" RUST_LOG=info timeout 2400 $B bench cmf/big/$f.cmf --tokens 64 --ctx 512 --ignore-eos --json > $l 2>&1
  echo "$tag ${cap}MB: exit $? | $(grep -hoE '"(decode_tok_s_steady|ttft_s)": [0-9.e+-]+' $l | tr '\n' ' ')| $(sed -E 's/\x1b\[[0-9;]*m//g' $l | grep -oE 'routed experts: [^|]*' | sort -u | cut -c1-200)"
}
run q2 ornith-1.5-122b-q2tp-v2 0
run q2 ornith-1.5-122b-q2tp-v2 32768
run q4 ornith-1.5-122b-q4tp 0
run q4 ornith-1.5-122b-q4tp 32768
echo VCACHE4_DONE
