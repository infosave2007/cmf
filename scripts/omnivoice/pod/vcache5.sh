#!/bin/bash
# The staging bank: a wide batch copies a system-memory bank to the card
# whole and its kernels read it there (vcache1-4: 512 positions in 36-63 s,
# about a second a banked layer). Ornith q4tp and q2tp-v2 within 32 GB and on
# the whole card, staging on / off (CMF_WGPU_EXPERT_STAGE=0), TTFT and decode
# after 512; perplexity with staging (CPU 3.049). After vcache4.sh.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
mkdir -p logs/vcache5
end=$((SECONDS + 14400))
until grep -qE "VCACHE4_DONE|VCACHE4_BUILD_FAILED" logs/vcache4.out 2>/dev/null || [ $SECONDS -gt $end ]; do sleep 30; done
bash scripts/build.sh fast > logs/vcache5/build.log 2>&1
grep -q BUILD_FAST_DONE logs/vcache5/build.log || { echo VCACHE5_BUILD_FAILED; grep -E '^error' -A8 logs/vcache5/build.log | head -40; exit 1; }
B=bin/cortiq-fast
run() { # tag file capMB(0 = the card) env...
  local tag=$1 f=$2 cap=$3; shift 3
  local l=logs/vcache5/${tag}_$cap.log x="CMF_GPU_VRAM_MB=$cap"
  [ $cap = 0 ] && x="CMF_X=1"
  flock gpu.lock env CMF_GPU=1 $x "$@" RUST_LOG=info timeout 2400 $B bench cmf/big/$f.cmf --tokens 64 --ctx 512 --ignore-eos --json > $l 2>&1
  echo "$tag ${cap}MB: exit $? | $(grep -hoE '"(decode_tok_s_steady|ttft_s)": [0-9.e+-]+' $l | tr '\n' ' ')| $(sed -E 's/\x1b\[[0-9;]*m//g' $l | grep -oE 'routed experts: [^|]*|staging bank[^|]*' | sort -u | tr '\n' ';' | cut -c1-260)"
}
for cap in 0 32768; do
  run q4 ornith-1.5-122b-q4tp $cap
  run q4_nostage ornith-1.5-122b-q4tp $cap CMF_WGPU_EXPERT_STAGE=0
done
for cap in 0 32768; do
  run q2 ornith-1.5-122b-q2tp-v2 $cap
done
flock gpu.lock env CMF_GPU=1 timeout 3600 $B ppl cmf/big/ornith-1.5-122b-q4tp.cmf --file prompts/rail.txt --tokens 256 > logs/vcache5/ppl_0.log 2>&1
echo "ppl vulkan whole card staging (CPU 3.049): exit $? | $(grep -hiE 'ppl|perplexity' logs/vcache5/ppl_0.log | tail -1 | cut -c1-200)"
echo VCACHE5_DONE
