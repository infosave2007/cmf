#!/bin/bash
# Wide prompt chunks (512) for a MoE file past the card, Vulkan and CUDA:
# Ornith-122B q4tp TTFT and decode after 512 within 32 GB on Vulkan (48-wide
# chunks: 55 s in vcache1) and within 24 GB / the whole card on CUDA, against
# CMF_PREFILL_CHUNK=48; then perplexity on Vulkan within 32 GB, where the
# card slots are on (CPU 3.049). After fnx2.sh.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
mkdir -p logs/vcache3
end=$((SECONDS + 14400))
until grep -qE "FNX2_DONE" logs/fnx2.out 2>/dev/null || [ $SECONDS -gt $end ]; do sleep 30; done
bash scripts/build.sh fast > logs/vcache3/build.log 2>&1
grep -q BUILD_FAST_DONE logs/vcache3/build.log || { echo VCACHE3_BUILD_FAILED; grep -E '^error' -A8 logs/vcache3/build.log | head -40; exit 1; }
B=bin/cortiq-fast
M=cmf/big/ornith-1.5-122b-q4tp.cmf
run() { # tag backend capMB(0 = the card) env...
  local tag=$1 be=$2 cap=$3; shift 3
  local l=logs/vcache3/${tag}_${be}_$cap.log x="CMF_GPU_VRAM_MB=$cap"
  [ $cap = 0 ] && x="CMF_X=1"
  flock gpu.lock env CMF_GPU=$be $x "$@" RUST_LOG=info timeout 2400 $B bench $M --tokens 64 --ctx 512 --ignore-eos --json > $l 2>&1
  echo "$tag $be ${cap}MB: exit $? | $(grep -hoE '"(decode_tok_s_steady|ttft_s|prefill_tok_s)": [0-9.e+-]+' $l | tr '\n' ' ')"
}
run wide 1 32768
run c48 1 32768 CMF_PREFILL_CHUNK=48
run wide cuda 24576
run c48 cuda 24576 CMF_PREFILL_CHUNK=48
run wide cuda 0
run c48 cuda 0 CMF_PREFILL_CHUNK=48
flock gpu.lock env CMF_GPU=1 CMF_GPU_VRAM_MB=32768 timeout 3600 $B ppl $M --file prompts/rail.txt --tokens 256 > logs/vcache3/ppl_32768.log 2>&1
echo "ppl vulkan 32768MB (CPU 3.049): exit $? | $(grep -hiE 'ppl|perplexity' logs/vcache3/ppl_32768.log | tail -1 | cut -c1-200)"
echo VCACHE3_DONE
