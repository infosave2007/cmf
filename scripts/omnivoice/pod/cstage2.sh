#!/bin/bash
# CUDA prompt over streamed experts on the grouped path, second form: only
# the chunk's picked experts staged after the top-k (staging every expert of
# a layer in cstage1 cut the 12 / 16 GB first token to 24-26 s but left the
# slots and the RAM tier cold for decode: 24 GB 4.4 tok/s instead of 32),
# Ornith q4tp within 12 / 16 / 24 GB and the whole card, TTFT and decode after
# 512, CMF_HYB_STREAM_STAGE=0 against it at 16 GB, perplexity at 24 GB. After
# glmv2.sh (its build carries the change).
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
mkdir -p logs/cstage2
bash scripts/build.sh fast > logs/cstage2/build.log 2>&1
grep -q BUILD_FAST_DONE logs/cstage2/build.log || { echo CSTAGE2_BUILD_FAILED; grep -E '^error|error:' -A8 logs/cstage2/build.log | head -60; exit 1; }
B=bin/cortiq-fast
M=cmf/big/ornith-1.5-122b-q4tp.cmf
run() { # tag capMB(0 = the card) env...
  local tag=$1 cap=$2; shift 2
  local l=logs/cstage2/${tag}_$cap.log x="CMF_GPU_VRAM_MB=$cap"
  [ $cap = 0 ] && x="CMF_X=1"
  flock gpu.lock env CMF_GPU=cuda $x "$@" RUST_LOG=info timeout 2400 $B bench $M --tokens 64 --ctx 512 --ignore-eos --json > $l 2>&1
  echo "$tag ${cap}MB: exit $? | $(grep -hoE '"(decode_tok_s_steady|ttft_s)": [0-9.e+-]+' $l | tr '\n' ' ')"
}
for cap in 12288 16384 24576 0; do run stage $cap; done
run small 16384 CMF_HYB_STREAM_STAGE=0
flock gpu.lock env CMF_GPU=cuda CMF_GPU_VRAM_MB=24576 timeout 3600 $B ppl $M --file prompts/rail.txt --tokens 256 > logs/cstage2/ppl.log 2>&1
echo "ppl cuda 24576MB (CPU 3.049): exit $? | $(grep -hiE 'ppl|perplexity' logs/cstage2/ppl.log | tail -1 | cut -c1-200)"
echo CSTAGE2_DONE
