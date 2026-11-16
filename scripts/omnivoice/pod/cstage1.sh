#!/bin/bash
# CUDA prompt over streamed experts on the grouped path: each layer's every
# expert staged into the stream's slots first (`stream_stage`; was the small
# path 16 rows at a time: TTFT 58-68 s within 12 / 16 GB, 16-22 s past them),
# Ornith q4tp within 12 / 16 / 24 GB and the whole card, TTFT and decode after
# 512, CMF_HYB_STREAM_STAGE=0 against it at 16 GB, perplexity at 24 GB. After
# glmv2.sh (its build carries the change).
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
mkdir -p logs/cstage1
end=$((SECONDS + 14400))
until grep -qE "GLMV2_DONE|GLMV2_BUILD_FAILED" logs/glmv2.out 2>/dev/null || [ $SECONDS -gt $end ]; do sleep 30; done
grep -q GLMV2_BUILD_FAILED logs/glmv2.out && { echo CSTAGE1_SKIPPED; exit 1; }
strings bin/cortiq-fast | grep -q CMF_HYB_STREAM_STAGE || { echo CSTAGE1_STALE_BINARY; exit 1; }
B=bin/cortiq-fast
M=cmf/big/ornith-1.5-122b-q4tp.cmf
run() { # tag capMB(0 = the card) env...
  local tag=$1 cap=$2; shift 2
  local l=logs/cstage1/${tag}_$cap.log x="CMF_GPU_VRAM_MB=$cap"
  [ $cap = 0 ] && x="CMF_X=1"
  flock gpu.lock env CMF_GPU=cuda $x "$@" RUST_LOG=info timeout 2400 $B bench $M --tokens 64 --ctx 512 --ignore-eos --json > $l 2>&1
  echo "$tag ${cap}MB: exit $? | $(grep -hoE '"(decode_tok_s_steady|ttft_s)": [0-9.e+-]+' $l | tr '\n' ' ')"
}
for cap in 12288 16384 24576 0; do run stage $cap; done
run small 16384 CMF_HYB_STREAM_STAGE=0
flock gpu.lock env CMF_GPU=cuda CMF_GPU_VRAM_MB=24576 timeout 3600 $B ppl $M --file prompts/rail.txt --tokens 256 > logs/cstage1/ppl.log 2>&1
echo "ppl cuda 24576MB (CPU 3.049): exit $? | $(grep -hiE 'ppl|perplexity' logs/cstage1/ppl.log | tail -1 | cut -c1-200)"
echo CSTAGE1_DONE
