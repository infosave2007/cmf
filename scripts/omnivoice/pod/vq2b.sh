#!/bin/bash
# Ornith q2tp-v2's prompt on Vulkan through the token-axis 2-bit gate/up
# kernel (was seven commands a row: 512 tokens in 7.0 / 9.5 s on the whole
# card / within 32 GB), TTFT and decode after 512, CMF_BATCH_Q2_MOE=0 against
# it, the perplexity within 32 GB (CPU 3.179). After vstr3.sh.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
mkdir -p logs/vq2b
end=$((SECONDS + 14400))
until grep -qE "VSTR3_DONE" logs/vstr3.out 2>/dev/null || [ $SECONDS -gt $end ]; do sleep 30; done
B=bin/cortiq-fast
M=cmf/big/ornith-1.5-122b-q2tp-v2.cmf
run() { # tag capMB(0 = the card) env...
  local tag=$1 cap=$2; shift 2
  local l=logs/vq2b/${tag}_$cap.log x="CMF_GPU_VRAM_MB=$cap"
  [ $cap = 0 ] && x="CMF_X=1"
  flock gpu.lock env CMF_GPU=1 $x "$@" RUST_LOG=info timeout 2400 $B bench $M --tokens 64 --ctx 512 --ignore-eos --json > $l 2>&1
  echo "$tag ${cap}MB: exit $? | $(grep -hoE '"(decode_tok_s_steady|ttft_s)": [0-9.e+-]+' $l | tr '\n' ' ')"
}
run q2b 0
run q2b 32768
run loop 32768 CMF_BATCH_Q2_MOE=0
flock gpu.lock env CMF_GPU=1 CMF_GPU_VRAM_MB=32768 timeout 3600 $B ppl $M --file prompts/rail.txt --tokens 256 > logs/vq2b/ppl.log 2>&1
echo "ppl vulkan 32768MB (CPU 3.179): exit $? | $(grep -hiE 'ppl|perplexity' logs/vq2b/ppl.log | tail -1 | cut -c1-200)"
echo VQ2B_DONE
