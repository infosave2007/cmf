#!/bin/bash
# Steady state of the expert stream: a 512-token prompt, then 128 tokens of
# decode (the slots and the page cache warm), the stream's hit-rate lines
# (CMF_CUDA_DEBUG), for MiMo-V2.6 and MiniMax-M3.1. bigs3: after bigs2, a
# build with MADV_WILLNEED over the missing experts before their builds.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
mkdir -p logs/bigs3
end=$((SECONDS + 14400))
until grep -qE "BIGS2_DONE" logs/bigs2.out 2>/dev/null || [ $SECONDS -gt $end ]; do sleep 30; done
bash scripts/build.sh fast > logs/bigs3/build.log 2>&1
grep -q BUILD_FAST_DONE logs/bigs3/build.log || { echo BIGS3_BUILD_FAILED; tail -30 logs/bigs3/build.log; exit 1; }
B=bin/cortiq-fast
for f in MiMo-V2.6-Flash-RL-q4tp minimax-m3.1-flash-q2tp-mix; do
  flock gpu.lock env CMF_GPU=cuda CMF_CUDA_DEBUG=1 timeout 3600 $B bench cmf/big/$f.cmf --tokens 128 --ctx 512 --ignore-eos --json > logs/bigs3/$f.log 2>&1
  echo "$f 512+128: exit $? | $(grep -hoE '"(decode_tok_s_steady|prefill_tok_s|ttft_s)": [0-9.e+-]+' logs/bigs3/$f.log | tr '\n' ' ')"
  grep -hE "expert stream: [0-9]+ slots" logs/bigs3/$f.log | tail -2 | cut -c1-200 | sed 's/^/      /'
done
echo BIGS3_DONE
