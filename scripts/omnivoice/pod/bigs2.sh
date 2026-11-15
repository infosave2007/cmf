#!/bin/bash
# Steady state of the expert stream: a 512-token prompt, then 128 tokens of
# decode (the slots and the page cache warm), the stream's hit-rate lines
# (CMF_CUDA_DEBUG), for MiMo-V2.6 and MiniMax-M3.1. After lfm5.sh's build.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
mkdir -p logs/bigs2
end=$((SECONDS + 14400))
until grep -qE "LFM5_DONE|LFM5_BUILD_FAILED" logs/lfm5.out 2>/dev/null || [ $SECONDS -gt $end ]; do sleep 30; done
B=bin/cortiq-fast
for f in MiMo-V2.6-Flash-RL-q4tp minimax-m3.1-flash-q2tp-mix; do
  flock gpu.lock env CMF_GPU=cuda CMF_CUDA_DEBUG=1 timeout 3600 $B bench cmf/big/$f.cmf --tokens 128 --ctx 512 --ignore-eos --json > logs/bigs2/$f.log 2>&1
  echo "$f 512+128: exit $? | $(grep -hoE '"(decode_tok_s_steady|prefill_tok_s|ttft_s)": [0-9.e+-]+' logs/bigs2/$f.log | tr '\n' ' ')"
  grep -hE "expert stream: [0-9]+ slots" logs/bigs2/$f.log | tail -2 | cut -c1-200 | sed 's/^/      /'
done
echo BIGS2_DONE
