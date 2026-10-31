#!/bin/bash
# Decode / prefill speed of LFM3 CMF files on the pod's GPU (bench --core,
# greedy, EOS ignored): a short prompt, then synthetic prompts of 1k and 4k.
#
# usage: pod_bench.sh MODEL.cmf [MODEL.cmf ...]   (CORTIQ from env)
set -u
B=${CORTIQ:-/root/lfm3/bin/cortiq}
export XDG_RUNTIME_DIR=/tmp
for M in "$@"; do
  echo "== $(basename "$M")"
  "$B" bench "$M" --core --ignore-eos --tokens 256 2>&1 | grep -E "Prompt:|Decode:" | sed 's/^ */short /'
  for C in 1024 4096; do
    "$B" bench "$M" --core --ignore-eos --tokens 128 --ctx $C 2>&1 | grep -iE "prefill|decode" | sed "s/^ */ctx$C /"
  done
done
echo BENCH_DONE
