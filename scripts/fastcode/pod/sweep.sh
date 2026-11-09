#!/bin/bash
# sweep.sh BIN TAG: stage costs of fc256 at a 4000-token prompt — prompt and decode tok/s with one stage left out
cd /root/fc; J=/root/fc/JOURNAL.md; export XDG_RUNTIME_DIR=/tmp
B=$1; T=$2
for S in none experts hc gdn qsa ple route; do
  BIN=$B TOK=48 timeout 300 ./bench.sh ${T}_$S out/fc256.cmf 4000 CMF_QWEN_PROFILE=out/fc256.profile CMF_QWEN_MTP=0 CMF_Q4_SKIP=$S > /dev/null 2>&1
  echo "$(date +%T) $T skip=$S $(sed -e "s/\x1b\[[0-9;]*m//g" logs/bench_${T}_$S.log | grep -a -o -E "prefill [0-9.]+ tok/s|Decode: +[0-9]+ tokens \| [0-9.]+ tok/s" | tr '\n' ' ')" >> $J
done
echo "$(date +%T) SWEEP_${T}_DONE" >> $J
