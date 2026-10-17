#!/bin/bash
# ppl10.sh: Qwen3-0.6B cortiq CPU ppl on the first 10 of the 40-window WikiText-2 test placement (512 tokens
# each) — the f16 run's log already holds them; every other file runs until window index 9 is logged, then
# is stopped by PID. Same windows for all files; the summary sums exact NLL / scored tokens.
set -u
W=/root/cmfbuild/q2tp_work
Q=$W/qwen06
L=/root/cmfbuild/q2tp_logs
B=$W/bin/cortiq_q4d
export CMF_GPU=0 CMF_THREADS=13 RAYON_NUM_THREADS=13 CMF_PPL_PROGRESS=1
for t in q2tp_v1 q2tp_v2 q2tp_gptq131k; do
  $B ppl $Q/$t.cmf --file $W/data/wiki.test.raw --windows 40 --window-len 512 > $L/ppl40_qwen06_$t.log 2>&1 &
  pid=$!
  until tr '\r' '\n' < $L/ppl40_qwen06_$t.log | grep -q "ppl-window: index=10 " || ! kill -0 $pid 2>/dev/null; do sleep 5; done
  kill $pid 2>/dev/null
  wait $pid 2>/dev/null
done
for t in f16 q2tp_v1 q2tp_v2 q2tp_gptq131k; do
  # the progress line carries running totals; index counts windows done
  tr '\r' '\n' < $L/ppl40_qwen06_$t.log | grep "ppl-window: index=10 " | head -1 | awk -v t=$t -v sz=$(stat -c %s $Q/$t.cmf) '
    { for (i = 1; i <= NF; i++) { if ($i ~ /^scored=/) sc = substr($i, 8) + 0; if ($i ~ /^exact_nll=/) nl = substr($i, 11) + 0 }
      printf "qwen06_w10 %s scored=%d nll=%.6f ppl=%.4f bytes=%d\n", t, sc, nl, exp(nl / sc), sz }' >> $L/gptq_runs.log
done
echo "=== ppl10 done $(date -u +%FT%TZ)" >> $L/gptq_runs.log
