#!/bin/bash
# hostppl.sh: host-path (CMF_GPU=0) code ppl over 128 tokens, turbo vs hot -> logs/hostppl.txt
cd /root/rip && . fnx/env.sh
{
for pair in turbo:fn/qwen38-flash-next-q2tp-turbo.cmf hot:fn/qwen38-flash-next-q2tp-hot-turbo.cmf; do
  n=${pair%%:*}; f=${pair#*:}; t0=$(date +%s)
  CMF_GPU=0 CMF_QWEN_DEVICE=0 $BIN ppl $f -f data/expert_store_v089.rs --tokens 128 > logs/hostppl_$n.log 2>&1
  echo "host $n: $(grep -a "PPL" logs/hostppl_$n.log | tail -1 | cut -c1-100) | $(( $(date +%s) - t0 )) s | $(grep -a -c -i "panic\|error" logs/hostppl_$n.log) panic/error lines"
done
echo HOSTPPL_DONE
} > logs/hostppl.txt 2>&1
