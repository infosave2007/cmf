#!/bin/bash
# benchchain.sh: the turbo file exact (CMF_QWEN_O1=off) vs its far field, 4k and 16k -> logs/benchchain.txt
cd /root/rip
export FILE=/root/rip/fn/qwen38-flash-next-q2tp-turbo.cmf
{
for c in 4000 16000; do
  fnx/bench.sh exact_$c $c CMF_QWEN_O1=off
  fnx/bench.sh turbo_$c $c
done
echo BENCHCHAIN_DONE
} > logs/benchchain.txt 2>&1
