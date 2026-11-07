#!/bin/bash
# final.sh: the quiet pass after everything else — exact vs far field at 1k/4k/16k/32k, the hot file at 1k/4k -> logs/final.txt
cd /root/rip && . fnx/env.sh
until grep -q HOTCHAIN_DONE logs/hotchain.txt 2>/dev/null; do sleep 30; done
HOT=/root/rip/fn/qwen38-flash-next-q2tp-hot-turbo.cmf
TURBO=/root/rip/fn/qwen38-flash-next-q2tp-turbo.cmf
warm() { t0=$(date +%s); cat $1 > /dev/null; echo "warmed $(basename $1) in $(( $(date +%s) - t0 )) s"; }
{
date -u; uptime
warm $TURBO
for c in 1000 4000 16000 32000; do
  FILE=$TURBO fnx/bench.sh exactq_$c $c CMF_QWEN_O1=off
  FILE=$TURBO fnx/bench.sh turboq_$c $c
done
warm $HOT
for c in 1000 4000; do FILE=$HOT fnx/bench.sh hotq_$c $c; done
echo FINAL_DONE
} > logs/final.txt 2>&1
