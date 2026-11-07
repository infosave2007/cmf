#!/bin/bash
# final3.sh: the rest of final2 after it was cut off — hot at 1k/4k, exact vs far field at 32k -> logs/final3.txt
cd /root/rip && . fnx/env.sh
BIN=/root/rip/cortiq.ft8
HOT=/root/rip/fn/qwen38-flash-next-q2tp-hot-turbo.cmf
TURBO=/root/rip/fn/qwen38-flash-next-q2tp-turbo.cmf
warm() { t0=$(date +%s); cat $1 > /dev/null; echo "warmed $(basename $1) in $(( $(date +%s) - t0 )) s"; }
{
echo "--- final3 $(date -u)"
warm $HOT
for c in 1000 4000; do BIN=$BIN FILE=$HOT fnx/bench.sh hotq_$c $c; done
warm $TURBO
for c in 32000; do
  BIN=$BIN FILE=$TURBO fnx/bench.sh exactq_$c $c CMF_QWEN_O1=off
  BIN=$BIN FILE=$TURBO fnx/bench.sh turboq_$c $c
done
echo FINAL3_DONE
} > logs/final3.txt 2>&1
