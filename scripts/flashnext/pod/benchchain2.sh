#!/bin/bash
# benchchain2.sh: after the rebuild — the turbo file exact vs far field at 16k -> logs/benchchain2.txt
cd /root/rip
until grep -q BUILD_DONE logs/build_ft2.log 2>/dev/null; do sleep 10; done
export FILE=/root/rip/fn/qwen38-flash-next-q2tp-turbo.cmf
{
/root/rip/cortiq.ft2 --version
fnx/bench.sh exact_16000 16000 CMF_QWEN_O1=off
fnx/bench.sh turbo_16000 16000
echo BENCHCHAIN2_DONE
} > logs/benchchain2.txt 2>&1
