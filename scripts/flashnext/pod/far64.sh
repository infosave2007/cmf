#!/bin/bash
# far64.sh: after the quiet pass — the turbo file exact vs far field at 64k (the exact KV may not fit) -> logs/far64.txt
cd /root/rip && . fnx/env.sh
until grep -q FINAL_DONE logs/final.txt 2>/dev/null; do sleep 30; done
export FILE=/root/rip/fn/qwen38-flash-next-q2tp-turbo.cmf
{
date -u
fnx/bench.sh turboq_64000 64000
fnx/bench.sh exactq_64000 64000 CMF_QWEN_O1=off
echo FAR64_DONE
} > logs/far64.txt 2>&1
