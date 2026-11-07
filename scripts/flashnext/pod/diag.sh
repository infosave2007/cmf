#!/bin/bash
# diag.sh: device ppl over 128 tokens, turbo vs hot (the host gave 23.241 vs 25.709) -> logs/diag.txt
cd /root/rip && . fnx/env.sh
{
FILE=/root/rip/fn/qwen38-flash-next-q2tp-turbo.cmf fnx/ppl.sh dev128_turbo 128
FILE=/root/rip/fn/qwen38-flash-next-q2tp-hot-turbo.cmf fnx/ppl.sh dev128_hot 128
echo DIAG_DONE
} > logs/diag.txt 2>&1
