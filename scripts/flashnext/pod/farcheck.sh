#!/bin/bash
# farcheck.sh: wiki ppl over 8192 tokens, exact vs sealed at 4096 -> logs/farcheck.txt
cd /root/rip
{
fnx/pplf.sh w8k_exact data/wiki.test.raw 8192 CMF_QWEN_O1=off
fnx/pplf.sh w8k_seal4k data/wiki.test.raw 8192 CMF_QWEN_O1=32,1800,4,4096
echo FARCHECK_DONE
} > logs/farcheck.txt 2>&1
