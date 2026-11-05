#!/bin/bash
# A prompt in one pass, the answer in two: the speed of a long prefill and what the
# continuation loses. fast_prompt.sh BIN LOG
cd /root/dory
export XDG_RUNTIME_DIR=/tmp
B=$1; L=$2
: > $L
Q=out/dory-q4tp-v2.cmf
T=out/dory-q4tp-v2-turbo.cmf
P() { CMF_NLL_CHUNK=64 CMF_NLL_SKIP=3584 timeout 1200 $B ppl $Q -f wiki_test.txt --windows 4 --window-len 4096 2>&1 | grep -E "PPL|panicked" | tail -1 | cut -c1-90; }
echo "ppl of the last 512 rows of 4096, prompt 2 passes per_loop: $(P)" | tee -a $L
echo "same, last_loop: $(CMF_DORY_KV=last_loop P)" | tee -a $L
echo "same, last_loop, prompt 1 pass: $(CMF_DORY_KV=last_loop CMF_DORY_PROMPT_LOOPS=1 P)" | tee -a $L
C() { r=$(timeout 1500 $B bench $T --core --tokens 30 --ignore-eos --ctx $1 2>&1 | grep -E "Prompt|Decode" | sed -e 's/  */ /g' | tr '\n' ' ' | cut -c1-130); echo "$2 ctx $1: $r" | tee -a $L; }
for c in 4000 16000 64000; do
  CMF_DORY_PROMPT_LOOPS=1 C $c "turbo prompt 1 pass"
done
echo DONE >> $L
