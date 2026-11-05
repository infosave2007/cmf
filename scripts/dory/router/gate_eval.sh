#!/bin/bash
# Quality of the early exit in the engine's real cache: perplexity row by row on the
# wiki windows, and greedy agreement on chat. gate_eval.sh [BIN] [MODEL] [LOG]
cd /root/dory
export XDG_RUNTIME_DIR=/tmp
B=${1:-./bin_gate}; M=${2:-out/dory-q4tp.cmf}; L=${3:-logs/gate.log}
: > $L
run() { # name, env...
  n=$1; shift
  r=$(env "$@" CMF_NLL_CHUNK=1 CMF_DORY_GATE_LOG=1 timeout 700 $B ppl $M -f wiki_test.txt --windows 12 --window-len 512 2>&1 | grep -E "dory gate|PPL" | tail -2 | tr '\n' ' ' | cut -c1-120)
  echo "$n: $r" >> $L
}
run ll_k2 CMF_DORY_KV=last_loop
run ll_k1 CMF_DORY_KV=last_loop CMF_DORY_LOOPS=1
run conf.9 CMF_DORY_KV=last_loop CMF_DORY_GATE=conf:0.9
run conf.7 CMF_DORY_KV=last_loop CMF_DORY_GATE=conf:0.7
run conf.5 CMF_DORY_KV=last_loop CMF_DORY_GATE=conf:0.5
for b in $(awk '{print $NF}' router/bias.txt | sed -n 2,4p); do
  run recon$b CMF_DORY_KV=last_loop CMF_DORY_GATE=recon:router/phi1.bin:$b
done
echo DONE >> $L
