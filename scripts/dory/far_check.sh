#!/bin/bash
# The far field's kernels, old against new: the same ppl, and the speed of a token.
# far_check.sh OLD NEW LOG
cd /root/dory
export XDG_RUNTIME_DIR=/tmp
A=$1; B=$2; L=$3
: > $L
Q=out/dory-q4tp-v2.cmf
export CMF_DORY_KV=last_loop CMF_DORY_O1=32,2048,4,2068
P() { CMF_NLL_SKIP=2500 CMF_NLL_CHUNK=1 timeout 900 $1 ppl $Q -f wiki_test.txt --windows 2 --window-len 3000 2>&1 | grep -E "PPL|panicked" | tail -1 | cut -c1-90; }
echo "old ppl: $(P $A)" | tee -a $L
echo "new ppl: $(P $B)" | tee -a $L
S() { timeout 300 $1 bench $Q --core --tokens 60 --ignore-eos --ctx 3000 2>&1 | grep -E "Decode" | sed -e 's/  */ /g' | cut -c1-90; }
echo "old: $(S $A)" | tee -a $L
echo "new: $(S $B)" | tee -a $L
for k in none far push farattend gate; do
  r=$(CMF_DORY_KEEP=$k timeout 200 $B bench $Q --core --tokens 60 --ignore-eos --ctx 3000 2>&1 | grep -oE "[0-9.]+ tok/s steady" | head -1 | cut -d' ' -f1)
  awk -v k=$k -v r="$r" 'BEGIN{printf "%-10s %8.1f tok/s %7.2f ms\n", k, r, 1000/r}' | tee -a $L
done
echo DONE >> $L
