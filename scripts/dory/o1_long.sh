#!/bin/bash
# Long context: two windows of 16384, the far field sealed at 8192 against exact and a cut.
cd /root/dory
export XDG_RUNTIME_DIR=/tmp
B=$1; M=$2; L=$3
: > $L
P() { CMF_NLL_CHUNK=64 timeout 2400 $B ppl $M -f wiki_test.txt --windows 2 --window-len 16384 2>&1 | grep -E "PPL|panicked" | tail -1 | cut -c1-90; }
echo "exact: $(P)" >> $L
echo "cut 2048: $(CMF_DORY_FULL_WINDOW=2048 P)" >> $L
echo "o1 32,2048,4,8192: $(CMF_DORY_O1=32,2048,4,8192 P)" >> $L
echo "o1 32,1024,4,8192: $(CMF_DORY_O1=32,1024,4,8192 P)" >> $L
echo DONE >> $L
