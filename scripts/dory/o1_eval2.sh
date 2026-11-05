#!/bin/bash
# What the far field is worth: full layers cut to a window against O(1) over the same
# window, on 4 x 4096. o1_eval2.sh BIN MODEL LOG
cd /root/dory
export XDG_RUNTIME_DIR=/tmp
B=$1; M=$2; L=$3
: > $L
P() { timeout 1500 $B ppl $M -f wiki_test.txt --windows 4 --window-len 4096 2>&1 | grep -E "PPL|panicked" | tail -1 | cut -c1-90; }
for w in 256 1024 2048; do echo "cut $w: $(CMF_DORY_FULL_WINDOW=$w P)" >> $L; done
echo "o1 32,2048,4,2048: $(CMF_DORY_O1=32,2048,4,2048 P)" >> $L
echo DONE >> $L
