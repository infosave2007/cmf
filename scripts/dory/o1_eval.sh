#!/bin/bash
# O(1) on the full layers on the device: perplexity over long windows, against exact.
# o1_eval.sh BIN MODEL LOG SPEC...
cd /root/dory
export XDG_RUNTIME_DIR=/tmp
B=$1; M=$2; L=$3; shift 3
: > $L
for spec in "$@"; do
  r=$(CMF_DORY_O1=$spec RUST_LOG=cortiq_engine=warn timeout 1500 $B ppl $M -f wiki_test.txt --windows 4 --window-len 4096 2>&1 | grep -E "PPL|WARN|panicked" | tail -2 | tr '\n' ' ' | cut -c1-170)
  echo "o1 $spec: $r" >> $L
done
echo DONE >> $L
