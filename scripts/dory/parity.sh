#!/bin/bash
# Quality and parity of the converted files on the device path.
cd /root/dory
export XDG_RUNTIME_DIR=/tmp
B=./src/target/release/cortiq
L=logs/parity.log
: > $L
for f in q8_2f q4tp; do
  echo "== $f ppl" >> $L
  CMF_PPL_IDS_OUT=wiki_ids.txt timeout 600 $B ppl out/dory-$f.cmf -f wiki_test.txt --windows 12 --window-len 512 2>&1 | grep -vE "ppl-window|PROGRESS|Opened" | tail -3 | cut -c1-200 >> $L
done
for q in "What is 17 * 23? Answer briefly." "Write a Python function that returns the n-th Fibonacci number."; do
  for f in q8_2f q4tp; do
    echo "== $f chat: $q" >> $L
    timeout 200 $B run out/dory-$f.cmf --prompt "$q" --greedy --max-tokens 200 2>&1 | grep -vE "PROGRESS|Opened|^$|Ready" | tail -12 | cut -c1-900 >> $L
  done
done
for f in q8_2f q4tp; do
  for c in 1000 4000; do
    echo "== $f ctx $c" >> $L
    timeout 300 $B bench out/dory-$f.cmf --core --tokens 60 --ignore-eos --ctx $c 2>&1 | grep -iE "tok/s|prefill|vram|kv" | tail -4 | cut -c1-200 >> $L
  done
done
echo DONE >> $L
