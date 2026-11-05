#!/bin/bash
# Quality and speed of both codecs on the device path. bench_all.sh LOG
cd /root/dory
export XDG_RUNTIME_DIR=/tmp
B=./src/target/release/cortiq
L=${1:-logs/bench_all.log}
: > $L
for f in q4tp q8_2f; do
  echo "== $f" >> $L
  timeout 60 $B bench out/dory-$f.cmf --core --tokens 120 --ignore-eos 2>&1 | grep -E "Prompt|Decode|Memory" | cut -c1-120 >> $L
  for c in 1000 4000 16000; do
    timeout 200 $B bench out/dory-$f.cmf --core --tokens 40 --ignore-eos --ctx $c 2>&1 | grep -E "Prompt|Decode|Memory" | cut -c1-120 >> $L
  done
  timeout 300 $B ppl out/dory-$f.cmf -f wiki_test.txt --windows 12 --window-len 512 2>&1 | grep -E "PPL" | cut -c1-120 >> $L
done
echo DONE >> $L
