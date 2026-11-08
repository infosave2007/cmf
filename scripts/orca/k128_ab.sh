#!/bin/bash
# The 128-row coop tile against the 64-row one and the old default (scalar tile), same file, in turn; then the
# greedy text of each arm on a 2000-token prompt. k128_ab.sh LOG MODEL
cd /root/o; export XDG_RUNTIME_DIR=/tmp CMF_GPU_PROBE=0
OLD=/root/o/src/target/release/cortiq; NEW=/root/o/target2/release/cortiq; L=$1; M=$2; : > $L
for rep in 1 2; do
  for arm in "OLD CMF_BATCH_COOP=0" "NEW CMF_COOP128=0" "NEW CMF_COOP128=1"; do
    set -- $arm; b=$OLD; [ $1 = NEW ] && b=$NEW
    r=$(env $2 $b bench $M --core --tokens 8 --ignore-eos --ctx 2000 2>&1 | sed -e 's/\x1b\[[0-9;]*m//g' | grep -E "Prompt|Decode" | tr '\n' ' ' | sed 's/  */ /g' | cut -c1-200)
    echo "bench2000 [$arm]: $r" | tee -a $L
  done
done
for arm in "OLD CMF_BATCH_COOP=0" "NEW CMF_COOP128=1"; do
  set -- $arm; b=$OLD; [ $1 = NEW ] && b=$NEW
  env $2 CMF_GRAPH_SPEC=0 $b run $M --prompt-file /tmp/p2k.txt --greedy --no-think --max-tokens 120 > /tmp/k.out 2>&1
  echo "run [$arm]: $(grep -E '^\[' /tmp/k.out | head -1 | cut -c1-120) | text $(grep -vE '^\[|^Loading|^Ready|^Prompt' /tmp/k.out | md5sum | cut -c1-8)" | tee -a $L
done
cd /root/o/src2 && . ~/.cargo/env && CARGO_TARGET_DIR=/root/o/target2 cargo test -j 16 --release -p cortiq-engine --lib -- wgpu_q4tp_mm_coop > /root/o/t128.log 2>&1; echo "test rc=$?" >> $L; grep -E "^test |test result" /root/o/t128.log >> $L
echo K128_DONE >> $L
