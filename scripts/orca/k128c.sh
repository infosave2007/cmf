#!/bin/bash
# vec4 activation loads in the 128-row tile: rebuild, A/B against the previous build, text, the parity tests.
cd /root/o; export XDG_RUNTIME_DIR=/tmp CMF_GPU_PROBE=0; . ~/.cargo/env
L=$1; M=$2; : > $L
cp /root/o/target2/release/cortiq /root/o/cortiq-k128b
(cd /root/o/src2 && CARGO_TARGET_DIR=/root/o/target2 cargo build -j 32 --release -p cortiq-cli > /root/o/build6.log 2>&1; echo "build6 rc=$?" >> $L)
A=/root/o/cortiq-k128b; N=/root/o/target2/release/cortiq
for rep in 1 2; do for b in $A $N; do
  r=$($b bench $M --core --tokens 8 --ignore-eos --ctx 2000 2>&1 | sed -e 's/\x1b\[[0-9;]*m//g' | grep -E "Prompt" | sed 's/  */ /g' | cut -c1-120)
  echo "bench2000 [$b]: $r" | tee -a $L
done; done
CMF_GRAPH_SPEC=0 $N run $M --prompt-file /tmp/p2k.txt --greedy --no-think --max-tokens 120 > /tmp/k.out 2>&1
echo "run [new]: $(grep -E '^\[' /tmp/k.out | head -1 | cut -c1-120) | text $(grep -vE '^\[|^Loading|^Ready|^Prompt' /tmp/k.out | md5sum | cut -c1-8)" | tee -a $L
cd /root/o/src2 && CARGO_TARGET_DIR=/root/o/target2 cargo test -j 32 --release -p cortiq-engine --features gpu --lib -- wgpu_q4tp_mm_coop > /root/o/t128.log 2>&1; echo "test rc=$?" >> $L; grep -E "test result|panicked" /root/o/t128.log >> $L
echo K128C_DONE >> $L
