#!/bin/bash
# The speculation-trial change: chat turns and one-shot prompts, old (cortiq-final) against new, in turn.
cd /root/o; export XDG_RUNTIME_DIR=/tmp CMF_GPU_PROBE=0; . ~/.cargo/env
L=$1; M=$2; : > $L
(cd /root/o/src2 && CARGO_TARGET_DIR=/root/o/target2 cargo build -j 32 --release -p cortiq-cli > /root/o/build7.log 2>&1; echo "build7 rc=$?" >> $L)
OLD=/root/o/cortiq-final; NEW=/root/o/target2/release/cortiq
for rep in 1 2; do for b in $OLD $NEW; do
  timeout 900 $b run $M --greedy --no-think --max-tokens 200 < /root/o/turns.txt > /root/o/tt.out 2>/dev/null
  echo "turns [$b] pass $rep: $(grep -oE 'decode [0-9.]+ tok/s' /root/o/tt.out | awk '{print $2}' | tr '\n' ' ')" | tee -a $L
done; done
AB=/root/o/src/scripts/dory/q27_dflash_ab.sh
for b in $OLD $NEW; do
  $AB $b $M p tab.log 0 220 code review logs prose ru -- mtp
  echo "one-shot [$b]: $(grep -oE 'decode [0-9.]+ tok/s' tab.log | awk '{print $2}' | tr '\n' ' ')" | tee -a $L
done
cd /root/o/src2 && CARGO_TARGET_DIR=/root/o/target2 cargo test -j 32 --release -p cortiq-engine --lib -- speculative_trial > /root/o/t7.log 2>&1; echo "test rc=$?" >> $L; grep -E "^test |test result" /root/o/t7.log >> $L
echo TRIAL_DONE >> $L
