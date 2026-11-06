#!/bin/bash
# stat-more.sh [PASSES]: more passes after stat-run.sh, to average the stand's run-to-run spread.
# 1. the five prompts, one answer a process, the release and the branch in turn, PASSES times (default 6);
# 2. the release with the depth pinned at 2 and 3, PASSES/2 times: what a fixed depth does on this card;
# 3. the chat turns of turns.txt in one process, the two binaries in turn, PASSES/2 times.
# Logs: st2-old.log, st2-new.log, st2-pin.log, turnsM<N>-old.out, turnsM<N>-new.out; st2.done at the end.
cd /root/q27; export XDG_RUNTIME_DIR=/tmp
M=m/qwen38-27b-turbo.cmf; n=${1:-6}; h=$(( (n + 1) / 2 ))
rm -f st2-old.log st2-new.log st2-pin.log st2.done
for rep in $(seq "$n"); do for p in code edit quote prose ru; do
  ./q27_dflash_ab.sh ./cortiq-old $M p st2-old.log 0 220 $p -- mtp
  ./q27_dflash_ab.sh ./cortiq $M p st2-new.log 0 220 $p -- mtp
done; done
for rep in $(seq "$h"); do
  ./q27_dflash_ab.sh ./cortiq-old $M p st2-pin.log 0 220 code edit quote prose ru -- k2 k3
done
for rep in $(seq "$h"); do for b in old new; do
  bin=./cortiq; [ $b = old ] && bin=./cortiq-old
  CMF_GRAPH_SPEC=1 CMF_GRAPH_SPEC_TIME=1 RUST_LOG=info timeout 400 $bin run $M --greedy --no-think --max-tokens 200 \
    < turns.txt > "turnsM$rep-$b.out" 2> "turnsM$rep-$b.err"
done; done
echo DONE > st2.done
