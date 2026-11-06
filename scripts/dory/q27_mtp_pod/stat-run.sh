#!/bin/bash
# stat-run.sh [PASSES]: the release (cortiq-old) against a branch (cortiq) on a card, the MTP chain under
# its default rule (stat-base.sh built both; p/p_NAME.txt and turns.txt are in place).
# 1. the five prompts without drafts, once;
# 2. the five prompts, one answer a process, the two binaries in turn, PASSES times (default 3);
# 3. the branch with the depth pinned at 2..5, once: the best a fixed depth does on this card;
# 4. the chat turns of turns.txt in one process, the two binaries in turn, PASSES times.
# Logs: st-plain.log, st-old.log, st-new.log, st-pin.log, turnsN-old.out, turnsN-new.out; st.done at the end.
cd /root/q27; export XDG_RUNTIME_DIR=/tmp
M=m/qwen38-27b-turbo.cmf; n=${1:-3}
rm -f st-plain.log st-old.log st-new.log st-pin.log st.done
./q27_dflash_ab.sh ./cortiq-old $M p st-plain.log 0 220 code edit quote prose ru -- plain
for rep in $(seq "$n"); do for p in code edit quote prose ru; do
  ./q27_dflash_ab.sh ./cortiq-old $M p st-old.log 0 220 $p -- mtp
  ./q27_dflash_ab.sh ./cortiq $M p st-new.log 0 220 $p -- mtp
  sed -e 's/\x1b\[[0-9;]*m//g' st-new.log.err | grep -E "^spec-k:|speculation (trial|re-check|stopped)" | cut -c1-220 >> st-new.log
done; done
./q27_dflash_ab.sh ./cortiq $M p st-pin.log 0 220 code edit quote prose ru -- k2 k3 k4 k5
for rep in $(seq "$n"); do for b in old new; do
  bin=./cortiq; [ $b = old ] && bin=./cortiq-old
  CMF_GRAPH_SPEC=1 CMF_GRAPH_SPEC_TIME=1 RUST_LOG=info timeout 400 $bin run $M --greedy --no-think --max-tokens 200 \
    < turns.txt > "turns$rep-$b.out" 2> "turns$rep-$b.err"
done; done
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader > st-smi.log
echo DONE > st.done
