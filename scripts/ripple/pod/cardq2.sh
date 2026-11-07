#!/bin/bash
# cardq2.sh: after the tables — the new binary, the two q4tp rows a parallel check disturbed, then
# the drafter on the real file: plain, depth pinned at 7, the adaptive depth, code and prose prompts.
cd /root/rip
export XDG_RUNTIME_DIR=/tmp
timeout 7200 bash -c "until grep -q CARDQ_DONE logs/cardq.out; do sleep 15; done"
bash scripts/build.sh > logs/build.out 2>&1; tail -n 1 logs/build.out
MODES="standard" bash scripts/final.sh logs/final_q4tp_redo.log out/ripple-65b-q4kv8.cmf 4000 16000
for p in code prose; do
  bash scripts/dflash.sh plain $p 160
  bash scripts/dflash.sh k7 $p 160 CMF_GRAPH_SPEC_K=7 CMF_DFLASH_TIME=1
  bash scripts/dflash.sh auto $p 160 CMF_DFLASH_TIME=1
done
echo CARDQ2_DONE
