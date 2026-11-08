#!/bin/bash
# chainG1.sh: after the walk/graph/coop A/B, the batched graph's stage profile at 4096 tokens
# (coop and scalar GEMMs) on a quiet card, then greedy-text parity of walk/graph/coop on
# 9 000- and 60 000-character prompts. Output logs/g_prof_*.log, logs/parity_*.out, chainG1.done.
cd /root/spark26; export XDG_RUNTIME_DIR=/tmp CMF_MAX_SEQ=140000
until grep -q GAB_DONE logs/gab_q1.txt 2>/dev/null; do sleep 5; done
F=hf/cmf/spark-x2.6-4b-q4tp.cmf
bash quiet.sh bash -c "
CMF_DENSE_GRAPH_PREFILL=1 CMF_BATCH_COOP=1 CMF_BATCH_TS=1 CMF_BATCH_KERNEL_TS=1 ./cortiq.graph bench $F --core --ignore-eos --tokens 4 --ctx 4096 > logs/g_prof_coop.log 2>&1
CMF_DENSE_GRAPH_PREFILL=1 CMF_BATCH_TS=1 CMF_BATCH_KERNEL_TS=1 ./cortiq.graph bench $F --core --ignore-eos --tokens 4 --ctx 4096 > logs/g_prof_scalar.log 2>&1
"
bash graph_parity.sh $F 9000 p9k > logs/parity_p9k.out 2>&1
bash graph_parity.sh $F 60000 p60k > logs/parity_p60k.out 2>&1
touch logs/chainG1.done
