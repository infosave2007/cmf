#!/bin/bash
# chainF.sh: the card's NVIDIA rows on the final binary (cortiq.fin), after chainG12: standard and
# turbo q4tp at 1000/16000/32000/64000 (`bench --core`, 60 tokens, peak VRAM and host RSS sampled),
# greedy-text parity on 9 000 and 60 000 characters, the stage profile at 16000 — all on a quiet card
# (the recall chain paused). Output logs/cb_fin_*.txt, logs/parity_fin_*.out, logs/gF_prof_16000.log,
# chainF.done.
cd /root/spark26; export XDG_RUNTIME_DIR=/tmp CMF_MAX_SEQ=140000
until [ -f logs/build_cortiq.fin.done ] && [ -f logs/chainG12.done ]; do sleep 5; done
[ -x cortiq.fin ] || { echo "no cortiq.fin"; touch logs/chainF.done; exit 1; }
B=/root/spark26/cortiq.fin; F=hf/cmf/spark-x2.6-4b-q4tp.cmf
BIN=$B bash quiet.sh bash ctxbench.sh $F fin_std_q4 1000 16000 32000 64000
BIN=$B bash quiet.sh bash ctxbench.sh out/spark-x2.6-4b-q4tp-turbo.cmf fin_turbo_q4 1000 16000 32000 64000
BIN=$B bash quiet.sh bash graph_parity.sh $F 9000 fin_9k > logs/parity_fin_9k.out 2>&1
BIN=$B bash quiet.sh bash graph_parity.sh $F 60000 fin_60k > logs/parity_fin_60k.out 2>&1
bash quiet.sh bash -c "CMF_GPU_TS=1 CMF_BATCH_TS=1 CMF_BATCH_KERNEL_TS=1 $B bench $F --core --ignore-eos --tokens 4 --ctx 16000 > logs/gF_prof_16000.log 2>&1"
touch logs/chainF.done
