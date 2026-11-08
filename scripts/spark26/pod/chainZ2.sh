#!/bin/bash
# chainZ2.sh: after chainZ: the card's NVIDIA rows on cortiq.zi (standard and turbo q4tp at
# 1000/16000/32000/64000, `bench --core`, 60 tokens, peak VRAM and host RSS sampled), then the
# turbo stage profile at 16000 with the host prefill profile — under quiet.sh (recall paused).
# Output logs/cb_zi_*.txt, logs/gZ_turbo_prof_16000.log, chainZ2.done.
cd /root/spark26; export XDG_RUNTIME_DIR=/tmp CMF_MAX_SEQ=140000
until [ -f logs/chainZ.done ]; do sleep 20; done
[ -x cortiq.zi ] || { echo "no cortiq.zi"; touch logs/chainZ2.done; exit 1; }
B=/root/spark26/cortiq.zi; F=hf/cmf/spark-x2.6-4b-q4tp.cmf
BIN=$B bash quiet.sh bash ctxbench.sh $F zi_std_q4 1000 16000 32000 64000
BIN=$B bash quiet.sh bash ctxbench.sh out/spark-x2.6-4b-q4tp-turbo.cmf zi_turbo_q4 1000 16000 32000 64000
bash quiet.sh bash -c "CMF_GPU_TS=1 CMF_BATCH_TS=1 CMF_BATCH_KERNEL_TS=1 CMF_PREFILL_PROF=1 $B bench out/spark-x2.6-4b-q4tp-turbo.cmf --core --ignore-eos --tokens 4 --ctx 16000 > logs/gZ_turbo_prof_16000.log 2>&1"
touch logs/chainZ2.done
