#!/bin/bash
# chainG7.sh: after chainG6 and the cortiq.xt5 build (coop attend + dequantize-once f16 GEMMs):
# coop512 with and without the f16 plane at 4096 and 16000 on a quiet card, the stage profile
# at 16000, greedy-text parity on 9 000 characters. Output logs/gab_xt5.txt,
# logs/g7_prof_16000.log, logs/parity_xt5_9k.out, chainG7.done. (No perplexity: `ppl` scores a
# window one position at a time through the token graph, ~16 min a run.)
cd /root/spark26; export XDG_RUNTIME_DIR=/tmp CMF_MAX_SEQ=140000
until [ -f logs/chainG6.done ] && [ -f logs/build_cortiq.xt5.done ]; do sleep 5; done
[ -x cortiq.xt5 ] || { echo "no cortiq.xt5"; touch logs/chainG7.done; exit 1; }
F=hf/cmf/spark-x2.6-4b-q4tp.cmf; B=./cortiq.xt5
BIN=$B VARIANTS="coop512:CMF_DENSE_GRAPH_PREFILL=1,CMF_BATCH_COOP=1,CMF_BATCH_K=512 coop512nof16:CMF_DENSE_GRAPH_PREFILL=1,CMF_BATCH_COOP=1,CMF_BATCH_K=512,CMF_BATCH_F16=0 coop256:CMF_DENSE_GRAPH_PREFILL=1,CMF_BATCH_COOP=1,CMF_BATCH_K=256" bash quiet.sh bash graph_ab.sh xt5 4096 16000
bash quiet.sh bash -c "CMF_DENSE_GRAPH_PREFILL=1 CMF_BATCH_COOP=1 CMF_BATCH_K=512 CMF_GPU_TS=1 CMF_BATCH_TS=1 CMF_BATCH_KERNEL_TS=1 $B bench $F --core --ignore-eos --tokens 4 --ctx 16000 > logs/g7_prof_16000.log 2>&1"
BIN=$B bash graph_parity.sh $F 9000 xt5_9k > logs/parity_xt5_9k.out 2>&1
touch logs/chainG7.done
