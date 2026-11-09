#!/bin/bash
# chainG8.sh: after chainG7 and the cortiq.xt6 build (matrix-unit attend with grouped blocks and
# a lazy rescale, tile-aligned sub-runs, one partial buffer a call): coop512 with the matrix-unit
# tile against the scalar tile at 4096 and 16000 on a quiet card, two group widths, the stage
# profile at 16000, greedy-text parity on 9 000 and 60 000 characters plus a forced-rescale
# parity (lazy = 0). Output logs/gab_xt6.txt, logs/g8_prof_16000.log, logs/parity_xt6_*.out,
# chainG8.done.
cd /root/spark26; export XDG_RUNTIME_DIR=/tmp CMF_MAX_SEQ=140000
until [ -f logs/chainG7.done ] && [ -f logs/build_cortiq.xt6.done ]; do sleep 5; done
[ -x cortiq.xt6 ] || { echo "no cortiq.xt6"; touch logs/chainG8.done; exit 1; }
F=hf/cmf/spark-x2.6-4b-q4tp.cmf; B=./cortiq.xt6
G="CMF_DENSE_GRAPH_PREFILL=1,CMF_BATCH_COOP=1,CMF_BATCH_K=512"
BIN=$B VARIANTS="coop512:$G scalar512:$G,CMF_ATTEND_XC=0 bpw8:$G,CMF_ATTEND_BPW=8 bpw64:$G,CMF_ATTEND_BPW=64" bash quiet.sh bash graph_ab.sh xt6 4096 16000
bash quiet.sh bash -c "CMF_DENSE_GRAPH_PREFILL=1 CMF_BATCH_COOP=1 CMF_BATCH_K=512 CMF_GPU_TS=1 CMF_BATCH_TS=1 CMF_BATCH_KERNEL_TS=1 $B bench $F --core --ignore-eos --tokens 4 --ctx 16000 > logs/g8_prof_16000.log 2>&1"
BIN=$B bash graph_parity.sh $F 9000 xt6_9k > logs/parity_xt6_9k.out 2>&1
BIN=$B bash graph_parity.sh $F 60000 xt6_60k > logs/parity_xt6_60k.out 2>&1
CMF_ATTEND_XC_LAZY=0 BIN=$B bash graph_parity.sh $F 9000 xt6_lazy0_9k > logs/parity_xt6_lazy0_9k.out 2>&1
touch logs/chainG8.done
