#!/bin/bash
# chain1.sh: exact-row fit over 40 prompts, both turbo files, then standard vs turbo at depth (q4tp).
cd /root/spark26; export XDG_RUNTIME_DIR=/tmp
bash fit.sh hf/cmf/spark-x2.6-4b-q4tp.cmf 40 21000 > logs/fit40.log 2>&1
cp keep_spark26.bin cmf/scripts/spark26/pod/keep_spark26.bin
./cortiq far-turbo hf/cmf/spark-x2.6-4b-q4tp.cmf out/spark-x2.6-4b-q4tp-turbo.cmf --exact-weights keep_spark26.bin > logs/ft_q4.log 2>&1
./cortiq far-turbo hf/cmf/spark-x2.6-4b-q8_2f.cmf out/spark-x2.6-4b-q8_2f-turbo.cmf --exact-weights keep_spark26.bin > logs/ft_q8.log 2>&1
bash ctxbench.sh hf/cmf/spark-x2.6-4b-q4tp.cmf std_q4 1000 16000 32000 64000
bash ctxbench.sh out/spark-x2.6-4b-q4tp-turbo.cmf turbo_q4 1000 16000 32000 64000
echo CHAIN1_DONE > logs/chain1.done
