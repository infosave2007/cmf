#!/bin/bash
# chain3.sh: turbo q4tp at depth, then q8_2f standard vs turbo, on the fixed binary.
cd /root/spark26; export XDG_RUNTIME_DIR=/tmp
bash ctxbench.sh out/spark-x2.6-4b-q4tp-turbo.cmf turbo_q4 1000 16000 32000 64000
bash ctxbench.sh hf/cmf/spark-x2.6-4b-q8_2f.cmf std_q8 1000 16000 32000 64000
bash ctxbench.sh out/spark-x2.6-4b-q8_2f-turbo.cmf turbo_q8 1000 16000 32000 64000
echo CHAIN3_DONE > logs/chain3.done
