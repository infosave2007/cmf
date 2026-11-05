#!/bin/bash
# GSM8K, greedy, direct answers, 60 questions: what the modes cost in right answers. gsm_all.sh BIN LOG
cd /root/dory
B=$1; L=$2
: > $L
Y=/root/dory/venv/bin/python
Q=out/dory-q4tp-v2.cmf
T=out/dory-q4tp-v2-turbo.cmf
G=gsm/main/test-00000-of-00001.parquet
E() { $Y scripts_gsm_eval.py $B "$@" | tee -a $L; }
cp src/scripts/dory/gsm_eval.py scripts_gsm_eval.py
E $Q $G 60 standard-k2
E $T $G 60 turbo
E $Q $G 60 k1 CMF_DORY_LOOPS=1
E $Q $G 60 conf.7 CMF_DORY_KV=last_loop CMF_DORY_GATE=conf:0.7
E $Q $G 60 k4 CMF_DORY_LOOPS=4
E out/dory-q8_2f.cmf $G 60 q8-k2
echo DONE >> $L
