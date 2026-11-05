#!/bin/bash
# Qwen3.8-27B: exact against the far field on the graph, speed and facts. q27_base.sh BIN LOG
cd /root/q
export XDG_RUNTIME_DIR=/tmp CMF_MAX_SEQ=70000
B=$1; L=$2
: > $L
M=m/qwen38-27b-q4tp.cmf
C() { r=$(timeout 1200 $B bench $M --core --tokens 40 --ignore-eos --ctx $1 2>&1 | grep -E "Prompt|Decode|device KV|panicked|declined" | sed -e 's/\x1b\[[0-9;]*m//g' -e 's/  */ /g' | tr '\n' ' ' | cut -c1-230); echo "$2 ctx $1: $r" | tee -a $L; }
F="CMF_O1=all CMF_O1_GPU=1 CMF_O1_WINDOW=1800 CMF_O1_SINK=4"
for c in 1000 4000 16000 32000; do
  C $c exact
  export CMF_O1=all CMF_O1_GPU=1 CMF_O1_WINDOW=1800 CMF_O1_SINK=4
  C $c far1800
  unset CMF_O1 CMF_O1_GPU CMF_O1_WINDOW CMF_O1_SINK
done
venv/bin/python src/scripts/dory/passkey2.py $B $M wiki_test.txt 100000 4 2>&1 | cut -c1-160 | tee -a $L
venv/bin/python src/scripts/dory/passkey2.py $B $M wiki_test.txt 100000 4 $F 2>&1 | cut -c1-160 | tee -a $L
echo DONE >> $L
