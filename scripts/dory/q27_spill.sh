#!/bin/bash
# Qwen3.8-27B with a ceiling: a first seal at 8192 rows, rows past it in RAM or in files.
cd /root/q
export XDG_RUNTIME_DIR=/tmp CMF_MAX_SEQ=70000
B=$1; L=$2
: > $L
V=src/scripts/dory/vram_peak.sh
M=m/qwen38-27b-q4tp.cmf
X="$B bench $M --core --tokens 40 --ignore-eos --ctx"
rm -rf /root/q/sp; mkdir -p /root/q/sp
export CMF_O1=all CMF_O1_GPU=1 CMF_O1_WINDOW=1800 CMF_O1_SINK=4 CMF_O1_EXACT=keep27.bin:248 CMF_O1_PREFIX_MAX=8192
CMF_O1_STAGES=0 $V "old attend, RAM 12000" $L /root/q/sp -- $X 12000
for c in 12000 32000 64000; do
  $V "far, RAM $c" $L /root/q/sp -- $X $c
  CMF_O1_SPILL=/root/q/sp $V "far, SSD $c" $L /root/q/sp -- $X $c
done
venv/bin/python src/scripts/dory/passkey2.py $B $M wiki_test.txt 100000 8 CMF_O1_SPILL=/root/q/sp 2>&1 | cut -c1-170 | tee -a $L
echo DONE >> $L
