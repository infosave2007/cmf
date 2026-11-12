#!/bin/bash
# Planted facts on the CUDA ceiling path (16 GB card): 60 000 and 100 000 characters, eight trials each.
cd /root/cuda
bin=${BIN:-bin/cortiq-cuda-fast}; T=/root/models/qwen38-27b-turbo.cmf; o=out/pk; mkdir -p $o
for ch in ${CHARS:-60000 100000}; do
  PASSKEY_FILE=$o/prompt.txt timeout 6000 python3 src/scripts/spark26/passkey_file.py $bin $T data/wiki.test.raw $ch 8 CMF_GPU=cuda CMF_GPU_PROBE=0 CMF_MAX_SEQ=70000 ${EXTRA:-} >> $o/pk_$ch.txt 2>&1
  tail -1 $o/pk_$ch.txt >> $o/summary.txt
done
echo PK_DONE >> $o/summary.txt
