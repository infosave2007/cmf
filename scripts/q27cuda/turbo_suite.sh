#!/bin/bash
# Qwen3.8-27B Turbo on CUDA, the turbo page's measurements on an RTX 3090:
# `cortiq bench --core` with the card's peak VRAM after prompts of 1 000 to
# 64 000 tokens (the standard file and the turbo one on CUDA, the turbo one
# over Vulkan on the same card), real answers (a summary of 30 000 and 100 000
# characters of Wikipedia, with and without drafts) and planted facts.
#   turbo_suite.sh <bin> <out-dir>      (SKIP_VK=1, SKIP_PK=1: without the Vulkan runs, the facts)
set -u
bin=$1; o=$2
cd /root/cuda
mkdir -p $o
Q=/root/models/qwen38-27b-q4tp.cmf
T=/root/models/qwen38-27b-turbo.cmf
B=src/scripts/cuda/pod/bench_one.sh
for c in 1000 4000 16000 32000 64000; do
  CMF_MAX_SEQ=70000 bash $B qt_std_cuda_$c $bin $Q $c CMF_GPU=cuda >> $o/bench.txt 2>&1
  CMF_MAX_SEQ=70000 bash $B qt_tur_cuda_$c $bin $T $c CMF_GPU=cuda >> $o/bench.txt 2>&1
  CMF_MAX_SEQ=70000 bash $B qt_tur_cuda_plain_$c $bin $T $c CMF_GPU=cuda CMF_GRAPH_SPEC=0 >> $o/bench.txt 2>&1
done
[ -z "${SKIP_VK:-}" ] && for c in 1000 4000 16000 32000 64000; do
  CMF_MAX_SEQ=70000 bash $B qt_tur_vk_$c $bin $T $c X=1 >> $o/bench.txt 2>&1
done
for chars in 30000 100000; do
  python3 - $chars > $o/real_$chars.txt <<PY
import sys
n=int(sys.argv[1]); text=open("data/wiki.test.raw",encoding="utf-8").read().replace("\r","")
print(text[5000:5000+n]+"\n\nIn one paragraph, what is the last article above about?")
PY
  for f in $Q $T; do
    for e in "CMF_GPU=cuda" "CMF_GPU=cuda CMF_GRAPH_SPEC=0"; do
      r=$(env $e CMF_GPU_PROBE=0 XDG_RUNTIME_DIR=/tmp CMF_MAX_SEQ=70000 timeout 1200 $bin run $f --prompt-file $o/real_$chars.txt --greedy --no-think --max-tokens 400 2>/dev/null | tail -3 | tr '\n' ' ' | cut -c1-240)
      echo "real $chars $(basename $f) [$e]: $r" >> $o/real.txt
    done
  done
done
[ -z "${SKIP_PK:-}" ] && PASSKEY_FILE=$o/pk.txt timeout 3000 python3 src/scripts/spark26/passkey_file.py $bin $T data/wiki.test.raw 100000 8 CMF_GPU=cuda CMF_GPU_PROBE=0 CMF_MAX_SEQ=70000 >> $o/passkey.txt 2>&1
echo TS_DONE > $o/done
