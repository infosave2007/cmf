#!/bin/bash
# Dory's turbo file with a fixed ceiling on CUDA against the base binary (Dory's
# 0.8.70 path, the prompt read whole) and the wgpu (Vulkan) path, release builds
# on one card:
#  - `cortiq bench --core` at 1k..64k tokens: first token, decode, the card's peak
#    (`scripts/cuda/pod/bench_one.sh`), the three binaries taking turns a length;
#  - the 12 real prompts (`scripts/dory/suite.py`, Vulkan's texts the baseline);
#  - summaries of 30 000 and 100 000 characters of wikitext (greedy, 200 tokens);
#  - facts planted in 60 000 and 100 000 characters, eight trials each;
#  - wikitext-2 perplexity, 12 windows of 512 tokens.
#   ceiling.sh <new-bin> <base-bin> <out-dir>
#   (SKIP_BENCH=1 SKIP_BASE=1 SKIP_VK=1 SKIP_REAL=1 SKIP_SUM=1 SKIP_PK=1 SKIP_PPL=1, CTXS="1000 ...")
set -u
cd /root/cuda
new=$(readlink -f $1); base=$(readlink -f $2); o=$(readlink -m $3)
mkdir -p $o
T=/root/models/dory/dory-turbo.cmf
B=src/scripts/cuda/pod/bench_one.sh
CTXS=${CTXS:-1000 4000 16000 32000 64000}
if [ -z "${SKIP_BENCH:-}" ]; then
  for c in $CTXS; do
    CMF_MAX_SEQ=70000 bash $B dc_new_$c $new $T $c CMF_GPU=cuda >> $o/bench.txt 2>&1
    [ -z "${SKIP_BASE:-}" ] && CMF_MAX_SEQ=70000 bash $B dc_base_$c $base $T $c CMF_GPU=cuda >> $o/bench.txt 2>&1
    [ -z "${SKIP_VK:-}" ] && CMF_MAX_SEQ=70000 timeout 3000 bash $B dc_vk_$c $new $T $c X=1 >> $o/bench.txt 2>&1
  done
  cp out/dc_*.json out/dc_*.vram out/dc_*.vram_trace $o/ 2>/dev/null
fi
if [ -z "${SKIP_REAL:-}" ]; then
  mkdir -p $o/real
  ( cd $o/real
    [ -z "${SKIP_VK:-}" ] && env X=1 CMF_GPU_PROBE=0 python3 /root/cuda/src/scripts/dory/suite.py $new $T turbo-vk
    env CMF_GPU=cuda CMF_GPU_PROBE=0 python3 /root/cuda/src/scripts/dory/suite.py $new $T turbo-cuda-new
    [ -z "${SKIP_BASE:-}" ] && env CMF_GPU=cuda CMF_GPU_PROBE=0 python3 /root/cuda/src/scripts/dory/suite.py $base $T turbo-cuda-base
  ) >> $o/real.txt 2>&1
fi
if [ -z "${SKIP_SUM:-}" ]; then
  for chars in 30000 100000; do
    python3 - $chars > $o/sum_$chars.txt <<PY
import sys
n=int(sys.argv[1]); text=open("data/wiki.test.raw",encoding="utf-8").read().replace("\r","")
print(text[5000:5000+n]+"\n\nIn one paragraph, what is the last article above about?")
PY
    for arm in new base vk; do
      e="CMF_GPU=cuda"; b=$new
      [ $arm = base ] && { [ -n "${SKIP_BASE:-}" ] && continue; b=$base; }
      [ $arm = vk ] && { [ -n "${SKIP_VK:-}" ] && continue; e="X=1"; }
      env $e CMF_GPU_PROBE=0 XDG_RUNTIME_DIR=/tmp CMF_MAX_SEQ=70000 timeout 2400 $b run $T --prompt-file $o/sum_$chars.txt --greedy --no-think --max-tokens 200 > $o/sum_${chars}_$arm.out 2> $o/sum_${chars}_$arm.err
      echo "sum $chars $arm: $(grep -h 'tokens · first' $o/sum_${chars}_$arm.out | cut -c1-90) text $(grep -v -E 'tokens · first token|^\[' $o/sum_${chars}_$arm.out | md5sum | cut -c1-8)" >> $o/sum.txt
    done
  done
fi
if [ -z "${SKIP_PK:-}" ]; then
  for ch in 60000 100000; do
    for arm in new base; do
      b=$new; [ $arm = base ] && { [ -n "${SKIP_BASE:-}" ] && continue; b=$base; }
      echo "== $arm $ch" >> $o/passkey.txt
      PASSKEY_FILE=$o/pk_${arm}_$ch.txt timeout 4000 python3 src/scripts/spark26/passkey_file.py $b $T data/wiki.test.raw $ch 8 CMF_GPU=cuda CMF_GPU_PROBE=0 CMF_MAX_SEQ=70000 >> $o/passkey.txt 2>&1
    done
  done
fi
if [ -z "${SKIP_PPL:-}" ]; then
  w=$o/wiki.txt; [ -f $w ] || head -c 400000 data/wiki.test.raw > $w
  for arm in new base vk; do
    e="CMF_GPU=cuda"; b=$new
    [ $arm = base ] && { [ -n "${SKIP_BASE:-}" ] && continue; b=$base; }
    [ $arm = vk ] && { [ -n "${SKIP_VK:-}" ] && continue; e="X=1"; }
    env $e CMF_GPU_PROBE=0 XDG_RUNTIME_DIR=/tmp timeout 1800 $b ppl $T -f $w --windows 12 --window-len 512 2>&1 | grep -E "PPL" | sed "s/^/turbo $arm /" >> $o/ppl.txt
  done
fi
echo CEILING_DONE > $o/done
