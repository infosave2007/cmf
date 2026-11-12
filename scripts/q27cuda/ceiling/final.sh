#!/bin/bash
# The ceiling branch's measurements on the 16 GB card (RTX 2000 Ada), release build:
#  bench --core rows 1k..64k (CUDA turbo with/without drafts, Vulkan turbo, CUDA standard file),
#  real answers (summaries of 30 000 / 100 000 characters), a chat that starts past the seal.
#   final.sh <bin> <out-dir>     (SKIP_VK=1 SKIP_STD=1 SKIP_REAL=1 SKIP_CHAT=1)
set -u
bin=$1; o=$2
cd /root/cuda
mkdir -p $o
Q=/root/models/qwen38-27b-q4tp.cmf
T=/root/models/qwen38-27b-turbo.cmf
B=src/scripts/cuda/pod/bench_one.sh
CTXS=${CTXS:-1000 4000 16000 32000 64000}
for c in $CTXS; do
  CMF_MAX_SEQ=70000 bash $B f_tur_cu_$c $bin $T $c CMF_GPU=cuda >> $o/bench.txt 2>&1
  CMF_MAX_SEQ=70000 bash $B f_tur_cu_plain_$c $bin $T $c CMF_GPU=cuda CMF_GRAPH_SPEC=0 >> $o/bench.txt 2>&1
done
[ -z "${SKIP_VK:-}" ] && for c in $CTXS; do
  CMF_MAX_SEQ=70000 timeout 2400 bash $B f_tur_vk_$c $bin $T $c X=1 >> $o/bench.txt 2>&1
done
[ -z "${SKIP_STD:-}" ] && [ -f $Q ] && for c in 1000 4000 16000; do
  CMF_MAX_SEQ=70000 timeout 1500 bash $B f_std_cu_$c $bin $Q $c CMF_GPU=cuda >> $o/bench.txt 2>&1
done
if [ -z "${SKIP_REAL:-}" ]; then
  for chars in 30000 100000; do
    python3 - $chars > $o/real_$chars.txt <<PY
import sys
n=int(sys.argv[1]); text=open("data/wiki.test.raw",encoding="utf-8").read().replace("\r","")
print(text[5000:5000+n]+"\n\nIn one paragraph, what is the last article above about?")
PY
    for e in "CMF_GPU=cuda" "CMF_GPU=cuda CMF_GRAPH_SPEC=0" "X=1"; do
      tag=$(echo $e | tr ' =' '__')
      env $e CMF_GPU_PROBE=0 XDG_RUNTIME_DIR=/tmp CMF_MAX_SEQ=70000 timeout 2400 $bin run $T --prompt-file $o/real_$chars.txt --greedy --no-think --max-tokens 400 > $o/real_${chars}_$tag.out 2> $o/real_${chars}_$tag.err
      echo "real $chars [$e]: $(tail -3 $o/real_${chars}_$tag.out | tr '\n' ' ' | cut -c1-400)" >> $o/real.txt
    done
  done
fi
if [ -z "${SKIP_CHAT:-}" ]; then
  python3 - > $o/chat_turns.txt <<PY
text=open("data/wiki.test.raw",encoding="utf-8").read().replace("\r","").replace("\n"," ")
print(text[5000:5000+40000]+" In one sentence: what is the last article above about?")
print("Who directed its music video?")
print("And at which number did it peak in the US?")
PY
  for e in "CMF_GPU=cuda" "X=1"; do
    tag=$(echo $e | tr ' =' '__')
    env $e CMF_GPU_PROBE=0 XDG_RUNTIME_DIR=/tmp CMF_MAX_SEQ=70000 timeout 2400 $bin run $T --greedy --no-think -n 64 < $o/chat_turns.txt > $o/chat_$tag.out 2> $o/chat_$tag.err
    echo "chat [$e]: $(grep -v '^$' $o/chat_$tag.out | tail -12 | tr '\n' ' ' | cut -c1-900)" >> $o/chat.txt
  done
fi
echo FINAL_DONE > $o/done
