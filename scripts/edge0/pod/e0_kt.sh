#!/bin/bash
# e0_kt.sh BIN TAG "ENV..." ...: kernel times of speculation round 40 of a short prompt (CMF_QWEN_NCU), per variant
BIN=$1; shift
cd /root/e0
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH XDG_RUNTIME_DIR=/tmp
while [ $# -ge 2 ]; do
  TAG=$1; ENVS=$2; shift 2
  env CMF_GPU=cuda CMF_QWEN_NCU=40 $ENVS $BIN run out/edge0-145b.cmf -p "Write a Python class for an LRU cache with get and put, with type hints and docstrings." -n 200 --no-think --temperature 0 > logs/kt_$TAG.out 2> logs/kt_$TAG.err
  echo "== $TAG [$ENVS] $(grep -ho "decode [0-9.]* tok/s" logs/kt_$TAG.out logs/kt_$TAG.err | head -1)"
  grep "kernel times of one round\|qwen4 cuda: kernel " logs/kt_$TAG.err | head -16 | sed "s/qwen4 cuda: //"
done
echo KT-DONE
