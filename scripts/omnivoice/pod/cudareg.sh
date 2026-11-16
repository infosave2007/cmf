#!/bin/bash
# CUDA after the hybrid's refactor (pair_dev / one_dev, the placement block):
# LFM3-1.7B (on the card) 4k prompt and greedy text CUDA == CPU, and
# Ornith-1.5-122B q2tp-v2 (mapped host memory) decode against 0.8.99's
# 38.5-38.9 tok/s, its text against the CPU. After lfm6.sh.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
mkdir -p logs/cudareg out/cudareg cmf/big
end=$((SECONDS + 14400))
until grep -qE "LFM6_DONE" logs/lfm6.out 2>/dev/null || [ $SECONDS -gt $end ]; do sleep 30; done
B=bin/cortiq-fast
o=cmf/big/ornith-1.5-122b-q2tp-v2.cmf
( [ -s $o ] || timeout 3600 venv/bin/hf download infosave/Ornith-1.5-122B-cmf ornith-1.5-122b-q2tp-v2.cmf --local-dir cmf/big > logs/cudareg/orn_dl.log 2>&1 ) &
dl=$!
f=cmf/dl/lfm3-1.7b-q4tp.cmf
flock gpu.lock env CMF_GPU=cuda timeout 1800 $B bench $f --tokens 32 --ctx 4096 --json > logs/cudareg/lfm_4k.log 2>&1
echo "lfm3 cuda 4k: exit $? | $(grep -hoE '"(decode_tok_s_steady|prefill_tok_s|ttft_s)": [0-9.e+-]+' logs/cudareg/lfm_4k.log | tr '\n' ' ')"
for be in cuda 0; do
  flock gpu.lock env CMF_GPU=$be timeout 1800 $B run $f --prompt-file prompts/long.txt -n 32 --greedy 2>/dev/null | grep -v '^\[' | grep -v '^Loading' > out/cudareg/lfm_$be.txt
done
if cmp -s out/cudareg/lfm_cuda.txt out/cudareg/lfm_0.txt; then echo "lfm3: CUDA text == CPU"; else echo "lfm3: CUDA text DIFFERS from CPU"; fi
wait $dl
if [ -s $o ]; then
  for k in 1 2; do
    flock gpu.lock env CMF_GPU=cuda CMF_CUDA_DEBUG=1 timeout 2400 $B bench $o --tokens 64 --ignore-eos --json > logs/cudareg/orn_d_$k.log 2>&1
    echo "ornith q2tp-v2 cuda decode run$k: exit $? | $(grep -hoE '"(decode_tok_s_steady|prefill_tok_s|ttft_s)": [0-9.e+-]+' logs/cudareg/orn_d_$k.log | tr '\n' ' ')"
  done
  P="Explain in two sentences why the sky is blue."
  for be in cuda 0; do
    flock gpu.lock env CMF_GPU=$be timeout 2400 $B run $o --prompt "$P" -n 24 --greedy 2>/dev/null | grep -v '^\[' > out/cudareg/orn_$be.txt
  done
  if cmp -s out/cudareg/orn_cuda.txt out/cudareg/orn_0.txt; then echo "ornith: CUDA text == CPU"; else echo "ornith: CUDA text DIFFERS from CPU"; diff out/cudareg/orn_cuda.txt out/cudareg/orn_0.txt | head -4 | cut -c1-160; fi
fi
echo CUDAREG_DONE
