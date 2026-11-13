#!/bin/bash
# smoke test of a converted Edge0 CMF: CPU reference, then CUDA; same prompt, greedy
M=${1:-/root/e0/out/edge0-145b.cmf}
TAG=${2:-145}
C=/root/e0/cmf/target/release/cortiq
cd /root/e0
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH XDG_RUNTIME_DIR=/tmp
P="Write a Python function that checks whether a string is a palindrome, ignoring case and punctuation. Code only."
CMF_GPU=0 CMF_QWEN_MTP=0 timeout 2400 $C run $M --prompt "$P" --greedy --no-think -n 64 > logs/smoke_cpu_$TAG.log 2>&1
echo "cpu rc=$?"
CMF_GPU=cuda CMF_QWEN_MTP=0 CMF_QWEN_PROF=1 timeout 2400 $C run $M --prompt "$P" --greedy --no-think -n 64 > logs/smoke_cuda_$TAG.log 2>&1
echo "cuda rc=$?"
echo SMOKE-DONE
