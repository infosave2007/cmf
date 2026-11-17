#!/bin/bash
# GLM-5.3 on the CUDA hybrid's whole-token graph (KDA, NoPE MLA, DeepSeek-V4
# mHC, routed experts with their clamp): texts against the CPU, decode after
# a short prompt and after 512 (host stack 1.76 / 1.93 tok/s in glm5).
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
mkdir -p logs/glmg1 out/glmg1
bash scripts/build.sh fast > logs/glmg1/build.log 2>&1
grep -q BUILD_FAST_DONE logs/glmg1/build.log || { echo GLMG1_BUILD_FAILED; grep -E '^error|error:' -A8 logs/glmg1/build.log | head -60; exit 1; }
B=bin/cortiq-fast; f=cmf/big/glm-5.3-flash-q2tp.cmf
P="Explain in two sentences why the sky is blue."
for be in cuda; do
  flock gpu.lock env CMF_GPU=$be CMF_CUDA_DEBUG=1 RUST_LOG=info timeout 3600 $B run $f --prompt "What is 2+2? Answer briefly." --no-think -n 16 --greedy > out/glmg1/two_$be.txt 2> logs/glmg1/two_$be.err
  flock gpu.lock env CMF_GPU=$be CMF_CUDA_DEBUG=1 RUST_LOG=info timeout 3600 $B run $f --prompt "$P" -n 24 --greedy > out/glmg1/sky_$be.txt 2> logs/glmg1/sky_$be.err
  echo "$be: 2+2 $(grep -v '^\[' out/glmg1/two_$be.txt | tail -n +5 | tr '\n' ' ' | cut -c1-60) | sky $(grep -v '^\[' out/glmg1/sky_$be.txt | tail -n +5 | tr '\n' ' ' | cut -c1-200)"
  echo "   $(sed -E 's/\x1b\[[0-9;]*m//g' logs/glmg1/two_$be.err | grep -iE 'glm5_next|declin|hybrid' | sort | uniq -c | head -8 | tr '\n' ';' | cut -c1-600)"
done
echo "cpu sky (glm5): $(grep -v '^\[' out/glm5/sky_0.txt | tail -n +5 | tr '\n' ' ' | cut -c1-200)"
flock gpu.lock env CMF_GPU=cuda RUST_LOG=info timeout 3600 $B bench $f --tokens 64 --ctx 512 --ignore-eos --json > logs/glmg1/p512.log 2>&1
echo "cuda 512+64: exit $? | $(grep -hoE '"(decode_tok_s_steady|prefill_tok_s|ttft_s)": [0-9.e+-]+' logs/glmg1/p512.log | tr '\n' ' ')"
echo GLMG1_DONE
