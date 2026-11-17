#!/bin/bash
# GLM-5.3-Flash on CUDA: the host runs mHC / KDA / DSA and routes, the card
# takes the routed experts (built into card slots from the mapped file) and
# the shared expert (`gpu::moe_row`). Texts against glm2's CPU answers, a
# 64-token bench and one after a 512-token prompt, the slots' hit lines.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
mkdir -p logs/glm3 out/glm3
bash scripts/build.sh fast > logs/glm3/build.log 2>&1
grep -q BUILD_FAST_DONE logs/glm3/build.log || { echo GLM3_BUILD_FAILED; tail -30 logs/glm3/build.log; exit 1; }
B=bin/cortiq-fast; f=cmf/big/glm-5.3-flash-q2tp.cmf
flock gpu.lock env CMF_GPU=cuda CMF_CUDA_DEBUG=1 timeout 3600 $B run $f --prompt "What is 2+2? Answer briefly." --no-think -n 16 --greedy > out/glm3/two.txt 2> logs/glm3/two.log
echo "cuda 2+2 exit $?: $(grep -v '^\[' out/glm3/two.txt | tail -n +5 | tr '\n' ' ' | cut -c1-200)"
sed -E 's/\x1b\[[0-9;]*m//g' logs/glm3/two.log | grep -iE "moe row|moe rows|declin|panick|error" | sed -E 's/^.*(INFO|WARN|ERROR) //' | sort | uniq -c | head -6 | cut -c1-200 | sed 's/^/      /'
P="Explain in two sentences why the sky is blue."
flock gpu.lock env CMF_GPU=cuda timeout 3600 $B run $f --prompt "$P" -n 24 --greedy 2>/dev/null | grep -v '^\[' > out/glm3/sky_cuda.txt
echo "cuda sky: $(tail -n +5 out/glm3/sky_cuda.txt | tr '\n' ' ' | cut -c1-240)"
if cmp -s out/glm3/sky_cuda.txt out/glm2/sky_cpu.txt; then echo "sky: CUDA text == CPU"; else echo "sky: CUDA text DIFFERS from CPU"; fi
flock gpu.lock env CMF_GPU=cuda CMF_CUDA_DEBUG=1 timeout 3600 $B bench $f --tokens 64 --ignore-eos --json > logs/glm3/d64.log 2>&1
echo "cuda 64: exit $? | $(grep -hoE '"(decode_tok_s_steady|prefill_tok_s|ttft_s)": [0-9.e+-]+' logs/glm3/d64.log | tr '\n' ' ')"
flock gpu.lock env CMF_GPU=cuda CMF_CUDA_DEBUG=1 timeout 3600 $B bench $f --tokens 64 --ctx 512 --ignore-eos --json > logs/glm3/p512.log 2>&1
echo "cuda 512+64: exit $? | $(grep -hoE '"(decode_tok_s_steady|prefill_tok_s|ttft_s)": [0-9.e+-]+' logs/glm3/p512.log | tr '\n' ' ')"
grep -hE "cuda moe rows" logs/glm3/*.log | tail -3 | cut -c1-200 | sed 's/^/      /'
echo GLM3_DONE
