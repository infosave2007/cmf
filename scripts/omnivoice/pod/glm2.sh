#!/bin/bash
# GLM-5.3-Flash q2tp with the glm5_next runtime ported from embryo-o1: the
# build, then the card's command and the sky prompt on the CPU (greedy), the
# speed of a short decode (bench on the CPU), and whether CUDA's run gives
# the same text (the stack is the host's; the GPU only takes matvecs).
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
mkdir -p logs/glm2 out/glm2 cmf/big
f=cmf/big/glm-5.3-flash-q2tp.cmf
( [ -s $f ] || timeout 7200 venv/bin/hf download infosave/GLM-5.3-Flash-cmf glm-5.3-flash-q2tp.cmf --local-dir cmf/big > logs/glm2/dl.log 2>&1 ) &
dl=$!
bash scripts/build.sh fast > logs/glm2/build.log 2>&1
grep -q BUILD_FAST_DONE logs/glm2/build.log || { echo GLM2_BUILD_FAILED; tail -30 logs/glm2/build.log; exit 1; }
wait $dl
[ -s $f ] || { echo GLM2_DOWNLOAD_FAILED; exit 1; }
B=bin/cortiq-fast
flock gpu.lock env CMF_GPU=0 RUST_LOG=info timeout 3600 $B run $f --prompt "What is 2+2? Answer briefly." --no-think -n 16 --greedy > out/glm2/two.txt 2> logs/glm2/two.log
echo "cpu 2+2 exit $?: $(grep -v '^\[' out/glm2/two.txt | tail -n +5 | tr '\n' ' ' | cut -c1-200)"
sed -E 's/\x1b\[[0-9;]*m//g' logs/glm2/two.log | grep -iE "Pipeline loaded|glm|error|panick" | cut -c1-200 | head -5 | sed 's/^/      /'
P="Explain in two sentences why the sky is blue."
flock gpu.lock env CMF_GPU=0 timeout 3600 $B run $f --prompt "$P" -n 24 --greedy 2>/dev/null | grep -v '^\[' > out/glm2/sky_cpu.txt
echo "cpu sky: $(tail -n +5 out/glm2/sky_cpu.txt | tr '\n' ' ' | cut -c1-240)"
flock gpu.lock env CMF_GPU=0 timeout 3600 $B bench $f --tokens 16 --json > logs/glm2/bench_cpu.log 2>&1
echo "cpu bench: exit $? | $(grep -hoE '"(decode_tok_s_steady|prefill_tok_s|ttft_s)": [0-9.e+-]+' logs/glm2/bench_cpu.log | tr '\n' ' ')"
flock gpu.lock env CMF_GPU=cuda timeout 3600 $B run $f --prompt "$P" -n 24 --greedy 2>/dev/null | grep -v '^\[' > out/glm2/sky_cuda.txt
if cmp -s out/glm2/sky_cuda.txt out/glm2/sky_cpu.txt; then echo "CUDA text == CPU"; else echo "CUDA text DIFFERS: $(tail -n +5 out/glm2/sky_cuda.txt | tr '\n' ' ' | cut -c1-200)"; fi
echo GLM2_DONE
