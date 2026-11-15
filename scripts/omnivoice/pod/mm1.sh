#!/bin/bash
# After glm1.sh: the build with the wgpu low-rank MLA query and the CUDA
# hybrid's SwiGLU-OAI (MiniMax M3), MiniMax-M3.1 fetched beside the build,
# then GLM-5.3 on Vulkan (decode, 4k, the declines, greedy text against the
# CPU text glm1 wrote), then MiniMax-M3.1 on CUDA and Vulkan the same way.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
mkdir -p logs/mm1 out/mm1
end=$((SECONDS + 10800))
until grep -qE "GLM1_DONE|GLM1_BUILD_FAILED|GLM1_DOWNLOAD_FAILED" logs/glm1.out 2>/dev/null || [ $SECONDS -gt $end ]; do sleep 30; done
m=cmf/big/minimax-m3.1-flash-q2tp-mix.cmf
( [ -s $m ] || timeout 7200 venv/bin/hf download infosave/MiniMax-M3.1-Flash-cmf minimax-m3.1-flash-q2tp-mix.cmf --local-dir cmf/big > logs/mm1/dl.log 2>&1 ) &
dl=$!
bash scripts/build.sh fast > logs/mm1/build.log 2>&1
grep -q BUILD_FAST_DONE logs/mm1/build.log || { echo MM1_BUILD_FAILED; tail -30 logs/mm1/build.log; exit 1; }
wait $dl
B=bin/cortiq-fast
P="Explain in two sentences why the sky is blue."
probe() { # tag file backend
  local tag=$1 f=$2 be=$3
  for k in 1 2; do
    flock gpu.lock env CMF_GPU=$be CMF_CUDA_DEBUG=1 timeout 2400 $B bench $f --tokens 64 --ignore-eos --json > logs/mm1/${tag}_d_${be}_$k.log 2>&1
    echo "$tag $be decode run$k: exit $? | $(grep -hoE '"(decode_tok_s_steady|prefill_tok_s|ttft_s)": [0-9.e+-]+' logs/mm1/${tag}_d_${be}_$k.log | tr '\n' ' ')"
  done
  # past the card and the host's memory a prompt runs about decode speed: 512, not 4096
  flock gpu.lock env CMF_GPU=$be CMF_CUDA_DEBUG=1 timeout 3600 $B bench $f --tokens 32 --ctx 512 --json > logs/mm1/${tag}_4k_${be}.log 2>&1
  echo "$tag $be 512: exit $? | $(grep -hoE '"(decode_tok_s_steady|prefill_tok_s|ttft_s)": [0-9.e+-]+' logs/mm1/${tag}_4k_${be}.log | tr '\n' ' ')"
  sed -E 's/\x1b\[[0-9;]*m//g' logs/mm1/${tag}_d_${be}_1.log logs/mm1/${tag}_4k_${be}.log | grep -iE 'declin|refus|fall.*back|per-op|not supported|out of memory|error' | sed -E 's/^.*(INFO|WARN|ERROR) //' | sort | uniq -c | sort -rn | head -6 | cut -c1-220 | sed 's/^/      /'
  flock gpu.lock env CMF_GPU=$be timeout 2400 $B run $f --prompt "$P" -n 32 --greedy 2>/dev/null | grep -v '^\[' > out/mm1/${tag}_$be.txt
}
g=cmf/big/glm-5.3-flash-q2tp.cmf
probe glm $g 1
if cmp -s out/mm1/glm_1.txt out/glm1/cpu.txt; then echo "glm: Vulkan text == CPU"; else echo "glm: Vulkan text DIFFERS from CPU"; diff out/mm1/glm_1.txt out/glm1/cpu.txt | head -6 | cut -c1-200; fi
if [ -s $m ]; then
  echo "minimax $(du -h $m | cut -f1)"
  probe mm $m cuda
  probe mm $m 1
  flock gpu.lock env CMF_GPU=0 timeout 3600 $B run $m --prompt "$P" -n 32 --greedy 2>/dev/null | grep -v '^\[' > out/mm1/mm_cpu.txt
  for be in cuda 1; do
    if cmp -s out/mm1/mm_$be.txt out/mm1/mm_cpu.txt; then echo "mm: $be text == CPU"; else echo "mm: $be text DIFFERS from CPU"; diff out/mm1/mm_$be.txt out/mm1/mm_cpu.txt | head -6 | cut -c1-200; fi
  done
else
  echo "MINIMAX DOWNLOAD FAILED"; tail -3 logs/mm1/dl.log
fi
echo MM1_DONE
