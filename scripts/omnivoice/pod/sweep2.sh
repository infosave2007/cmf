#!/bin/bash
# CUDA across card sizes with the RAM tier: the slots' misses built from
# raw experts in page-locked host memory (DMA, no page faults). Ornith-122B q4tp at
# 12/16/24/32 GB (decode after 512), the 12 GB text against the CPU; at 12 GB
# the pinned experts read over the bus and the plain stream for the A/B. After ppl1.sh.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
mkdir -p logs/sweep2 out/sweep2
end=$((SECONDS + 14400))
until grep -qE "PPL1_DONE" logs/ppl1.out 2>/dev/null || [ $SECONDS -gt $end ]; do sleep 30; done
bash scripts/build.sh fast > logs/sweep2/build.log 2>&1
grep -q BUILD_FAST_DONE logs/sweep2/build.log || { echo SWEEP2_BUILD_FAILED; tail -30 logs/sweep2/build.log; exit 1; }
B=bin/cortiq-fast; orn=cmf/big/ornith-1.5-122b-q4tp.cmf; fnx=cmf/big/qwen38-flash-next-fast-code.cmf
P="Explain in two sentences why the sky is blue."
run() { # tag capMB extra-env [file]
  local l=logs/sweep2/$1_$2.log f=${4:-$orn}
  flock gpu.lock env CMF_GPU=cuda CMF_GPU_VRAM_MB=$2 CMF_CUDA_DEBUG=1 RUST_LOG=info $3 timeout 2400 $B bench $f --tokens 64 --ctx 512 --ignore-eos --json > $l 2>&1
  local rc=$?
  echo "$1 ${2}MB: exit $rc | $(grep -hoE '"(decode_tok_s_steady|ttft_s)": [0-9.e+-]+' $l | tr '\n' ' ')| $(sed -E 's/\x1b\[[0-9;]*m//g' $l | grep -oE "cuda hybrid: [0-9.]+ GB of a plan.s dense[^|]*|expert stream of [0-9]+ card slots[^)]*\)|expert stream: [0-9]+ slots.*" | sort -u | tail -3 | tr '\n' ';' | cut -c1-300)"
}
for cap in 12288 16384 24576 32768; do run ram $cap ""; done
run hostread 12288 "CMF_HYB_RAM_TIER=0 CMF_HYB_HOST_TIER=1"
run stream 12288 CMF_HYB_RAM_TIER=0
for cap in 12288 16384; do run fnx_zc $cap CMF_QWEN_ZERO_COPY=1 $fnx; done
flock gpu.lock env CMF_GPU=cuda CMF_GPU_VRAM_MB=12288 timeout 2400 $B run $orn --prompt "$P" -n 24 --greedy 2>/dev/null | grep -v '^\[' > out/sweep2/orn_cuda_12288.txt
if cmp -s out/sweep2/orn_cuda_12288.txt out/orn4/sky_0.txt; then echo "orn cuda 12 GB: text == CPU"; else echo "orn cuda 12 GB: text DIFFERS: $(tail -n +5 out/sweep2/orn_cuda_12288.txt | tr '\n' ' ' | cut -c1-140)"; fi
echo SWEEP2_DONE
