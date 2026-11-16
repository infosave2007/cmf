#!/bin/bash
# Big MoE models across card sizes (CMF_GPU_VRAM_MB emulates a 12/16/24/32 GB
# card on the A40, the container's 46 GB of RAM as it is): decode after a
# 512-token prompt on CUDA and Vulkan, the placement each took, and at 12 GB
# the greedy text against the CPU. GLM-5.3 on CUDA at 16 and 24 GB. After glm6.sh.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
mkdir -p logs/sweep1 out/sweep1
end=$((SECONDS + 14400))
until grep -qE "GLM6_DONE|GLM6_BUILD_FAILED" logs/glm6.out 2>/dev/null || [ $SECONDS -gt $end ]; do sleep 30; done
bash scripts/build.sh fast > logs/sweep1/build.log 2>&1
grep -q BUILD_FAST_DONE logs/sweep1/build.log || { echo SWEEP1_BUILD_FAILED; tail -30 logs/sweep1/build.log; exit 1; }
B=bin/cortiq-fast
P="Explain in two sentences why the sky is blue."
run() { # tag file backend capMB
  local tag=$1 f=$2 be=$3 cap=$4 l=logs/sweep1/$1_$3_$4.log
  flock gpu.lock env CMF_GPU=$be CMF_GPU_VRAM_MB=$cap CMF_CUDA_DEBUG=1 RUST_LOG=info timeout 2400 $B bench $f --tokens 64 --ctx 512 --ignore-eos --json > $l 2>&1
  local rc=$?
  local place=$(sed -E 's/\x1b\[[0-9;]*m//g' $l | grep -oE "cuda hybrid: [0-9.]+ GB of a plan.s[^|]*|expert stream of [0-9]+ card slots[^)]*\)|device prefix [0-9]+ of [0-9]+ layers|routed experts past the card[^(]*|cuda moe rows: [0-9]+ card slots[^)]*\)|token graph declined: [^(]*" | sort -u | head -3 | tr '\n' ';' | cut -c1-260)
  echo "$tag $be ${cap}MB: exit $rc | $(grep -hoE '"(decode_tok_s_steady|ttft_s)": [0-9.e+-]+' $l | tr '\n' ' ')| $place"
}
txt() { # tag file backend capMB
  flock gpu.lock env CMF_GPU=$3 CMF_GPU_VRAM_MB=$4 timeout 2400 $B run $2 --prompt "$P" -n 24 --greedy 2>/dev/null | grep -v '^\[' > out/sweep1/$1_$3_$4.txt
}
orn=cmf/big/ornith-1.5-122b-q4tp.cmf; fnx=cmf/big/qwen38-flash-next-fast-code.cmf; glm=cmf/big/glm-5.3-flash-q2tp.cmf
[ -s out/sweep1/fnx_cpu.txt ] || flock gpu.lock env CMF_GPU=0 timeout 3600 $B run $fnx --prompt "$P" -n 24 --greedy 2>/dev/null | grep -v '^\[' > out/sweep1/fnx_cpu.txt
cp -f out/orn4/sky_0.txt out/sweep1/orn_cpu.txt 2>/dev/null
for cap in 12288 16384 24576 32768; do
  for be in cuda 1; do
    run orn $orn $be $cap
    run fnx $fnx $be $cap
  done
done
for be in cuda 1; do
  for m in orn fnx; do
    f=$orn; [ $m = fnx ] && f=$fnx
    txt $m $f $be 12288
    if cmp -s out/sweep1/${m}_${be}_12288.txt out/sweep1/${m}_cpu.txt; then echo "$m $be 12 GB: text == CPU"; else echo "$m $be 12 GB: text DIFFERS: $(tail -n +5 out/sweep1/${m}_${be}_12288.txt | tr '\n' ' ' | cut -c1-140)"; fi
  done
done
for cap in 16384 24576; do run glm $glm cuda $cap; done
echo SWEEP1_DONE
