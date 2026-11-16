#!/bin/bash
# Ornith-1.5-122B q4tp on Vulkan with routed experts past the card in
# GPU-visible system memory (the token graph no longer ends at a device
# prefix): decode 64 tokens, 512+64, the declines, greedy text against the
# CPU and against CUDA. After glm5.sh (its build has the change).
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
mkdir -p logs/orn4 out/orn4
end=$((SECONDS + 14400))
until grep -qE "GLM5_DONE|GLM5_BUILD_FAILED" logs/glm5.out 2>/dev/null || [ $SECONDS -gt $end ]; do sleep 30; done
bash scripts/build.sh fast > logs/orn4/build.log 2>&1
grep -q BUILD_FAST_DONE logs/orn4/build.log || { echo ORN4_BUILD_FAILED; tail -30 logs/orn4/build.log; exit 1; }
B=bin/cortiq-fast; f=cmf/big/ornith-1.5-122b-q4tp.cmf
flock gpu.lock env CMF_GPU=1 RUST_LOG=info timeout 2400 $B bench $f --tokens 64 --ignore-eos --json > logs/orn4/d64.log 2>&1
echo "vulkan 64: exit $? | $(grep -hoE '"(decode_tok_s_steady|prefill_tok_s|ttft_s)": [0-9.e+-]+' logs/orn4/d64.log | tr '\n' ' ')"
sed -E 's/\x1b\[[0-9;]*m//g' logs/orn4/d64.log | grep -iE 'declin|refus|prefix|system memory|budget|out of memory|panick' | sed -E 's/^.*(INFO|WARN|ERROR) //' | cut -c1-200 | sort | uniq -c | sort -rn | head -8 | sed 's/^/      /'
flock gpu.lock env CMF_GPU=1 timeout 3600 $B bench $f --tokens 64 --ctx 512 --ignore-eos --json > logs/orn4/p512.log 2>&1
echo "vulkan 512+64: exit $? | $(grep -hoE '"(decode_tok_s_steady|prefill_tok_s|ttft_s)": [0-9.e+-]+' logs/orn4/p512.log | tr '\n' ' ')"
P="Explain in two sentences why the sky is blue."
for be in 1 0; do
  flock gpu.lock env CMF_GPU=$be timeout 2400 $B run $f --prompt "$P" -n 24 --greedy 2>/dev/null | grep -v '^\[' > out/orn4/sky_$be.txt
done
echo "vulkan: $(tail -n +5 out/orn4/sky_1.txt | tr '\n' ' ' | cut -c1-200)"
if cmp -s out/orn4/sky_1.txt out/orn4/sky_0.txt; then echo "Vulkan text == CPU"; else echo "Vulkan text DIFFERS from CPU: $(tail -n +5 out/orn4/sky_0.txt | tr '\n' ' ' | cut -c1-160)"; fi
echo ORN4_DONE
