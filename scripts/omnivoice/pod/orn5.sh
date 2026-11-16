#!/bin/bash
# Ornith-1.5-122B q4tp on Vulkan with routed experts past the card in
# GPU-visible system memory (the token graph no longer ends at a device
# prefix): decode 64 tokens, 512+64, the declines, greedy text against the
# CPU. orn5: the dense parts placed first, experts in what is left (two
# passes). After flash1.sh.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
mkdir -p logs/orn5 out/orn5
end=$((SECONDS + 14400))
until grep -qE "FLASH1_DONE|FLASH1_DOWNLOAD_FAILED" logs/flash1.out 2>/dev/null || [ $SECONDS -gt $end ]; do sleep 30; done
bash scripts/build.sh fast > logs/orn5/build.log 2>&1
grep -q BUILD_FAST_DONE logs/orn5/build.log || { echo ORN5_BUILD_FAILED; tail -30 logs/orn5/build.log; exit 1; }
B=bin/cortiq-fast; f=cmf/big/ornith-1.5-122b-q4tp.cmf
flock gpu.lock env CMF_GPU=1 RUST_LOG=info timeout 2400 $B bench $f --tokens 64 --ignore-eos --json > logs/orn5/d64.log 2>&1
echo "vulkan 64: exit $? | $(grep -hoE '"(decode_tok_s_steady|prefill_tok_s|ttft_s)": [0-9.e+-]+' logs/orn5/d64.log | tr '\n' ' ')"
sed -E 's/\x1b\[[0-9;]*m//g' logs/orn5/d64.log | grep -iE 'declin|refus|prefix|system memory|budget|out of memory|panick' | sed -E 's/^.*(INFO|WARN|ERROR) //' | cut -c1-200 | sort | uniq -c | sort -rn | head -8 | sed 's/^/      /'
flock gpu.lock env CMF_GPU=1 timeout 3600 $B bench $f --tokens 64 --ctx 512 --ignore-eos --json > logs/orn5/p512.log 2>&1
echo "vulkan 512+64: exit $? | $(grep -hoE '"(decode_tok_s_steady|prefill_tok_s|ttft_s)": [0-9.e+-]+' logs/orn5/p512.log | tr '\n' ' ')"
P="Explain in two sentences why the sky is blue."
for be in 1 0; do
  flock gpu.lock env CMF_GPU=$be timeout 2400 $B run $f --prompt "$P" -n 24 --greedy 2>/dev/null | grep -v '^\[' > out/orn5/sky_$be.txt
done
echo "vulkan: $(tail -n +5 out/orn5/sky_1.txt | tr '\n' ' ' | cut -c1-200)"
if cmp -s out/orn5/sky_1.txt out/orn5/sky_0.txt; then echo "Vulkan text == CPU"; else echo "Vulkan text DIFFERS from CPU: $(tail -n +5 out/orn5/sky_0.txt | tr '\n' ' ' | cut -c1-160)"; fi
echo ORN5_DONE
