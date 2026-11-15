#!/bin/bash
# Ripple's 4k prompt on CUDA by prefill chunk (default / 512 / 1024), then
# Edge0-120B on Vulkan with the latent-bank refusal (fast build).
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
bash scripts/build.sh fast > logs/q5_build.log 2>&1
grep -q BUILD_FAST_DONE logs/q5_build.log || { echo Q5_BUILD_FAILED; tail -30 logs/q5_build.log; exit 1; }
B=bin/cortiq-fast
mkdir -p logs/q5
for ch in 512 1024; do
  flock gpu.lock env CMF_GPU=cuda CMF_PREFILL_CHUNK=$ch timeout 1800 $B bench cmf/big/ripple-65b-q4tp.cmf --tokens 32 --ctx 4096 --json > logs/q5/rip_4k_$ch.log 2>&1
  echo "ripple 4k chunk $ch: exit $? | $(grep -hoE '"(prefill_tok_s|ttft_s|decode_tok_s_steady)": [0-9.e+-]+' logs/q5/rip_4k_$ch.log | tr '\n' ' ')"
done
flock gpu.lock env CMF_GPU=1 RUST_BACKTRACE=1 timeout 2400 $B bench cmf/big/edge0-120b.cmf --tokens 32 --ignore-eos --json > logs/q5/edge0_vk.log 2>&1
echo "edge0 vk: exit $? | $(grep -hoE '"(decode_tok_s_steady|prefill_tok_s|ttft_s)": [0-9.e+-]+' logs/q5/edge0_vk.log | tr '\n' ' ')"
sed -E 's/\x1b\[[0-9;]*m//g' logs/q5/edge0_vk.log | grep -iE 'refused layer|latent experts|system memory|device path off|panicked|wgpu error' | sed -E 's/^.*(INFO|WARN|ERROR) //' | sort | uniq -c | head -6
echo Q5_DONE
