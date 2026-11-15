#!/bin/bash
# Ripple-65B on the CUDA hybrid arm (value residual + clamped SwiGLU): fast
# build, decode (64 tokens, two runs) and a 4096-token prompt on CUDA, the
# greedy text against the probe's Vulkan one (out/big/ripple-65b-q4tp_1.txt).
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
bash scripts/build.sh fast > logs/rip_build.log 2>&1
grep -q BUILD_FAST_DONE logs/rip_build.log || { echo RIP_BUILD_FAILED; tail -40 logs/rip_build.log; exit 1; }
mkdir -p cmf/rip logs/rip out/rip
f=cmf/rip/ripple-65b-q4tp.cmf
[ -s $f ] || timeout 7200 venv/bin/hf download infosave/Ripple-65B-Preview-cmf ripple-65b-q4tp.cmf --local-dir cmf/rip > logs/rip/dl.log 2>&1
B=bin/cortiq-fast
for k in 1 2; do
  flock gpu.lock env CMF_GPU=cuda CMF_CUDA_DEBUG=1 timeout 2400 $B bench $f --tokens 64 --ignore-eos --json > logs/rip/d_$k.log 2>&1
  echo "decode run$k: exit $? | $(grep -hoE '"(decode_tok_s_steady|prefill_tok_s|ttft_s)": [0-9.e+-]+' logs/rip/d_$k.log | tr '\n' ' ')"
done
sed -E 's/\x1b\[[0-9;]*m//g' logs/rip/d_1.log | grep -iE 'declin|refus|per-op|error|hybrid graphs' | sort | uniq -c | head -8
flock gpu.lock env CMF_GPU=cuda timeout 2400 $B run $f --prompt "Explain in two sentences why the sky is blue." -n 48 --greedy 2>/dev/null | grep -v '^\[' > out/rip/cuda.txt
if cmp -s out/rip/cuda.txt out/big/ripple-65b-q4tp_1.txt; then echo "CUDA text == Vulkan"; else echo "CUDA text DIFFERS"; diff out/rip/cuda.txt out/big/ripple-65b-q4tp_1.txt | head -8; fi
flock gpu.lock env CMF_GPU=cuda timeout 3600 $B bench $f --tokens 32 --ctx 4096 --json > logs/rip/4k.log 2>&1
echo "4k: exit $? | $(grep -hoE '"(decode_tok_s_steady|prefill_tok_s|ttft_s)": [0-9.e+-]+' logs/rip/4k.log | tr '\n' ' ')"
echo RIP1_DONE
