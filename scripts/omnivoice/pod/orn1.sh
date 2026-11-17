#!/bin/bash
# Ornith-1.5-122B q2tp-v2 on the CUDA hybrid arm with routed experts past the
# card in mapped host memory: after rip1, a fast build, decode (64 tokens, two
# runs), a 4096-token prompt, the greedy text against the probe's Vulkan one.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
end=$(( $(date +%s) + 7200 ))
until grep -q RIP1_DONE logs/rip1.out 2>/dev/null; do [ "$(date +%s)" -gt $end ] && break; sleep 30; done
bash scripts/build.sh fast > logs/orn_build.log 2>&1
grep -q BUILD_FAST_DONE logs/orn_build.log || { echo ORN_BUILD_FAILED; tail -40 logs/orn_build.log; exit 1; }
mkdir -p cmf/orn logs/orn out/orn
f=cmf/orn/ornith-1.5-122b-q2tp-v2.cmf
[ -s $f ] || timeout 7200 venv/bin/hf download infosave/Ornith-1.5-122B-cmf ornith-1.5-122b-q2tp-v2.cmf --local-dir cmf/orn > logs/orn/dl.log 2>&1
B=bin/cortiq-fast
for k in 1 2; do
  flock gpu.lock env CMF_GPU=cuda CMF_CUDA_DEBUG=1 CMF_CUDA_MEM=1 timeout 2400 $B bench $f --tokens 64 --ignore-eos --json > logs/orn/d_$k.log 2>&1
  echo "decode run$k: exit $? | $(grep -hoE '"(decode_tok_s_steady|prefill_tok_s|ttft_s)": [0-9.e+-]+' logs/orn/d_$k.log | tr '\n' ' ')"
done
sed -E 's/\x1b\[[0-9;]*m//g' logs/orn/d_1.log | grep -iE 'declin|refus|per-op|error|mapped host|hybrid graphs|on the CPU' | sed -E 's/^.*(INFO|WARN|ERROR) //' | sort | uniq -c | head -10
flock gpu.lock env CMF_GPU=cuda timeout 2400 $B run $f --prompt "Explain in two sentences why the sky is blue." -n 48 --greedy 2>/dev/null | grep -v '^\[' > out/orn/cuda.txt
if cmp -s out/orn/cuda.txt out/big/ornith-1.5-122b-q2tp-v2_1.txt; then echo "CUDA text == Vulkan"; else echo "CUDA text DIFFERS"; head -c 400 out/orn/cuda.txt; echo; head -c 400 out/big/ornith-1.5-122b-q2tp-v2_1.txt; echo; fi
flock gpu.lock env CMF_GPU=cuda timeout 3600 $B bench $f --tokens 32 --ctx 4096 --json > logs/orn/4k.log 2>&1
echo "4k: exit $? | $(grep -hoE '"(decode_tok_s_steady|prefill_tok_s|ttft_s)": [0-9.e+-]+' logs/orn/4k.log | tr '\n' ' ')"
echo ORN1_DONE
