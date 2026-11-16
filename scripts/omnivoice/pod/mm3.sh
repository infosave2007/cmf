#!/bin/bash
# MiniMax-M3.1 (426B, 157 GB) on the CUDA hybrid with the expert stream:
# the dense part on the card, routed experts built into card slots after
# each layer's router. The CPU greedy text first (the reference), then CUDA
# decode (64 tokens, two runs), a 512-token prompt, the stream's hit rate
# (CMF_CUDA_DEBUG) and the greedy text against the CPU's.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
mkdir -p logs/mm3 out/mm3
bash scripts/build.sh fast > logs/mm3/build.log 2>&1
grep -q BUILD_FAST_DONE logs/mm3/build.log || { echo MM3_BUILD_FAILED; tail -30 logs/mm3/build.log; exit 1; }
B=bin/cortiq-fast; m=cmf/big/minimax-m3.1-flash-q2tp-mix.cmf
P="Explain in two sentences why the sky is blue."
[ -s out/mm3/cpu.txt ] || flock gpu.lock env CMF_GPU=0 timeout 3600 $B run $m --prompt "$P" -n 24 --greedy 2>/dev/null | grep -v '^\[' > out/mm3/cpu.txt
echo "cpu text: $(tail -n +6 out/mm3/cpu.txt | tr '\n' ' ' | cut -c1-200)"
flock gpu.lock env CMF_GPU=cuda CMF_CUDA_DEBUG=1 timeout 2400 $B run $m --prompt "$P" -n 24 --greedy > out/mm3/cuda.txt 2> logs/mm3/run_cuda.log
echo "cuda text exit $?: $(grep -v '^\[' out/mm3/cuda.txt | tail -n +6 | tr '\n' ' ' | cut -c1-200)"
if cmp -s <(grep -v '^\[' out/mm3/cuda.txt) out/mm3/cpu.txt; then echo "CUDA text == CPU"; else echo "CUDA text DIFFERS from CPU"; fi
sed -E 's/\x1b\[[0-9;]*m//g' logs/mm3/run_cuda.log | grep -iE 'declin|refus|stream|hybrid:|error|panick' | sed -E 's/^.*(INFO|WARN|ERROR) //' | sort | uniq -c | sort -rn | head -10 | cut -c1-230 | sed 's/^/      /'
for k in 1 2; do
  flock gpu.lock env CMF_GPU=cuda CMF_CUDA_DEBUG=1 timeout 2400 $B bench $m --tokens 64 --ignore-eos --json > logs/mm3/d_$k.log 2>&1
  echo "decode run$k: exit $? | $(grep -hoE '"(decode_tok_s_steady|prefill_tok_s|ttft_s)": [0-9.e+-]+' logs/mm3/d_$k.log | tr '\n' ' ') | $(grep -hoE 'expert stream: .*' logs/mm3/d_$k.log | tail -1)"
done
flock gpu.lock env CMF_GPU=cuda CMF_CUDA_DEBUG=1 timeout 3600 $B bench $m --tokens 32 --ctx 512 --json > logs/mm3/p512.log 2>&1
echo "512: exit $? | $(grep -hoE '"(decode_tok_s_steady|prefill_tok_s|ttft_s)": [0-9.e+-]+' logs/mm3/p512.log | tr '\n' ' ') | $(grep -hoE 'expert stream: .*' logs/mm3/p512.log | tail -1)"
echo MM3_DONE
