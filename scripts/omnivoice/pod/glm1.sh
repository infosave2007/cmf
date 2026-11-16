#!/bin/bash
# GLM-5.3-Flash q2tp (MLA with a low-rank query, 109 GB) on the CUDA hybrid
# arm after the q_a branch: the download while ov4 holds the card, then the
# build, decode (64 tokens, two runs), a 4096-token prompt, the declines, and
# greedy text CUDA against the CPU.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
mkdir -p logs/glm1 out/glm1 cmf/big
f=cmf/big/glm-5.3-flash-q2tp.cmf
[ -s $f ] || timeout 7200 venv/bin/hf download infosave/GLM-5.3-Flash-cmf glm-5.3-flash-q2tp.cmf --local-dir cmf/big > logs/glm1/dl.log 2>&1
[ -s $f ] || { echo GLM1_DOWNLOAD_FAILED; exit 1; }
echo "downloaded $(du -h $f | cut -f1)"
end=$((SECONDS + 7200))
until grep -qE "OV4_DONE|OV4_BUILD_FAILED" logs/ov4.out 2>/dev/null || [ $SECONDS -gt $end ]; do sleep 30; done
bash scripts/build.sh fast > logs/glm1/build.log 2>&1
grep -q BUILD_FAST_DONE logs/glm1/build.log || { echo GLM1_BUILD_FAILED; tail -30 logs/glm1/build.log; exit 1; }
B=bin/cortiq-fast
for k in 1 2; do
  flock gpu.lock env CMF_GPU=cuda CMF_CUDA_DEBUG=1 timeout 2400 $B bench $f --tokens 64 --ignore-eos --json > logs/glm1/d_$k.log 2>&1
  echo "decode run$k: exit $? | $(grep -hoE '"(decode_tok_s_steady|prefill_tok_s|ttft_s)": [0-9.e+-]+' logs/glm1/d_$k.log | tr '\n' ' ')"
done
flock gpu.lock env CMF_GPU=cuda CMF_CUDA_DEBUG=1 timeout 3600 $B bench $f --tokens 32 --ctx 4096 --json > logs/glm1/4k.log 2>&1
echo "4k: exit $? | $(grep -hoE '"(decode_tok_s_steady|prefill_tok_s|ttft_s)": [0-9.e+-]+' logs/glm1/4k.log | tr '\n' ' ')"
sed -E 's/\x1b\[[0-9;]*m//g' logs/glm1/d_1.log logs/glm1/4k.log | grep -iE 'declin|refus|fall.*back|per-op|not supported|out of memory|error|arena|resident' | sed -E 's/^.*(INFO|WARN|ERROR) //' | sort | uniq -c | sort -rn | head -10 | cut -c1-220 | sed 's/^/      /'
P="Explain in two sentences why the sky is blue."
flock gpu.lock env CMF_GPU=cuda timeout 2400 $B run $f --prompt "$P" -n 32 --greedy 2>/dev/null | grep -v '^\[' > out/glm1/cuda.txt
flock gpu.lock env CMF_GPU=0 timeout 3600 $B run $f --prompt "$P" -n 32 --greedy 2>/dev/null | grep -v '^\[' > out/glm1/cpu.txt
if cmp -s out/glm1/cuda.txt out/glm1/cpu.txt; then echo "CUDA text == CPU"; else echo "CUDA text DIFFERS from CPU"; diff out/glm1/cuda.txt out/glm1/cpu.txt | head -8 | cut -c1-200; fi
echo GLM1_DONE
