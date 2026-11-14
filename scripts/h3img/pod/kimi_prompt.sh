#!/bin/bash
# Kimi-Linear-48B-A3B on the CUDA hybrid prompt path (KDA and MLA layers in
# GEMMs instead of 16-row small-path chunks): long-prompt text against the
# CPU, then the 4k prompt and decode, twice each.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
bash scripts/build.sh release > logs/kimi_build.log 2>&1 || { echo BUILD_RELEASE_FAILED; tail -40 logs/kimi_build.log; exit 1; }
echo "build ok"
f=cmf/dl/kimi48-code.cmf
mkdir -p cmf/dl
[ -s $f ] || timeout 3600 venv/bin/hf download infosave/Kimi-Linear-48B-A3B-Code-CMF kimi48-code.cmf --local-dir cmf/dl > logs/kimi_dl.log 2>&1 \
  || { echo KIMI_DL_FAILED; tail -3 logs/kimi_dl.log; exit 1; }
long="$(head -c 2400 srcrs/docs/ppl_nat.txt) Summarize the text above in three sentences."
for be in cuda 0; do
  flock gpu.lock env CMF_GPU=$be timeout 1800 bin/cortiq run $f --prompt "$long" -n 48 --greedy 2>logs/kimi_long_$be.err | grep -v '^\[' > out/kimi_long_$be.txt
  grep -iE "declin|failed|refused" logs/kimi_long_$be.err | sort -u | head -4
done
cmp -s out/kimi_long_cuda.txt out/kimi_long_0.txt && echo "kimi long prompt: CUDA text == CPU" || { echo "kimi long prompt: CUDA text DIFFERS"; tail -4 out/kimi_long_cuda.txt; echo ...; tail -4 out/kimi_long_0.txt; }
keys='"(decode_tok_s_steady|prefill_tok_s|ttft_s)": *[0-9.]*'
for i in 1 2; do
  flock gpu.lock env CMF_GPU=cuda CMF_CUDA_DEBUG=1 timeout 1800 bin/cortiq bench $f --tokens 32 --ctx 4096 --json > logs/kimi_4k_$i.log 2>&1
  echo "kimi 4k cuda run$i: $(grep -oE "$keys" logs/kimi_4k_$i.log | tr '\n' ' ')"
done
grep -iE "declin|failed|refused|prompt path" logs/kimi_4k_1.log | sort -u | head -5
for i in 1 2; do
  flock gpu.lock env CMF_GPU=cuda timeout 900 bin/cortiq bench $f --tokens 64 --json > logs/kimi_d_$i.log 2>&1
  echo "kimi decode cuda run$i: $(grep -oE "$keys" logs/kimi_d_$i.log | tr '\n' ' ')"
done
echo KIMI_DONE
