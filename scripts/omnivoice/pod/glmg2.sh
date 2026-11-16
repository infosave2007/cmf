#!/bin/bash
# GLM-5.3 on the CUDA graph: perplexity against the CPU (256 tokens of
# prompts/rail.txt) and decode after 512 within 16 / 24 GB (host stack
# 1.13 / 1.32 tok/s in sweep1). After glmg1.sh.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
mkdir -p logs/glmg2
B=bin/cortiq-fast; f=cmf/big/glm-5.3-flash-q2tp.cmf
for v in cuda cpu; do
  case $v in cpu) e="CMF_GPU=0";; cuda) e="CMF_GPU=cuda";; esac
  flock gpu.lock env $e timeout 5400 $B ppl $f --file prompts/rail.txt --tokens 256 > logs/glmg2/ppl_$v.log 2>&1
  echo "ppl $v: exit $? | $(grep -hiE 'ppl|perplexity' logs/glmg2/ppl_$v.log | tail -1 | cut -c1-200)"
done
for cap in 16384 24576; do
  flock gpu.lock env CMF_GPU=cuda CMF_GPU_VRAM_MB=$cap RUST_LOG=info timeout 3600 $B bench $f --tokens 64 --ctx 512 --ignore-eos --json > logs/glmg2/p512_$cap.log 2>&1
  echo "cuda ${cap}MB 512+64: exit $? | $(grep -hoE '"(decode_tok_s_steady|prefill_tok_s|ttft_s)": [0-9.e+-]+' logs/glmg2/p512_$cap.log | tr '\n' ' ')"
done
echo GLMG2_DONE
