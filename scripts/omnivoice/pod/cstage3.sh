#!/bin/bash
# CUDA's staged prompt past its warm-up: decode 256 tokens after 512 within
# 24 GB, staged (the chunk's picks after the top-k) against the small path
# (CMF_HYB_STREAM_STAGE=0); 64 tokens gave 11.7 against 32 tok/s.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
mkdir -p logs/cstage3
B=bin/cortiq-fast
M=cmf/big/ornith-1.5-122b-q4tp.cmf
for v in stage small; do
  case $v in stage) e="CMF_X=1";; small) e="CMF_HYB_STREAM_STAGE=0";; esac
  l=logs/cstage3/$v.log
  flock gpu.lock env CMF_GPU=cuda CMF_GPU_VRAM_MB=24576 $e CMF_CUDA_DEBUG=1 RUST_LOG=info timeout 3600 $B bench $M --tokens 256 --ctx 512 --ignore-eos --json > $l 2>&1
  echo "$v 24576MB 512+256: exit $? | $(grep -hoE '"(decode_tok_s_steady|ttft_s)": [0-9.e+-]+' $l | tr '\n' ' ')| $(grep -E 'expert stream:' $l | tail -1 | cut -c1-200)"
done
echo CSTAGE3_DONE
