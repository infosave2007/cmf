#!/bin/bash
# Flash-Next on CUDA within 24 / 32 GB: zero copy (the static arena, misses
# read over the bus; 15.8 / 30.7 tok/s in sweep4) against admitting every miss
# (CMF_QWEN_ZERO_COPY=0, an LRU arena), with the arena's counters
# (CMF_QWEN_PROF=1), decode after 512.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
mkdir -p logs/fnx1
B=bin/cortiq-fast
run() { # tag capMB env...
  local tag=$1 cap=$2; shift 2
  local l=logs/fnx1/${tag}_$cap.log
  flock gpu.lock env CMF_GPU=cuda CMF_GPU_VRAM_MB=$cap "$@" RUST_LOG=info timeout 1800 $B bench cmf/big/qwen38-flash-next-fast-code.cmf --tokens 64 --ctx 512 --ignore-eos --json > $l 2>&1
  echo "$tag ${cap}MB: exit $? | $(grep -hoE '"(decode_tok_s_steady|ttft_s)": [0-9.e+-]+' $l | tr '\n' ' ')"
}
for cap in 24576 32768; do
  run zc $cap CMF_QWEN_ZERO_COPY=1
  run lru $cap CMF_QWEN_ZERO_COPY=0
  run lru_a4 $cap CMF_QWEN_ZERO_COPY=0 CMF_QWEN_ADMIT=4
done
echo FNX1_DONE
