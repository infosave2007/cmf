#!/bin/bash
# mm13's GLM-5.3 CUDA 3072-token prompt on the card ran out of card memory
# on its second 512-row chunk (a 32 MB scratch) and fell to the host. Is that
# the prompt path's (new) or the mirrors' (as before)? The same prompt with
# CMF_HYB_MOE_BIG=0 (0.9.0's small path for GLM), and with the new path
# within 24 GB, each with the card's used memory sampled; Ornith-122B q4tp
# CUDA within 12 GB twice more (mm13 9.03 tok/s against mm11's 10.45).
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
mkdir -p logs/mm14
B=bin/cortiq-fast
pick() { sed -E 's/\x1b\[[0-9;]*m//g' $1 | grep -hoE "expert stream of .*|graph declined: .*|batch graph failed.*|host replays.*|panicked.*|CUDA_ERROR[A-Z_]*" | sort -u | cut -c1-150 | tr '\n' '|'; }
bench() { # tag file ctx tmo env...
  local tag=$1 f=$2 ctx=$3 tmo=$4; shift 4
  local l=logs/mm14/$tag.log
  ( while sleep 2; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; done ) > logs/mm14/$tag.mem 2>/dev/null &
  local smp=$!
  flock gpu.lock env "$@" CMF_STREAM_STATS=1 RUST_LOG=info timeout $tmo $B bench $f --tokens 64 --ctx $ctx --ignore-eos --json > $l 2>&1
  local rc=$?
  kill $smp 2>/dev/null
  echo "$tag: exit $rc | max card MiB $(sort -n logs/mm14/$tag.mem | tail -1) | $(grep -hoE '"(decode_tok_s_steady|prefill_tok_s|ttft_s)": [0-9.e+-]+' $l | tr '\n' ' ') | $(pick $l)"
}
o=cmf/big/ornith-1.5-122b-q4tp.cmf
g=cmf/big/glm-5.3-flash-q2tp.cmf
bench orn_cuda_12288_a $o 512 900 CMF_GPU=cuda CMF_GPU_VRAM_MB=12288
bench orn_cuda_12288_b $o 512 900 CMF_GPU=cuda CMF_GPU_VRAM_MB=12288
bench glm_cuda_card_3k_small $g 3072 1500 CMF_GPU=cuda CMF_X=1 CMF_HYB_MOE_BIG=0
bench glm_cuda_24g_3k $g 3072 1500 CMF_GPU=cuda CMF_GPU_VRAM_MB=24576
echo MM14_DONE
