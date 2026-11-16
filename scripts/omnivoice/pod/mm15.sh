#!/bin/bash
# GLM-5.3's DSA mirrors in the CUDA plan's reserve: a DSA layer's K/V mirror
# stops at topk rows, now sized to that at once and held beside the slots
# (mm13/mm14: on the whole A40 its 3072-token prompt fell to the host, at
# row 512 on the prompt path and at row 1536 on 0.9.0's small path). GLM CUDA
# 3072 tokens on the card and within 24 GB, 512 on the card and within 24 GB
# (mm13: 23.0 / 10.55 tokens/s, first token 5.6 / 5.9 s), perplexity within
# 24 GB (CPU 2.715), "2+2" on the card; the card's copy rate from VMM host
# memory now (vmmprobe 19.5 GB/s before; Ornith within 12 GB read 9.0 tok/s
# in mm13/mm14 against mm11's 10.45 on the same code path).
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
mkdir -p logs/mm15 out/mm15
bash scripts/build.sh fast > logs/mm15/build.log 2>&1
grep -q BUILD_FAST_DONE logs/mm15/build.log || { echo MM15_BUILD_FAILED; grep -E '^error|error:' -A8 logs/mm15/build.log | head -40; exit 1; }
B=bin/cortiq-fast
strings $B | grep -q "a DSA layer selecting within the chunk" || { echo MM15_STALE_BINARY; exit 1; }
echo MM15_BUILT
flock gpu.lock timeout 300 python3 scripts/vmmprobe.py > logs/mm15/vmmprobe.log 2>&1
echo "vmmprobe: $(grep -E 'VMM HOST:|cuMemHostAlloc' logs/mm15/vmmprobe.log | cut -c1-220 | tr '\n' '|')"
pick() { sed -E 's/\x1b\[[0-9;]*m//g' $1 | grep -hoE "expert stream of .*|graph declined: .*|batch graph failed.*|batch graph declined.*|host replays.*|stopped at position.*|panicked.*|CUDA_ERROR[A-Z_]*" | sort -u | cut -c1-150 | tr '\n' '|'; }
bench() { # tag file ctx tmo env...
  local tag=$1 f=$2 ctx=$3 tmo=$4; shift 4
  local l=logs/mm15/$tag.log
  ( while sleep 2; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; done ) > logs/mm15/$tag.mem 2>/dev/null &
  local smp=$!
  flock gpu.lock env "$@" CMF_STREAM_STATS=1 RUST_LOG=info timeout $tmo $B bench $f --tokens 64 --ctx $ctx --ignore-eos --json > $l 2>&1
  local rc=$?
  kill $smp 2>/dev/null
  echo "$tag: exit $rc | max card MiB $(sort -n logs/mm15/$tag.mem | tail -1) | $(grep -hoE '"(decode_tok_s_steady|prefill_tok_s|ttft_s)": [0-9.e+-]+' $l | tr '\n' ' ') | $(pick $l)"
}
ppl() { # tag file env...
  local tag=$1 f=$2; shift 2
  flock gpu.lock env "$@" RUST_LOG=info timeout 1800 $B ppl $f --file prompts/rail.txt --tokens 256 > logs/mm15/$tag.log 2>&1
  echo "$tag: exit $? | $(grep -hiE 'ppl|perplexity' logs/mm15/$tag.log | tail -1 | cut -c1-80) | $(pick logs/mm15/$tag.log)"
}
two() { # tag file env...
  local tag=$1 f=$2; shift 2
  flock gpu.lock env "$@" timeout 1800 $B run $f --prompt "What is 2+2? Answer briefly." --no-think -n 16 --greedy > out/mm15/$tag.txt 2> logs/mm15/$tag.log
  echo "$tag: exit $? | $(grep -v '^\[' out/mm15/$tag.txt | tail -n +5 | tr '\n' ' ' | cut -c1-120)"
}
g=cmf/big/glm-5.3-flash-q2tp.cmf
bench glm_cuda_card_3k $g 3072 1200 CMF_GPU=cuda CMF_X=1
bench glm_cuda_card $g 512 900 CMF_GPU=cuda CMF_X=1
bench glm_cuda_24g $g 512 900 CMF_GPU=cuda CMF_GPU_VRAM_MB=24576
bench glm_cuda_24g_3k $g 3072 1200 CMF_GPU=cuda CMF_GPU_VRAM_MB=24576
ppl glm_cuda_24g_ppl $g CMF_GPU=cuda CMF_GPU_VRAM_MB=24576
two glm_cuda_card_two $g CMF_GPU=cuda CMF_X=1
echo MM15_DONE
