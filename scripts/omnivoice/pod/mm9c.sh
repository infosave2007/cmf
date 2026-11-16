#!/bin/bash
# In place of mm9b.sh (stopped while it waited): the CUDA hybrid's RAM tier
# and host arenas in VMM host memory (cuMemCreate at a host location) when
# the container's limit counts page-locked memory and not that (vmmprobe:
# +513 MB for 512 MB page-locked, +1 MB VMM, 19.5 GB/s to the card) — the
# tier then holds every expert, as Vulkan's system-memory banks do. Plus the
# card's reserve at an eighth of it, 1-3 GiB. Ornith-122B q4tp CUDA within
# 12 / 16 / 24 / 32 GB and on the card, its perplexity within 12 GB, Vulkan
# within 32 GB; Apex-35B CUDA within 8 / 12 GB (and 4096 tokens within 8 GB
# on both backends); GLM-5.3 CUDA on the card and within 24 GB; MiniMax-M3.1
# and MiMo-V2.6 on CUDA on the card (with the "2+2" check) and within 24 GB.
# Starts after issue #153's Edge0 runs (they append I153C_DONE).
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
mkdir -p logs/mm9c out/mm9c
end=$((SECONDS + 43200))
until grep -qE "I153C_DONE" logs/i153c.out 2>/dev/null || [ $SECONDS -gt $end ]; do sleep 30; done
bash scripts/build.sh fast > logs/mm9c/build.log 2>&1
grep -q BUILD_FAST_DONE logs/mm9c/build.log || { echo MM9C_BUILD_FAILED; grep -E '^error|error:' -A8 logs/mm9c/build.log | head -40; exit 1; }
B=bin/cortiq-fast
pick() { sed -E 's/\x1b\[[0-9;]*m//g' $1 | grep -hoE "container's usage grew .*|RAM tier of .*|RAM tier filled at load: .*|prompt chunks [a-z ]+: .*|expert stream of .*|routed experts: .*|graph declined: .*|VMM host memory of .*|panicked.*|out of memory|Out of Memory" | sort -u | cut -c1-150 | tr '\n' '|'; }
bench() { # tag file ctx env...
  local tag=$1 f=$2 ctx=$3; shift 3
  local l=logs/mm9c/$tag.log
  flock gpu.lock env "$@" CMF_STREAM_STATS=1 RUST_LOG=info timeout 3600 $B bench $f --tokens 64 --ctx $ctx --ignore-eos --json > $l 2>&1
  echo "$tag: exit $? | $(grep -hoE '"(decode_tok_s_steady|prefill_tok_s|ttft_s)": [0-9.e+-]+' $l | tr '\n' ' ') | $(pick $l)"
}
ppl() { # tag file env...
  local tag=$1 f=$2; shift 2
  flock gpu.lock env "$@" RUST_LOG=info timeout 3600 $B ppl $f --file prompts/rail.txt --tokens 256 > logs/mm9c/$tag.log 2>&1
  echo "$tag: exit $? | $(grep -hiE 'ppl|perplexity' logs/mm9c/$tag.log | tail -1 | cut -c1-80) | $(pick logs/mm9c/$tag.log)"
}
two() { # tag file env...
  local tag=$1 f=$2; shift 2
  flock gpu.lock env "$@" timeout 3600 $B run $f --prompt "What is 2+2? Answer briefly." --no-think -n 16 --greedy > out/mm9c/$tag.txt 2> logs/mm9c/$tag.log
  echo "$tag: exit $? | $(grep -v '^\[' out/mm9c/$tag.txt | tail -n +5 | tr '\n' ' ' | cut -c1-120)"
}
o=cmf/big/ornith-1.5-122b-q4tp.cmf
for cap in 12288 16384 24576 32768; do bench orn_cuda_$cap $o 512 CMF_GPU=cuda CMF_GPU_VRAM_MB=$cap; done
bench orn_cuda_card $o 512 CMF_GPU=cuda CMF_X=1
ppl orn_cuda_12288_ppl $o CMF_GPU=cuda CMF_GPU_VRAM_MB=12288
bench orn_vk_32768 $o 512 CMF_GPU=1 CMF_GPU_VRAM_MB=32768
a=cmf/big/apex-35b-v1.0-q4tp.cmf
if [ -s $a ]; then
  bench apex_cuda_8g $a 512 CMF_GPU=cuda CMF_GPU_VRAM_MB=8192
  bench apex_cuda_12g $a 512 CMF_GPU=cuda CMF_GPU_VRAM_MB=12288
  bench apex_cuda_8g_4k $a 4096 CMF_GPU=cuda CMF_GPU_VRAM_MB=8192
  bench apex_vk_8g_4k $a 4096 CMF_GPU=1 CMF_GPU_VRAM_MB=8192
  ppl apex_cuda_8g_ppl $a CMF_GPU=cuda CMF_GPU_VRAM_MB=8192
fi
g=cmf/big/glm-5.3-flash-q2tp.cmf
if [ -s $g ]; then
  bench glm_cuda_card $g 512 CMF_GPU=cuda CMF_X=1
  bench glm_cuda_24g $g 512 CMF_GPU=cuda CMF_GPU_VRAM_MB=24576
fi
rm -f $a $g
m=cmf/big/minimax-m3.1-flash-q2tp-mix.cmf
[ -s $m ] || timeout 14400 venv/bin/hf download infosave/MiniMax-M3.1-Flash-cmf minimax-m3.1-flash-q2tp-mix.cmf --local-dir cmf/big > logs/mm9c/dl.log 2>&1
if ls -la $m; then
  bench minimax_cuda_card $m 512 CMF_GPU=cuda CMF_X=1
  bench minimax_cuda_24g $m 512 CMF_GPU=cuda CMF_GPU_VRAM_MB=24576
  two minimax_cuda_two $m CMF_GPU=cuda
  rm -f $m
else
  echo "MM9C_DOWNLOAD_FAILED: $(tail -3 logs/mm9c/dl.log)"
fi
mi=cmf/big/MiMo-V2.6-Flash-RL-q4tp.cmf
[ -s $mi ] || timeout 10800 venv/bin/hf download infosave/MiMo-V2.6-Flash-RL-cmf MiMo-V2.6-Flash-RL-q4tp.cmf --local-dir cmf/big > logs/mm9c/dl_mimo.log 2>&1
if ls -la $mi; then
  bench mimo_cuda_card $mi 512 CMF_GPU=cuda CMF_X=1
  bench mimo_cuda_24g $mi 512 CMF_GPU=cuda CMF_GPU_VRAM_MB=24576
  two mimo_cuda_two $mi CMF_GPU=cuda
  rm -f $mi
else
  echo "MM9C_MIMO_DOWNLOAD_FAILED: $(tail -3 logs/mm9c/dl_mimo.log)"
fi
echo MM9C_DONE
