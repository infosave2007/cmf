#!/bin/bash
# mm12.sh with its pick() line mended (it stopped before its runs).
# mm11.sh, continued after a fix: its whole-card perplexity run was killed by
# the OOM killer — the pinned probe read +1 MB for 256 MB because the
# container sat at its limit and the charge evicted as much page cache, so
# the RAM tier took page-locked memory past the limit. The probes (CUDA and
# Vulkan) now read the usage less the page cache, and VMM host memory wins
# whenever it falls outside. mm11 measured Ornith q4tp CUDA within 12 / 16 /
# 24 / 32 GB and on the card at 10.5 / 10.9 / 40.1 / 43.5 / 44.6 tokens/s
# (first token 1.8-2.5 s), perplexity 3.049 within 12 GB. Here: Ornith's
# whole-card perplexity and Vulkan on the card (its probe changed), then
# mm11's Apex, GLM-5.3, MiniMax-M3.1 and MiMo-V2.6 runs.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
mkdir -p logs/mm12b out/mm12b
# mm12.sh's build may still run (its script stopped on a broken line)
while pgrep -f "scripts/build.sh fast" > /dev/null; do sleep 10; done
bash scripts/build.sh fast > logs/mm12b/build.log 2>&1
grep -q BUILD_FAST_DONE logs/mm12b/build.log || { echo MM12B_BUILD_FAILED; grep -E '^error|error:' -A8 logs/mm12b/build.log | head -40; exit 1; }
B=bin/cortiq-fast
strings $B | grep -q "less its page cache" || { echo MM12B_STALE_BINARY; exit 1; }
pick() { sed -E 's/\x1b\[[0-9;]*m//g' $1 | grep -hoE "container's usage less its page cache grew .*|GPU-visible system memory [a-z ]+the container's limit.*|RAM tier of .*|RAM tier filled at load: .*|prompt chunks [a-z ]+: .*|expert stream of .*|routed experts: .*|graph declined: .*|VMM host memory of .*|panicked.*|out of memory|Out of Memory|CUDA_ERROR[A-Z_]*" | sort -u | cut -c1-150 | tr '\n' '|'; }
bench() { # tag file ctx env...
  local tag=$1 f=$2 ctx=$3; shift 3
  local l=logs/mm12b/$tag.log
  flock gpu.lock env "$@" CMF_STREAM_STATS=1 RUST_LOG=info timeout 3600 $B bench $f --tokens 64 --ctx $ctx --ignore-eos --json > $l 2>&1
  echo "$tag: exit $? | $(grep -hoE '"(decode_tok_s_steady|prefill_tok_s|ttft_s)": [0-9.e+-]+' $l | tr '\n' ' ') | $(pick $l)"
}
ppl() { # tag file env...
  local tag=$1 f=$2; shift 2
  flock gpu.lock env "$@" RUST_LOG=info timeout 3600 $B ppl $f --file prompts/rail.txt --tokens 256 > logs/mm12b/$tag.log 2>&1
  echo "$tag: exit $? | $(grep -hiE 'ppl|perplexity' logs/mm12b/$tag.log | tail -1 | cut -c1-80) | $(pick logs/mm12b/$tag.log)"
}
two() { # tag file env...
  local tag=$1 f=$2; shift 2
  flock gpu.lock env "$@" timeout 3600 $B run $f --prompt "What is 2+2? Answer briefly." --no-think -n 16 --greedy > out/mm12b/$tag.txt 2> logs/mm12b/$tag.log
  echo "$tag: exit $? | $(grep -v '^\[' out/mm12b/$tag.txt | tail -n +5 | tr '\n' ' ' | cut -c1-120)"
}
room_for() { # bytes: wait (up to 8 h) until the disk has them and 15 GB more
  local need=$(( $1 / 1024 + 15 * 1024 * 1024 )) end=$((SECONDS + 28800))
  until [ "$(df --output=avail -k /root | tail -1)" -gt $need ] || [ $SECONDS -gt $end ]; do sleep 60; done
  [ "$(df --output=avail -k /root | tail -1)" -gt $need ]
}
o=cmf/big/ornith-1.5-122b-q4tp.cmf
ppl orn_cuda_card_ppl $o CMF_GPU=cuda CMF_X=1
bench orn_vk_card $o 512 CMF_GPU=1 CMF_X=1
a=cmf/big/apex-35b-v1.0-q4tp.cmf
if [ -s $a ]; then
  bench apex_cuda_8g $a 512 CMF_GPU=cuda CMF_GPU_VRAM_MB=8192
  bench apex_cuda_12g $a 512 CMF_GPU=cuda CMF_GPU_VRAM_MB=12288
  bench apex_cuda_8g_4k $a 4096 CMF_GPU=cuda CMF_GPU_VRAM_MB=8192
  ppl apex_cuda_8g_ppl $a CMF_GPU=cuda CMF_GPU_VRAM_MB=8192
fi
g=cmf/big/glm-5.3-flash-q2tp.cmf
if [ -s $g ]; then
  bench glm_cuda_24g $g 512 CMF_GPU=cuda CMF_GPU_VRAM_MB=24576
  bench glm_cuda_card $g 512 CMF_GPU=cuda CMF_X=1
  ppl glm_cuda_24g_ppl $g CMF_GPU=cuda CMF_GPU_VRAM_MB=24576
fi
echo MM12B_PART1_DONE
rm -f $a $g
m=cmf/big/minimax-m3.1-flash-q2tp-mix.cmf
if room_for 157000000000; then
  timeout 14400 venv/bin/hf download infosave/MiniMax-M3.1-Flash-cmf minimax-m3.1-flash-q2tp-mix.cmf --local-dir cmf/big > logs/mm12b/dl.log 2>&1
fi
if ls -la $m; then
  bench minimax_cuda_24g_novmm $m 512 CMF_GPU=cuda CMF_GPU_VRAM_MB=24576 CMF_HYB_PINNED_PROBE=0
  bench minimax_cuda_card $m 512 CMF_GPU=cuda CMF_X=1
  two minimax_cuda_two $m CMF_GPU=cuda
  rm -f $m
else
  echo "MM12B_DOWNLOAD_FAILED: $(df -h /root | tail -1) $(tail -2 logs/mm12b/dl.log 2>/dev/null)"
fi
mi=cmf/big/MiMo-V2.6-Flash-RL-q4tp.cmf
if room_for 164000000000; then
  timeout 10800 venv/bin/hf download infosave/MiMo-V2.6-Flash-RL-cmf MiMo-V2.6-Flash-RL-q4tp.cmf --local-dir cmf/big > logs/mm12b/dl_mimo.log 2>&1
fi
if ls -la $mi; then
  bench mimo_cuda_24g_novmm $mi 512 CMF_GPU=cuda CMF_GPU_VRAM_MB=24576 CMF_HYB_PINNED_PROBE=0
  bench mimo_cuda_card $mi 512 CMF_GPU=cuda CMF_X=1
  two mimo_cuda_two $mi CMF_GPU=cuda
  rm -f $mi
else
  echo "MM12B_MIMO_DOWNLOAD_FAILED: $(df -h /root | tail -1) $(tail -2 logs/mm12b/dl_mimo.log 2>/dev/null)"
fi
echo MM12B_DONE
