#!/bin/bash
# The three CUDA pieces left out of 0.9.0, measured: the RAM tier and host
# arenas in VMM host memory (cuMemCreate at a host location, which the pod's
# container limit does not count: the tier then holds every expert), the
# card's reserve at an eighth of it (1-3 GiB, was 3 GiB), and a prompt call
# on the small path handing its rest to the staged path once most picks were
# built. Ornith-122B q4tp CUDA within 12 / 16 / 24 / 32 GB and on the card
# (0.9.0: 6.79 / 7.05 / 28.7 / - / 37.3 tokens/s, first token ~10 s), its
# perplexity (CPU 3.049) within 12 GB and on the card, and 12 GB with each
# piece off; Apex-35B CUDA within 8 / 12 GB (0.9.0: 34.3 / 88.7) and a
# 4096-token prompt within 8 GB, perplexity (CPU 4.188); GLM-5.3 CUDA within
# 24 GB (0.9.0: 6.4 tokens/s, first token 100 s; Vulkan 9.8 and 15 s) and on
# the card (17.1, 38 s), perplexity (CPU 2.715); then MiniMax-M3.1 and
# MiMo-V2.6 on CUDA within 24 GB with VMM off (the switch: 398 / 297 s on the
# small path) and on the card with every piece on, "2+2". Downloads wait for
# the disk (issue #153's Edge0 shares it).
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
mkdir -p logs/mm11 out/mm11
bash scripts/build.sh fast > logs/mm11/build.log 2>&1
grep -q BUILD_FAST_DONE logs/mm11/build.log || { echo MM11_BUILD_FAILED; grep -E '^error|error:' -A8 logs/mm11/build.log | head -40; exit 1; }
B=bin/cortiq-fast
strings $B | grep -q "VMM host memory" || { echo MM11_STALE_BINARY; exit 1; }
pick() { sed -E 's/\x1b\[[0-9;]*m//g' $1 | grep -hoE "container's usage grew .*|RAM tier of .*|RAM tier filled at load: .*|prompt chunks [a-z ]+: .*|expert stream of .*|routed experts: .*|graph declined: .*|VMM host memory of .*|panicked.*|out of memory|Out of Memory|CUDA_ERROR[A-Z_]*" | sort -u | cut -c1-150 | tr '\n' '|'; }
bench() { # tag file ctx env...
  local tag=$1 f=$2 ctx=$3; shift 3
  local l=logs/mm11/$tag.log
  flock gpu.lock env "$@" CMF_STREAM_STATS=1 RUST_LOG=info timeout 3600 $B bench $f --tokens 64 --ctx $ctx --ignore-eos --json > $l 2>&1
  echo "$tag: exit $? | $(grep -hoE '"(decode_tok_s_steady|prefill_tok_s|ttft_s)": [0-9.e+-]+' $l | tr '\n' ' ') | $(pick $l)"
}
ppl() { # tag file env...
  local tag=$1 f=$2; shift 2
  flock gpu.lock env "$@" RUST_LOG=info timeout 3600 $B ppl $f --file prompts/rail.txt --tokens 256 > logs/mm11/$tag.log 2>&1
  echo "$tag: exit $? | $(grep -hiE 'ppl|perplexity' logs/mm11/$tag.log | tail -1 | cut -c1-80) | $(pick logs/mm11/$tag.log)"
}
two() { # tag file env...
  local tag=$1 f=$2; shift 2
  flock gpu.lock env "$@" timeout 3600 $B run $f --prompt "What is 2+2? Answer briefly." --no-think -n 16 --greedy > out/mm11/$tag.txt 2> logs/mm11/$tag.log
  echo "$tag: exit $? | $(grep -v '^\[' out/mm11/$tag.txt | tail -n +5 | tr '\n' ' ' | cut -c1-120)"
}
room_for() { # bytes: wait (up to 8 h) until the disk has them and 15 GB more
  local need=$(( $1 / 1024 + 15 * 1024 * 1024 )) end=$((SECONDS + 28800))
  until [ "$(df --output=avail -k /root | tail -1)" -gt $need ] || [ $SECONDS -gt $end ]; do sleep 60; done
  [ "$(df --output=avail -k /root | tail -1)" -gt $need ]
}
o=cmf/big/ornith-1.5-122b-q4tp.cmf
for cap in 12288 16384 24576 32768; do bench orn_cuda_$cap $o 512 CMF_GPU=cuda CMF_GPU_VRAM_MB=$cap; done
bench orn_cuda_card $o 512 CMF_GPU=cuda CMF_X=1
ppl orn_cuda_12288_ppl $o CMF_GPU=cuda CMF_GPU_VRAM_MB=12288
ppl orn_cuda_card_ppl $o CMF_GPU=cuda CMF_X=1
bench orn_cuda_12288_novmm $o 512 CMF_GPU=cuda CMF_GPU_VRAM_MB=12288 CMF_HYB_PINNED_PROBE=0
bench orn_cuda_12288_res3g $o 512 CMF_GPU=cuda CMF_GPU_VRAM_MB=12288 CMF_HYB_RESERVE_MB=3072
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
echo MM11_PART1_DONE
rm -f $a $g
m=cmf/big/minimax-m3.1-flash-q2tp-mix.cmf
if room_for 157000000000; then
  timeout 14400 venv/bin/hf download infosave/MiniMax-M3.1-Flash-cmf minimax-m3.1-flash-q2tp-mix.cmf --local-dir cmf/big > logs/mm11/dl.log 2>&1
fi
if ls -la $m; then
  bench minimax_cuda_24g_novmm $m 512 CMF_GPU=cuda CMF_GPU_VRAM_MB=24576 CMF_HYB_PINNED_PROBE=0
  bench minimax_cuda_card $m 512 CMF_GPU=cuda CMF_X=1
  two minimax_cuda_two $m CMF_GPU=cuda
  rm -f $m
else
  echo "MM11_DOWNLOAD_FAILED: $(df -h /root | tail -1) $(tail -2 logs/mm11/dl.log 2>/dev/null)"
fi
mi=cmf/big/MiMo-V2.6-Flash-RL-q4tp.cmf
if room_for 164000000000; then
  timeout 10800 venv/bin/hf download infosave/MiMo-V2.6-Flash-RL-cmf MiMo-V2.6-Flash-RL-q4tp.cmf --local-dir cmf/big > logs/mm11/dl_mimo.log 2>&1
fi
if ls -la $mi; then
  bench mimo_cuda_24g_novmm $mi 512 CMF_GPU=cuda CMF_GPU_VRAM_MB=24576 CMF_HYB_PINNED_PROBE=0
  bench mimo_cuda_card $mi 512 CMF_GPU=cuda CMF_X=1
  two mimo_cuda_two $mi CMF_GPU=cuda
  rm -f $mi
else
  echo "MM11_MIMO_DOWNLOAD_FAILED: $(df -h /root | tail -1) $(tail -2 logs/mm11/dl_mimo.log 2>/dev/null)"
fi
echo MM11_DONE
