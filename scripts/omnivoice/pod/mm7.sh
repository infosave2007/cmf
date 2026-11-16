#!/bin/bash
# mm6.sh's build again: GLM-5.3 streamed on Vulkan with the pools seeded
# from the prompt and the RAM tier left alone (glmw: seeding with the tier's
# demotion decoded 3.9 tok/s, unseeded 5.8); then MiniMax goes and MiMo-V2.6
# comes back: Vulkan text (2+2 → 4) and 512+64 on the whole card and within
# 24 GB with the token graph's own expert placement and no layer prefix
# (mimo3: TTFT 78 s, the prompt walked the host past the leading layers);
# CUDA on the card and within 24 GB with the RAM tier filled at load; then
# Flash-Next on Vulkan at 12/16/24/32 GB and the card under the new bank
# placement (vfnx: 14.5/17.6/29.9/32.4/43.9). After mm6.sh.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
mkdir -p logs/mm7 out/mm7
end=$((SECONDS + 43200))
until grep -qE "MM6_DONE|MM6_BUILD_FAILED|MM6_STALE_BINARY|MM6_DOWNLOAD_FAILED" logs/mm6.out 2>/dev/null || [ $SECONDS -gt $end ]; do sleep 30; done
B=bin/cortiq-fast
strings $B | grep -q CMF_WGPU_EXPERT_WARM_TIER || { echo MM7_STALE_BINARY; exit 1; }
bench() { # tag file env...
  local tag=$1 f=$2; shift 2
  local l=logs/mm7/$tag.log
  flock gpu.lock env "$@" CMF_STREAM_STATS=1 RUST_LOG=info timeout 3600 $B bench $f --tokens 64 --ctx 512 --ignore-eos --json > $l 2>&1
  echo "$tag: exit $? | $(grep -hoE '"(decode_tok_s_steady|prefill_tok_s|ttft_s)": [0-9.e+-]+' $l | tr '\n' ' ') | $(sed -E 's/\x1b\[[0-9;]*m//g' $l | grep -hoE 'RAM tier filled at load: .*|MiMo MoE placement: [a-z]+|expert stream: .*|routed experts: .*|graph declined: .*|layer prefix.*|panicked.*|Out of Memory' | sort -u | cut -c1-140 | tr '\n' '|')"
}
g=cmf/big/glm-5.3-flash-q2tp.cmf
bench glm_vk_stream_warm $g CMF_GPU=1 CMF_WGPU_EXPERT_STREAM=1 CMF_WGPU_EXPERT_WARM=1
rm -f cmf/big/minimax-m3.1-flash-q2tp-mix.cmf
mi=cmf/big/MiMo-V2.6-Flash-RL-q4tp.cmf
[ -s $mi ] || timeout 10800 venv/bin/hf download infosave/MiMo-V2.6-Flash-RL-cmf MiMo-V2.6-Flash-RL-q4tp.cmf --local-dir cmf/big > logs/mm7/dl_mimo.log 2>&1
if ls -la $mi; then
  flock gpu.lock env CMF_GPU=1 RUST_LOG=info timeout 3600 $B run $mi --prompt "What is 2+2? Answer briefly." -n 24 --greedy > out/mm7/mimo_vk.txt 2> logs/mm7/mimo_vk.err
  echo "mimo vulkan 2+2: $(grep -v '^\[' out/mm7/mimo_vk.txt | tail -n +5 | tr '\n' ' ' | cut -c1-120)"
  bench mimo_vk_card $mi CMF_GPU=1 CMF_X=1
  bench mimo_vk_24g $mi CMF_GPU=1 CMF_GPU_VRAM_MB=24576
  bench mimo_cuda_card $mi CMF_GPU=cuda CMF_X=1
  bench mimo_cuda_24g $mi CMF_GPU=cuda CMF_GPU_VRAM_MB=24576
  rm -f $mi
else
  echo "MIMO_DOWNLOAD_FAILED: $(tail -3 logs/mm7/dl_mimo.log)"
fi
fx=cmf/big/qwen38-flash-next-fast-code.cmf
[ -s $fx ] || timeout 3600 venv/bin/hf download infosave/Qwen3.8-Flash-Next-Fast-Code-cmf qwen38-flash-next-fast-code.cmf --local-dir cmf/big > logs/mm7/dl_fnx.log 2>&1
if ls -la $fx; then
  for cap in 12288 16384 24576 32768; do bench fnx_vk_$cap $fx CMF_GPU=1 CMF_GPU_VRAM_MB=$cap; done
  bench fnx_vk_card $fx CMF_GPU=1 CMF_X=1
else
  echo "FNX_DOWNLOAD_FAILED: $(tail -3 logs/mm7/dl_fnx.log)"
fi
echo MM7_DONE
