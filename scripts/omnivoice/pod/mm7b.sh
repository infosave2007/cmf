#!/bin/bash
# mm7.sh stopped before it deleted MiniMax; this one finishes its queue and
# appends to logs/mm7.out (mm8.sh and i153.sh wait on its markers). Rebuilt
# first: the CUDA prompt stages a streamed layer also when the small path
# would churn the slots (a 16-row chunk's expected picks past twice the
# layer's share: MiniMax-M3.1's first token 375 s on the small path, 35 s
# staged), and the bench's pair micro-benchmark is skipped when routed
# experts live past the card (MiniMax on the whole A40 ran out of memory in
# it). MiniMax again on Vulkan's whole card and CUDA (card, 24 GB); then
# mm7.sh's MiMo and Flash-Next runs. GLM's streamed bench of mm7.sh (still
# running when this starts) is summarized from its log.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
mkdir -p logs/mm7 out/mm7
end=$((SECONDS + 7200))
while kill -0 2700471 2>/dev/null && [ $SECONDS -lt $end ]; do sleep 15; done
l=logs/mm7/glm_vk_stream_warm.log
echo "glm_vk_stream_warm (pools seeded, tier left alone): $(grep -hoE '"(decode_tok_s_steady|prefill_tok_s|ttft_s)": [0-9.e+-]+' $l | tr '\n' ' ') | $(sed -E 's/\x1b\[[0-9;]*m//g' $l | grep -hoE 'expert stream: .*' | sort -u | cut -c1-140)"
bash scripts/build.sh fast > logs/mm7/build.log 2>&1
grep -q BUILD_FAST_DONE logs/mm7/build.log || { echo MM7B_BUILD_FAILED; grep -E '^error|error:' -A8 logs/mm7/build.log | head -40; echo MM7_DONE; exit 1; }
B=bin/cortiq-fast
strings $B | grep -q "routed experts past the card have no per-op route" || { echo MM7_STALE_BINARY; exit 1; }
bench() { # tag file env...
  local tag=$1 f=$2; shift 2
  local l=logs/mm7/$tag.log
  flock gpu.lock env "$@" CMF_STREAM_STATS=1 RUST_LOG=info timeout 3600 $B bench $f --tokens 64 --ctx 512 --ignore-eos --json > $l 2>&1
  echo "$tag: exit $? | $(grep -hoE '"(decode_tok_s_steady|prefill_tok_s|ttft_s)": [0-9.e+-]+' $l | tr '\n' ' ') | $(sed -E 's/\x1b\[[0-9;]*m//g' $l | grep -hoE 'RAM tier filled at load: .*|MiMo MoE placement: [a-z]+|expert stream: .*|routed experts: .*|graph declined: .*|pair micro-benchmark skipped.*|panicked.*|Out of Memory' | sort -u | cut -c1-140 | tr '\n' '|')"
}
m=cmf/big/minimax-m3.1-flash-q2tp-mix.cmf
if [ -s $m ]; then
  bench minimax_vk_card $m CMF_GPU=1 CMF_X=1
  bench minimax_cuda_card $m CMF_GPU=cuda CMF_X=1
  bench minimax_cuda_24g $m CMF_GPU=cuda CMF_GPU_VRAM_MB=24576
  rm -f $m
fi
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
