#!/bin/bash
# mm7b.sh stopped during MiMo's whole-card Vulkan bench; this one appends to
# logs/mm7.out (mm8.sh and i153.sh wait on its markers). Rebuilt first: the
# CUDA prompt stages a streamed layer when the small path's prompt chunks so
# far built most of their picks (the stream measures it; MiniMax-M3.1 and
# MiMo-V2.6 have the same expected spread, 363 s against 54 s on the small
# path). MiMo on Vulkan within 24 GB, on CUDA on the card and within 24 GB;
# GLM-5.3 on CUDA on the card; then mm7.sh's Flash-Next runs.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
mkdir -p logs/mm7 out/mm7
end=$((SECONDS + 7200))
while kill -0 2706617 2>/dev/null && [ $SECONDS -lt $end ]; do sleep 15; done
bench_line() { # tag log
  echo "$1: $(grep -hoE '"(decode_tok_s_steady|prefill_tok_s|ttft_s)": [0-9.e+-]+' $2 | tr '\n' ' ') | $(sed -E 's/\x1b\[[0-9;]*m//g' $2 | grep -hoE 'prompt chunks [a-z ]+: .*|RAM tier filled at load: .*|MiMo MoE placement: [a-z]+|expert stream: .*|routed experts: .*|graph declined: .*|layer prefix.*|pair micro-benchmark skipped.*|panicked.*|Out of Memory' | sort -u | cut -c1-140 | tr '\n' '|')"
}
bench_line mimo_vk_card logs/mm7/mimo_vk_card.log
bash scripts/build.sh fast > logs/mm7/build_c.log 2>&1
grep -q BUILD_FAST_DONE logs/mm7/build_c.log || { echo MM7C_BUILD_FAILED; grep -E '^error|error:' -A8 logs/mm7/build_c.log | head -40; echo MM7_DONE; exit 1; }
B=bin/cortiq-fast
strings $B | grep -q "prompt chunks staged: the small path built" || { echo MM7_STALE_BINARY; exit 1; }
bench() { # tag file env...
  local tag=$1 f=$2; shift 2
  local l=logs/mm7/$tag.log
  flock gpu.lock env "$@" CMF_STREAM_STATS=1 RUST_LOG=info timeout 3600 $B bench $f --tokens 64 --ctx 512 --ignore-eos --json > $l 2>&1
  echo -n "exit $? "; bench_line $tag $l
}
mi=cmf/big/MiMo-V2.6-Flash-RL-q4tp.cmf
if [ -s $mi ]; then
  bench mimo_vk_24g $mi CMF_GPU=1 CMF_GPU_VRAM_MB=24576
  bench mimo_cuda_card $mi CMF_GPU=cuda CMF_X=1
  bench mimo_cuda_24g $mi CMF_GPU=cuda CMF_GPU_VRAM_MB=24576
  rm -f $mi
fi
bench glm_cuda_card cmf/big/glm-5.3-flash-q2tp.cmf CMF_GPU=cuda CMF_X=1
fx=cmf/big/qwen38-flash-next-fast-code.cmf
[ -s $fx ] || timeout 3600 venv/bin/hf download infosave/Qwen3.8-Flash-Next-Fast-Code-cmf qwen38-flash-next-fast-code.cmf --local-dir cmf/big > logs/mm7/dl_fnx.log 2>&1
if ls -la $fx; then
  for cap in 12288 16384 24576 32768; do bench fnx_vk_$cap $fx CMF_GPU=1 CMF_GPU_VRAM_MB=$cap; done
  bench fnx_vk_card $fx CMF_GPU=1 CMF_X=1
else
  echo "FNX_DOWNLOAD_FAILED: $(tail -3 logs/mm7/dl_fnx.log)"
fi
echo MM7_DONE
