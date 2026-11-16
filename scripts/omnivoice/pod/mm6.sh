#!/bin/bash
# Rebuilt: an f16 K/V mirror only for a geometry with f16 kernels on both
# graphs (MiniMax-M3.1's mirror made the token and batch graphs decline, its
# experts ran on the host: Vulkan 0.17 tok/s, TTFT 1508 s within 24 GB, the
# whole card out of memory); the CUDA hybrid probes whether page-locked host
# memory counts against the container's limit (Vulkan's banks do not) and
# sizes its RAM tier from MemAvailable when it does not. Checks: q2tp-v2
# perplexity grouped vs per-row at 256 tokens (3.180); Ornith q4tp CUDA at
# 12/16/24 GB and the card (was 6.30/6.53/25.3/37.5); GLM CUDA (16.1, TTFT
# 40.6); MiniMax again (re-download): texts vs CPU, Vulkan and CUDA at the
# card / 24 / 16 GB. After glmw.sh.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
mkdir -p logs/mm6 out/mm6
end=$((SECONDS + 43200))
until grep -qE "GLMW_DONE" logs/glmw.out 2>/dev/null || [ $SECONDS -gt $end ]; do sleep 30; done
bash scripts/build.sh fast > logs/mm6/build.log 2>&1
grep -q BUILD_FAST_DONE logs/mm6/build.log || { echo MM6_BUILD_FAILED; grep -E '^error|error:' -A8 logs/mm6/build.log | head -40; exit 1; }
B=bin/cortiq-fast
strings $B | grep -q CMF_HYB_PINNED_PROBE || { echo MM6_STALE_BINARY; exit 1; }
q2=cmf/big/ornith-1.5-122b-q2tp-v2.cmf
for x in CMF_X=1 CMF_MOEG_Q2=0; do
  flock gpu.lock env CMF_GPU=1 $x RUST_LOG=info timeout 3600 $B ppl $q2 --file prompts/rail.txt --tokens 256 > logs/mm6/ppl_$x.log 2>&1
  echo "q2tp-v2 vulkan ppl $x (3.180): exit $? | $(grep -hiE 'ppl|perplexity' logs/mm6/ppl_$x.log | tail -1 | cut -c1-120)"
done
rm -f $q2
bench() { # tag file env...
  local tag=$1 f=$2; shift 2
  local l=logs/mm6/$tag.log
  flock gpu.lock env "$@" CMF_STREAM_STATS=1 RUST_LOG=info timeout 3600 $B bench $f --tokens 64 --ctx 512 --ignore-eos --json > $l 2>&1
  echo "$tag: exit $? | $(grep -hoE '"(decode_tok_s_steady|prefill_tok_s|ttft_s)": [0-9.e+-]+' $l | tr '\n' ' ') | $(sed -E 's/\x1b\[[0-9;]*m//g' $l | grep -hoE 'page-locked host memory [a-z ]+|RAM tier of .*|expert stream: .*|routed experts: .*|graph declined: .*|panicked.*|Out of Memory' | sort -u | cut -c1-140 | tr '\n' '|')"
}
o=cmf/big/ornith-1.5-122b-q4tp.cmf
for cap in 12288 16384 24576; do bench orn_cuda_$cap $o CMF_GPU=cuda CMF_GPU_VRAM_MB=$cap; done
bench orn_cuda_card $o CMF_GPU=cuda CMF_X=1
g=cmf/big/glm-5.3-flash-q2tp.cmf
bench glm_cuda_card $g CMF_GPU=cuda CMF_X=1
bench glm_cuda_card_nostage $g CMF_GPU=cuda CMF_HYB_STREAM_STAGE=0
bench glm_cuda_24g $g CMF_GPU=cuda CMF_GPU_VRAM_MB=24576
m=cmf/big/minimax-m3.1-flash-q2tp-mix.cmf
[ -s $m ] || timeout 14400 venv/bin/hf download infosave/MiniMax-M3.1-Flash-cmf minimax-m3.1-flash-q2tp-mix.cmf --local-dir cmf/big > logs/mm6/dl.log 2>&1
ls -la $m || { echo MM6_DOWNLOAD_FAILED; tail -5 logs/mm6/dl.log; exit 1; }
P="Explain in two sentences why the sky is blue."
for be in 1 cuda; do
  flock gpu.lock env CMF_GPU=$be RUST_LOG=info timeout 3600 $B run $m --prompt "$P" -n 24 --greedy > out/mm6/t_$be.txt 2> logs/mm6/t_$be.err
  echo "minimax $be text exit $?: $(grep -v '^\[' out/mm6/t_$be.txt | tail -n +6 | tr '\n' ' ' | cut -c1-120)"
  if cmp -s <(grep -v '^\[' out/mm6/t_$be.txt) out/mm3/cpu.txt; then echo "   == CPU"; else echo "   DIFFERS from CPU"; fi
done
bench minimax_vk_card $m CMF_GPU=1 CMF_X=1
bench minimax_vk_24g $m CMF_GPU=1 CMF_GPU_VRAM_MB=24576
bench minimax_vk_16g $m CMF_GPU=1 CMF_GPU_VRAM_MB=16384
bench minimax_cuda_card $m CMF_GPU=cuda CMF_X=1
bench minimax_cuda_24g $m CMF_GPU=cuda CMF_GPU_VRAM_MB=24576
bench minimax_cuda_16g $m CMF_GPU=cuda CMF_GPU_VRAM_MB=16384
echo MM6_DONE
