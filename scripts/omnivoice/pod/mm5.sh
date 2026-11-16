#!/bin/bash
# Rebuilt: wgpu runs the whole stack when every layer's dense part fits the
# card and the experts can live in system memory (no layer prefix: MiMo's
# prompt walked the host past the leading layers, TTFT 78 s); CUDA stages a
# prompt's streamed layers only when the streamed experts are at most twice
# the host's room (MiMo / GLM decoded slower after a staged prompt), and a
# staged chunk copies back only the ranks a layer keeps (Ornith within 24 GB:
# TTFT 12.3 s against 10.5). MiniMax-M3.1 on Vulkan and CUDA again, GLM on
# CUDA (the small path again: 17.7 tok/s, TTFT 36.5 s), Ornith CUDA 24 GB;
# then MiniMax goes. After mm4.sh.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
mkdir -p logs/mm5 out/mm5
end=$((SECONDS + 43200))
until grep -qE "MM4_DONE|MM4_DOWNLOAD_FAILED|MM4_BUILD_FAILED" logs/mm4.out 2>/dev/null || [ $SECONDS -gt $end ]; do sleep 30; done
bash scripts/build.sh fast > logs/mm5/build.log 2>&1
grep -q BUILD_FAST_DONE logs/mm5/build.log || { echo MM5_BUILD_FAILED; grep -E '^error|error:' -A8 logs/mm5/build.log | head -40; exit 1; }
B=bin/cortiq-fast
m=cmf/big/minimax-m3.1-flash-q2tp-mix.cmf
P="Explain in two sentences why the sky is blue."
bench() { # tag file env...
  local tag=$1 f=$2; shift 2
  local l=logs/mm5/$tag.log
  flock gpu.lock env "$@" CMF_CUDA_DEBUG=1 RUST_LOG=info timeout 3600 $B bench $f --tokens 64 --ctx 512 --ignore-eos --json > $l 2>&1
  echo "$tag: exit $? | $(grep -hoE '"(decode_tok_s_steady|prefill_tok_s|ttft_s)": [0-9.e+-]+' $l | tr '\n' ' ') | $(sed -E 's/\x1b\[[0-9;]*m//g' $l | grep -hoE 'routed experts: .*|batch graph declined: .*' | sort -u | cut -c1-120 | tr '\n' '|')"
}
if [ -s $m ]; then
  flock gpu.lock env CMF_GPU=1 RUST_LOG=info timeout 3600 $B run $m --prompt "$P" -n 24 --greedy > out/mm5/t_1.txt 2> logs/mm5/t_1.err
  echo "minimax vulkan text: $(grep -v '^\[' out/mm5/t_1.txt | tail -n +6 | tr '\n' ' ' | cut -c1-160)"
  if cmp -s <(grep -v '^\[' out/mm5/t_1.txt) out/mm3/cpu.txt; then echo "   == CPU"; else echo "   DIFFERS from CPU"; fi
  bench minimax_vk_card $m CMF_GPU=1
  bench minimax_cuda_card $m CMF_GPU=cuda
  bench minimax_vk_24g $m CMF_GPU=1 CMF_GPU_VRAM_MB=24576
fi
bench glm_cuda_card cmf/big/glm-5.3-flash-q2tp.cmf CMF_GPU=cuda
bench orn_cuda_24g cmf/big/ornith-1.5-122b-q4tp.cmf CMF_GPU=cuda CMF_GPU_VRAM_MB=24576
rm -f $m
echo MM5_DONE
