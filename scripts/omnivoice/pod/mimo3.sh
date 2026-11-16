#!/bin/bash
# MiMo-V2.6 q4tp (164 GB) on Vulkan again with the RAM tier filled on several
# threads and the next layer read ahead (mimo2.sh, one thread: whole card 1.85
# tok/s, TTFT 180 s). Decode after 512 over 64 and 256 with the stream's
# counters, and within 24 GB. Builds first. After vsmall2.sh.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
mkdir -p logs/mimo3
end=$((SECONDS + 21600))
until grep -qE "VSMALL2_DONE|VSMALL2_STALE_BINARY" logs/vsmall2.out 2>/dev/null || [ $SECONDS -gt $end ]; do sleep 30; done
# rebuilt: CUDA's RAM tier keeps a built expert's card layout (one DMA copy
# for its next build instead of a repack with its waits: 0.4 ms a build,
# ~345 builds a token within 12 GB)
bash scripts/build.sh fast > logs/mimo3/build.log 2>&1
grep -q BUILD_FAST_DONE logs/mimo3/build.log || { echo MIMO3_STALE_BINARY; grep -E '^error|error:' -A8 logs/mimo3/build.log | head -40; exit 1; }
B=bin/cortiq-fast; f=cmf/big/MiMo-V2.6-Flash-RL-q4tp.cmf
strings $B | grep -q CMF_HYB_TIER_PACK || { echo MIMO3_STALE_BINARY; exit 1; }
# Ornith q4tp on CUDA first (its pages are warm from vsmall2.sh): perplexity
# within 12 GB (CPU 3.049) and decode after 512, packed tier against not
o=cmf/big/ornith-1.5-122b-q4tp.cmf
flock gpu.lock env CMF_GPU=cuda CMF_GPU_VRAM_MB=12288 RUST_LOG=info timeout 3600 $B ppl $o --file prompts/rail.txt --tokens 512 > logs/mimo3/orn_ppl12.log 2>&1
echo "ornith cuda 12288MB ppl (3.049): exit $? | $(grep -hiE 'ppl|perplexity' logs/mimo3/orn_ppl12.log | tail -1 | cut -c1-120)"
for x in CMF_X=1 CMF_WGPU_EXPERT_OUT=fit; do
  l=logs/mimo3/orn_vk0_${x%%=*}.log
  flock gpu.lock env CMF_GPU=1 $x RUST_LOG=info timeout 3600 $B bench $o --tokens 256 --ctx 512 --ignore-eos --json > $l 2>&1
  echo "ornith vulkan card $x 512+256: exit $? | $(grep -hoE '"(decode_tok_s_steady|ttft_s)": [0-9.e+-]+' $l | tr '\n' ' ') | $(sed -E 's/\x1b\[[0-9;]*m//g' $l | grep -hoE 'routed experts: .*|system memory (falls|counts).*' | sort -u | cut -c1-150 | tr '\n' '|')"
done
for cap in 12288 16384 24576 0; do
  for pk in 1 0; do
    [ $cap = 24576 ] || [ $cap = 0 ] && [ $pk = 0 ] && continue
    l=logs/mimo3/orn_${cap}_pack$pk.log; x="CMF_GPU_VRAM_MB=$cap"; [ $cap = 0 ] && x="CMF_X=1"
    flock gpu.lock env CMF_GPU=cuda $x CMF_HYB_TIER_PACK=$pk CMF_CUDA_DEBUG=1 RUST_LOG=info timeout 3600 $B bench $o --tokens 64 --ctx 512 --ignore-eos --json > $l 2>&1
    echo "ornith cuda ${cap}MB pack=$pk 512+64: exit $? | $(grep -hoE '"(decode_tok_s_steady|ttft_s)": [0-9.e+-]+' $l | tr '\n' ' ') | $(grep -hoE 'cuda hybrid expert stream: .*' $l | tail -1 | cut -c1-200)"
  done
done
run() { # tokens capMB(0 = the card)
  local n=$1 cap=$2
  local l=logs/mimo3/v_${n}_$cap.log x="CMF_GPU_VRAM_MB=$cap"
  [ $cap = 0 ] && x="CMF_X=1"
  flock gpu.lock env CMF_GPU=1 $x CMF_STREAM_STATS=1 RUST_LOG=info timeout 3600 $B bench $f --tokens $n --ctx 512 --ignore-eos --json > $l 2>&1
  echo "vulkan ${cap}MB 512+$n: exit $? | $(grep -hoE '"(decode_tok_s_steady|prefill_tok_s|ttft_s)": [0-9.e+-]+' $l | tr '\n' ' ') | $(sed -E 's/\x1b\[[0-9;]*m//g' $l | grep -hoE 'routed experts: .*|system memory (falls|counts).*' | sort -u | cut -c1-150 | tr '\n' '|')"
}
run 64 0
run 256 0
run 64 24576
l=logs/mimo3/s_64_0.log
flock gpu.lock env CMF_GPU=1 CMF_WGPU_EXPERT_STREAM=1 CMF_STREAM_STATS=1 RUST_LOG=info timeout 3600 $B bench $f --tokens 64 --ctx 512 --ignore-eos --json > $l 2>&1
echo "vulkan stream 512+64: exit $? | $(grep -hoE '"(decode_tok_s_steady|prefill_tok_s|ttft_s)": [0-9.e+-]+' $l | tr '\n' ' ') | $(grep -hoE 'wgpu expert stream: .*' $l | tail -1)"
# CUDA: staged prompt picks with the RAM tier's share kept (mimo2: 5.2 tok/s,
# TTFT 18.8 s; the small path in 0.8.99's campaign 13.6) against the small path
for st in 1 0; do
  l=logs/mimo3/c_64_stage$st.log
  flock gpu.lock env CMF_GPU=cuda CMF_HYB_STREAM_STAGE=$st RUST_LOG=info timeout 3600 $B bench $f --tokens 64 --ctx 512 --ignore-eos --json > $l 2>&1
  echo "cuda stage=$st 512+64: exit $? | $(grep -hoE '"(decode_tok_s_steady|prefill_tok_s|ttft_s)": [0-9.e+-]+' $l | tr '\n' ' ')"
done
echo MIMO3_DONE
