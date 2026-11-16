#!/bin/bash
# Rebuilt: the CUDA RAM tier takes every expert when they fit three quarters
# of the host's room (was half, the rest for the page cache) and is then
# filled and packed at load (ExpertStream::prepack). The pod's container
# holds no 62 GB of Ornith's experts in page-locked memory (it counts against
# the limit, unlike Vulkan's system-memory banks), so Apex-35B q4tp (19.8 GB)
# within 8 / 12 GB: greedy text and perplexity against the CPU, decode after
# 512 and that prompt's first token with the tier filled at load and as picks
# come (CMF_HYB_TIER_PREPACK=0), Vulkan at the same caps. After mm7.sh.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
mkdir -p logs/mm8 out/mm8
end=$((SECONDS + 43200))
until grep -qE "MM7_DONE|MM7_STALE_BINARY" logs/mm7.out 2>/dev/null || [ $SECONDS -gt $end ]; do sleep 30; done
bash scripts/build.sh fast > logs/mm8/build.log 2>&1
grep -q BUILD_FAST_DONE logs/mm8/build.log || { echo MM8_BUILD_FAILED; grep -E '^error|error:' -A8 logs/mm8/build.log | head -40; exit 1; }
B=bin/cortiq-fast
strings $B | grep -q CMF_HYB_TIER_PREPACK || { echo MM8_STALE_BINARY; exit 1; }
a=cmf/big/apex-35b-v1.0-q4tp.cmf
[ -s $a ] || timeout 3600 venv/bin/hf download infosave/Apex-35B-V1.0-cmf apex-35b-v1.0-q4tp.cmf --local-dir cmf/big > logs/mm8/dl.log 2>&1
ls -la $a || { echo MM8_DOWNLOAD_FAILED; tail -3 logs/mm8/dl.log; exit 1; }
P="Explain in two sentences why the sky is blue."
CMF_GPU=0 timeout 1800 $B run $a --prompt "$P" -n 32 --greedy 2>/dev/null | grep -v '^\[' > out/mm8/cpu.txt
echo "cpu text: $(tail -n +4 out/mm8/cpu.txt | tr '\n' ' ' | cut -c1-140)"
for x in "CMF_GPU=cuda CMF_GPU_VRAM_MB=8192" "CMF_GPU=1 CMF_GPU_VRAM_MB=8192"; do
  t=out/mm8/t_$(echo $x | tr ' =' '__').txt
  flock gpu.lock env $x RUST_LOG=info timeout 1800 $B run $a --prompt "$P" -n 32 --greedy 2>logs/mm8/t.err | grep -v '^\[' > $t
  if cmp -s $t out/mm8/cpu.txt; then echo "$x text == CPU"; else echo "$x text DIFFERS: $(tail -n +4 $t | tr '\n' ' ' | cut -c1-140)"; fi
done
CMF_GPU=0 timeout 3600 $B ppl $a --file prompts/rail.txt --tokens 256 > logs/mm8/ppl_cpu.log 2>&1
echo "ppl cpu: $(grep -hiE 'ppl|perplexity' logs/mm8/ppl_cpu.log | tail -1 | cut -c1-80)"
flock gpu.lock env CMF_GPU=cuda CMF_GPU_VRAM_MB=8192 RUST_LOG=info timeout 3600 $B ppl $a --file prompts/rail.txt --tokens 256 > logs/mm8/ppl_cuda8.log 2>&1
echo "ppl cuda 8 GB: $(grep -hiE 'ppl|perplexity' logs/mm8/ppl_cuda8.log | tail -1 | cut -c1-80)"
bench() { # tag env...
  local tag=$1; shift
  local l=logs/mm8/$tag.log
  flock gpu.lock env "$@" CMF_STREAM_STATS=1 RUST_LOG=info timeout 3600 $B bench $a --tokens 64 --ctx 512 --ignore-eos --json > $l 2>&1
  echo "$tag: exit $? | $(grep -hoE '"(decode_tok_s_steady|prefill_tok_s|ttft_s)": [0-9.e+-]+' $l | tr '\n' ' ') | $(sed -E 's/\x1b\[[0-9;]*m//g' $l | grep -hoE 'RAM tier of .*|RAM tier filled at load: .*|cuda hybrid expert stream: .*|routed experts: .*|graph declined: .*|panicked.*' | sort -u | cut -c1-150 | tr '\n' '|')"
}
for cap in 8192 12288; do
  bench cuda_$cap CMF_GPU=cuda CMF_GPU_VRAM_MB=$cap
  bench cuda_${cap}_lazy CMF_GPU=cuda CMF_GPU_VRAM_MB=$cap CMF_HYB_TIER_PREPACK=0
  bench vk_$cap CMF_GPU=1 CMF_GPU_VRAM_MB=$cap
done
bench cuda_card CMF_GPU=cuda CMF_X=1
bench vk_card CMF_GPU=1 CMF_X=1
echo MM8_DONE
