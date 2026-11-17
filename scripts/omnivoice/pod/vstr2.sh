#!/bin/bash
# Vulkan experts streamed through card slots, second form: pools a group of
# layers each (one binding held 1304 slots: 1.7 tok/s in vstr1) over a RAM
# tier in GPU-visible system memory. Ornith-122B q4tp within 24 GB (in place
# 8.8-9.6), 16 GB, decode and TTFT after 512; the perplexity at 24 GB.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
mkdir -p logs/vstr2
bash scripts/build.sh fast > logs/vstr2/build.log 2>&1
grep -q BUILD_FAST_DONE logs/vstr2/build.log || { echo VSTR2_BUILD_FAILED; grep -E '^error|error:' -A8 logs/vstr2/build.log | head -60; exit 1; }
strings bin/cortiq-fast | grep -q CMF_WGPU_EXPERT_STREAM || { echo VSTR2_STALE_BINARY; exit 1; }
B=bin/cortiq-fast
M=cmf/big/ornith-1.5-122b-q4tp.cmf
run() { # tag capMB env...
  local tag=$1 cap=$2; shift 2
  local l=logs/vstr2/${tag}_$cap.log
  flock gpu.lock env CMF_GPU=1 CMF_GPU_VRAM_MB=$cap "$@" RUST_LOG=info timeout 2400 $B bench $M --tokens 64 --ctx 512 --ignore-eos --json > $l 2>&1
  echo "$tag ${cap}MB: exit $? | $(grep -hoE '"(decode_tok_s_steady|ttft_s)": [0-9.e+-]+' $l | tr '\n' ' ')| $(sed -E 's/\x1b\[[0-9;]*m//g' $l | grep -oE 'routed experts: [^|]*|expert stream: [^|]*|declin[^|]*' | sort | uniq -c | tr '\n' ';' | cut -c1-400)"
}
run str 24576
run str 16384
run notier 24576 CMF_WGPU_EXPERT_TIER=0
flock gpu.lock env CMF_GPU=1 CMF_GPU_VRAM_MB=24576 timeout 3600 $B ppl $M --file prompts/rail.txt --tokens 256 > logs/vstr2/ppl_24576.log 2>&1
echo "ppl vulkan stream 24576MB (CPU 3.049): exit $? | $(grep -hiE 'ppl|perplexity' logs/vstr2/ppl_24576.log | tail -1 | cut -c1-200)"
echo VSTR2_DONE
