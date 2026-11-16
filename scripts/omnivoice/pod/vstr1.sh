#!/bin/bash
# Vulkan experts streamed through card slots from the file (`moe_stream`),
# for a stack past the card and the host's room: Ornith-122B q4tp within 24 /
# 32 GB on the 46 GB container (in place: 9.6 / 15.4 tok/s in vfinal / vcache5),
# decode and TTFT after 512, CMF_WGPU_EXPERT_STREAM=0 against it at 24 GB, the
# perplexity at 24 GB (CPU 3.049).
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
mkdir -p logs/vstr1
bash scripts/build.sh fast > logs/vstr1/build.log 2>&1
grep -q BUILD_FAST_DONE logs/vstr1/build.log || { echo VSTR1_BUILD_FAILED; grep -E '^error|error:' -A8 logs/vstr1/build.log | head -60; exit 1; }
strings bin/cortiq-fast | grep -q CMF_WGPU_EXPERT_STREAM || { echo VSTR1_STALE_BINARY; exit 1; }
B=bin/cortiq-fast
M=cmf/big/ornith-1.5-122b-q4tp.cmf
run() { # tag capMB env...
  local tag=$1 cap=$2; shift 2
  local l=logs/vstr1/${tag}_$cap.log
  flock gpu.lock env CMF_GPU=1 CMF_GPU_VRAM_MB=$cap "$@" RUST_LOG=info timeout 2400 $B bench $M --tokens 64 --ctx 512 --ignore-eos --json > $l 2>&1
  echo "$tag ${cap}MB: exit $? | $(grep -hoE '"(decode_tok_s_steady|ttft_s)": [0-9.e+-]+' $l | tr '\n' ' ')| $(sed -E 's/\x1b\[[0-9;]*m//g' $l | grep -oE 'routed experts: [^|]*|expert stream: [^|]*|declin[^|]*' | sort | uniq -c | tr '\n' ';' | cut -c1-400)"
}
run str 24576
run str 32768
run off 24576 CMF_WGPU_EXPERT_STREAM=0
flock gpu.lock env CMF_GPU=1 CMF_GPU_VRAM_MB=24576 timeout 3600 $B ppl $M --file prompts/rail.txt --tokens 256 > logs/vstr1/ppl_24576.log 2>&1
echo "ppl vulkan stream 24576MB (CPU 3.049): exit $? | $(grep -hiE 'ppl|perplexity' logs/vstr1/ppl_24576.log | tail -1 | cut -c1-200)"
echo VSTR1_DONE
