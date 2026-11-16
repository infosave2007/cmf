#!/bin/bash
# CUDA on the small cards: Ornith q4tp within 12 / 16 GB streams through few
# slots (5.7 / 7.0 tok/s) where Vulkan reads RAM-resident banks in place
# (7.9 / 8.3). The CUDA host tier (experts in mapped page-locked memory read
# over the bus, CMF_HYB_HOST_TIER=1) against the stream, decode after 512.
# After glmv1.sh.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
mkdir -p logs/csmall1
end=$((SECONDS + 14400))
until grep -qE "GLMV1_DONE|GLMV1_BUILD_FAILED" logs/glmv1.out 2>/dev/null || [ $SECONDS -gt $end ]; do sleep 30; done
B=bin/cortiq-fast
M=cmf/big/ornith-1.5-122b-q4tp.cmf
run() { # tag capMB env...
  local tag=$1 cap=$2; shift 2
  local l=logs/csmall1/${tag}_$cap.log
  flock gpu.lock env CMF_GPU=cuda CMF_GPU_VRAM_MB=$cap "$@" RUST_LOG=info timeout 2400 $B bench $M --tokens 64 --ctx 512 --ignore-eos --json > $l 2>&1
  echo "$tag ${cap}MB: exit $? | $(grep -hoE '"(decode_tok_s_steady|ttft_s)": [0-9.e+-]+' $l | tr '\n' ' ')| $(sed -E 's/\x1b\[[0-9;]*m//g' $l | grep -oE 'cuda hybrid: [0-9.]+ GB of a plan.s[^|;]*' | sort -u | head -2 | tr '\n' ';' | cut -c1-300)"
}
for cap in 12288 16384; do
  run host $cap CMF_HYB_HOST_TIER=1
  run stream $cap
done
echo CSMALL1_DONE
