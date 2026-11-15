#!/bin/bash
# Flash-Next CUDA within 12 / 16 GB, zero copy against admitting every miss,
# run back to back (fnx1 showed sweep1/sweep4's Flash-Next numbers were taken
# with its pages evicted by Ornith before it). After vcache2.sh.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
mkdir -p logs/fnx2
end=$((SECONDS + 14400))
until grep -qE "VCACHE2_DONE|VCACHE2_SKIPPED" logs/vcache2.out 2>/dev/null || [ $SECONDS -gt $end ]; do sleep 30; done
B=bin/cortiq-fast
M=cmf/big/qwen38-flash-next-fast-code.cmf
# warm the file's pages once
flock gpu.lock env CMF_GPU=cuda CMF_GPU_VRAM_MB=16384 timeout 1800 $B bench $M --tokens 8 --ctx 64 --ignore-eos --json > logs/fnx2/warm.log 2>&1
run() { # tag capMB env...
  local tag=$1 cap=$2; shift 2
  local l=logs/fnx2/${tag}_$cap.log x="CMF_GPU_VRAM_MB=$cap"
  [ $cap = 0 ] && x="CMF_X=1"
  flock gpu.lock env CMF_GPU=cuda $x "$@" RUST_LOG=info timeout 1800 $B bench $M --tokens 64 --ctx 512 --ignore-eos --json > $l 2>&1
  echo "$tag ${cap}MB: exit $? | $(grep -hoE '"(decode_tok_s_steady|ttft_s)": [0-9.e+-]+' $l | tr '\n' ' ')"
}
for cap in 12288 16384; do
  run zc $cap CMF_QWEN_ZERO_COPY=1
  run lru $cap CMF_QWEN_ZERO_COPY=0
done
run zc 0 CMF_QWEN_ZERO_COPY=1
echo FNX2_DONE
