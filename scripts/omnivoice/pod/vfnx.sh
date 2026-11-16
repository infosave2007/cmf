#!/bin/bash
# Flash-Next on Vulkan measured cleanly (sweep1 ran it right after Ornith,
# its pages evicted): the file warmed once, then 12 / 16 / 24 / 32 GB and the
# whole card back to back, decode after 512. After vq2b.sh.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
mkdir -p logs/vfnx
end=$((SECONDS + 14400))
until grep -qE "VQ2B_DONE" logs/vq2b.out 2>/dev/null || [ $SECONDS -gt $end ]; do sleep 30; done
B=bin/cortiq-fast
M=cmf/big/qwen38-flash-next-fast-code.cmf
flock gpu.lock env CMF_GPU=1 CMF_GPU_VRAM_MB=16384 timeout 1800 $B bench $M --tokens 8 --ctx 64 --ignore-eos --json > logs/vfnx/warm.log 2>&1
for cap in 12288 16384 24576 32768 0; do
  l=logs/vfnx/v_$cap.log x="CMF_GPU_VRAM_MB=$cap"; [ $cap = 0 ] && x="CMF_X=1"
  flock gpu.lock env CMF_GPU=1 $x RUST_LOG=info timeout 1800 $B bench $M --tokens 64 --ctx 512 --ignore-eos --json > $l 2>&1
  echo "vulkan ${cap}MB: exit $? | $(grep -hoE '"(decode_tok_s_steady|ttft_s)": [0-9.e+-]+' $l | tr '\n' ' ')"
done
echo VFNX_DONE
