#!/bin/bash
# Flash-Next on Vulkan within 32 GB twice more: mm7 gave 28.4 tok/s against
# vfnx's 32.4 on the qwen4 path these changes do not touch (the same plan in
# both logs). Then Flash-Next goes, once i153b.sh (issue #153's timestamps,
# same file) is done too. After mm9.sh.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
mkdir -p logs/mm10
end=$((SECONDS + 43200))
until grep -qE "MM9_DONE|MM9_[A-Z_]*FAILED|MM9_STALE_BINARY" logs/mm9.out 2>/dev/null || [ $SECONDS -gt $end ]; do sleep 30; done
grep -q I153_DONE logs/i153.out || { echo MM10_I153_NOT_DONE; exit 1; }
B=bin/cortiq-fast; fx=cmf/big/qwen38-flash-next-fast-code.cmf
for r in 1 2; do
  l=logs/mm10/fnx_vk_32768_$r.log
  flock gpu.lock env CMF_GPU=1 CMF_GPU_VRAM_MB=32768 RUST_LOG=info timeout 1800 $B bench $fx --tokens 64 --ctx 512 --ignore-eos --json > $l 2>&1
  echo "fnx_vk_32768 run $r: exit $? | $(grep -hoE '"(decode_tok_s_steady|prefill_tok_s|ttft_s)": [0-9.e+-]+' $l | tr '\n' ' ')"
done
until grep -q I153B_DONE logs/i153b.out 2>/dev/null || [ $SECONDS -gt $end ]; do sleep 30; done
grep -q I153B_DONE logs/i153b.out 2>/dev/null && rm -f $fx
echo MM10_DONE
