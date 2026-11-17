#!/bin/bash
# The matrix's missing cells on the final build: Vulkan on the whole card for
# Ornith q4tp (two passes, head reserved) and Flash-Next, decode after 512.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
mkdir -p logs/fill1
B=bin/cortiq-fast
for f in ornith-1.5-122b-q4tp qwen38-flash-next-fast-code; do
  l=logs/fill1/$f.log
  flock gpu.lock env CMF_GPU=1 RUST_LOG=info timeout 2400 $B bench cmf/big/$f.cmf --tokens 64 --ctx 512 --ignore-eos --json > $l 2>&1
  echo "$f vulkan whole card: exit $? | $(grep -hoE '"(decode_tok_s_steady|ttft_s)": [0-9.e+-]+' $l | tr '\n' ' ')| $(sed -E 's/\x1b\[[0-9;]*m//g' $l | grep -cE 'head-only: lm_head resolve') head-only declines"
done
echo FILL1_DONE
