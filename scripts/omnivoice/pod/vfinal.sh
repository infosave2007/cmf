#!/bin/bash
# The final Vulkan matrix for Ornith q4tp on the small cards (12 / 16 / 24 GB:
# the pod's RAM holds too few banks there for card slots, the staging bank
# still serves the prompt), TTFT and decode after 512. After the q2 diag.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
mkdir -p logs/vfinal
end=$((SECONDS + 7200))
until [ -s logs/diag/q2.done ] || [ $SECONDS -gt $end ]; do sleep 30; done
B=bin/cortiq-fast
for cap in 12288 16384 24576; do
  l=logs/vfinal/q4_$cap.log
  flock gpu.lock env CMF_GPU=1 CMF_GPU_VRAM_MB=$cap RUST_LOG=info timeout 2400 $B bench cmf/big/ornith-1.5-122b-q4tp.cmf --tokens 64 --ctx 512 --ignore-eos --json > $l 2>&1
  echo "q4 ${cap}MB: exit $? | $(grep -hoE '"(decode_tok_s_steady|ttft_s)": [0-9.e+-]+' $l | tr '\n' ' ')| $(sed -E 's/\x1b\[[0-9;]*m//g' $l | grep -oE 'routed experts: [^|]*' | sort -u | cut -c1-200)"
done
echo VFINAL_DONE
