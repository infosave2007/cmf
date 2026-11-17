#!/bin/bash
# GLM-5.3 on CUDA with the RAM tier under its card slots (`moe_row`): decode
# after 512 at 16, 24 GB and the whole card, the slots' and the tier's hit
# lines. After sweep2.sh (same build).
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
mkdir -p logs/sweep3
end=$((SECONDS + 14400))
until grep -qE "SWEEP2_DONE|SWEEP2_BUILD_FAILED" logs/sweep2.out 2>/dev/null || [ $SECONDS -gt $end ]; do sleep 30; done
B=bin/cortiq-fast; glm=cmf/big/glm-5.3-flash-q2tp.cmf
for cap in 16384 24576 0; do
  x="CMF_GPU_VRAM_MB=$cap"; [ $cap = 0 ] && x="CMF_X=1"
  l=logs/sweep3/glm_$cap.log
  flock gpu.lock env CMF_GPU=cuda $x CMF_CUDA_DEBUG=1 RUST_LOG=info timeout 3600 $B bench $glm --tokens 64 --ctx 512 --ignore-eos --json > $l 2>&1
  echo "glm cuda ${cap}MB: exit $? | $(grep -hoE '"(decode_tok_s_steady|ttft_s)": [0-9.e+-]+' $l | tr '\n' ' ')| $(sed -E 's/\x1b\[[0-9;]*m//g' $l | grep -oE 'cuda moe rows: [0-9]+ card slots[^)]*\)|RAM tier of [0-9]+ experts[^)]*\)|cuda moe rows: [0-9]+ slots.*' | sort -u | tail -3 | tr '\n' ';' | cut -c1-300)"
done
echo SWEEP3_DONE
