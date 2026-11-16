#!/bin/bash
# The reserve's share of the slots, interleaved A/B (one run drifts ±8 %:
# Ornith within 12 GB read 8.9-10.45 on one code path, the card's copy rate
# from VMM host memory unchanged at 19.5 GB/s). GLM-5.3 CUDA within 24 GB:
# the DSA mirrors in the reserve (mm15: 1140 slots, 8.65 tokens/s) against
# mm13's slots (1319, 10.55) by CMF_HYB_RESERVE_MB=1412 (1412 MB + the mirrors'
# 1660 = mm13's 3072). Ornith-122B q4tp within 12 GB and Apex-35B q4tp within
# 8 GB: the reserve at an eighth of the card (1.5 / 1 GiB) against 0.9.0's 3 GiB.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
mkdir -p logs/mm16
B=bin/cortiq-fast
a=cmf/big/apex-35b-v1.0-q4tp.cmf
[ -s $a ] || timeout 3600 venv/bin/hf download infosave/Apex-35B-V1.0-cmf apex-35b-v1.0-q4tp.cmf --local-dir cmf/big > logs/mm16/dl.log 2>&1 &
dl=$!
bench() { # tag file ctx env...
  local tag=$1 f=$2 ctx=$3; shift 3
  local l=logs/mm16/$tag.log
  flock gpu.lock env "$@" CMF_STREAM_STATS=1 RUST_LOG=info timeout 900 $B bench $f --tokens 64 --ctx $ctx --ignore-eos --json > $l 2>&1
  echo "$tag: exit $? | $(grep -hoE '"(decode_tok_s_steady|ttft_s)": [0-9.e+-]+' $l | tr '\n' ' ') | $(sed -E 's/\x1b\[[0-9;]*m//g' $l | grep -hoE 'expert stream of [0-9]+ card slots|graph declined: .*|batch graph failed.*|host replays.*' | sort -u | tr '\n' '|')"
}
g=cmf/big/glm-5.3-flash-q2tp.cmf
o=cmf/big/ornith-1.5-122b-q4tp.cmf
for i in 1 2; do
  bench glm_24g_new_$i $g 512 CMF_GPU=cuda CMF_GPU_VRAM_MB=24576
  bench glm_24g_old_$i $g 512 CMF_GPU=cuda CMF_GPU_VRAM_MB=24576 CMF_HYB_RESERVE_MB=1412
done
for i in 1 2; do
  bench orn_12g_r8_$i $o 512 CMF_GPU=cuda CMF_GPU_VRAM_MB=12288
  bench orn_12g_r3g_$i $o 512 CMF_GPU=cuda CMF_GPU_VRAM_MB=12288 CMF_HYB_RESERVE_MB=3072
done
wait $dl
ls -la $a || { echo "MM16_APEX_DOWNLOAD_FAILED: $(tail -2 logs/mm16/dl.log)"; echo MM16_DONE; exit 0; }
for i in 1 2; do
  bench apex_8g_r8_$i $a 512 CMF_GPU=cuda CMF_GPU_VRAM_MB=8192
  bench apex_8g_r3g_$i $a 512 CMF_GPU=cuda CMF_GPU_VRAM_MB=8192 CMF_HYB_RESERVE_MB=3072
done
echo MM16_DONE
