#!/bin/bash
# Ornith-1.5-122B q4tp (64 GB) on Vulkan, past the 46 GB card: why the MoE
# block falls to the CPU. Residency lines (CMF_MOE_RES), the declines, decode
# 64 tokens and after a 512-token prompt. After glm4.sh.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
mkdir -p logs/orn3 cmf/big
end=$((SECONDS + 14400))
until grep -qE "GLM4_DONE" logs/glm4.out 2>/dev/null || [ $SECONDS -gt $end ]; do sleep 30; done
f=cmf/big/ornith-1.5-122b-q4tp.cmf
[ -s $f ] || timeout 3600 venv/bin/hf download infosave/Ornith-1.5-122B-cmf ornith-1.5-122b-q4tp.cmf --local-dir cmf/big > logs/orn3/dl.log 2>&1
[ -s $f ] || { echo ORN3_DOWNLOAD_FAILED; exit 1; }
B=bin/cortiq-fast
flock gpu.lock env CMF_GPU=1 CMF_MOE_RES=1 RUST_LOG=info timeout 2400 $B bench $f --tokens 64 --ignore-eos --json > logs/orn3/d64.log 2>&1
echo "vulkan 64: exit $? | $(grep -hoE '"(decode_tok_s_steady|prefill_tok_s|ttft_s)": [0-9.e+-]+' logs/orn3/d64.log | tr '\n' ' ')"
sed -E 's/\x1b\[[0-9;]*m//g' logs/orn3/d64.log | grep -iE 'declin|refus|budget|probe|resid|tier|dynamic|moe' | sed -E 's/^.*(INFO|WARN|ERROR) //' | cut -c1-200 | sort | uniq -c | sort -rn | head -14 | sed 's/^/      /'
flock gpu.lock env CMF_GPU=1 CMF_MOE_RES=1 timeout 3600 $B bench $f --tokens 64 --ctx 512 --ignore-eos --json > logs/orn3/p512.log 2>&1
echo "vulkan 512+64: exit $? | $(grep -hoE '"(decode_tok_s_steady|prefill_tok_s|ttft_s)": [0-9.e+-]+' logs/orn3/p512.log | tr '\n' ' ')"
grep -hoE "residency: .*" logs/orn3/p512.log | tail -2 | cut -c1-220 | sed 's/^/      /'
echo ORN3_DONE
