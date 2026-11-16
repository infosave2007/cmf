#!/bin/bash
# GLM-5.3 on Vulkan with DSA on the token graph (past index_topk) and the
# prompt's cache rows on the batch graph: with the selection cut to 64
# (CMF_GLM_DSA_TOPK=64) the 256-token perplexity runs the selection (CPU 2.728,
# CUDA 2.730); the prompt in 512-row chunks (was 256: TTFT 210 s), decode
# after 512. After vfnx.sh.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
mkdir -p logs/glmv2
end=$((SECONDS + 14400))
until grep -qE "VFNX_DONE" logs/vfnx.out 2>/dev/null || [ $SECONDS -gt $end ]; do sleep 30; done
bash scripts/build.sh fast > logs/glmv2/build.log 2>&1
grep -q BUILD_FAST_DONE logs/glmv2/build.log || { echo GLMV2_BUILD_FAILED; grep -E '^error|error:' -A8 logs/glmv2/build.log | head -60; exit 1; }
B=bin/cortiq-fast; f=cmf/big/glm-5.3-flash-q2tp.cmf
flock gpu.lock env CMF_GPU=1 CMF_GLM_DSA_TOPK=64 RUST_LOG=info timeout 5400 $B ppl $f --file prompts/rail.txt --tokens 256 > logs/glmv2/ppl64.log 2>&1
echo "ppl topk=64 vulkan (CPU 2.728): exit $? | $(grep -hiE 'ppl|perplexity' logs/glmv2/ppl64.log | tail -1 | cut -c1-200)"
sed -E 's/\x1b\[[0-9;]*m//g' logs/glmv2/ppl64.log | grep -iE 'glm5_next|declin|replay|refus|DSA' | cut -c30- | sort | uniq -c | head -6
flock gpu.lock env CMF_GPU=1 RUST_LOG=info timeout 3600 $B bench $f --tokens 64 --ctx 512 --ignore-eos --json > logs/glmv2/p512.log 2>&1
echo "vulkan 512+64: exit $? | $(grep -hoE '"(decode_tok_s_steady|prefill_tok_s|ttft_s)": [0-9.e+-]+' logs/glmv2/p512.log | tr '\n' ' ')"
echo GLMV2_DONE
