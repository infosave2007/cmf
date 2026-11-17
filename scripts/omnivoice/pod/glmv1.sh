#!/bin/bash
# GLM-5.3 on Vulkan: the stack on the wgpu graphs (KDA / NoPE MLA as Kimi's,
# DeepSeek-V4's four-stream mHC kernels, the 512-expert select, the routed
# experts' clamp, the experts streamed through card slots over a RAM tier:
# 116 GB file, 46 GB container). Texts, perplexity (CUDA / CPU 2.715), decode
# after 512. After vstr2.sh.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
mkdir -p logs/glmv1 out/glmv1
end=$((SECONDS + 10800))
until grep -qE "VSTR2_DONE|VSTR2_BUILD_FAILED|VSTR2_STALE" logs/vstr2.out 2>/dev/null || [ $SECONDS -gt $end ]; do sleep 30; done
bash scripts/build.sh fast > logs/glmv1/build.log 2>&1
grep -q BUILD_FAST_DONE logs/glmv1/build.log || { echo GLMV1_BUILD_FAILED; grep -E '^error|error:' -A8 logs/glmv1/build.log | head -60; exit 1; }
B=bin/cortiq-fast; f=cmf/big/glm-5.3-flash-q2tp.cmf
flock gpu.lock env CMF_GPU=1 RUST_LOG=info timeout 3600 $B run $f --prompt "What is 2+2? Answer briefly." --no-think -n 16 --greedy > out/glmv1/two.txt 2> logs/glmv1/two.err
echo "vulkan 2+2: $(grep -v '^\[' out/glmv1/two.txt | tail -n +5 | tr '\n' ' ' | cut -c1-80)"
echo "   $(sed -E 's/\x1b\[[0-9;]*m//g' logs/glmv1/two.err | grep -iE 'glm5_next|declin|refus|routed experts|expert stream|WARN' | cut -c30- | sort | uniq -c | head -10 | tr '\n' ';' | cut -c1-900)"
flock gpu.lock env CMF_GPU=1 RUST_LOG=info timeout 5400 $B ppl $f --file prompts/rail.txt --tokens 256 > logs/glmv1/ppl.log 2>&1
echo "ppl vulkan (CPU 2.715): exit $? | $(grep -hiE 'ppl|perplexity' logs/glmv1/ppl.log | tail -1 | cut -c1-200)"
flock gpu.lock env CMF_GPU=1 RUST_LOG=info timeout 3600 $B bench $f --tokens 64 --ctx 512 --ignore-eos --json > logs/glmv1/p512.log 2>&1
echo "vulkan 512+64: exit $? | $(grep -hoE '"(decode_tok_s_steady|prefill_tok_s|ttft_s)": [0-9.e+-]+' logs/glmv1/p512.log | tr '\n' ' ')"
echo GLMV1_DONE
