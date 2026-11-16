#!/bin/bash
# GLM-5.3's prompt on the CUDA hybrid's batch graph (was a token graph a
# position: 512 tokens in 60 s): the texts, TTFT and decode after 512 on the
# whole card. After glmg2.sh.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
mkdir -p logs/glmg3 out/glmg3
end=$((SECONDS + 14400))
until grep -q GLMG2_DONE logs/glmg2.out 2>/dev/null || [ $SECONDS -gt $end ]; do sleep 30; done
bash scripts/build.sh fast > logs/glmg3/build.log 2>&1
grep -q BUILD_FAST_DONE logs/glmg3/build.log || { echo GLMG3_BUILD_FAILED; grep -E '^error|error:' -A8 logs/glmg3/build.log | head -60; exit 1; }
strings bin/cortiq-fast | grep -q CMF_GLM_PROMPT_CHUNK || { echo GLMG3_STALE_BINARY; exit 1; }
B=bin/cortiq-fast; f=cmf/big/glm-5.3-flash-q2tp.cmf
P="Explain in two sentences why the sky is blue."
flock gpu.lock env CMF_GPU=cuda RUST_LOG=info timeout 3600 $B run $f --prompt "What is 2+2? Answer briefly." --no-think -n 16 --greedy > out/glmg3/two.txt 2> logs/glmg3/two.err
flock gpu.lock env CMF_GPU=cuda RUST_LOG=info timeout 3600 $B run $f --prompt "$P" -n 24 --greedy > out/glmg3/sky.txt 2> logs/glmg3/sky.err
echo "cuda: 2+2 $(grep -v '^\[' out/glmg3/two.txt | tail -n +5 | tr '\n' ' ' | cut -c1-60) | sky $(grep -v '^\[' out/glmg3/sky.txt | tail -n +5 | tr '\n' ' ' | cut -c1-200)"
echo "   $(sed -E 's/\x1b\[[0-9;]*m//g' logs/glmg3/sky.err | grep -E 'glm5_next|declin|refus' | cut -c30- | sort | uniq -c | head -6 | tr '\n' ';' | cut -c1-500)"
flock gpu.lock env CMF_GPU=cuda RUST_LOG=info timeout 3600 $B bench $f --tokens 64 --ctx 512 --ignore-eos --json > logs/glmg3/p512.log 2>&1
echo "cuda 512+64: exit $? | $(grep -hoE '"(decode_tok_s_steady|prefill_tok_s|ttft_s)": [0-9.e+-]+' logs/glmg3/p512.log | tr '\n' ' ')"
echo GLMG3_DONE
