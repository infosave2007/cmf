#!/bin/bash
# GLM-5.3 on Vulkan: by default its banks in GPU-visible system memory (the
# driver's, outside the container's limit on this pod) under card slots;
# the stream (CMF_WGPU_EXPERT_STREAM=1) with the RAM tier filled on several threads (one thread's
# page faults read the file at ~1.2 GB/s: the 512-token prompt stages every
# layer's 288 experts, TTFT 92.5 s) and the next streamed layer read ahead
# while the card runs the current one. Greedy text with the pools seeded from the prompt and without (equal),
# perplexity (CPU 2.715), decode after
# 512 over 64 and 256 tokens with the stream's counters; CUDA over 256 for the
# steady state beside it. Uses mimo3.sh's build. After mimo3.sh.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
mkdir -p logs/glmv3
end=$((SECONDS + 21600))
until grep -qE "MIMO3_DONE|MIMO3_STALE_BINARY" logs/mimo3.out 2>/dev/null || [ $SECONDS -gt $end ]; do sleep 30; done
B=bin/cortiq-fast; f=cmf/big/glm-5.3-flash-q2tp.cmf
strings $B | grep -q CMF_WGPU_EXPERT_FILL_THREADS || { echo GLMV3_STALE_BINARY; exit 1; }
# the pools seeded from the prompt's staging bank must not change a token
P="Explain in two sentences why the sky is blue."
for w in 1 0; do
  flock gpu.lock env CMF_GPU=1 CMF_WGPU_EXPERT_STREAM=1 CMF_WGPU_EXPERT_WARM=$w timeout 3600 $B run $f --prompt "$P" -n 24 --greedy 2>/dev/null | grep -v '^\[' > logs/glmv3/text_w$w.txt
done
echo "text: $(tail -n +5 logs/glmv3/text_w1.txt | tr '\n' ' ' | cut -c1-160)"
if cmp -s logs/glmv3/text_w1.txt logs/glmv3/text_w0.txt; then echo "   stream warm == cold"; else echo "   stream warm DIFFERS from cold"; fi
flock gpu.lock env CMF_GPU=1 RUST_LOG=info timeout 3600 $B run $f --prompt "$P" -n 24 --greedy 2>logs/glmv3/text_banks.err | grep -v '^\[' > logs/glmv3/text_banks.txt
if cmp -s logs/glmv3/text_w1.txt logs/glmv3/text_banks.txt; then echo "   banks == stream"; else echo "   banks DIFFER from stream: $(tail -n +5 logs/glmv3/text_banks.txt | tr '\n' ' ' | cut -c1-120)"; fi
sed -E 's/\x1b\[[0-9;]*m//g' logs/glmv3/text_banks.err | grep -hoE 'routed experts: .*|system memory (falls|counts).*' | sort -u | cut -c1-200
flock gpu.lock env CMF_GPU=1 RUST_LOG=info timeout 5400 $B ppl $f --file prompts/rail.txt --tokens 256 > logs/glmv3/ppl.log 2>&1
echo "ppl vulkan (CPU 2.715): exit $? | $(grep -hiE 'ppl|perplexity' logs/glmv3/ppl.log | tail -1 | cut -c1-200)"
for n in 64 256; do
  flock gpu.lock env CMF_GPU=1 CMF_STREAM_STATS=1 RUST_LOG=info timeout 3600 $B bench $f --tokens $n --ctx 512 --ignore-eos --json > logs/glmv3/v_$n.log 2>&1
  echo "vulkan 512+$n: exit $? | $(grep -hoE '"(decode_tok_s_steady|prefill_tok_s|ttft_s)": [0-9.e+-]+' logs/glmv3/v_$n.log | tr '\n' ' ') | $(sed -E 's/\x1b\[[0-9;]*m//g' logs/glmv3/v_$n.log | grep -hoE 'routed experts: .*' | tail -1 | cut -c1-150)"
done
for w in 1 0; do
  flock gpu.lock env CMF_GPU=1 CMF_WGPU_EXPERT_STREAM=1 CMF_WGPU_EXPERT_WARM=$w CMF_STREAM_STATS=1 RUST_LOG=info timeout 3600 $B bench $f --tokens 64 --ctx 512 --ignore-eos --json > logs/glmv3/s_64_w$w.log 2>&1
  echo "vulkan stream warm=$w 512+64: exit $? | $(grep -hoE '"(decode_tok_s_steady|prefill_tok_s|ttft_s)": [0-9.e+-]+' logs/glmv3/s_64_w$w.log | tr '\n' ' ') | $(grep -hoE 'wgpu expert stream: .*' logs/glmv3/s_64_w$w.log | tail -1)"
done
flock gpu.lock env CMF_GPU=cuda RUST_LOG=info timeout 3600 $B bench $f --tokens 256 --ctx 512 --ignore-eos --json > logs/glmv3/c_256.log 2>&1
echo "cuda 512+256: exit $? | $(grep -hoE '"(decode_tok_s_steady|prefill_tok_s|ttft_s)": [0-9.e+-]+' logs/glmv3/c_256.log | tr '\n' ' ')"
# within 24 GB with the staged prompt picks (16 / 24 GB were 4.1 / 5.7 on the small path)
for be in cuda 1; do
  flock gpu.lock env CMF_GPU=$be CMF_GPU_VRAM_MB=24576 CMF_STREAM_STATS=1 RUST_LOG=info timeout 3600 $B bench $f --tokens 64 --ctx 512 --ignore-eos --json > logs/glmv3/${be}_24g.log 2>&1
  echo "$be 24576MB 512+64: exit $? | $(grep -hoE '"(decode_tok_s_steady|prefill_tok_s|ttft_s)": [0-9.e+-]+' logs/glmv3/${be}_24g.log | tr '\n' ' ') | $(grep -hoE 'wgpu expert stream: .*' logs/glmv3/${be}_24g.log | tail -1)"
done
echo GLMV3_DONE
