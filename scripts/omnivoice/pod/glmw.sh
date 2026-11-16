#!/bin/bash
# The Vulkan stream's pools seeded from a prompt's staging bank, again after
# the staging bank got COPY_SRC (glmv3: the seeded run panicked): GLM-5.3's
# greedy text streamed with seeding and without (equal), and a 512+64 decode
# with seeding beside the cold one. Uses mm5.sh's build. After vq2g.sh.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
mkdir -p logs/glmw
end=$((SECONDS + 43200))
until grep -qE "VQ2G_DONE|VQ2G_DOWNLOAD_FAILED|VQ2G_STALE_BINARY" logs/vq2g.out 2>/dev/null || [ $SECONDS -gt $end ]; do sleep 30; done
B=bin/cortiq-fast; f=cmf/big/glm-5.3-flash-q2tp.cmf
P="Explain in two sentences why the sky is blue."
for w in 1 0; do
  flock gpu.lock env CMF_GPU=1 CMF_WGPU_EXPERT_STREAM=1 CMF_WGPU_EXPERT_WARM=$w timeout 3600 $B run $f --prompt "$P" -n 24 --greedy 2>logs/glmw/text_w$w.err | grep -v '^\[' > logs/glmw/text_w$w.txt
done
echo "text: $(tail -n +5 logs/glmw/text_w1.txt | tr '\n' ' ' | cut -c1-160)"
if cmp -s logs/glmw/text_w1.txt logs/glmw/text_w0.txt; then echo "   stream warm == cold"; else echo "   stream warm DIFFERS from cold"; grep -iE "panick|Caused" -A3 logs/glmw/text_w1.err | head -8; fi
for w in 1 0; do
  flock gpu.lock env CMF_GPU=1 CMF_WGPU_EXPERT_STREAM=1 CMF_WGPU_EXPERT_WARM=$w CMF_STREAM_STATS=1 RUST_LOG=info timeout 3600 $B bench $f --tokens 64 --ctx 512 --ignore-eos --json > logs/glmw/s_w$w.log 2>&1
  echo "vulkan stream warm=$w 512+64: exit $? | $(grep -hoE '"(decode_tok_s_steady|prefill_tok_s|ttft_s)": [0-9.e+-]+' logs/glmw/s_w$w.log | tr '\n' ' ') | $(grep -hoE 'wgpu expert stream: .*' logs/glmw/s_w$w.log | tail -1)"
done
echo GLMW_DONE
