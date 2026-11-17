#!/bin/bash
# Round 7: every narrow plane product of the CUDA-graph arm (the depth
# decoder) against exact f32 rows of the same weight, seed 2 (which stops at
# frame 7 with planes and runs to the end without them).
set -u
cd /root/h3i || exit 1
bash scripts/build.sh release > logs/build_mc7.log 2>&1 || { echo MC7_BUILD_FAILED; exit 1; }
MC_SEED=2 MC_TAG=_s2 MC_SECONDS=2 MC_STEPS=1 MC_ARMS="cuda_graph_verify" bash scripts/music_cuda.sh | grep -v -E "MC_DONE|\(32770,\)"
grep -a "cuda mm verify" logs/mc/cuda_graph_verify_s2.err | head -40
echo "bad lines: $(grep -a -c 'cuda mm verify #' logs/mc/cuda_graph_verify_s2.err)"
echo MC7_DONE
