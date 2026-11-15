#!/bin/bash
# Round 5: the graph against the host at every position (same input, own
# cache) on CUDA and Vulkan — where the CUDA graph leaves the host; then the
# CUDA FFN on the card and the cached VRAM reading, 4 s and 10 s songs.
set -u
cd /root/h3i || exit 1
bash scripts/build.sh release > logs/build_mc5.log 2>&1 || { echo MC5_BUILD_FAILED; exit 1; }
rm -f out/mc/*.logits
MC_SEED=2 MC_TAG=_s2 MC_SECONDS=2 MC_STEPS=1 MC_ARMS="cuda_graph_check vk_graph_check" bash scripts/music_cuda.sh | grep -v MC_DONE
for t in cuda_graph_check_s2 vk_graph_check_s2; do
  echo "== $t"; grep -a "graph check" logs/mc/$t.err | awk 'NR<=3 || (NR>=27 && NR<=40) || NR%10==0' | cut -c1-160
done
MC_SECONDS=4 MC_STEPS=4 MC_TAG=_4s MC_ARMS="cuda_prof cuda_graph_prof" bash scripts/music_cuda.sh | grep -v MC_DONE
for t in cuda_prof_4s cuda_graph_prof_4s; do
  echo "== $t"; tr '\r' '\n' < logs/mc/$t.err | grep -a -A14 -E "denoise split|cuda h3 product" | grep -a -v "^ *denoise [0-9]" | head -24
done
MC_SECONDS=10 MC_STEPS=8 MC_TAG=_10s MC_ARMS="cuda cuda_graph" bash scripts/music_cuda.sh
echo MC5_DONE
