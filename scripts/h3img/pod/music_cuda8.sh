#!/bin/bash
# Round 8: the depth decoder as the CFG pair with guided codes, the first
# sample's frame dropped (both as ComfyUI's ar.py). Unit test, early-stop
# seeds on the CUDA graph, then 10 s / 8 steps on CUDA (graph, host AR) and
# Vulkan, with whisper on the songs.
set -u
cd /root/h3i || exit 1
. "$HOME/.cargo/env"
bash scripts/build.sh release > logs/build_mc8.log 2>&1 || { echo MC8_BUILD_FAILED; exit 1; }
( cd srcrs && CMF_GPU=0 CMF_MUSIC3_TE=/root/h3i/cmf/dl/minimax-music3-q4tp.cmf timeout 2400 cargo test --release -p cortiq-engine --lib music3_rvq -- --nocapture 2>&1 | grep -E "^test |test result|panicked|assert" | head -20 )
rm -f out/mc/*.logits
for seed in 7 2 1 3; do
  MC_SEED=$seed MC_TAG=_s$seed MC_SECONDS=4 MC_STEPS=1 MC_ARMS="cuda_graph" bash scripts/music_cuda.sh | grep -v -E "MC_DONE|\(32770,\)|rel vs ref"
done
MC_SECONDS=10 MC_STEPS=8 MC_TAG=_10s MC_ARMS="cuda_graph cuda vk" bash scripts/music_cuda.sh | grep -v -E "MC_DONE|\(32770,\)|rel vs ref"
for t in cuda_graph_10s cuda_10s vk_10s; do
  printf '%s whisper: ' $t
  flock gpu.lock env CMF_GPU=cuda timeout 600 bin/cortiq transcribe cmf/dl/whisper-large-v3-turbo-q8.cmf out/mc/$t.wav 2>/dev/null | tr '\n' ' ' | cut -c1-200; echo
done
echo MC8_DONE
