#!/bin/bash
# Round 10: the CUDA graph now reads the hidden back and is the CUDA default.
# Early-stop seeds (long and short lyric), then the original probe — 10 s,
# 8 steps — on CUDA (graph), CUDA (host AR) and Vulkan, whisper on each.
set -u
cd /root/h3i || exit 1
bash scripts/build.sh release > logs/build_mc10.log 2>&1 || { echo MC10_BUILD_FAILED; exit 1; }
rm -f out/mc/*.logits
long='[verse] The sea is calm tonight ^ The harbor lights are low ^ I hear the gentle water ^ Singing soft and slow ^ [chorus] Hold me by the shore ^ Until the morning light ^ Hold me by the shore ^ The sea is calm tonight'
for seed in 2 1 7; do
  MC_LYRICS="$long" MC_SEED=$seed MC_TAG=_long_s$seed MC_SECONDS=10 MC_STEPS=1 MC_ARMS="cuda" bash scripts/music_cuda.sh | grep -v -E "MC_DONE|\(32770,\)|rel vs ref"
done
for seed in 2 1; do
  MC_SEED=$seed MC_TAG=_s$seed MC_SECONDS=4 MC_STEPS=1 MC_ARMS="cuda" bash scripts/music_cuda.sh | grep -v -E "MC_DONE|\(32770,\)|rel vs ref"
done
MC_SECONDS=10 MC_STEPS=8 MC_TAG=_10s MC_ARMS="cuda cuda_host vk" bash scripts/music_cuda.sh | grep -v -E "MC_DONE|\(32770,\)|rel vs ref"
MC_LYRICS="$long" MC_SECONDS=10 MC_STEPS=8 MC_TAG=_long10 MC_ARMS="cuda vk" bash scripts/music_cuda.sh | grep -v -E "MC_DONE|\(32770,\)|rel vs ref"
for t in cuda_10s cuda_host_10s vk_10s cuda_long10 vk_long10; do
  printf '%s whisper: ' $t
  flock gpu.lock env CMF_GPU=cuda timeout 600 bin/cortiq transcribe cmf/dl/whisper-large-v3-turbo-q8.cmf out/mc/$t.wav 2>/dev/null | tr '\n' ' ' | cut -c1-240; echo
done
echo MC10_DONE
