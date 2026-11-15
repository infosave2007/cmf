#!/bin/bash
# Round 9: is the early stop the model's own answer under exact arithmetic?
# The exact host (CMF_SDOT=0, CPU) on the seeds the CUDA graph stops early,
# and a longer lyric on the CUDA graph and on the default host.
set -u
cd /root/h3i || exit 1
. "$HOME/.cargo/env"
( cd srcrs && CMF_GPU=0 CMF_MUSIC3_TE=/root/h3i/cmf/dl/minimax-music3-q4tp.cmf timeout 2400 cargo test --release -p cortiq-engine --lib music3_rvq -- --nocapture 2>&1 | grep -E "^test |test result|panicked|differ" | head -20 )
bash scripts/build.sh release > logs/build_mc9.log 2>&1 || { echo MC9_BUILD_FAILED; exit 1; }
for seed in 2 1; do
  MC_SEED=$seed MC_TAG=_s$seed MC_SECONDS=4 MC_STEPS=1 MC_ARMS="cpu_exact" bash scripts/music_cuda.sh | grep -v -E "MC_DONE|\(32770,\)|rel vs ref"
done
long='[verse] The sea is calm tonight ^ The harbor lights are low ^ I hear the gentle water ^ Singing soft and slow ^ [chorus] Hold me by the shore ^ Until the morning light ^ Hold me by the shore ^ The sea is calm tonight'
for seed in 2 1 7; do
  MC_LYRICS="$long" MC_SEED=$seed MC_TAG=_long_s$seed MC_SECONDS=10 MC_STEPS=1 MC_ARMS="cuda_graph vk" bash scripts/music_cuda.sh | grep -v -E "MC_DONE|\(32770,\)|rel vs ref"
done
echo MC9_DONE
