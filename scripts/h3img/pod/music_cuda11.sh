#!/bin/bash
# Round 11: the final build — unit tests (music3 on the CPU, the CUDA h3
# products) and one 10 s / 8 step song on CUDA with the defaults.
set -u
cd /root/h3i || exit 1
. "$HOME/.cargo/env"
bash scripts/build.sh release > logs/build_mc11.log 2>&1 || { echo MC11_BUILD_FAILED; exit 1; }
( cd srcrs && CMF_GPU=0 CMF_MUSIC3_TE=/root/h3i/cmf/dl/minimax-music3-q4tp.cmf timeout 2400 cargo test --release -p cortiq-engine --lib music3 -- --nocapture 2>&1 | grep -E "^test |test result|panicked" | head -20 )
( cd srcrs && flock /root/h3i/gpu.lock timeout 2400 cargo test --release -p cortiq-engine --features cuda --lib gpu_cuda::h3 -- --test-threads=1 2>&1 | grep -E "^test |test result|panicked" | head -30 )
MC_SECONDS=10 MC_STEPS=8 MC_TAG=_final MC_ARMS="cuda" bash scripts/music_cuda.sh | grep -v -E "MC_DONE|\(32770,\)|rel vs ref"
echo MC11_DONE
