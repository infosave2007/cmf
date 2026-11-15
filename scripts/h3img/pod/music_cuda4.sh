#!/bin/bash
# Round 4: f32 narrow products — do CUDA-graph songs still end early (the
# seeds that stopped at 16 and 4 frames), and where the CUDA AR and denoise
# spend their time with and without the growing plane cache.
set -u
cd /root/h3i || exit 1
bash scripts/build.sh release > logs/build_mc4.log 2>&1 || { echo MC4_BUILD_FAILED; exit 1; }
rm -f out/mc/*.logits
for seed in 7 2; do
  MC_SEED=$seed MC_TAG=_s$seed MC_SECONDS=4 MC_STEPS=1 MC_ARMS="cuda_graph" bash scripts/music_cuda.sh | grep -v MC_DONE
done
MC_SECONDS=4 MC_STEPS=4 MC_TAG=_4s MC_ARMS="cuda_prof cuda_prof_fixed cuda_graph_prof vk_prof" bash scripts/music_cuda.sh
for t in cuda_prof_4s cuda_prof_fixed_4s cuda_graph_prof_4s vk_prof_4s; do
  echo "== $t"; tr '\r' '\n' < logs/mc/$t.err | grep -a -A14 -E "denoise split|cuda h3 product" | grep -a -v "^ *denoise [0-9]" | head -24
done
echo MC4_DONE
