#!/bin/bash
# Round 3: does the CUDA token graph end songs early (several seeds, against
# the Vulkan graph), and what the growing plane cache does to a 10 s song.
set -u
cd /root/h3i || exit 1
bash scripts/build.sh release > logs/build_mc3.log 2>&1 || { echo MC3_BUILD_FAILED; exit 1; }
rm -f out/mc/*.logits
for seed in 7 1 2 3; do
  MC_SEED=$seed MC_TAG=_s$seed MC_SECONDS=4 MC_STEPS=1 MC_ARMS="cuda_graph vk_graph" bash scripts/music_cuda.sh | grep -v MC_DONE
done
MC_SECONDS=2 MC_STEPS=1 MC_ARMS="cpu" MC_TAG=_ref bash scripts/music_cuda.sh | grep -v MC_DONE
MC_SECONDS=10 MC_STEPS=8 MC_TAG=_10s MC_ARMS="cuda vk" bash scripts/music_cuda.sh
echo MC3_DONE
