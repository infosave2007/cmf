#!/bin/bash
# Round 6: is it the plane product in the depth decoder that ends CUDA-graph
# songs early? The same seeds with no narrow planes at all (cap 0).
set -u
cd /root/h3i || exit 1
rm -f out/mc/*.logits
for seed in 7 2; do
  MC_SEED=$seed MC_TAG=_s$seed MC_SECONDS=4 MC_STEPS=1 MC_ARMS="cuda_graph cuda_graph_noplane" bash scripts/music_cuda.sh | grep -v -E "MC_DONE|\(32770,\)"
done
echo MC6_DONE
