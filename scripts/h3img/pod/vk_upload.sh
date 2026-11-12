#!/bin/bash
# How long the DiT's weights take to reach the card on Vulkan (the chain's
# first call builds them: its "build" time), by upload chunk and threads.
set -u
root=${H3I_ROOT:-/root/h3i}
cd "$root" || exit 1
one() { # tag env...
  local tag=$1; shift
  env "$@" MIN_AREA=1024 CMF_H3_WGPU_PROF=1 STEPS=1 bash scripts/gen.sh "$tag" vulkan 1024 1024 > /dev/null
  echo "$tag [$*] $(grep -m1 "1178 rows x 50 blocks" out/$tag.log | sed -E 's/.*blocks: ([0-9.]+) s \(build ([0-9.]+) s.*/chain \1 s, build \2 s/') | $(grep -E "image\(s\), " out/$tag.log | sed -E 's/.*\((text.*)\)/\1/')"
}
one up_default
one up_c256 CMF_GPU_UPLOAD_CHUNK_MB=256
one up_c16 CMF_GPU_UPLOAD_CHUNK_MB=16
one up_t4 CMF_GPU_UPLOAD_THREADS=4
one up_t16 CMF_GPU_UPLOAD_THREADS=16
one up_wb CMF_GPU_UPLOAD=queue
echo VK_UPLOAD_DONE
