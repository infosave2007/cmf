#!/bin/bash
# Where the Vulkan VAE decode chain declines on MOSS-Image's Flux2 decoder:
# the fast build, then two 512 x 512 steps over Vulkan with the profile on.
#   vae_diag.sh [MODEL]
set -u
root=${H3I_ROOT:-/root/h3i}
cd "$root" || exit 1
bash scripts/build.sh fast > logs/vd_build.log 2>&1 || { echo BUILD_FAILED; tail -30 logs/vd_build.log; exit 1; }
CMF_ZIMAGE_PROF=1 MODEL=${1:-cmf/moss-keep.cmf} bash scripts/moss.sh vd2_vk vulkan 512 2 | head -1
tr '\r' '\n' < out/vd2_vk.log | grep -E "zimage|MOSS image" | head -20 | cut -c1-200
echo VAE_DIAG_DONE
