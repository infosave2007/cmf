#!/bin/bash
# SAM3 release measurement: release build, one image on CUDA and Vulkan, the
# 8-frame video on CUDA and Vulkan (host figures from sam3_base / sam3_video).
set -u
root=${H3I_ROOT:-/root/h3i}
cd "$root" || exit 1
bash scripts/build.sh release > logs/sam3f_build.log 2>&1 || { echo BUILD_FAILED; tail -30 logs/sam3f_build.log; exit 1; }
BIN=cortiq BACKENDS="cuda wgpu" bash scripts/sam3_cuda.sh sam3r
BIN=cortiq BACKENDS="cuda wgpu" bash scripts/sam3_video.sh
nvidia-smi --query-gpu=memory.used --format=csv,noheader
echo SAM3_FINAL_DONE
