#!/bin/bash
# Qwen-Image-2.1 before its CUDA arm (release build of 0.8.81): Vulkan at
# 512 and 1024 (40 steps), CUDA at 512 for 2 steps (the DiT on the host).
set -u
root=${H3I_ROOT:-/root/h3i}
cd "$root" || exit 1
export BIN=$root/bin/cortiq
f=cmf/qwen-image-2.1.cmf
[ -f "$f" ] || venv/bin/hf download infosave/Qwen-Image-2.1-cmf qwen-image-2.1.cmf --local-dir cmf > logs/q21_dl.log 2>&1
ls -la "$f" || { echo DOWNLOAD_FAILED; exit 1; }
bash scripts/qi21.sh q21b_vk_512 vulkan 512 40
bash scripts/qi21.sh q21b_vk_1024 vulkan 1024 40
bash scripts/qi21.sh q21b_cuda_512 cuda 512 2
echo Q21_BASE_DONE
