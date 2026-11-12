#!/bin/bash
# Z-Image-Turbo before its CUDA arm (release build of 0.8.82): Vulkan at
# 512 and 1024 (8 steps), CUDA at 512 for 1 step (the DiT on the host).
set -u
root=${H3I_ROOT:-/root/h3i}
cd "$root" || exit 1
export BIN=$root/bin/cortiq
f=cmf/z-image-turbo.cmf
[ -f "$f" ] || venv/bin/hf download infosave/Z-Image-Turbo-cmf z-image-turbo.cmf --local-dir cmf > logs/zi_dl.log 2>&1
ls -la "$f" || { echo DOWNLOAD_FAILED; exit 1; }
bash scripts/zi.sh zib_vk_512 vulkan 512 8
bash scripts/zi.sh zib_vk_1024 vulkan 1024 8
bash scripts/zi.sh zib_cuda_512 cuda 512 2
echo ZI_BASE_DONE
