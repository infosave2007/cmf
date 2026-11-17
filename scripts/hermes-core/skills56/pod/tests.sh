#!/bin/bash
# The skill-mask tests of the Embryo trainer on this pod: CUDA, native Vulkan, the software Vulkan adapter (CPU).
#   tests.sh cuda | vulkan | cpu        log: /root/sk/tests.<name>.log
cd /root/sk/cmf && source /root/.cargo/env
export LD_LIBRARY_PATH=/usr/local/cuda/lib64 XDG_RUNTIME_DIR=/tmp
case "$1" in
  cuda)   feat=cuda;   dir=target ;;
  vulkan) feat=vulkan; dir=target-vk ;;
  # the software adapter must be the only one visible: with a card present CMF_GPU_SOFTWARE=1 alone still takes the card
  cpu)    feat=vulkan; dir=target-vk; export CMF_GPU_SOFTWARE=1 VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json ;;
esac
{
  echo "== $1: $(date -u +%T)"
  XDG_RUNTIME_DIR=/tmp vulkaninfo --summary 2>&1 | grep -E "deviceName" | head -3
  cargo test --release -p cortiq-embryo --features $feat --target-dir $dir --test skill_mask_experts -- --nocapture --test-threads 1
  echo "MASK_EXIT $?"
  cargo test --release -p cortiq-embryo --features $feat --target-dir $dir --test skill_bake_v2 -- --test-threads 1
  echo "BAKE_EXIT $?"
  cargo test --release -p cortiq-embryo --features $feat --target-dir $dir --lib skill_mask -- --nocapture --test-threads 1
  echo "KERNEL_EXIT $?"
  echo "== $1 done: $(date -u +%T)"
} > /root/sk/tests.$1.log 2>&1
