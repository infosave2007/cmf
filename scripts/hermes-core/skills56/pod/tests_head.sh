#!/bin/bash
# The committed trainer (without the expert masks) on the software Vulkan adapter only: is a failure there older than the change?
cd /root/sk/cmf-head && source /root/.cargo/env
export XDG_RUNTIME_DIR=/tmp CMF_GPU_SOFTWARE=1 VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json
{
  echo "== head on llvmpipe: $(date -u +%T)"
  cargo test --release -p cortiq-embryo --features vulkan --target-dir target-vk --test skill_bake_v2 -- --test-threads 1
  echo "BAKE_EXIT $?"
  echo "== done: $(date -u +%T)"
} > /root/sk/tests.cpu-head.log 2>&1
