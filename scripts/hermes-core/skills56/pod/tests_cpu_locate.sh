#!/bin/bash
# Which kernels of the Vulkan trainer compute something else on the software adapter (llvmpipe)?
# The kernel, operator and model parity tests of the trainer with only the software adapter visible.
#   tests_cpu_locate.sh [test files...]     log: /root/sk/tests.cpu-locate.log
cd /root/sk/cmf && source /root/.cargo/env
export XDG_RUNTIME_DIR=/tmp CMF_GPU_SOFTWARE=1 VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json
T="${*:-kernels_parity vulkan_correctness vulkan_operator_contract vulkan_forward_contract vulkan_model_parity vulkan_gdn_wy gdn_mixer vulkan_bounded_anchor vulkan_scatter_chain vulkan_sumsq_contract vulkan_gemm64 vulkan_hk_gemm moe_desc_kernels vulkan_head_table model_gradcheck}"
{
  echo "== locate on llvmpipe: $(date -u +%T)"
  for t in $T; do
    echo "## $t"
    timeout 900 cargo test --release -p cortiq-embryo --features vulkan --target-dir target-vk --test $t -- --test-threads 1 2>&1 | grep -E "^test |test result|panicked|^  left|^ right|assert" -A 1 | grep -v "^--" | cut -c1-260
    echo "EXIT $t ${PIPESTATUS[0]}"
  done
  echo "== done: $(date -u +%T)"
} > /root/sk/tests.cpu-locate.log 2>&1
