#!/bin/bash
# The final binary: outputs against Vulkan (short and long prompts), a chat,
# two server requests, one more pass of the CUDA rows of the table, the other
# prompt modes, and a 32K prompt on both paths.
root=${CUDA_RUN_ROOT:-/root/cuda}
cd "$root" || exit 1
CHECK_SUFFIX=_final ./scripts/chain_cuda_check.sh bin/cortiq-cuda q4tp q8_2f
PROMPTS_DIR=$root/prompts_long CHECK_SUFFIX=_final ./scripts/chain_cuda_check.sh bin/cortiq-cuda q4tp q8_2f
./scripts/chat_check.sh | head -3
./scripts/serve_check.sh | tail -5
b=./scripts/bench_one.sh; cu=bin/cortiq-cuda
q4=models/spark-x2.6-4b-q4tp-turbo.cmf; q8=models/spark-x2.6-4b-q8_2f-turbo.cmf
for ctx in 1000 4000 16000; do
  $b fin2_cu_q4_${ctx}_r1 $cu $q4 $ctx CMF_GPU=cuda
  $b fin2_cu_q8_${ctx}_r1 $cu $q8 $ctx CMF_GPU=cuda
done
$b fin2_cutf_q4_4000_r1 $cu $q4 4000 CMF_GPU=cuda CMF_CUDA_PREC=tf32
$b fin2_cub2k_q4_4000_r1 $cu $q4 4000 CMF_GPU=cuda CMF_CUDA_BATCH=2048
$b fin2_cub2k_q4_16000_r1 $cu $q4 16000 CMF_GPU=cuda CMF_CUDA_BATCH=2048
$b fin2_cu32k_q4_32000_r1 $cu $q4 32000 CMF_GPU=cuda CMF_MAX_SEQ=40000
$b fin2_vk32k_q4_32000_r1 bin/cortiq-vk $q4 32000 CMF_O1=off CMF_MAX_SEQ=40000
echo CHAIN_FINAL_VERIFY_DONE
