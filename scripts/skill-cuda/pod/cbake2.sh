#!/bin/bash
# cbake2.sh TAG BASE [extra bake args...]: DTG-MA bake of the coding skill on the Cortiq-Coder chat
# records (answer-span loss: the assistant's final answer, thinking off), runs/TAG
set -u
tag=$1; base=$2; shift 2
B=${B:-/root/h3i/bin/cortiq-bake-r}
cd /root/h3i/skill/code
mkdir -p runs/$tag
export CMF_BAKE_MATS_LAYERS=${CMF_BAKE_MATS_LAYERS:-0} CMF_GPU=${BAKE_GPU:-cuda} XDG_RUNTIME_DIR=/tmp CMF_GPU_PROBE=0 CMF_BAKE_A_STRICT=0 GLIBC_TUNABLES=glibc.malloc.hugetlb=1
timeout ${BAKE_TIMEOUT:-10800} $B skill bake $base --files chat_train.txt --held-files chat_val.txt \
  --held ${HELD:-20} --calib-chunks ${CALIB:-410} --chunk ${CHUNK:-4096} --length-buckets --drop-oversize \
  --record-start '<|im_start|>system' --answer-after $'<think>\n\n</think>\n\n' --answer-end '<|im_end|>' \
  --soft-fold --no-defrag --mask-name code --checkpoint-dir runs/$tag/ckpt \
  --archaeology-report runs/$tag/archaeology.json --output runs/$tag/specialist.cmf "$@" > runs/$tag/bake.log 2>&1
echo "BAKE_EXIT $tag $?"
