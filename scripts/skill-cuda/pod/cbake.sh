#!/bin/bash
# cbake.sh TAG BASE [extra bake args...]: DTG-MA bake of the coding skill (all-token LM loss over the
# Cortiq-Coder corpus), log + archaeology + mask checkpoints in runs/TAG
set -u
tag=$1; base=$2; shift 2
B=${B:-/root/h3i/bin/cortiq-skill6}
cd /root/h3i/skill/code
mkdir -p runs/$tag
export CMF_BAKE_MATS_LAYERS=${CMF_BAKE_MATS_LAYERS:-0} CMF_GPU=${BAKE_GPU:-cuda} XDG_RUNTIME_DIR=/tmp CMF_GPU_PROBE=0 CMF_BAKE_A_STRICT=0 GLIBC_TUNABLES=glibc.malloc.hugetlb=1
timeout ${BAKE_TIMEOUT:-10800} $B skill bake $base --files code_train.txt --held-files code_val.txt \
  --held ${HELD:-24} --calib-chunks ${CALIB:-6000} --chunk ${CHUNK:-512} \
  --soft-fold --no-defrag --mask-name code --checkpoint-dir runs/$tag/ckpt \
  --archaeology-report runs/$tag/archaeology.json --output runs/$tag/specialist.cmf "$@" > runs/$tag/bake.log 2>&1
echo "BAKE_EXIT $tag $?"
