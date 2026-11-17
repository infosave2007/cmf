#!/bin/bash
# bake.sh TAG BASE [extra bake args...]: DTG-MA bake of the emotion skill (answer span), log + archaeology in runs/
set -u
tag=$1; base=$2; shift 2
B=${B:-/root/h3i/bin/cortiq-skill}
cd /root/h3i/skill/emo
mkdir -p runs/$tag
export CMF_BAKE_MATS_LAYERS=${CMF_BAKE_MATS_LAYERS:-0} CMF_GPU=${BAKE_GPU:-1} XDG_RUNTIME_DIR=/tmp CMF_GPU_PROBE=0 CMF_BAKE_A_STRICT=0 GLIBC_TUNABLES=glibc.malloc.hugetlb=1
timeout ${BAKE_TIMEOUT:-7200} $B skill bake $base --files train.txt --held-files ${HELD_FILE:-held.txt} --held-tags ${HELD_TAGS:-held_tags.jsonl} \
  --held ${HELD:-400} --calib-chunks 16000 --chunk 128 --length-buckets \
  --record-start '<|im_start|>system' --answer-after $'<think>\n\n</think>\n\n' --answer-end '<|im_end|>' \
  --soft-fold --no-defrag --mask-name emotion --checkpoint-dir runs/$tag/ckpt \
  --archaeology-report runs/$tag/archaeology.json --output runs/$tag/specialist.cmf "$@" > runs/$tag/bake.log 2>&1
echo "BAKE_EXIT $tag $?"
