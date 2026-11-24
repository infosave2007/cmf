#!/bin/bash
# bake.sh TARGET TAG [extra bake args]: the frozen bake recipe of PROTOCOL.json for TARGET (dir | range), runs/TAG
cd ~/mskill
t=$1; tag=$2; shift 2
case $t in dir) ft=DOWN,UP;; range) ft=LOW,HIGH;; *) echo "target?"; exit 2;; esac
d=data/prep/$t; r=runs/$tag; mkdir -p $r
CMF_GPU=cuda CMF_GPU_PROBE=0 timeout ${BAKE_TIMEOUT:-7200} ~/cmf-skillode/target/release/cortiq skill bake ~/skill-ode/cmf/qwen35-0.8b-q8_2f.cmf \
  --files $d/train_corpus.txt --held-files $d/held_corpus.txt --held-tags $d/held_tags.jsonl --held 480 \
  --calib-chunks 12000 --chunk 512 --length-buckets --drop-oversize --record-start '<|im_start|>system' \
  --focus-tokens $ft --checkpoint-accuracy --steps-a ${STEPS:-750} --batch 8 --eval-every ${EVAL:-50} --steps-b 0 \
  --mask-init 2.0 --lr-a 0.1 --l1-init 0.56 --l1-step 0 --tau 0.001 --through-attn \
  --soft-fold --no-defrag --mask-name mkt_$t --checkpoint-dir $r/ckpt --skip-runtime-gate \
  --output $r/specialist.cmf "$@" > $r/bake.log 2>&1
echo "BAKE_EXIT $tag $?"
