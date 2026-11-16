#!/bin/bash
# run4b_more.sh: 500 more Phase-A steps of the 4B emotion mask (from step 500), export, overlay, test
cd /root/h3i/skill/emo
export B=/root/h3i/bin/cortiq-skill9 BAKE_GPU=cuda
HELD=400 CMF_BAKE_SKIP_BASELINE=1 flock /root/h3i/gpu.lock ./bake.sh e5 ../cmf/qwen35-4B-q4tp.cmf --mask-from runs/e4a/ckpt/phase-a-step-500.json --steps-a 500 --eval-every 100 --steps-b 0 --batch 32 --mask-init 4.0 --lr-a 0.05 --l1-init 0.0 --l1-step 0.0 --skip-runtime-gate
grep -E "supplied|\[A\] step|=== " runs/e5/bake.log | cut -c1-200
cd ..
$B skill export emo/runs/e5/specialist.cmf --base cmf/qwen35-4B-q4tp.cmf --id emotion -o emo/emotion4b5.skill.cmf 2>&1 | grep -v INFO | tail -1
rm -f cmf/qwen35-4B-emotion5.cmf
$B skill apply cmf/qwen35-4B-q4tp.cmf emo/emotion4b5.skill.cmf --overlay -o cmf/qwen35-4B-emotion5.cmf 2>&1 | grep -v INFO | tail -1
cd emo
timeout 1500 flock /root/h3i/gpu.lock ./serve_eval.sh e5-skill-cuda ../cmf/qwen35-4B-emotion5.cmf cuda emotion 2000
echo RUN4B_MORE_DONE
