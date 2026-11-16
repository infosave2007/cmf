#!/bin/bash
# run4b_fcd.sh: FCD on the 4B emotion mask with a smaller step (lr 1e-6; 3e-6 hurt): 2 layers, 2 steps,
# from the 1000-step mask (run4b_more.sh); serve accuracy only if FCD beat the mask. Then 500 more
# Phase-A steps (1500 in all), export, overlay, accuracy.
cd /root/h3i/skill/emo
export B=/root/h3i/bin/cortiq-skill9 BAKE_GPU=cuda
until grep -q RUN4B_MORE_DONE run4b_more.out 2>/dev/null; do sleep 20; done
HELD=400 CMF_BAKE_SKIP_BASELINE=1 flock /root/h3i/gpu.lock ./bake.sh e6 ../cmf/qwen35-4B-q4tp.cmf --mask-from runs/e5/ckpt/phase-a-step-500.json --steps-a 0 --steps-b 2 --eval-every 1 --fcd-layers 2 --lr-b 0.000001 --batch 32 --mask-init 4.0 --skip-runtime-gate
grep -E "supplied|\[B\]|=== " runs/e6/bake.log | cut -c1-200
verdict=$(grep "=== bake" runs/e6/bake.log | python3 -c "
import sys,re
l=sys.stdin.read(); m=re.search(r'mask ([0-9.]+) \| mask\+FCD ([0-9.]+)',l)
print('better' if m and float(m.group(2))<float(m.group(1)) else 'same')")
echo "FCD: $verdict"
if [ "$verdict" = better ]; then
  cd ..
  $B skill export emo/runs/e6/specialist.cmf --base cmf/qwen35-4B-q4tp.cmf --id emotion -o emo/emotion4b6.skill.cmf 2>&1 | grep -v INFO | tail -1
  rm -f cmf/qwen35-4B-emotion6.cmf
  $B skill apply cmf/qwen35-4B-q4tp.cmf emo/emotion4b6.skill.cmf --overlay -o cmf/qwen35-4B-emotion6.cmf 2>&1 | grep -v INFO | tail -1
  cd emo
  timeout 1500 flock /root/h3i/gpu.lock ./serve_eval.sh e6-skill-cuda ../cmf/qwen35-4B-emotion6.cmf cuda emotion 2000
fi
echo RUN4B_FCD_DONE
cd /root/h3i/skill/emo
HELD=400 CMF_BAKE_SKIP_BASELINE=1 flock /root/h3i/gpu.lock ./bake.sh e7 ../cmf/qwen35-4B-q4tp.cmf --mask-from runs/e5/ckpt/phase-a-step-500.json --steps-a 500 --eval-every 100 --steps-b 0 --batch 32 --mask-init 4.0 --lr-a 0.05 --l1-init 0.0 --l1-step 0.0 --skip-runtime-gate
grep -E "supplied|\[A\] step|=== " runs/e7/bake.log | cut -c1-200
cd ..
$B skill export emo/runs/e7/specialist.cmf --base cmf/qwen35-4B-q4tp.cmf --id emotion -o emo/emotion4b7.skill.cmf 2>&1 | grep -v INFO | tail -1
rm -f cmf/qwen35-4B-emotion7.cmf
$B skill apply cmf/qwen35-4B-q4tp.cmf emo/emotion4b7.skill.cmf --overlay -o cmf/qwen35-4B-emotion7.cmf 2>&1 | grep -v INFO | tail -1
cd emo
timeout 1500 flock /root/h3i/gpu.lock ./serve_eval.sh e7-skill-cuda ../cmf/qwen35-4B-emotion7.cmf cuda emotion 2000
echo RUN4B_E7_DONE
