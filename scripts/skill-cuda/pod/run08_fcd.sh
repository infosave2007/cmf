#!/bin/bash
# run08_fcd.sh: FCD with lr 1e-6 (the 4B's winner) on the 0.8B emotion mask (step 500): 2 layers, 2 steps;
# export, overlay, the 2000-text accuracy if FCD beat the mask.
cd /root/h3i/skill/emo
export B=/root/h3i/bin/cortiq-skill9 BAKE_GPU=cuda
HELD=400 CMF_BAKE_SKIP_BASELINE=1 flock /root/h3i/gpu.lock ./bake.sh e8 ../cmf/qwen35-0.8B-q4tp.cmf --mask-from runs/e2/ckpt/phase-a-step-500.json --steps-a 0 --steps-b 2 --eval-every 1 --fcd-layers 2 --lr-b 0.000001 --batch 64 --mask-init 4.0 --skip-runtime-gate
grep -E "supplied|\[B\]|=== " runs/e8/bake.log | cut -c1-200
verdict=$(grep "=== bake" runs/e8/bake.log | python3 -c "
import sys,re
l=sys.stdin.read(); m=re.search(r'mask ([0-9.]+) \| mask\+FCD ([0-9.]+)',l)
print('better' if m and float(m.group(2))<float(m.group(1)) else 'same')")
echo "FCD: $verdict"
if [ "$verdict" = better ]; then
  cd ..
  $B skill export emo/runs/e8/specialist.cmf --base cmf/qwen35-0.8B-q4tp.cmf --id emotion -o emo/emotion8.skill.cmf 2>&1 | grep -v INFO | tail -1
  rm -f cmf/qwen35-0.8B-emotion8.cmf
  $B skill apply cmf/qwen35-0.8B-q4tp.cmf emo/emotion8.skill.cmf --overlay -o cmf/qwen35-0.8B-emotion8.cmf 2>&1 | grep -v INFO | tail -1
  cd emo
  timeout 1500 flock /root/h3i/gpu.lock ./serve_eval.sh e8-skill-cuda ../cmf/qwen35-0.8B-emotion8.cmf cuda emotion 2000
fi
echo RUN08_FCD_DONE
