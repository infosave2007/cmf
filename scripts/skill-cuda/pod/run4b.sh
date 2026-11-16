#!/bin/bash
# run4b.sh: the emotion skill on Qwen3.5-4B — Phase A, then FCD (2 layers, 2 steps), export, overlay
cd /root/h3i/skill/emo
export B=/root/h3i/bin/cortiq-skill9 BAKE_GPU=cuda
HELD=400 flock /root/h3i/gpu.lock ./bake.sh e4a ../cmf/qwen35-4B-q4tp.cmf --steps-a 500 --eval-every 100 --steps-b 0 --batch 32 --mask-init 4.0 --lr-a 0.05 --l1-init 0.0 --l1-step 0.0 --skip-runtime-gate --analysis-only
grep -E "baseline|\[A\] step|=== " runs/e4a/bake.log | cut -c1-200
best=$(grep -o "bottom [0-9.]*@" runs/e4a/bake.log | tail -1)
step=$(ls runs/e4a/ckpt | sed "s/phase-a-step-//; s/.json//" | sort -n | tail -1)
# the checkpoint the bake ranked best: the step whose hard-PPL equals the final bottom
for f in $(ls runs/e4a/ckpt | sort -t- -k4 -n); do :; done
bstep=$(grep "\[A\] step" runs/e4a/bake.log | awk -F"hard-PPL=" "{split(\$2,a,\" \"); print \$0}" | python3 -c "
import sys,re
best=None
for l in sys.stdin:
    m=re.search(r\"step (\d+): .*hard-PPL=([0-9.]+)\",l)
    if m:
        s,p=int(m.group(1)),float(m.group(2))
        if best is None or p<best[1]: best=(s,p)
print(best[0])")
echo "best phase-A step $bstep"
HELD=400 CMF_BAKE_SKIP_BASELINE=1 flock /root/h3i/gpu.lock ./bake.sh e4 ../cmf/qwen35-4B-q4tp.cmf --mask-from runs/e4a/ckpt/phase-a-step-$bstep.json --steps-a 0 --steps-b 2 --eval-every 1 --fcd-layers 2 --lr-b 0.000003 --batch 32 --mask-init 4.0 --skip-runtime-gate
grep -E "supplied|\[B\]|=== " runs/e4/bake.log | cut -c1-200
cd ..
$B skill export emo/runs/e4/specialist.cmf --base cmf/qwen35-4B-q4tp.cmf --id emotion -o emo/emotion4b.skill.cmf 2>&1 | grep -v INFO | tail -1
rm -f cmf/qwen35-4B-emotion.cmf
$B skill apply cmf/qwen35-4B-q4tp.cmf emo/emotion4b.skill.cmf --overlay -o cmf/qwen35-4B-emotion.cmf 2>&1 | grep -v INFO | tail -2
echo RUN4B_DONE
