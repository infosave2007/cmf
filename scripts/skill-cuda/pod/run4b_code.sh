#!/bin/bash
# run4b_code.sh: the coding skill on Qwen3.5-4B (answer span), Phase A then FCD (2 layers, 2 steps, lr 1e-6)
cd /root/h3i/skill/code
export B=/root/h3i/bin/cortiq-skill9
CALIB=390 flock /root/h3i/gpu.lock ./cbake2.sh k4a ../cmf/qwen35-4B-q4tp.cmf --steps-a 300 --eval-every 50 --steps-b 0 --batch 1 --mask-init 4.0 --lr-a 0.02 --l1-init 0.0 --l1-step 0.0 --skip-runtime-gate --analysis-only
grep -E "baseline|\[A\] step|=== " runs/k4a/bake.log | cut -c1-200
bstep=$(grep "\[A\] step" runs/k4a/bake.log | python3 -c "
import sys,re
best=None
for l in sys.stdin:
    m=re.search(r\"step (\d+): .*hard-PPL=([0-9.]+)\",l)
    if m:
        s,p=int(m.group(1)),float(m.group(2))
        if best is None or p<best[1]: best=(s,p)
print(best[0])")
CALIB=390 CMF_BAKE_SKIP_BASELINE=1 flock /root/h3i/gpu.lock ./cbake2.sh k4 ../cmf/qwen35-4B-q4tp.cmf --mask-from runs/k4a/ckpt/phase-a-step-$bstep.json --steps-a 0 --steps-b 2 --eval-every 1 --fcd-layers 2 --lr-b 0.000001 --batch 1 --mask-init 4.0 --skip-runtime-gate
grep -E "supplied|\[B\]|=== " runs/k4/bake.log | cut -c1-200
echo RUN4B_CODE_DONE
