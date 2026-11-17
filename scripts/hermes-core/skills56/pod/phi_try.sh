#!/bin/bash
# A two-step bake that reaches the runtime check fast: how far is the trainer's φ on this device
# from the runtime's CPU φ? Both unit φ of every holdout go to /root/sk/run/phi-TAG.diag.jsonl.
#   phi_try.sh TAG [ENV=VALUE ...]     e.g. phi_try.sh native0 CMF_CUDA_NATIVE=0
#   E=/root/sk/cmf/target-vk/release/cortiq-embryo phi_try.sh vk     (the Vulkan build)
TAG="$1"; shift
for kv in "$@"; do export "$kv"; done
export CMF_BAKE_PHI_DIAG=/root/sk/run/phi-$TAG.diag.jsonl
cd /root/sk && ./bake.sh phi-$TAG f0 s1 cards --layers 1 2 3 4 5 6 7 --phi-layer 0 --mask-init 4.0 --lr-a 0.05 --l1 0 --soft-fold --lm-frac 0 \
  --steps-a 2 --eval-every 1 --steps-b 1 --lr-b 0.000001 --fcd-layers 2 --eval-every-b 1 --batch 16 --dev-batches 1 --final-batches 1 > /root/sk/run/phi-$TAG.out 2>&1
grep -oE "runtime check[^(]*|BAKE_EXIT.*|max_unit_phi[^,}]*" /root/sk/run/phi-$TAG.bake.log | cut -c1-300 | head -6
python3 -c "
import json,sys
r=[json.loads(l) for l in open(sys.argv[1])]; d=sorted(x['delta'] for x in r)
print('n',len(d),'median %.3e p90 %.3e max %.3e'%(d[len(d)//2],d[int(len(d)*.9)],d[-1]))
for x in sorted(r,key=lambda x:-x['delta'])[:3]: print('  #%d %.3e span %s'%(x['k'],x['delta'],x['span']))
" /root/sk/run/phi-$TAG.diag.jsonl
rm -f /root/sk/run/phi-$TAG.cmf
