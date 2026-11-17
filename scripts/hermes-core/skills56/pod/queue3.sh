#!/bin/bash
# After E2: skill 1 baked by the recipe onto the bases E1 and E2 grew into (genome exports of their last
# checkpoints), their R answers (CPU, while the card waits), then E1b — E2 without the teacher (B16, the same
# 7633 steps and tokens): the teacher's share at equal batch and tokens.
# Log /root/sk/run/queue3.log; marker queue3.done
cd /root/sk/run
L=/root/sk/run/queue3.log
export LD_LIBRARY_PATH=/usr/local/cuda/lib64 CMF_VULKAN_HEAD_TABLE=0
E=/root/sk/cmf/target/release/cortiq-embryo; M=/root/sk/mix; D=/root/sk/data
R="--layers 1 2 3 4 5 6 7 --phi-layer 0 --mask-init 4.0 --lr-a 0.05 --l1 0 --soft-fold --lm-frac 0 --steps-a 1000 --eval-every 100 --steps-b 2 --lr-b 0.000001 --fcd-layers 2 --eval-every-b 1 --batch 16 --dev-batches 16 --final-batches 16"
{
  until [ -f e2.done ] || [ -f e2.failed ]; do sleep 20; done
  echo "== queue3 $(date -u +%T)"
  for n in e1 e2; do
    $E export --ckpt $n-end.ckpt --tokenizer $D/gen1.tokenizer.json --out $n-end.cmf 2>&1 | tail -1
    rm -f f$n.cmf
    $E export --ckpt $n-end.ckpt --tokenizer $D/gen1.tokenizer.json --out f$n.cmf --genome-id embryo56-$n --genome-status sealed 2>&1 | tail -1
    CKPT=/root/sk/run/$n-end.ckpt /root/sk/bake.sh c1-$n f$n s1 cards $R
  done
  for n in e1 e2; do SECONDS=0; /root/sk/measure.sh c1-$n cards; echo "MEASURE c1-$n $SECONDS s"; done
  echo "== E1b $(date -u +%T)"
  $E birth --resume $D/keep-final-healed.ckpt --steps 23133 --out e1b.ckpt \
    --shard $M/en2.u16:0.6 $M/ru2.u16:0.2 $M/code.u16:0.12 $M/math2.u16:0.08 --val $D/rawval-tails-2m.u16 \
    --batch 16 --seq 1024 --warmup 300 --lr 1e-4 --lr-restart --eval-every 1908 --save-every 1908 \
    --carry --carry-reset-every 16 --carry-eot 32760 --dropless --seed 17 \
    --retrieval-rate 3 --retrieval-dists 128,2048 --retrieval-pairs 4 > e1b.log 2>&1 || echo "E1b FAILED"
  cp e1b.ckpt e1b-end.ckpt
  $E export --ckpt e1b-end.ckpt --tokenizer $D/gen1.tokenizer.json --out e1b-end.cmf 2>&1 | tail -1
  grep -E "val loss" e1b.log
  echo "== queue3 done $(date -u +%T)"
  touch queue3.done
} >> $L 2>&1
