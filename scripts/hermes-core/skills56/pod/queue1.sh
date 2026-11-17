#!/bin/bash
# Skill 1 on Embryo-56M on this pod, one run after another: the trainer's tests on CUDA, the bake
# by the recipe of the Mac (c1), the same with the mask over the routed experts too (c1e),
# then both measured on set R (answers only; grading in Docker on the Mac) and general text.
# Marker: /root/sk/run/queue1.done (or queue1.failed). Log: /root/sk/run/queue1.log
cd /root/sk
R="--layers 1 2 3 4 5 6 7 --phi-layer 0 --mask-init 4.0 --lr-a 0.05 --l1 0 --soft-fold --lm-frac 0 --steps-a 1000 --eval-every 100 --steps-b 2 --lr-b 0.000001 --fcd-layers 2 --eval-every-b 1 --batch 16 --dev-batches 16 --final-batches 16"
{
  echo "== queue1 $(date -u +%T)"
  ./tests.sh cuda; grep -E "EXIT|test result" /root/sk/tests.cuda.log
  ./bake.sh c1 f0 s1 cards $R
  ./bake.sh c1e f0 s1 cards $R --mask-experts
  for n in c1 c1e; do
    [ -f run/$n.cmf ] || { echo "NO run/$n.cmf"; touch run/queue1.failed; exit 1; }
    SECONDS=0; ./measure.sh $n cards; echo "MEASURE $n ${SECONDS} s"
  done
  echo "== queue1 done $(date -u +%T)"
  touch run/queue1.done
} > run/queue1.log 2>&1
