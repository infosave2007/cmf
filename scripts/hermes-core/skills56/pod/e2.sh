#!/bin/bash
# E2, logit distillation: E1 (same start, data, shares, seed, carry, retrieval tasks) with the 340M Hermes core
# (base1 step 408200, same vocabulary and head clusters) as the teacher of the hierarchical head:
# loss (1-w)·CE + w·CE(p_teacher), w 0.5, plain CE on the first 64 positions of a window (the teacher reads
# each window alone). Compared with E1 at the same pod time.
#   e2.sh B TB STEPS        B: student batch, TB: teacher micro-batch, STEPS: steps from 15500
# Log /root/sk/run/e2.log; markers e2.done / e2.failed; snapshot e2-end.ckpt
cd /root/sk/run
export LD_LIBRARY_PATH=/usr/local/cuda/lib64 CMF_VULKAN_HEAD_TABLE=0
E=/root/sk/cmf/target/release/cortiq-embryo; M=/root/sk/mix; D=/root/sk/data
B="$1"; TB="$2"; N="$3"; END=$((15500 + N)); EV=$(( (N / 4) > 0 ? N / 4 : 1 ))
T=/root/sk/teacher/runs/hermes_core_a40_20261114/ckpt/base1-step408200.ckpt
{
  echo "== e2 B$B teacher micro-batch $TB, steps 15500 -> $END ($((N * B * 1024 / 1000000)) M tokens) $(date -u +%T)"
  $E birth --resume $D/keep-final-healed.ckpt --steps $END --out e2.ckpt \
    --shard $M/en2.u16:0.6 $M/ru2.u16:0.2 $M/code.u16:0.12 $M/math2.u16:0.08 --val $D/rawval-tails-2m.u16 \
    --batch $B --seq 1024 --warmup 300 --lr 1e-4 --lr-restart --eval-every $EV --save-every $EV \
    --carry --carry-reset-every 16 --carry-eot 32760 --dropless --seed 17 \
    --retrieval-rate 3 --retrieval-dists 128,2048 --retrieval-pairs 4 \
    --kd-from $T --kd-w 0.5 --kd-batch $TB --kd-skip 64 || { touch e2.failed; exit 1; }
  cp e2.ckpt e2-end.ckpt
  echo "== e2 done $(date -u +%T)"
  touch e2.done
} >> e2.log 2>&1
