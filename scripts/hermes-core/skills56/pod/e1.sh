#!/bin/bash
# E1, "is the 56M under-trained?": the release 56M (chat SFT, step 15500) trained further on general text it has
# not seen — FineWeb-Edu sample/10BT 001-002, FineWeb-2 rus 000_00001 (400 M), FineMath 4plus 00001 — and its own
# code shard, in the shares it was raised on (0.6/0.2/0.12/0.08), with the retrieval tasks of its long run.
# 250 M tokens: B32 x T1024 = 32768 tokens a step, 7630 steps (15500 -> 23130), rewarm to 1e-4, cosine to 1e-5;
# a snapshot at about 125 M tokens (the save at step 19313). ~72.6 k tok/s on the L40S (CMF_VULKAN_HEAD_TABLE=0): about an hour.
# Log /root/sk/run/e1.log; markers e1.done / e1.failed.
cd /root/sk/run
export LD_LIBRARY_PATH=/usr/local/cuda/lib64 CMF_VULKAN_HEAD_TABLE=0
E=/root/sk/cmf/target/release/cortiq-embryo; M=/root/sk/mix; D=/root/sk/data
until [ -f $M/shard.done ]; do sleep 20; done
COMMON="--shard $M/en2.u16:0.6 $M/ru2.u16:0.2 $M/code.u16:0.12 $M/math2.u16:0.08 --val $D/rawval-tails-2m.u16 \
 --batch 32 --seq 1024 --warmup 300 --lr 1e-4 --lr-restart --eval-every 1907 --save-every 1907 \
 --carry --carry-reset-every 16 --carry-eot 32760 --dropless --seed 17 \
 --retrieval-rate 3 --retrieval-dists 128,2048 --retrieval-pairs 4"
{
  echo "== e1 $(date -u +%T)"
  # one run (one cosine 15500 -> 23130); the save at step 19313 (tick 3814 = 2 x 1907) is the 125 M snapshot,
  # copied by the watcher below once the trainer has written it (saves go through a .tmp and a rename)
  ( until grep -q "saved .*(step 1931[0-9])" e1.log; do sleep 10; [ -f e1.done ] || [ -f e1.failed ] && exit; done
    cp e1.ckpt e1-mid.ckpt && echo "== e1 mid snapshot $(date -u +%T)" ) &
  $E birth --resume $D/keep-final-healed.ckpt --steps 23130 --out e1.ckpt $COMMON || { touch e1.failed; exit 1; }
  cp e1.ckpt e1-end.ckpt
  echo "== e1 done $(date -u +%T)"
  touch e1.done
} >> e1.log 2>&1
