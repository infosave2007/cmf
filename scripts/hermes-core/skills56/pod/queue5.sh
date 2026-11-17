#!/bin/bash
# After E3: E4 — E1 (same start, fresh text, seed, carry, retrieval, B32, 7630 steps, 250 M tokens) with 16 % of the
# stream from the 340M's chat shard b_chat_misc (cmf-im-v1 transcripts: system chats, everyday, rewrite, summarize):
# can one stage keep the chat format the skills need? Then skill 1 on its genome export, R answers (6 threads beside
# the GPU), the KLM index on the GPU. Log /root/sk/run/queue5.log; marker queue5.done
cd /root/sk/run
L=/root/sk/run/queue5.log
export LD_LIBRARY_PATH=/usr/local/cuda/lib64 CMF_VULKAN_HEAD_TABLE=0 EMBRYO_LOGLIK_BIN=/root/sk/cmf/target-kd/release/cortiq-embryo
E=/root/sk/cmf/target-kd/release/cortiq-embryo; M=/root/sk/mix; D=/root/sk/data
R="--layers 1 2 3 4 5 6 7 --phi-layer 0 --mask-init 4.0 --lr-a 0.05 --l1 0 --soft-fold --lm-frac 0 --steps-a 1000 --eval-every 100 --steps-b 2 --lr-b 0.000001 --fcd-layers 2 --eval-every-b 1 --batch 16 --dev-batches 16 --final-batches 16"
until [ -f queue4.done ]; do sleep 20; done
{
  echo "== E4 $(date -u +%T)"
  $E birth --resume $D/keep-final-healed.ckpt --steps 23130 --out e4.ckpt \
    --shard $M/en2.u16:0.5 $M/ru2.u16:0.17 $M/code.u16:0.1 $M/math2.u16:0.07 $M/b_chat_misc.u16:0.16 --val $D/rawval-tails-2m.u16 \
    --batch 32 --seq 1024 --warmup 300 --lr 1e-4 --lr-restart --eval-every 1907 --save-every 1907 \
    --carry --carry-reset-every 16 --carry-eot 32760 --dropless --seed 17 \
    --retrieval-rate 3 --retrieval-dists 128,2048 --retrieval-pairs 4 > e4.log 2>&1 || { echo "E4 FAILED"; tail -3 e4.log; exit 1; }
  grep -E "val loss" e4.log
  cp e4.ckpt e4-end.ckpt
  $E export --ckpt e4-end.ckpt --tokenizer $D/gen1.tokenizer.json --out e4-end.cmf 2>&1 | tail -1
  rm -f fe4.cmf
  $E export --ckpt e4-end.ckpt --tokenizer $D/gen1.tokenizer.json --out fe4.cmf --genome-id embryo56-e4 --genome-status sealed 2>&1 | tail -1
  E=$E CKPT=/root/sk/run/e4-end.ckpt /root/sk/bake.sh c1-e4 fe4 s1 cards $R
  ( CMF_THREADS=6 nice -n 10 /root/sk/measure.sh c1-e4 cards > m-c1-e4.log 2>&1 ) &
  /root/sk/klm_gpu.sh e4-end=/root/sk/run/e4-end.ckpt
  wait
  tail -2 m-c1-e4.log
  echo "== queue5 done $(date -u +%T)"
  touch queue5.done
} >> $L 2>&1
