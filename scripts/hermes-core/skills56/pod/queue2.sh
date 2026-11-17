#!/bin/bash
# After E1: the trainer with logit distillation, its test on CUDA, the starting point's validation in E1's
# configuration, a probe of the student batch / teacher micro-batch that fits and runs fastest, then E2 for
# E1's pod time (57 min of steps). Log /root/sk/run/queue2.log; marker queue2.started (E2 running).
cd /root/sk
L=/root/sk/run/queue2.log
export LD_LIBRARY_PATH=/usr/local/cuda/lib64 CMF_VULKAN_HEAD_TABLE=0 XDG_RUNTIME_DIR=/tmp
E=/root/sk/cmf/target/release/cortiq-embryo; M=/root/sk/mix; D=/root/sk/data
T=/root/sk/teacher/runs/hermes_core_a40_20261114/ckpt/base1-step408200.ckpt
COMMON="--shard $M/en2.u16:0.6 $M/ru2.u16:0.2 $M/code.u16:0.12 $M/math2.u16:0.08 --val $D/rawval-tails-2m.u16 \
 --seq 1024 --warmup 300 --lr 1e-4 --lr-restart --carry --carry-reset-every 16 --carry-eot 32760 --dropless --seed 17 \
 --retrieval-rate 3 --retrieval-dists 128,2048 --retrieval-pairs 4"
{
  until [ -f run/e1.done ] || [ -f run/e1.failed ]; do sleep 20; done
  echo "== queue2 $(date -u +%T)"
  ./build.sh; tail -1 /root/sk/build.log
  source /root/.cargo/env
  (cd cmf && cargo test --release -p cortiq-embryo --features cuda --test kd_head -- --nocapture 2>&1 | grep -E "alpha|test result|panicked")
  echo "== the starting point in E1's validation (one step at the warmup lr)"
  $E birth --resume $D/keep-final-healed.ckpt --steps 15501 --batch 32 --eval-every 1907 --save-every 100000 --out run/probe.ckpt $COMMON 2>&1 | grep -E "val loss"
  best=""; bt=0
  for cfg in "16 4" "16 2" "8 8" "8 4" "8 2"; do
    set -- $cfg
    out=$(timeout 600 $E birth --resume $D/keep-final-healed.ckpt --steps 15525 --batch $1 --eval-every 100000 --save-every 100000 \
      --out run/probe.ckpt $COMMON --kd-from $T --kd-w 0.5 --kd-batch $2 --kd-skip 64 2>&1)
    ms=$(echo "$out" | grep -E "^step" | tail -1 | grep -oE "[0-9]+ ms" | grep -oE "[0-9]+")
    echo "probe B$1 teacher $2: $(echo "$out" | grep -E "^step" | tail -1 | cut -c1-140) $(echo "$out" | grep -iE "out of memory|panicked" | head -1 | cut -c1-120)"
    if [ -n "$ms" ]; then
      tps=$(( $1 * 1024 * 1000 / ms )); echo "   $tps tok/s"
      [ $tps -gt $bt ] && { bt=$tps; best="$1 $2 $ms"; }
    fi
  done
  rm -f run/probe.ckpt
  [ -z "$best" ] && { echo "NO KD CONFIG FITS"; exit 1; }
  set -- $best
  steps=$(( 57 * 60 * 1000 / $3 ))
  echo "== E2 with B$1, teacher micro-batch $2: $3 ms a step, $steps steps for 57 min"
  touch run/queue2.started
  ./e2.sh $1 $2 $steps
} >> $L 2>&1
