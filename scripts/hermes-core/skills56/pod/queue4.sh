#!/bin/bash
# After E1b: the KLM index of E1b on the GPU; E3 on the E1 base and, as its control, on the release 56M (the same
# 2000 chat SFT steps); their R answers on 6 threads beside the GPU work, and their KLM index.
# Log /root/sk/run/queue4.log; marker queue4.done
cd /root/sk
L=/root/sk/run/queue4.log
export EMBRYO_LOGLIK_BIN=/root/sk/cmf/target-kd/release/cortiq-embryo
{
  until [ -f run/queue3.done ]; do sleep 20; done
  until ! pgrep -f "klm_gpu.s[h]" >/dev/null; do sleep 20; done
  echo "== queue4 $(date -u +%T)"
  ./klm_gpu.sh e1b-end=/root/sk/run/e1b-end.ckpt
  ./e3.sh e1s /root/sk/run/e1-end.ckpt
  ( CMF_THREADS=6 nice -n 10 ./measure.sh c1-e1s cards > run/m-c1-e1s.log 2>&1 ) &
  ./e3.sh f0s /root/sk/data/keep-final-healed.ckpt
  wait
  ( CMF_THREADS=6 nice -n 10 ./measure.sh c1-f0s cards > run/m-c1-f0s.log 2>&1 ) &
  ./klm_gpu.sh e1s=/root/sk/run/e1s-sft.ckpt f0s=/root/sk/run/f0s-sft.ckpt
  wait
  tail -2 run/m-c1-e1s.log run/m-c1-f0s.log
  echo "== queue4 done $(date -u +%T)"
  touch run/queue4.done
} >> $L 2>&1
