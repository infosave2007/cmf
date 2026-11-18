#!/bin/bash
# vulkan_rest.sh: the Vulkan half of final_chain.sh past its first turbo rows, then a CUDA image request through
# the server (the first CUDA serve check had a broken image payload).
set -u
cd /root/sn
B=bin/cortiq
export XDG_RUNTIME_DIR=/tmp CMF_MAX_SEQ=270000
Q4=out/sensenova-loop-9b-q4tp.cmf; T=out/sensenova-loop-9b-turbo.cmf; BK=vulkan
cp logs/bench_vulkan_turbo.txt logs/bench_vulkan_turbo_a.txt
BENCH_ENV=CMF_MTP=0 bash bench.sh $B $T ${BK}_turbo_b $BK 64000 128000
mkdir -p /root/sn/spill
BENCH_ENV="CMF_MTP=0 CMF_O1_SPILL=/root/sn/spill" bash bench.sh $B $T ${BK}_turbo250 $BK 250000
echo "== $BK real answers $(date)"
bash mtp_survey.sh $B $Q4 $BK final_${BK}_q4 4
echo "== $BK facts $(date)"
for c in 60000 120000 240000; do
  for f in $T $Q4; do
    [ $f = $Q4 ] && [ $c = 240000 ] && continue
    echo "-- facts $c $(basename $f) $BK"
    python3 -I src/scripts/snloop/facts_engine.py $B $f ref/facts${c}.json
  done
done
echo "== $BK serve $(date)"
bash serve_check.sh $B $Q4 $BK ${BK}_q4
bash serve_check.sh $B $T $BK ${BK}_turbo
echo "== cuda serve image $(date)"
bash serve_check.sh $B $Q4 cuda cuda_q4_img
echo VULKAN_REST_DONE $(date)
