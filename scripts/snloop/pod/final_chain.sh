#!/bin/bash
# final_chain.sh Q4SRC: the published files from the chosen q4tp, the turbo file, their private backup, then the
# card's measurements on the release build (CUDA, then Vulkan): bench rows without drafts, real answers with and
# without drafts, planted facts, the server. HF_TOKEN in the environment (stdin of the ssh), never on disk.
set -u
Q4SRC=$1
cd /root/sn
B=bin/cortiq
export XDG_RUNTIME_DIR=/tmp CMF_MAX_SEQ=270000
Q4=out/sensenova-loop-9b-q4tp.cmf; Q8=out/sensenova-loop-9b-q8_2f.cmf; T=out/sensenova-loop-9b-turbo.cmf
[ -f $Q4 ] || ln $Q4SRC $Q4
[ -f $Q8 ] || ln out/sn9-q8_2f.cmf $Q8
[ -f $T ] || $B far-turbo $Q4 $T --exact-weights turbo/keep_sn9_long.bin 2>&1 | tail -1
ls -la $Q4 $Q8 $T
for f in $Q4 $Q8 $T; do sha256sum $f > $f.sha256; done
python3 -I src/scripts/snloop/hf_up.py infosave/cmf-experiments snloop-2026-11-18 $Q4 $T turbo/keep_sn9_long.bin 2>&1 | grep -E "VERIFIED|MISMATCH|private"
unset HF_TOKEN
for c in 120000 240000; do
  [ -f ref/facts${c}.json ] || FACTS_NO_REF=1 python3 -I src/scripts/snloop/facts_ref.py dl/SenseNova-Loop-9B data/wiki.test.raw $c 4 ref/facts${c}.json
done
for BK in cuda vulkan; do
  echo "== $BK bench $(date)"
  BENCH_ENV=CMF_MTP=0 bash bench.sh $B $Q4 ${BK}_q4 $BK 1000 4000 16000 32000 64000 128000
  BENCH_ENV=CMF_MTP=0 bash bench.sh $B $Q8 ${BK}_q8 $BK 1000 4000 16000
  BENCH_ENV=CMF_MTP=0 bash bench.sh $B $T ${BK}_turbo $BK 16000 32000 64000 128000
  # 250 000 rows: their keys and values (~114 KB a row) go to files, not the 50 GB container
  mkdir -p /root/sn/spill
  BENCH_ENV="CMF_MTP=0 CMF_O1_SPILL=/root/sn/spill" bash bench.sh $B $T ${BK}_turbo250 $BK 250000
  echo "== $BK real answers $(date)"
  bash mtp_survey.sh $B $Q4 $BK final_${BK}_q4 4
  echo "== $BK facts $(date)"
  E=""; [ $BK = cuda ] && E="CMF_GPU=cuda"
  for c in 60000 120000 240000; do
    for f in $T $Q4; do
      [ $BK = vulkan ] && [ $f = $Q4 ] && [ $c = 240000 ] && continue
      echo "-- facts $c $(basename $f) $BK"
      env $E python3 -I src/scripts/snloop/facts_engine.py $B $f ref/facts${c}.json
    done
  done
  echo "== $BK serve $(date)"
  bash serve_check.sh $B $Q4 $BK ${BK}_q4
  bash serve_check.sh $B $T $BK ${BK}_turbo
done
echo FINAL_CHAIN_DONE $(date)
