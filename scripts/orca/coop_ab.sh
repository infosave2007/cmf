#!/bin/bash
# Prefill A/B on the card: CMF_BATCH_COOP (tensor-core GEMM for the batched prompt) and the chunk width, in turn.
# coop_ab.sh LOG MODEL
cd /root/o; export XDG_RUNTIME_DIR=/tmp CMF_GPU_PROBE=0
B=/root/o/src/target/release/cortiq; L=$1; M=$2; : > $L
python3 - > /tmp/p2k.txt <<'PY'
t=open("/root/o/wiki_test.txt",encoding="utf-8").read()
print(t[30000:38000]+"\n\nSummarize the article above in three sentences.")
PY
for rep in 1 2; do
for arm in "CMF_BATCH_COOP=0" "CMF_BATCH_COOP=1" "CMF_BATCH_COOP=1 CMF_BATCH_K=256"; do
  r=$(env $arm $B bench $M --core --tokens 20 --ignore-eos --ctx 2000 2>&1 | sed -e 's/\x1b\[[0-9;]*m//g' | grep -E "Prompt|Decode" | tr '\n' ' ' | sed 's/  */ /g' | cut -c1-200)
  echo "bench2000 [$arm]: $r" | tee -a $L
done; done
for arm in "CMF_BATCH_COOP=0" "CMF_BATCH_COOP=1"; do
  env $arm CMF_GRAPH_SPEC=0 $B run $M --prompt-file /tmp/p2k.txt --greedy --no-think --max-tokens 120 > /tmp/ab.out 2>&1
  echo "run [$arm]: $(grep -E '^\[' /tmp/ab.out | head -1 | cut -c1-120) | text $(grep -vE '^\[|^Loading|^Ready|^Prompt' /tmp/ab.out | md5sum | cut -c1-8)" | tee -a $L
  grep -vE '^\[|^Loading|^Ready|^Prompt' /tmp/ab.out | tr '\n' ' ' | cut -c1-300 >> $L; echo >> $L
done
echo COOP_AB_DONE >> $L
