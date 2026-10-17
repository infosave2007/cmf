#!/bin/bash
# eval.sh TAG MODEL: raw-dump a real CMF (the q4x cortiq build, read-only use: it has `dequant --dtype raw`),
# torch ppl on WikiText-2 test (160 + 12 windows of 512, the q2tp study's evaluator, outputs redirected to
# /root/eng_q2tp), then delete the dump. Serialised by our own lock.
set -u
TAG=$1; M=$2
B=/root/q4x/src/target/release/cortiq
PY=/root/q4x/venv/bin/python
E=/root/eng_q2tp
D=$E/dumps/$TAG
export CMF_GPU=0 CMF_THREADS=4 RAYON_NUM_THREADS=4
mkdir -p $D $E/logs $E/pw
exec 9> $E/eval.lock
flock 9
$B dequant $M --name model. --all --dtype raw --out $D 2> $D/listing.raw
$B dequant $M --name lm_head --all --dtype raw --out $D 2>> $D/listing.raw
grep -P "\t" $D/listing.raw > $D/listing.tsv
$B info $M --tensors '' > $E/logs/info_$TAG.txt 2>&1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY $E/scripts/ppl_torch2.py --dump $D --tag $TAG > $E/logs/tppl_$TAG.log 2>&1
echo "$(date -u +%T) $TAG $(tail -1 $E/logs/tppl_$TAG.log)" >> $E/logs/eval.log
rm -rf $D
