#!/bin/bash
# After the second fetch: documents for the remaining sources, own tokenizer, then one shard per source.
C=/root/corpus; O=$C/shards.log; : > $O; T=/root/cmf/target/release/cortiq-embryo; export XDG_RUNTIME_DIR=/tmp
S=/root/cmf/scripts/hermes-core
deadline=$(( $(date +%s) + 14400 ))
until grep -q FETCH2-DONE $C/fetch2.log || [ $(date +%s) -gt $deadline ]; do sleep 30; done
python3 $S/corpus_build.py $C/docs se_ru_so,se_so_ops >> $O 2>&1
mkdir -p $C/shards
ALL=$(ls $C/docs/*.jsonl.gz | grep -v '\.val\.' | tr '\n' ' ')
$T train-tokenizer --inputs $ALL --out $C/hermes.tokenizer.json --vocab 32768 >> $O 2>&1
for f in $C/docs/*.jsonl.gz; do
  n=$(basename $f .jsonl.gz)
  $T shard --tokenizer $C/hermes.tokenizer.json --inputs $f --out $C/shards/$n.u16 >> $O 2>&1
done
echo "tokens per shard:" >> $O
for f in $C/shards/*.u16; do echo "$(basename $f .u16) $(( $(stat -c %s $f) / 2 ))" >> $O; done
echo SHARDS-DONE >> $O
