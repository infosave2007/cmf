#!/bin/bash
# wikitext-2 test perplexity of LFM3 CMF files on the pod: the first 4096
# tokens without BOS (ids identical to transformers' tok(text)) and with BOS
# (the card number), each compared per position against the transformers
# float32 reference (lfm3_ref.py ppl over ref/ids4096*.txt).
#
# usage: pod_ppl.sh MODEL.cmf [MODEL.cmf ...]   (CORTIQ, TAG, CMF_GPU from env)
set -u
B=${CORTIQ:-/root/lfm3/bin/cortiq}
R=/root/lfm3
export XDG_RUNTIME_DIR=/tmp
for M in "$@"; do
  n=$(basename "$M" .cmf)${TAG:+-$TAG}
  echo "== $n"
  CMF_PPL_TRACE=1 "$B" ppl "$M" -f $R/data/wiki.test.raw --tokens 4096 > $R/logs/ppl_$n.log 2>&1
  CMF_PPL_TRACE=1 "$B" ppl "$M" -f $R/data/wiki_bos.txt --tokens 4096 > $R/logs/ppl_bos_$n.log 2>&1
  python3 -I $R/scripts/ppl_cmp.py $R/ref/ppl_f32.npz $R/logs/ppl_$n.log
  python3 -I $R/scripts/ppl_cmp.py $R/ref/ppl_bos_f32.npz $R/logs/ppl_bos_$n.log
done
echo PPL_DONE
