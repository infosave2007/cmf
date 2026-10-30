#!/bin/bash
# wikitext-2 test perplexity of LFM3 CMF files on the pod: the first 4096
# tokens WITHOUT BOS (ids identical to transformers' tok(text), compared
# per position against lfm3_ref.py) and WITH BOS (the card number).
#
# usage: pod_ppl.sh MODEL.cmf [MODEL.cmf ...]   (CORTIQ, TAG, CMF_GPU from env)
set -u
B=${CORTIQ:-/root/lfm3/bin/cortiq}
D=/root/lfm3/data
L=/root/lfm3/logs
export XDG_RUNTIME_DIR=/tmp
for M in "$@"; do
  n=$(basename "$M" .cmf)${TAG:+-$TAG}
  echo "== $n"
  CMF_PPL_TRACE=1 "$B" ppl "$M" -f $D/wiki.test.raw --tokens 4096 > $L/ppl_$n.log 2>&1
  grep -E "^PPL|rror|panic" $L/ppl_$n.log
  "$B" ppl "$M" -f $D/wiki_bos.txt --tokens 4096 2>&1 | grep -E "^PPL|rror|panic" | sed 's/^/BOS /'
done
echo PPL_DONE
