#!/bin/bash
# wikitext-2 test perplexity of Apex CMF files on the pod's GPU: the first
# 4096 tokens with BOS (the conversion gate) and 12 x 512 windows. The ids
# `cortiq ppl` scored go to ids4096.txt for the PyTorch reference.
#
# usage: pod_ppl.sh MODEL.cmf [MODEL.cmf ...]
set -u
B=${CORTIQ:-/root/apex/bin/cortiq}
T=/root/apex/data/wiki.test.raw
export XDG_RUNTIME_DIR=/tmp
for M in "$@"; do
  echo "== $(basename "$M")"
  CMF_PPL_IDS_OUT=/root/apex/ids4096.txt "$B" ppl "$M" -f "$T" --tokens 4096 2>&1 | grep -E "PPL|rror|panic"
  "$B" ppl "$M" -f "$T" --windows 12 --window-len 512 2>&1 | grep -E "PPL|windows|rror|panic"
done
echo PPL_DONE
