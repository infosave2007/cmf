#!/bin/bash
# ppl_lean.sh FILE...: wikitext-2 first 2048 tokens and the held-out chat text (4096 tokens) on CUDA.
set -u
cd /root/sn
for f in "$@"; do
  echo "$(basename $f .cmf) [wiki 2048] $(CMF_GPU=cuda bin/cortiq-fast ppl $f --file data/wiki.test.raw --tokens 2048 2>&1 | grep -i ppl | tail -1)"
  echo "$(basename $f .cmf) [chat_eval 4096] $(CMF_GPU=cuda bin/cortiq-fast ppl $f --file data/chat_eval.txt --tokens 4096 2>&1 | grep -i ppl | tail -1)"
done
