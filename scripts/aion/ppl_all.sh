#!/usr/bin/env bash
# ppl_all.sh FILE TAG [ENV...]: wiki/code/python 12x512 + wiki 4x4096
source /root/aion/scripts/env.sh
F=$1; TAG=$2; shift 2
for t in "wiki data/wiki.test.raw 12 512" "code data/code_eval.txt 12 512" "py data/python_eval.txt 12 512" "wiki4k data/wiki.test.raw 4 4096"; do
  set -- $t
  s=$(date +%s)
  r=$(env ${PPL_ENV:-} $B ppl $F --file $2 --windows $3 --window-len $4 2>&1 | grep -E "PPL|ppl" | tail -2 | tr '\n' ' ')
  echo "$TAG $1 $r ($(( $(date +%s) - s )) s)"
done
