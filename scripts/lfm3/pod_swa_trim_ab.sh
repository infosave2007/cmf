#!/bin/bash
# Sliding-window KV trim on LFM3 (the 5 sliding layers keep only the rows
# their 256-token window can still read) against CMF_SWA_TRIM=0: perplexity,
# greedy output on a long prompt, KV memory and speed at 8k and 16k.
#
# usage: pod_swa_trim_ab.sh MODEL.cmf   (CORTIQ from env)
set -u
B=${CORTIQ:-/root/lfm3/bin/cortiq}
M=$1
R=/root/lfm3
export XDG_RUNTIME_DIR=/tmp
P="$(head -c 14000 $R/data/wiki.test.raw) Summarize the text above in one sentence."
for t in 1 0; do
  echo "== CMF_SWA_TRIM=$t"
  CMF_SWA_TRIM=$t "$B" ppl "$M" -f $R/data/wiki_bos.txt --tokens 4096 2>&1 | grep -a "PPL ="
  CMF_SWA_TRIM=$t "$B" run "$M" --prompt "$P" --no-think --greedy --max-tokens 96 2>/dev/null \
    | sed -n '/^Prompt:/,$p' | grep -v "tokens ·" > $R/logs/swa_greedy_$t.txt
done
cmp $R/logs/swa_greedy_1.txt $R/logs/swa_greedy_0.txt && echo "GREEDY IDENTICAL" || echo "GREEDY DIFFERS"
for r in 1 2; do
  for C in 8192 16384; do
    for t in 0 1; do
      printf "r$r ctx$C trim=$t: "
      CMF_SWA_TRIM=$t "$B" bench "$M" --core --ignore-eos --tokens 64 --ctx $C 2>&1 \
        | grep -aE "Prompt:|Decode:|Memory:" | sed 's/^ *//' | tr '\n' ' '
      echo
    done
  done
done
echo AB_DONE
