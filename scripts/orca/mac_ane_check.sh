#!/bin/bash
# Is decode after an ANE prompt slower, or was it heat? 3600 tokens, ANE first then GPU, 90 s pauses. LOG
R=/Users/oleg/dev/orca-runs/mac; B=/Users/oleg/dev/cmf-orca/target/release/cortiq; M=$R/orcacyber-nano-1.0-turbo.cmf; L=$1; : > $L
for arm in "ANE CMF_ANE=1" "GPU CMF_ANE=0" "ANE CMF_ANE=1"; do
  set -- $arm; sleep 90
  env CMF_GRAPH_SPEC=1 $2 $B run $M --prompt-file $R/p3600.txt --greedy --no-think --max-tokens 120 > $R/ac.out 2>/dev/null
  echo "$1: $(grep -E '^\[' $R/ac.out | head -2 | tr '\n' ' ' | cut -c1-170) | text $(grep -vE '^\[' $R/ac.out | md5 -q | cut -c1-8)" | tee -a $L
done
echo ANE_DONE >> $L
