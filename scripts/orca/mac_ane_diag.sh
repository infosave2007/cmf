#!/bin/bash
# Page faults, CPU time and footprint of the 3 600-token run with and without the ANE. LOG
R=/Users/oleg/dev/orca-runs/mac; B=/Users/oleg/dev/cmf-orca/target/release/cortiq; M=$R/orcacyber-nano-1.0-turbo.cmf; L=$1; : > $L
for arm in "GPU CMF_ANE=0" "ANE CMF_ANE=1"; do
  set -- $arm; sleep 90
  env CMF_GRAPH_SPEC=1 $2 /usr/bin/time -l $B run $M --prompt-file $R/p3600.txt --greedy --no-think --max-tokens 120 > $R/ad.out 2> $R/ad.err
  echo "$1: $(grep -E '^\[' $R/ad.out | head -1 | cut -c1-110) | $(grep -E 'real|user|sys' $R/ad.err | head -1 | sed 's/  */ /g') | faults $(grep 'page faults' $R/ad.err | awk '{print $1}') reclaims $(grep 'page reclaims' $R/ad.err | awk '{print $1}') | footprint $(grep 'peak memory footprint' $R/ad.err | awk '{printf "%.2f GB", $1/1073741824}') | invol cs $(grep 'involuntary' $R/ad.err | awk '{print $1}')" | tee -a $L
done
vm_stat | head -8 >> $L
echo DIAG_DONE >> $L
