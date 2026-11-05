#!/bin/bash
# The host path on the pod's CPU: standard, turbo, q8. cpu_final.sh BIN LOG
cd /root/dory
B=$1; L=$2
: > $L
export CMF_GPU=0
for f in out/dory-q4tp-v2.cmf out/dory-q4tp-v2-turbo.cmf out/dory-q8_2f.cmf; do
  for c in 256 4000; do
    r=$(timeout 900 $B bench $f --core --tokens 30 --ignore-eos --ctx $c 2>&1 | grep -E "Prompt|Decode" | sed -e 's/  */ /g' | tr '\n' ' ' | cut -c1-130)
    echo "$(basename $f) ctx $c: $r" | tee -a $L
  done
done
echo DONE >> $L
