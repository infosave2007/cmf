#!/bin/bash
# The comparison table on a quiet card: real prompts and growing contexts,
# the standard file against the turbo one. final_speed.sh BIN LOG
cd /root/dory
export XDG_RUNTIME_DIR=/tmp
B=$1; L=$2
: > $L
S() { /root/dory/venv/bin/python suite.py $B "$@" | tee -a $L; }
C() { # file, tokens, ctx, name
  r=$(timeout 1500 $B bench $1 --core --tokens $2 --ignore-eos --ctx $3 2>&1 | grep -E "Prompt|Decode|device KV|Device:" | sed -e 's/  */ /g' | tr '\n' ' ' | cut -c1-210)
  echo "$4 ctx $3: $r" | tee -a $L
}
Q=out/dory-q4tp-v2.cmf
T=out/dory-q4tp-v2-turbo.cmf
Q8=out/dory-q8_2f.cmf
S $Q standard-q4tp-v2
S $T turbo-q4tp-v2
S $Q8 standard-q8_2f
for c in 1000 4000 16000 64000; do
  C $Q 30 $c standard
  C $T 30 $c turbo
done
C $Q8 30 1000 standard-q8
C $Q8 30 4000 standard-q8
echo DONE >> $L
