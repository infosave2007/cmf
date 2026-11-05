#!/bin/bash
# The table of the Qwen3.8-27B turbo page, on a quiet card. q27_final.sh BIN LOG
cd /root/q
export XDG_RUNTIME_DIR=/tmp
B=$1; L=$2
: > $L
V=src/scripts/dory/vram_peak.sh
Q=m/qwen38-27b-q4tp.cmf
T=m/qwen38-27b-turbo.cmf
X() { echo "bench $1 --core --tokens 60 --ignore-eos --ctx $2"; }
$B far-turbo $Q $T --exact-weights keep27.bin 2>&1 | tail -1 | cut -c1-120 | tee -a $L
for c in 1000 4000 16000 32000; do $V "standard $c" $L /root/q/sp -- $B $(X $Q $c); done
CMF_MAX_SEQ=70000 $V "standard 64000" $L /root/q/sp -- $B $(X $Q 64000)
for c in 1000 4000 16000 32000 64000; do $V "turbo $c" $L /root/q/sp -- $B $(X $T $c); done
export CMF_O1_SPILL=/root/q/sp
for c in 16000 32000 64000; do $V "turbo ssd $c" $L /root/q/sp -- $B $(X $T $c); done
unset CMF_O1_SPILL
export CMF_GPU_VRAM_MB=15000
for c in 4000 16000 32000; do $V "turbo 16GB $c" $L /root/q/sp -- $B $(X $T $c); done
CMF_O1_SPILL=/root/q/sp $V "turbo 16GB ssd 64000" $L /root/q/sp -- $B $(X $T 64000)
unset CMF_GPU_VRAM_MB
# Real text: a summary of a long article, tokens of the answer as they come.
for chars in 30000 100000; do
  python3 - $chars > /tmp/pf.txt <<PY
import sys
n=int(sys.argv[1]); text=open("wiki_test.txt",encoding="utf-8").read().replace("\r","")
print(text[5000:5000+n]+"\n\nIn one paragraph, what is the last article above about?")
PY
  for f in $Q $T; do
    r=$(CMF_MAX_SEQ=70000 timeout 900 $B run $f --prompt "$(cat /tmp/pf.txt)" --greedy --no-think --max-tokens 300 --ignore-eos 2>/dev/null | tail -2 | tr '\n' ' ' | cut -c1-170)
    echo "real $chars chars $f: $r" | tee -a $L
  done
done
echo DONE >> $L
