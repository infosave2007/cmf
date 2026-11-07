#!/bin/bash
# Long-window perplexity (2 x 16384 tokens of WikiText-2): exact vs the far field sealed at 8192, and a real
# 400-token summary after 30 000 / 100 000 characters, standard vs turbo.
cd /root/cherrypit; export XDG_RUNTIME_DIR=/tmp CMF_MAX_SEQ=70000
S=hfrepo/CherryPit-1.2-2B-q4tp.cmf; T=models/CherryPit-1.2-2B-turbo.cmf; W=/root/rip/data/wiki.test.raw; L=logs/turboq.txt; : > $L
P() { timeout 1800 ./cortiq ppl $S -f $W --windows 2 --window-len 16384 2>&1 | grep -E "PPL|panicked" | tail -1; }
echo "ppl exact: $(P)" | tee -a $L
echo "ppl far: $(CMF_O1=all CMF_O1_GPU=1 CMF_O1_WINDOW=1800 CMF_O1_SINK=4 CMF_O1_PREFILL=8192 P)" | tee -a $L
for chars in 30000 100000; do
  python3 - $chars > /tmp/pf.txt <<PY
import sys
n=int(sys.argv[1]); text=open("$W",encoding="utf-8").read().replace("\r","")
print(text[5000:5000+n]+"\n\nWrite a detailed summary of the last article above, in several paragraphs.")
PY
  for f in $S $T; do
    r=$(timeout 1200 ./cortiq run $f --prompt-file /tmp/pf.txt --greedy --no-think --max-tokens 400 2>/dev/null | tail -2 | tr "\n" " " | cut -c1-200)
    echo "real $chars chars $f: $r" | tee -a $L
  done
done
echo TQ_DONE >> $L
