#!/bin/bash
# Real answers, facts and long-window perplexity of the turbo file. q27_quality.sh BIN LOG
cd /root/q
export XDG_RUNTIME_DIR=/tmp
B=$1; L=$2
: > $L
Q=m/qwen38-27b-q4tp.cmf
T=m/qwen38-27b-turbo.cmf
for chars in 30000 100000; do
  python3 - $chars > /tmp/pf.txt <<PY
import sys
n=int(sys.argv[1]); text=open("wiki_test.txt",encoding="utf-8").read().replace("\r","")
print(text[5000:5000+n]+"\n\nWrite a detailed summary of the last article above, in several paragraphs.")
PY
  for f in $Q $T; do
    r=$(CMF_MAX_SEQ=70000 timeout 900 $B run $f --prompt-file /tmp/pf.txt --greedy --no-think --max-tokens 400 2>/dev/null | tail -2 | tr '\n' ' ' | cut -c1-170)
    echo "real $chars chars $f: $r" | tee -a $L
  done
done
# The fact and the scores for the page's figure.
python3 - <<PY > /tmp/fact.txt
text=open("wiki_test.txt",encoding="utf-8").read().replace("\r","")
a=text[5000:65000]
print(a[:36000]+"\n\nRemember this pass key: 653269. It is important.\n\n"+a[36000:]+"\n\nWhat was the pass key given above? Reply with the digits only.")
PY
CMF_DORY_EXACT_LOG=/tmp/q27_scores.bin CMF_O1_TOKENS_DUMP=/tmp/q27_tokens.bin timeout 400 $B run $T --prompt-file /tmp/fact.txt --greedy --no-think --max-tokens 12 2>/dev/null | tail -c 200 | tr '\n' ' ' | tee -a $L
echo >> $L
venv/bin/python - <<PY | tee -a $L
import numpy as np
from tokenizers import Tokenizer
t=Tokenizer.from_file("tok/tokenizer.json")
ids=np.fromfile("/tmp/q27_tokens.bin",dtype=np.uint32).tolist()
sc=np.fromfile("/tmp/q27_scores.bin",dtype=np.float32)
hits=[i for i in range(len(ids)-6) if "pass key" in t.decode(ids[i:i+4]) and i<len(sc)]
print("rows",len(ids),"scores",len(sc),"fact rows from",hits[:3])
if hits:
    h=hits[0]; r=np.argsort(np.argsort(-sc))
    print("places",[int(r[j]) for j in range(h,min(h+24,len(sc)))], repr(t.decode(ids[h:h+24])))
PY
# Facts on the small-card path with an SSD, then on the big-card one.
venv/bin/python src/scripts/dory/passkey2.py $B $T wiki_test.txt 100000 8 CMF_GPU_VRAM_MB=15000 CMF_O1_SPILL=/root/q/sp 2>&1 | cut -c1-170 | tee -a $L
venv/bin/python src/scripts/dory/passkey2.py $B $T wiki_test.txt 100000 4 2>&1 | cut -c1-170 | tee -a $L
# Two long windows: exact, and with the far field past the first 8192 rows.
P() { timeout 1200 $B ppl $Q -f wiki_test.txt --windows 2 --window-len 16384 2>&1 | grep -E "PPL|panicked" | tail -1 | cut -c1-110; }
echo "ppl exact: $(CMF_MAX_SEQ=70000 P)" | tee -a $L
echo "ppl far: $(CMF_MAX_SEQ=70000 CMF_O1=all CMF_O1_GPU=1 CMF_O1_WINDOW=1800 CMF_O1_SINK=4 CMF_O1_PREFILL=8192 P)" | tee -a $L
echo DONE >> $L
