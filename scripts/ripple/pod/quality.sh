#!/bin/bash
# quality.sh FILE LOG: the page's exact-rows figure and the fact check of a turbo file. One prompt of
# 60 000 characters with a planted pass key dumps every prompt row's score (CMF_DORY_EXACT_LOG) and the
# rows' ids (CMF_O1_TOKENS_DUMP); the fact's rows and their places follow. Then facts of four kinds at
# random places (scripts/dory/passkey2.py), eight trials.
set -uo pipefail
cd /root/rip
export XDG_RUNTIME_DIR=/tmp CMF_MAX_SEQ=70000
F=$1; L=$2
python3 - <<PY > /tmp/fact.txt
text=open("data/wiki.test.raw",encoding="utf-8").read().replace("\r","")
a=text[5000:65000]
print(a[:36000]+"\n\nRemember this pass key: 653269. It is important.\n\n"+a[36000:]+"\n\nWhat was the pass key given above? Reply with the digits only.")
PY
CMF_DORY_EXACT_LOG=/root/rip/out/scores.bin CMF_O1_TOKENS_DUMP=/root/rip/out/tokens.bin PASSKEY_TOKENS=300 timeout 900 ./cortiq run $F --prompt-file /tmp/fact.txt --greedy --no-think --max-tokens 300 2>/dev/null | tail -c 300 | tr '\n' ' ' | tee -a $L
echo >> $L
python3 - <<PY | tee -a $L
import numpy as np
from tokenizers import Tokenizer
t=Tokenizer.from_file("src/tokenizer.json")
ids=np.fromfile("out/tokens.bin",dtype=np.uint32).tolist()
sc=np.fromfile("out/scores.bin",dtype=np.float32)
hits=[i for i in range(len(ids)-6) if "pass key" in t.decode(ids[i:i+4]) and i<len(sc)]
print("rows",len(ids),"scores",len(sc),"fact rows from",hits[:3])
if hits:
    h=hits[0]; r=np.argsort(np.argsort(-sc))
    print("places",[int(r[j]) for j in range(h,min(h+24,len(sc)))], repr(t.decode(ids[h:h+24])))
PY
bash scripts/facts.sh $F 60000 8 2>&1 | tee -a $L
echo QUALITY_DONE | tee -a $L
