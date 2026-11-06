#!/bin/bash
# The final binary: the ceiling at 64000, the same text as the old binary, facts in files and past 58000 rows.
# dory_final.sh OLD NEW LOG
# From the directory of spill.sh.
export XDG_RUNTIME_DIR=/tmp CMF_MAX_SEQ=70000 RUST_LOG=cortiq_engine=info
OLD=$1; B=$2; L=$3
: > $L
V=src/scripts/dory/vram_peak.sh
T=m/dory/dory-turbo.cmf
X="$B bench $T --core --tokens 40 --ignore-eos --ctx"
rm -rf $PWD/sp; mkdir -p $PWD/sp
seals() { grep -E "O\(1\) sealed|rebuilt over" /tmp/vp.out | sed -e 's/\x1b\[[0-9;]*m//g' -e 's/.*dory: /   /' | cut -c1-110 | tail -2 | tee -a $L; }
timeout 900 $V "RAM 64000" $L $PWD/sp -- $X 64000; seals
CMF_O1_SPILL=$PWD/sp timeout 900 $V "SSD 64000" $L $PWD/sp -- $X 64000; seals
echo "files left: $(ls $PWD/sp | wc -l)" | tee -a $L
# The same greedy text: old rows on the card, new rows in RAM and in files.
python3 - <<PY > /tmp/p100.txt
text=open("wiki_test.txt",encoding="utf-8").read().replace("\r","")
print(text[5000:105000]+"\n\nIn one paragraph, what is the last article above about?")
PY
R() { CMF_GPU_PROBE=0 timeout 600 $1 run $T --prompt-file /tmp/p100.txt --greedy --no-think --max-tokens 120 2>/dev/null; }
R $OLD > /tmp/t_old.txt
R $B > /tmp/t_ram.txt
CMF_O1_SPILL=$PWD/sp R $B > /tmp/t_ssd.txt
python3 - <<PY | tee -a $L
o=open("/tmp/t_old.txt").read().split("\n[")[0]; r=open("/tmp/t_ram.txt").read().split("\n[")[0]; s=open("/tmp/t_ssd.txt").read().split("\n[")[0]
print("text same, old and RAM:", o==r, "RAM and files:", r==s, "chars", len(o))
print(repr(o[-200:]))
if o!=r:
    n=next((i for i,(x,y) in enumerate(zip(o,r)) if x!=y), -1); print("first difference at", n, repr(o[max(0,n-50):n+50]), repr(r[max(0,n-50):n+50]))
PY
timeout 1500 venv/bin/python src/scripts/dory/passkey2.py $B $T wiki_test.txt 100000 8 CMF_O1_SPILL=$PWD/sp 2>&1 | cut -c1-170 | tee -a $L
# Past 58000 rows, through a server: a single argument stops at 128 KB.
CMF_O1_SPILL=$PWD/sp $B serve $T --port 8077 > /tmp/serve.log 2>&1 &
S=$!
for i in $(seq 1 60); do curl -s localhost:8077/health >/dev/null 2>&1 && break; sleep 1; done
timeout 900 venv/bin/python src/scripts/dory/passkey_serve.py 8077 wiki_test.txt 250000 4 2>&1 | cut -c1-170 | tee -a $L
kill $S; sleep 1
echo "serve: $(grep -a -c 'rebuilt over' /tmp/serve.log) rebuilds, files left $(ls $PWD/sp | wc -l), VRAM now $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)" | tee -a $L
echo DONE >> $L
