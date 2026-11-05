#!/bin/bash
# The turbo file of Qwen3.8-27B: short prompts, long ones, drafts on and off.
cd /root/q
export XDG_RUNTIME_DIR=/tmp CMF_MAX_SEQ=70000
B=$1; L=$2
: > $L
V=src/scripts/dory/vram_peak.sh
T=m/qwen38-27b-turbo.cmf
$B far-turbo m/qwen38-27b-q4tp.cmf $T --exact-weights keep27.bin 2>&1 | tail -1 | cut -c1-120 | tee -a $L
X="$B bench $T --core --tokens 60 --ignore-eos --ctx"
for c in 1000 4000 12000 32000; do
  $V "turbo $c" $L /root/q/sp -- $X $c
done
CMF_O1_SPEC=0 $V "turbo no drafts 12000" $L /root/q/sp -- $X 12000
python3 - <<PY > /tmp/p27.txt
text=open("wiki_test.txt",encoding="utf-8").read().replace("\r","")
print(text[5000:65000]+"\n\nIn one paragraph, what is the last article above about?")
PY
R() { timeout 400 $B run $T --prompt "$(cat /tmp/p27.txt)" --greedy --no-think --max-tokens 120 2>/dev/null; }
R > /tmp/spec_on.txt
CMF_O1_SPEC=0 R > /tmp/spec_off.txt
python3 - <<PY | tee -a $L
a=open("/tmp/spec_on.txt").read(); b=open("/tmp/spec_off.txt").read()
ta=a.split("\n[")[0]; tb=b.split("\n[")[0]
print("text same with drafts and without:", ta==tb, len(ta), len(tb))
n=next((i for i,(x,y) in enumerate(zip(ta,tb)) if x!=y), -1)
if n>=0: print("first difference:", repr(ta[n-60:n+60]), repr(tb[n-60:n+60]))
print("on :", a[-130:].replace("\n"," "))
print("off:", b[-130:].replace("\n"," "))
PY
echo DONE >> $L
