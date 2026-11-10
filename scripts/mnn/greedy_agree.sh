#!/bin/bash
# Greedy answers (96 tokens) of each CMF file to the 48 chat_eval.py requests, and how many equal the f16 file's.
# greedy_agree.sh FILE...   (the first file is the reference)
set -u
R=$HOME/models/mnn-pocket; B=${BIN:-$R/bin/cortiq}
P=$R/data/prompts.txt
[ -f $P ] || ~/dev/vision-test/venv/bin/python -I -c "
import ast,sys
src=open('$HOME/dev/cmf-mnn/scripts/mnn/chat_eval.py',encoding='utf-8').read()
t=ast.parse(src); L={}
for n in t.body:
  if isinstance(n,ast.Assign) and getattr(n.targets[0],'id',None) in ('EN','ZH'): L[n.targets[0].id]=ast.literal_eval(n.value)
open('$P','w',encoding='utf-8').write('\n'.join(p for pr in zip(L['EN'],L['ZH']) for p in pr)+'\n')"
ref=""
for F in "$@"; do
  tag=$(basename $F .cmf); D=$R/logs/greedy_$tag; mkdir -p $D; i=0
  while IFS= read -r q; do
    i=$((i+1))
    $B run $F --greedy -n 96 --prompt "$q" 2>/dev/null < /dev/null | awk 'f&&/^\[[0-9]+ tokens/{exit} f{print} /^Prompt: /{f=1;getline}' > $D/$i.txt
  done < $P
  if [ -z "$ref" ]; then ref=$D; echo "$tag: reference"; continue; fi
  same=0; for k in $(seq 1 $i); do cmp -s $ref/$k.txt $D/$k.txt && same=$((same+1)); done
  pre=$(python3 -I -c "
import os,sys
a,b,n=sys.argv[1],sys.argv[2],int(sys.argv[3]); t=0
for k in range(1,n+1):
  x=open(f'{a}/{k}.txt','rb').read(); y=open(f'{b}/{k}.txt','rb').read(); j=0
  while j<min(len(x),len(y)) and x[j]==y[j]: j+=1
  t+=j/max(len(x),1)
print(f'{100*t/n:.0f}')" $ref $D $i)
  echo "$tag: $same/$i greedy answers identical to $(basename $ref), mean common prefix $pre % of the reference answer"
done
