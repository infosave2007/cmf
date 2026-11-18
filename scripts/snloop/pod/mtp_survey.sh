#!/bin/bash
# mtp_survey.sh BIN FILE BACKEND TAG [N]: greedy answers (no thinking, 256 tokens) to the first N prompts of each kind
# in prompts_mixed.json, with MTP drafts and without (CMF_MTP=0); decode tok/s and acceptance per prompt.
set -u
B=$1; F=$2; BK=$3; TAG=$4; N=${5:-4}
export XDG_RUNTIME_DIR=/tmp
[ $BK = cuda ] && export CMF_GPU=cuda
O=/root/sn/logs/mtp_$TAG; mkdir -p $O; : > $O/summary.txt
python3 -I -c "
import json,sys
d=json.load(open('/root/sn/src/scripts/snloop/prompts_mixed.json'))
for k,v in d.items():
    for q in v[:$N]: print(k+'\t'+q)
" > $O/prompts.tsv
i=0
while IFS=$'\t' read -r kind q; do
  for mode in spec plain; do
    E=""; [ $mode = plain ] && E="CMF_MTP=0"
    env $E timeout 900 $B run $F --no-think --greedy --max-tokens 256 --prompt "$q" > $O/$i.$mode.out 2>&1 < /dev/null
  done
  s=$(grep -oE "decode [0-9.]+ tok/s" $O/$i.spec.out | tail -1); p=$(grep -oE "decode [0-9.]+ tok/s" $O/$i.plain.out | tail -1)
  a=$(grep -oE "MTP draft: [0-9]+ of [0-9]+ accepted \([0-9]+%\)" $O/$i.spec.out | tail -1)
  same=$(diff <(sed -n '/^Prompt:/,/^\[/p' $O/$i.spec.out | grep -v '^\[') <(sed -n '/^Prompt:/,/^\[/p' $O/$i.plain.out | grep -v '^\[') > /dev/null && echo same || echo DIFF)
  echo "$i $kind | spec $s | plain $p | $a | text $same" | tee -a $O/summary.txt
  i=$((i+1))
done < $O/prompts.tsv
echo MTP_SURVEY_DONE >> $O/summary.txt
