#!/bin/bash
# The final turbo file: speed on a quiet card, the quality of a long window, facts at 64000 rows.
cd /root/dory
export XDG_RUNTIME_DIR=/tmp
B=./bin_x
L=logs/final3.log
: > $L
T=out/dory-turbo.cmf
Y=venv/bin/python
C() { r=$(timeout 1800 $B bench $1 --core --tokens 30 --ignore-eos --ctx $2 2>&1 | grep -E "Prompt|Decode|device KV" | sed -e 's/  */ /g' | tr '\n' ' ' | cut -c1-190); echo "$3 ctx $2: $r" | tee -a $L; }
rm -f suite_base_dory-turbo.cmf.json
$Y src/scripts/dory/suite.py $B $T turbo | tee -a $L
for c in 1000 4000 16000 32000 64000; do C $T $c turbo; done
echo "ppl 2x16384 turbo geometry, two passes on every row: $(CMF_DORY_LOOPS=2 CMF_DORY_KV=last_loop CMF_DORY_O1=32,1536,4,8192 CMF_DORY_EXACT=router/keep.bin:512 CMF_NLL_CHUNK=64 timeout 1500 $B ppl out/dory-q4tp-v2.cmf -f wiki_test.txt --windows 2 --window-len 16384 2>&1 | grep -E "PPL|panicked" | tail -1 | cut -c1-90)" | tee -a $L
$Y src/scripts/dory/passkey2.py $B $T wiki_test.txt 250000 4 2>&1 | cut -c1-170 | tee -a $L
echo DONE >> $L
