#!/bin/bash
# Real prompts (MTP chain, pinned depths, no drafts) on both turbo files, then a long real answer and planted facts
# on the chosen one (/root/o/chosen). No prefill-kernel dependence in the answers. chain_s2b.sh LOG
cd /root/o
export XDG_RUNTIME_DIR=/tmp CMF_GPU_PROBE=0 CMF_BATCH_COOP=1
B=/root/o/src/target/release/cortiq
L=$1; : > $L
AB=/root/o/src/scripts/dory/q27_dflash_ab.sh
T=/root/o/m/orca-turbo.cmf; GT=/root/o/m/orca-gguf-turbo.cmf
$AB $B $T p sab.log 0 220 code review logs prose ru -- plain mtp k2 k3 k4
echo "== rebuilt turbo" >> $L; cat sab.log >> $L
$AB $B $GT p sabg.log 0 220 code review logs prose ru -- plain mtp k2 k3 k4
echo "== GGUF-direct turbo" >> $L; cat sabg.log >> $L
echo MTP_DONE >> $L
for i in $(seq 1 30); do [ -f /root/o/chosen ] && break; sleep 10; done
C=$(cat /root/o/chosen 2>/dev/null || echo $T); echo "chosen: $C" | tee -a $L
python3 - > /tmp/pf.txt <<'PY'
text=open("/root/o/wiki_test.txt",encoding="utf-8").read().replace("\r","")
print(text[5000:105000]+"\n\nIn one paragraph, what is the last article above about?")
PY
timeout 2400 $B run $C --prompt-file /tmp/pf.txt --greedy --no-think --max-tokens 300 > /tmp/real.txt 2>&1
echo "real 100000 chars: $(sed -e 's/\x1b\[[0-9;]*m//g' /tmp/real.txt | grep -v '^\s*$' | tail -8 | tr '\n' ' ' | cut -c1-1400)" | tee -a $L
python3 /root/o/src/scripts/dory/passkey2.py $B $C wiki_test.txt 100000 6 2>&1 | tail -1 | sed 's/^/facts: /' | tee -a $L
python3 /root/o/src/scripts/dory/passkey2.py $B $C wiki_test.txt 100000 2 CMF_O1_EXACT=none:0 2>&1 | tail -1 | sed 's/^/facts without exact rows: /' | tee -a $L
echo CHAIN_S2B_DONE >> $L
