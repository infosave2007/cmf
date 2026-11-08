#!/bin/bash
# OrcaCyber-Nano final numbers on the RTX 2000 Ada 16 GB with the final binary: bench standard vs turbo (VRAM/RAM peaks),
# real prompts plain vs MTP, a real answer over 100 000 characters, planted facts. final.sh LOG BINARY
cd /root/o
export XDG_RUNTIME_DIR=/tmp CMF_GPU_PROBE=0
L=$1; B=$2; : > $L
Q=/root/o/m/orca-gguf-q4tp.cmf; T=/root/o/m/orca-gguf-turbo.cmf
V=/root/o/src/scripts/dory/vram_peak.sh
AB=/root/o/src/scripts/dory/q27_dflash_ab.sh
mkdir -p sp
echo "binary: $($B --version) ($B); $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader)" | tee -a $L
X() { echo "bench $1 --core --tokens 60 --ignore-eos --ctx $2"; }
for c in 1000 4000 16000 32000; do $V "turbo $c" $L sp -- $B $(X $T $c); done
CMF_MAX_SEQ=70000 $V "turbo 64000" $L sp -- $B $(X $T 64000)
for c in 1000 4000 16000; do $V "standard $c" $L sp -- $B $(X $Q $c); done
CMF_MAX_SEQ=40000 $V "standard 32000" $L sp -- $B $(X $Q 32000)
CMF_MAX_SEQ=70000 $V "standard 64000" $L sp -- $B $(X $Q 64000)
$AB $B $T p fab.log 0 220 code review logs prose ru -- plain mtp
echo "== real prompts (turbo)" >> $L; cat fab.log >> $L
python3 - > /tmp/pf.txt <<'PY'
text=open("/root/o/wiki_test.txt",encoding="utf-8").read().replace("\r","")
print(text[5000:105000]+"\n\nIn one paragraph, what is the last article above about?")
PY
CMF_GRAPH_SPEC_TIME=1 timeout 2400 $B run $T --prompt-file /tmp/pf.txt --greedy --no-think --max-tokens 300 > /tmp/real.txt 2>&1
echo "real 100000 chars turbo: $(sed -e 's/\x1b\[[0-9;]*m//g' /tmp/real.txt | grep -v '^\s*$' | tail -8 | tr '\n' ' ' | cut -c1-1500)" | tee -a $L
python3 /root/o/src/scripts/dory/passkey2.py $B $T wiki_test.txt 100000 6 2>&1 | tail -1 | sed 's/^/turbo facts: /' | tee -a $L
python3 /root/o/src/scripts/dory/passkey2.py $B $T wiki_test.txt 100000 2 CMF_O1_EXACT=none:0 2>&1 | tail -1 | sed 's/^/turbo without exact rows facts: /' | tee -a $L
echo FINAL_DONE >> $L
