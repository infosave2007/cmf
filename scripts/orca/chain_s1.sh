#!/bin/bash
# OrcaCyber-Nano, step 3: quality controls (the base's published Turbo, the GGUF-direct file again, wikitext and
# code), the GGUF-direct turbo file. chain_s1.sh LOG
cd /root/o
export XDG_RUNTIME_DIR=/tmp CMF_GPU_PROBE=0
B=/root/o/src/target/release/cortiq
L=$1; : > $L
GG=/root/o/m/OrcaCyber-Nano-1.0-AI-Defense-Mix-15.7G.gguf
GQ=/root/o/m/orca-gguf-q4tp.cmf; GT=/root/o/m/orca-gguf-turbo.cmf
RQ=/root/o/m/orca-q4tp.cmf; T=/root/o/m/orca-turbo.cmf; QT=/root/o/m/qwen38-27b-turbo.cmf
K=/root/o/src/scripts/dory/router/keep27.bin
rm -rf /root/o/orca-bf16   # reproducible: Qwen bf16 (1 min) + rebuild.py (7 min)
df -h / | tail -1 | tee -a $L
python3 dl.py infosave/Qwen3.8-27B-Turbo-cmf /root/o/m qwen38-27b-turbo.cmf | tail -1 | tee -a $L
python3 dl.py orcarouter/OrcaCyber-Nano-1.0 /root/o/m OrcaCyber-Nano-1.0-AI-Defense-Mix-15.7G.gguf | tail -1 | tee -a $L
( $B import-gguf $GG --quant q4tp --output $GQ --tokenizer-dir /root/o/tokdir > /root/o/imp2.out 2> /root/o/imp2.err; echo "import2 rc=$?" >> $L ) &
IP=$!
# A code corpus: the importer source (Rust), 12 windows of 512 tokens like wikitext.
cat /root/o/src/crates/cortiq-cli/src/gguf.rs /root/o/src/crates/cortiq-cli/src/convert.rs | head -c 400000 > /root/o/code_test.txt
P() { echo "ppl $2 $1: $($B ppl $1 -f $2 --windows 12 --window-len 512 2>&1 | grep -iE "perplexity|ppl" | tail -1)" | tee -a $L; }
P $QT wiki_test.txt
wait $IP
ls -l $GQ | tee -a $L
rm -f $GG
for f in $GQ $RQ $QT; do P $f code_test.txt; done
rm -f $QT
$B far-turbo $GQ $GT --exact-weights $K 2>&1 | tail -1 | cut -c1-200 | tee -a $L
ls -l $GT | tee -a $L
echo "ppl turbo files (exact under the seal):" | tee -a $L
P $T wiki_test.txt
P $GT wiki_test.txt
df -h / | tail -1 | tee -a $L
echo CHAIN_S1_DONE >> $L
