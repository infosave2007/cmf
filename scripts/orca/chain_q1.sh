#!/bin/bash
# OrcaCyber-Nano, step 2: rebuild the weights on the CPU while the GPU reads the GGUF-direct q4tp (quality only:
# the CPU is busy, no speed numbers here); then q4tp from the rebuilt weights, the turbo file, the same checks.
# chain_q1.sh LOG
cd /root/o
export XDG_RUNTIME_DIR=/tmp CMF_GPU_PROBE=0
B=/root/o/src/target/release/cortiq
L=$1; : > $L
GG=/root/o/m/OrcaCyber-Nano-1.0-AI-Defense-Mix-15.7G.gguf
GQ=/root/o/m/orca-gguf-q4tp.cmf; RQ=/root/o/m/orca-q4tp.cmf; T=/root/o/m/orca-turbo.cmf
K=/root/o/src/scripts/dory/router/keep27.bin
echo "binary: $($B --version)" | tee -a $L
[ -f /root/o/m/orca-q4tp.cmf ] && [ ! -f $GQ ] && mv /root/o/m/orca-q4tp.cmf $GQ
OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 setsid nohup python3 rebuild.py $GG /root/o/qwen /root/o/orca-bf16 --workers 6 --delete-base > /root/o/rebuild.log 2>&1 < /dev/null &
RB=$!
A() { echo "== $1: $2" >> $L; timeout 900 $B run $1 --prompt "$2" --greedy --no-think --max-tokens ${3:-200} > /tmp/ans.txt 2>&1; sed -e 's/\x1b\[[0-9;]*m//g' /tmp/ans.txt | grep -v "^\s*$" | tail -12 >> $L; }
Q() {
  A $1 "Who are you? Answer in one sentence."
  A $1 "What does this C function do, and is it safe? void f(char *s){char b[16]; strcpy(b,s); puts(b);}" 260
  A $1 "Объясни простыми словами, что такое SQL-инъекция, в трёх предложениях."
  A $1 "用两句话解释什么是缓冲区溢出。"
  echo "ppl $1: $($B ppl $1 -f wiki_test.txt --windows 12 --window-len 512 2>&1 | grep -iE "perplexity|ppl" | tail -1)" | tee -a $L
}
Q $GQ
while kill -0 $RB 2>/dev/null; do sleep 20; done
tail -3 /root/o/rebuild.log | tee -a $L
grep -c '"base + low-rank edit"' /root/o/orca-bf16/rebuild_report.jsonl | sed 's/^/edited matrices: /' | tee -a $L
# Disk: 100 GB. The GGUF (14 s to fetch again) and the GGUF-direct file (measured above) make room for convert.
rm -f $GG $GQ
cp /root/o/tokdir/tokenizer.json /root/o/tokdir/tokenizer_config.json /root/o/tokdir/chat_template.jinja /root/o/qtok/generation_config.json /root/o/orca-bf16/
df -h / | tail -1 | tee -a $L
t0=$(date +%s)
$B convert --model /root/o/orca-bf16 --quant q4tp --output $RQ > /root/o/conv.out 2>&1
echo "convert rc=$? $(( $(date +%s)-t0 )) s" | tee -a $L
tail -3 /root/o/conv.out | cut -c1-200 | tee -a $L
ls -l $GQ $RQ | tee -a $L
Q $RQ
$B far-turbo $RQ $T --exact-weights $K 2>&1 | tail -2 | cut -c1-200 | tee -a $L
ls -l $T | tee -a $L
Q $T
echo CHAIN_Q1_DONE >> $L
