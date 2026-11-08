#!/bin/bash
# Card answers of the published file, stdout only (no timing logs). answers.sh LOG BINARY MODEL
cd /root/o; export XDG_RUNTIME_DIR=/tmp CMF_GPU_PROBE=0
L=$1; B=$2; M=$3; : > $L
for i in $(seq 1 360); do grep -q FINAL_DONE /root/o/final.log 2>/dev/null && break; sleep 10; done
A() { echo "== $1" >> $L; timeout 900 $B run $M --prompt "$1" --greedy --no-think --max-tokens ${2:-300} 2>/dev/null | sed -e 's/\x1b\[[0-9;]*m//g' | grep -vE "^Loading|^Ready|^Prompt" >> $L; }
A "Who are you? Answer in one sentence." 60
A "What does this C function do, and is it safe? void f(char *s){char b[16]; strcpy(b,s); puts(b);}" 400
A "Объясни простыми словами, что такое SQL-инъекция, в трёх предложениях." 200
A "用两句话解释什么是缓冲区溢出。" 150
echo ANSWERS_DONE >> $L
