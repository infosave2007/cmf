#!/bin/bash
# t1.sh MODEL [SKILL]: greedy 64 tokens, CPU vs CUDA; prints texts + cuda decline lines
set -u
B=${B:-/root/h3i/bin/cortiq-skill}
M=$1; S=${2:-}
P=${P:-"Write a Python function that checks whether a number is prime, then explain it."}
cd /root/h3i/skill
sk=(); [ -n "$S" ] && sk=(--skill "$S")
tag=$(basename $M .cmf)${S:+-$S}
CMF_GPU=0 timeout 600 $B run $M -p "$P" -n ${N:-64} --greedy "${sk[@]}" > out/$tag.cpu.txt 2> out/$tag.cpu.err
flock /root/h3i/gpu.lock env CMF_GPU=cuda CMF_CUDA_DEBUG=1 timeout 600 $B run $M -p "$P" -n ${N:-64} --greedy "${sk[@]}" > out/$tag.cuda.txt 2> out/$tag.cuda.err
echo "== $tag"
for b in cpu cuda; do echo "-- $b"; grep -v -E "^\s*$|INFO|Ready|Loaded|Opened" out/$tag.$b.txt | tail -12; done
cmp -s <(sed -n "/Ready/,\$p" out/$tag.cpu.txt | grep -v -E "tok/s|ms|Ready") <(sed -n "/Ready/,\$p" out/$tag.cuda.txt | grep -v -E "tok/s|ms|Ready") && echo SAME_TEXT || echo DIFF_TEXT
grep -i -E "declin|refus|fallback|host|error|fail" out/$tag.cuda.err | sort | uniq -c | sort -rn | head -12
