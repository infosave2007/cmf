#!/bin/bash
# chainM.sh: the merged tree (spark26-turbo + master with OrcaCyber-Nano #99) on the card. Builds
# cortiq.zi4 on cores 8-47, then under quiet.sh: prompt A/B zi3 (before the merge) against zi4 at 4096
# and 16 000 tokens, and greedy text parity zi3 vs zi4 on 9 000 characters (SAME_TEXT expected: the
# merge only moved the in-kernel-dequant arm, which a 512-row Spark chunk does not take). Then on 8
# pinned host cores (CMF_GPU=0) the decode lane width of the CPU q4tp matvec (CMF_Q4TP_LANES 2/4/8) at
# 1000 tokens with the CPU stage timers. Output logs/gab_merge.txt, logs/par_merge.txt,
# logs/lanes_ab.txt, chainM.done.
cd /root/spark26; export XDG_RUNTIME_DIR=/tmp CMF_MAX_SEQ=140000
rm -f logs/chainM.done
sed -i 's/^CARGO_PROFILE_RELEASE_LTO=off CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 cargo build/CARGO_PROFILE_RELEASE_LTO=off CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 taskset -c 8-47 cargo build/' build_to.sh
bash build_to.sh zi4 > logs/build_zi4.out 2>&1
[ -x zi4 ] && mv zi4 cortiq.zi4
F=hf/cmf/spark-x2.6-4b-q4tp.cmf
: > logs/gab_merge.txt
for n in 4096 16000; do
  for B in cortiq.zi3 cortiq.zi4; do
    bash quiet.sh bash -c "./$B bench $F --core --ignore-eos --tokens 16 --ctx $n > logs/gab_merge.$B.$n.log 2>&1"
    echo "$n | $B | $(grep -E "Prompt:|Decode:" logs/gab_merge.$B.$n.log | tr '\n' ' ' | tr -s ' ')" >> logs/gab_merge.txt
  done
done
venv/bin/python -c "t=open('data/wiki.test.raw').read(); open('logs/p_merge.txt','w').write(t[:9000]+'\n\nSummarize the text above.')"
for B in cortiq.zi3 cortiq.zi4; do
  bash quiet.sh bash -c "./$B run $F --prompt-file logs/p_merge.txt --no-think --greedy -n 48 > logs/par_merge.$B.txt 2> logs/par_merge.$B.err"
done
if diff <(sed '$d' logs/par_merge.cortiq.zi3.txt) <(sed '$d' logs/par_merge.cortiq.zi4.txt) > /dev/null; then
  echo "merge: SAME_TEXT" > logs/par_merge.txt
else
  { echo "merge: DIFF_TEXT"; diff <(sed '$d' logs/par_merge.cortiq.zi3.txt) <(sed '$d' logs/par_merge.cortiq.zi4.txt) | head -6; } > logs/par_merge.txt
fi
: > logs/lanes_ab.txt
for L in 4 2 8; do
  CMF_GPU=0 CMF_THREADS=8 CMF_Q4TP_LANES=$L CMF_CPU_PROF=1 taskset -c 0-7 ./cortiq.cpu3 bench $F --core --ignore-eos --tokens 24 --ctx 1000 > logs/lanes_ab.$L.log 2>&1
  echo "lanes $L | $(grep -E "Decode:|ffn_gate_up|ffn_down|qkv |attn_core|attn_o" logs/lanes_ab.$L.log | tr '\n' ' ' | tr -s ' ')" >> logs/lanes_ab.txt
done
touch logs/chainM.done
