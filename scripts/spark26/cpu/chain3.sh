#!/bin/bash
# chain3.sh: the optimized CPU build (cortiq2: AVX2 batched chunk attend, two-row FMA q4tp batch
# kernel, pooled activation split, split-K decode attend) against the first build (cortiq) on the
# 8-core Zen 1 test server. Waits for llama.cpp's Q4_K_M rows (logs/cpu_llama_q4km.{pp,tg}.log), stops chain2 there, then
# runs on a quiet machine, one job at a time:
#  1. A/B at 2000 tokens: old, new, and new with each change switched off (CMF_Q4TP_FMA=0,
#     CMF_CPU_ATTEND=0, CMF_ATTEND_SPLITK=0) -> logs/ab2.txt
#  2. perplexity old vs new: 8 chunks of 512 raw wiki.test tokens, second half scored -> logs/ppl_ab.txt
#  3. greedy text old vs new after 6000 characters -> logs/text_ab.txt (SAME_TEXT / DIFF_TEXT)
#  4. the card rows with cortiq2: q4tp 1000/4000/16000, the turbo file (far-turbo by cortiq2)
#     1000/4000/16000, q8_2f 1000/4000 -> logs/cpu2_*.txt
#  5. llama.cpp Q8_0: prompt 1000/4000, decode at depth 0/1000/4000 -> logs/cpu_llama_q80.txt
# Marker logs/chain3.done.
cd /root/spark26cpu; export CMF_GPU=0 CMF_THREADS=8 CMF_MAX_SEQ=140000
# chain2's summary line went to logs/cpu_llama_.txt (a `local` that read $T before setting it); its Q4_K_M
# rows are complete once the Q8_0 run starts, which chain3 then stops and redoes in a shorter form.
until [ -e logs/cpu_llama_q80.pp.log ]; do sleep 5; done
for p in $(pgrep -f "bash chain2.sh"); do kill $p 2>/dev/null; done
sleep 2; pkill -x llama-bench; sleep 3
B0=./cortiq; B1=./cortiq2; F=spark-x2.6-4b-q4tp.cmf
: > logs/ab2.txt
for v in "old|$B0|" "new|$B1|" "no_fma|$B1|CMF_Q4TP_FMA=0" "no_chunk_attend|$B1|CMF_CPU_ATTEND=0" "no_splitk|$B1|CMF_ATTEND_SPLITK=0"; do
  IFS='|' read -r name bin envs <<< "$v"
  env $envs $bin bench $F --core --ignore-eos --tokens 32 --ctx 2000 > logs/ab2.$name.log 2>&1
  echo "$name | $(grep -E "Prompt:|Decode:" logs/ab2.$name.log | tr '\n' ' ' | tr -s ' ')" >> logs/ab2.txt
done
: > logs/ppl_ab.txt
for bin in $B0 $B1; do
  echo "== $bin" >> logs/ppl_ab.txt
  $bin ppl $F data/wiki.test.raw --windows 8 --window-len 512 --window-stride 512 --window-score-from 256 2>&1 | grep -E "windows|PPL" >> logs/ppl_ab.txt
done
venv/bin/python -c "t=open('data/wiki.test.raw').read(); open('logs/p_text.txt','w').write(t[:6000]+'\n\nSummarize the text above.')"
for n in 0 1; do
  bin=$B0; [ $n = 1 ] && bin=$B1
  $bin run $F --prompt-file logs/p_text.txt --no-think --greedy -n 48 > logs/text_ab.$n.txt 2> logs/text_ab.$n.err
done
if diff <(sed '$d' logs/text_ab.0.txt) <(sed '$d' logs/text_ab.1.txt) > /dev/null; then
  echo "text: SAME_TEXT" > logs/text_ab.txt
else
  { echo "text: DIFF_TEXT"; diff <(sed '$d' logs/text_ab.0.txt) <(sed '$d' logs/text_ab.1.txt) | head -8; } > logs/text_ab.txt
fi
sample() { # sample PID -> peak VmHWM kB
  local h=0 w; while kill -0 $1 2>/dev/null; do w=$(awk '/VmHWM/{print $2}' /proc/$1/status 2>/dev/null); [ -n "$w" ] && [ "$w" -gt "$h" ] && h=$w; sleep 0.5; done; echo $h; }
cb() { # cb FILE TAG CTX...
  local F=$1 T=$2; shift 2; local O=logs/cpu2_$T.txt; : > $O
  for n in "$@"; do
    $B1 bench $F --core --ignore-eos --tokens 60 --ctx $n > logs/cpu2_$T.$n.log 2>&1 & local pid=$!
    local h=$(sample $pid)
    echo "$n | rss_kb=$h | $(grep -E "TTFT|Prompt|Decode" logs/cpu2_$T.$n.log | tr "\n" " " | tr -s " ")" | tee -a $O
  done
  echo CB_DONE >> $O
}
cb $F std_q4 1000 4000 16000
$B1 far-turbo $F spark-x2.6-4b-q4tp-turbo.cmf --exact-weights keep_spark26.bin > logs/far_turbo.log 2>&1
[ -f spark-x2.6-4b-q4tp-turbo.cmf ] && cb spark-x2.6-4b-q4tp-turbo.cmf turbo_q4 1000 4000 16000
cb spark-x2.6-4b-q8_2f.cmf std_q8 1000 4000
LB=llama.cpp/build/bin/llama-bench; O=logs/cpu_llama_q80.txt; : > $O
$LB -m Spark-X2.6-4B-Q8_0.gguf -t 8 -p 1000,4000 -n 0 -r 1 -o md > logs/cpu_llama_q80.pp.log 2>&1 & pid=$!
h=$(sample $pid); echo "pp rss_kb=$h" >> $O; grep -E "pp|model" logs/cpu_llama_q80.pp.log >> $O
$LB -m Spark-X2.6-4B-Q8_0.gguf -t 8 -p 0 -n 60 -d 0,1000,4000 -r 1 -o md > logs/cpu_llama_q80.tg.log 2>&1
grep -E "tg" logs/cpu_llama_q80.tg.log >> $O; echo LB_DONE >> $O
touch logs/chain3.done
