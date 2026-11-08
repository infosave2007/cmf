#!/bin/bash
# chain6.sh: chain5 without its perplexity step (moved to the pod's x86 cores, where the same code runs
# five times faster): for 1000, 4000, 16000 tokens, cortiq q4tp (60 tokens after the prompt, peak RSS),
# then llama.cpp Q4_K_M prompt (llama-bench -p N, peak RSS) and decode at that depth (-d N -n 60), back to
# back. Ends by touching chain3.done for chain4. Output logs/final_rows.txt, chain6.done.
cd /root/spark26cpu; export CMF_GPU=0 CMF_THREADS=8 CMF_MAX_SEQ=140000
[ -x cortiq2.new ] && mv cortiq2.new cortiq2
sample() { local h=0 w; while kill -0 $1 2>/dev/null; do w=$(awk '/VmHWM/{print $2}' /proc/$1/status 2>/dev/null); [ -n "$w" ] && [ "$w" -gt "$h" ] && h=$w; sleep 0.5; done; echo $h; }
LB=llama.cpp/build/bin/llama-bench; G=Spark-X2.6-4B-Q4_K_M.gguf; O=logs/final_rows.txt; : > $O
for n in 1000 4000 16000; do
  ./cortiq2 bench spark-x2.6-4b-q4tp.cmf --core --ignore-eos --tokens 60 --ctx $n > logs/final_cq.$n.log 2>&1 & pid=$!
  h=$(sample $pid)
  echo "cortiq $n | rss_kb=$h | $(grep -E "Prompt:|Decode:" logs/final_cq.$n.log | tr '\n' ' ' | tr -s ' ')" >> $O
  $LB -m $G -t 8 -p $n -n 0 -r 1 -o md > logs/final_ll_pp.$n.log 2>&1 & pid=$!
  h=$(sample $pid)
  echo "llama pp $n | rss_kb=$h | $(grep -E "pp$n" logs/final_ll_pp.$n.log | tr -s ' ')" >> $O
  $LB -m $G -t 8 -p 0 -n 60 -d $n -r 1 -o md > logs/final_ll_tg.$n.log 2>&1 & pid=$!
  h=$(sample $pid)
  echo "llama tg@$n | rss_kb=$h | $(grep -E "tg60" logs/final_ll_tg.$n.log | tr -s ' ')" >> $O
done
echo ROWS_DONE >> $O
touch logs/chain3.done logs/chain6.done
