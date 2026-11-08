#!/bin/bash
# chain4.sh: after chain3: (1) how the optimized build scales on this VM's 8 vCPUs at 1000 tokens: 8
# threads, 4 threads on vCPUs 0-3, 4 threads on 0,2,4,6, and 8 threads with the stage timers;
# (2) llama.cpp Q4_K_M's peak resident memory on a 16 000-token prompt (its first summary was
# overwritten): llama-bench -p 16000 -n 0, VmHWM sampled. Output logs/threads.txt,
# logs/cpu_llama_q4km_rss.txt, chain4.done.
cd /root/spark26cpu; export CMF_GPU=0 CMF_MAX_SEQ=140000
until [ -f logs/chain3.done ]; do sleep 20; done
F=spark-x2.6-4b-q4tp.cmf; : > logs/threads.txt
for v in "t8|8|0-7|" "t4_0123|4|0-3|" "t4_0246|4|0,2,4,6|" "t8_prof|8|0-7|CMF_CPU_PROF=1"; do
  IFS='|' read -r name th cpus envs <<< "$v"
  env $envs CMF_THREADS=$th taskset -c $cpus ./cortiq2 bench $F --core --ignore-eos --tokens 16 --ctx 1000 > logs/threads.$name.log 2>&1
  echo "$name | $(grep -E "Prompt:|Decode:" logs/threads.$name.log | tr '\n' ' ' | tr -s ' ')" >> logs/threads.txt
done
sample() { local h=0 w; while kill -0 $1 2>/dev/null; do w=$(awk '/VmHWM/{print $2}' /proc/$1/status 2>/dev/null); [ -n "$w" ] && [ "$w" -gt "$h" ] && h=$w; sleep 0.5; done; echo $h; }
llama.cpp/build/bin/llama-bench -m Spark-X2.6-4B-Q4_K_M.gguf -t 8 -p 16000 -n 0 -r 1 -o md > logs/cpu_llama_q4km_rss.log 2>&1 & pid=$!
h=$(sample $pid)
{ echo "pp16000 rss_kb=$h"; grep pp16000 logs/cpu_llama_q4km_rss.log; } > logs/cpu_llama_q4km_rss.txt
touch logs/chain4.done
