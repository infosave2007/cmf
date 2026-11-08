#!/bin/bash
# chain2.sh: once the standard q4tp rows are in (CB_DONE in cpu_std_q4.txt): stop the first chain
# (its next steps, llama-bench and the turbo conversion, move here and to chain3), run the CPU stage
# profile of a 1000-token prompt (CMF_CPU_PROF=1, 8 and 4 threads), the memory-bandwidth probe
# (membw.c, 1 and 8 threads), then llama.cpp Q4_K_M and Q8_0 (llama-bench, 8 threads: prompt
# 1000/4000/16000, decode at depth 0/1000/4000/16000). The cortiq turbo and 8-bit rows wait for the
# rebuilt binary (chain3). Output logs/cpu_prof_1000_t8.log, logs/cpu_prof_1000_t4.log,
# logs/membw.txt, logs/cpu_llama_q4km.txt, logs/cpu_llama_q80.txt, chain2.done.
cd /root/spark26cpu; mkdir -p logs; export CMF_GPU=0 CMF_MAX_SEQ=140000
until grep -q CB_DONE logs/cpu_std_q4.txt 2>/dev/null; do sleep 5; done
# Stop the first chain by PID (the script names; this shell's own argv holds neither).
for s in cpubench.sh cpuprof.sh; do for p in $(pgrep -f "bash $s"); do kill $p 2>/dev/null; done; done
sleep 2; pkill -x llama-bench; sleep 2
CMF_CPU_PROF=1 CMF_THREADS=8 ./cortiq bench spark-x2.6-4b-q4tp.cmf --core --ignore-eos --tokens 16 --ctx 1000 > logs/cpu_prof_1000_t8.log 2>&1
CMF_CPU_PROF=1 CMF_THREADS=4 ./cortiq bench spark-x2.6-4b-q4tp.cmf --core --ignore-eos --tokens 16 --ctx 1000 > logs/cpu_prof_1000_t4.log 2>&1
gcc -O2 -fopenmp -o membw membw.c > logs/membw.txt 2>&1
{ OMP_NUM_THREADS=1 ./membw; OMP_NUM_THREADS=8 ./membw; } >> logs/membw.txt 2>&1
sample() { # sample PID -> peak VmHWM kB
  local h=0 w; while kill -0 $1 2>/dev/null; do w=$(awk '/VmHWM/{print $2}' /proc/$1/status 2>/dev/null); [ -n "$w" ] && [ "$w" -gt "$h" ] && h=$w; sleep 0.5; done; echo $h; }
LB=llama.cpp/build/bin/llama-bench
lb() { # lb GGUF TAG
  local G=$1 T=$2; local O=logs/cpu_llama_$T.txt; : > $O
  $LB -m $G -t 8 -p 1000,4000,16000 -n 0 -r 1 -o md > logs/cpu_llama_$T.pp.log 2>&1 & local pid=$!
  local h=$(sample $pid)
  echo "pp rss_kb=$h" >> $O; grep -E "pp|model" logs/cpu_llama_$T.pp.log >> $O
  $LB -m $G -t 8 -p 0 -n 60 -d 0,1000,4000,16000 -r 1 -o md > logs/cpu_llama_$T.tg.log 2>&1
  grep -E "tg" logs/cpu_llama_$T.tg.log >> $O
  echo LB_DONE >> $O
}
lb Spark-X2.6-4B-Q4_K_M.gguf q4km
lb Spark-X2.6-4B-Q8_0.gguf q80
touch logs/chain2.done
