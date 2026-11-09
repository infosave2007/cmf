#!/bin/bash
# chain4.sh: after the final rows (chain5 touches chain3.done): how the optimized build scales on this
# VM's 8 vCPUs at 1000 tokens: 8 threads, 4 threads on vCPUs 0-3, 4 threads on 0,2,4,6, and 8 threads
# with the stage timers. Output logs/threads.txt, chain4.done.
cd /root/spark26cpu; export CMF_GPU=0 CMF_MAX_SEQ=140000
until [ -f logs/chain3.done ]; do sleep 20; done
F=spark-x2.6-4b-q4tp.cmf; : > logs/threads.txt
for v in "t8|8|0-7|" "t4_0123|4|0-3|" "t4_0246|4|0,2,4,6|" "t8_prof|8|0-7|CMF_CPU_PROF=1"; do
  IFS='|' read -r name th cpus envs <<< "$v"
  env $envs CMF_THREADS=$th taskset -c $cpus ./cortiq2 bench $F --core --ignore-eos --tokens 16 --ctx 1000 > logs/threads.$name.log 2>&1
  echo "$name | $(grep -E "Prompt:|Decode:" logs/threads.$name.log | tr '\n' ' ' | tr -s ' ')" >> logs/threads.txt
done
touch logs/chain4.done
