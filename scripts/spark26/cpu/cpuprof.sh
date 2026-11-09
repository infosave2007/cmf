#!/bin/bash
# cpuprof.sh: once the q4tp rows and llama.cpp's Q4_K_M rows are in (the far-turbo conversion is the
# only neighbour then): the in-process CPU stage timers (CMF_CPU_PROF=1) of a 1000-token prompt and
# 16 decode tokens on the q4tp file, 8 threads and 4 threads. The split between stages survives a
# busy neighbour; the absolute numbers are the chain's. Output logs/cpu_prof_1000_t8.log,
# logs/cpu_prof_1000_t4.log, cpuprof.done.
cd /root/spark26cpu; export CMF_GPU=0 CMF_MAX_SEQ=140000
until grep -q LB_DONE logs/cpu_llama_q4km.txt 2>/dev/null; do sleep 30; done
CMF_CPU_PROF=1 CMF_THREADS=8 ./cortiq bench spark-x2.6-4b-q4tp.cmf --core --ignore-eos --tokens 16 --ctx 1000 > logs/cpu_prof_1000_t8.log 2>&1
CMF_CPU_PROF=1 CMF_THREADS=4 ./cortiq bench spark-x2.6-4b-q4tp.cmf --core --ignore-eos --tokens 16 --ctx 1000 > logs/cpu_prof_1000_t4.log 2>&1
touch logs/cpuprof.done
