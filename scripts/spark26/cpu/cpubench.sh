#!/bin/bash
# cpubench.sh: Spark-X2.6-4B on the 8-core CPU server. Waits for the cortiq build and for each model
# file to finish uploading (sizes from the Mac), then runs: cortiq q4tp standard at 1000/4000/16000
# (`bench --core --ignore-eos --tokens 60`, peak RSS sampled), llama.cpp Q4_K_M (llama-bench, 8
# threads: prompt 1000/4000/16000, decode at depth 0/1000/4000/16000), the turbo file built here with
# far-turbo at 4000/16000/32000, then the 8-bit pair at 1000/4000. Output logs/cpu_*.txt, cpubench.done.
cd /root/spark26cpu; mkdir -p logs; export CMF_GPU=0 CMF_THREADS=8 CMF_MAX_SEQ=140000
sz() { stat -c %s "$1" 2>/dev/null || echo 0; }
waitf() { until [ "$(sz $1)" -ge "$2" ]; do sleep 20; done; sleep 5; }
until [ -x cortiq ]; do sleep 20; done
# A quiet machine: the llama.cpp build (-j 8) must be over before any timing starts.
until grep -q "llama build" logs/setup.log; do sleep 20; done
waitf spark-x2.6-4b-q4tp.cmf 2161419978
B=./cortiq
sample() { # sample PID -> peak VmHWM kB
  local h=0 w; while kill -0 $1 2>/dev/null; do w=$(awk '/VmHWM/{print $2}' /proc/$1/status 2>/dev/null); [ -n "$w" ] && [ "$w" -gt "$h" ] && h=$w; sleep 0.5; done; echo $h; }
cb() { # cb FILE TAG CTX...
  local F=$1 T=$2; shift 2; local O=logs/cpu_$T.txt; : > $O
  for n in "$@"; do
    $B bench $F --core --ignore-eos --tokens 60 --ctx $n > logs/cpu_$T.$n.log 2>&1 & local pid=$!
    local h=$(sample $pid)
    echo "$n | rss_kb=$h | $(grep -E "TTFT|Prompt|Decode" logs/cpu_$T.$n.log | tr "\n" " " | tr -s " ")" | tee -a $O
  done
  echo CB_DONE >> $O
}
LB=llama.cpp/build/bin/llama-bench
lb() { # lb GGUF TAG
  local G=$1 T=$2; local O=logs/cpu_llama_$T.txt; : > $O
  until [ -x $LB ]; do sleep 20; done
  $LB -m $G -t 8 -p 1000,4000,16000 -n 0 -r 1 -o md > logs/cpu_llama_$T.pp.log 2>&1 & local pid=$!
  local h=$(sample $pid)
  echo "pp rss_kb=$h" >> $O; grep -E "pp|model" logs/cpu_llama_$T.pp.log >> $O
  $LB -m $G -t 8 -p 0 -n 60 -d 0,1000,4000,16000 -r 1 -o md > logs/cpu_llama_$T.tg.log 2>&1
  grep -E "tg" logs/cpu_llama_$T.tg.log >> $O
  echo LB_DONE >> $O
}
cb spark-x2.6-4b-q4tp.cmf std_q4 1000 4000 16000
waitf Spark-X2.6-4B-Q4_K_M.gguf 3543441312
lb Spark-X2.6-4B-Q4_K_M.gguf q4km
$B far-turbo spark-x2.6-4b-q4tp.cmf spark-x2.6-4b-q4tp-turbo.cmf --exact-weights keep_spark26.bin > logs/far_turbo.log 2>&1
[ -f spark-x2.6-4b-q4tp-turbo.cmf ] && cb spark-x2.6-4b-q4tp-turbo.cmf turbo_q4 4000 16000 32000
waitf spark-x2.6-4b-q8_2f.cmf 4135951050
cb spark-x2.6-4b-q8_2f.cmf std_q8 1000 4000
waitf Spark-X2.6-4B-Q8_0.gguf 4375021824
lb Spark-X2.6-4B-Q8_0.gguf q80
touch logs/cpubench.done
