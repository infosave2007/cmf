#!/bin/bash
# mac_suite.sh CORTIQ DIR: cortiq on the Mac, the same prompts as the MLX and llama.cpp runs
# (p2k.txt, p14k.txt in DIR), 128 greedy tokens; first token, prompt tok/s, decode, peak RSS.
# A minute of rest before each run: a fanless MacBook Air slows under back-to-back runs.
B=$1; D=$2; cd "$D"
run() { # tag file prompt env...
  local tag=$1 f=$2 p=$3; shift 3
  python3 -c "import time; time.sleep(60)"
  env "$@" CMF_PREFILL_PROF=1 /usr/bin/time -l "$B" run "$f" --prompt-file "$p" --no-think --greedy -n 128 > suite_out.txt 2> suite_err.txt
  local n=$(grep -o "metal-prefill: [0-9]* of [0-9]*" suite_err.txt | tail -1 | awk '{print $4}')
  [ -z "$n" ] && n=$(grep -o "prefill: [0-9]* tokens" suite_err.txt | tail -1 | awk '{print $2}')
  local line=$(tail -1 suite_out.txt)
  local ft=$(echo "$line" | grep -o "first token [0-9.]*" | awk '{print $3}')
  local dec=$(echo "$line" | grep -o "decode [0-9.]*" | awk '{print $2}')
  local rss=$(grep "maximum resident set size" suite_err.txt | awk '{printf "%.2f", $1/1073741824}')
  echo "$tag | $p | prompt $n tok | first token $ft s | prompt $(python3 -c "print(round(${n:-0}/${ft:-1},1))") tok/s | decode $dec tok/s | rss $rss GB"
}
for p in p2k.txt p14k.txt; do
  run "q4tp" hf/spark-x2.6-4b-q4tp.cmf $p X=1
  run "q4tp+ANE" hf/spark-x2.6-4b-q4tp.cmf $p CMF_ANE=1
  run "q8_2f" hf/spark-x2.6-4b-q8_2f.cmf $p X=1
done
run "q4tp-turbo" hf/spark-x2.6-4b-q4tp-turbo.cmf p14k.txt X=1
run "q4tp-turbo+ANE" hf/spark-x2.6-4b-q4tp-turbo.cmf p14k.txt CMF_ANE=1
echo SUITE_DONE
