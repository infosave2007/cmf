#!/bin/bash
# MNN-Pocket-0.3B → CMF on the Mac: mac.sh STEP
#   convert   f16 reference + q8_2f
#   q4tp      GPTQ q4tp from the f16 file (embeddings/head q8_2f)
#   ppl FILE  WikiText-2 test, 12 windows x 512 (ids dumped next to the file)
#   bench FILE TAG   bench --core: Metal x3, Metal 2k prompt, CPU x3
#   chat FILE TAG    greedy answers on a fixed prompt set
set -u
R=$HOME/models/mnn-pocket
B=${BIN:-$R/bin/cortiq}
mkdir -p $R/out $R/logs
case $1 in
convert)
  /usr/bin/time -l $B convert --model $R/src --quant f16 --output $R/out/MNN-Pocket-0.3B-f16.cmf \
    --also-output q8_2f=$R/out/MNN-Pocket-0.3B-q8_2f.cmf > $R/logs/convert.log 2>&1
  tail -5 $R/logs/convert.log; ls -l $R/out ;;
q4tp)
  /usr/bin/time -l nice -n 10 $B quantize-gptq $R/out/MNN-Pocket-0.3B-f16.cmf --codec q4tp \
    --calib-model $R/out/MNN-Pocket-0.3B-q8_2f.cmf ${CALIB:---calib $HOME/dev/cherrypit/data/wiki.train.raw} \
    --tokens ${TOKENS:-4096} --window 2048 --lambda 0.01 \
    --tensor-quant lm_head.weight=q8_2f --tensor-quant model.embed_tokens.weight=q8_2f "${@:3}" \
    --output ${2:-$R/out/MNN-Pocket-0.3B-q4tp.cmf} > $R/logs/gptq_$(basename ${2:-q4tp}).log 2>&1
  tail -5 $R/logs/gptq_$(basename ${2:-q4tp}).log ;;
ppl)
  CMF_PPL_IDS_OUT=$R/logs/ppl_ids.txt $B ppl "$2" --file $HOME/hf/wikitext2_test.txt --windows 12 --window-len 512 \
    ${@:3} 2>&1 | grep -E "PPL|ppl" | tail -2 ;;
bench)
  gpu() { ioreg -r -d 1 -w 0 -c IOAccelerator 2>/dev/null | grep -o '"Device Utilization %"=[0-9]*' | head -1 | cut -d= -f2; }
  one() { local label=$1; shift; local log=$R/logs/bench_$TAG.$label.log
    echo "== $label (gpu busy before: $(gpu)%, load $(uptime | awk -F'load averages: ' '{print $2}'))"
    /usr/bin/time -l env "$@" > $log 2>&1
    grep -E "tok/s|TTFT|maximum resident" $log | grep -v INFO; }
  FILE=$2; TAG=$3
  for i in 1 2 3; do one metal$i $B bench $FILE --core --ignore-eos --tokens 256; done
  one metal_ctx2k $B bench $FILE --core --ignore-eos --ctx 2048 --tokens 128
  for i in 1 2 3; do one cpu$i CMF_GPU=0 $B bench $FILE --core --ignore-eos --tokens 128; done
  one cpu_ctx2k CMF_GPU=0 $B bench $FILE --core --ignore-eos --ctx 2048 --tokens 64 ;;
chat)
  FILE=$2; OUT=$R/logs/chat_$3; mkdir -p $OUT
  run() { local name=$1 n=$2 p=$3
    $B run $FILE --greedy -n $n ${EXTRA:-} --prompt "$p" > $OUT/$name.txt 2> $OUT/$name.err
    echo "== $name"; sed -n '1,40p' $OUT/$name.txt; }
  run who 80 "Who are you?"
  run capital 80 "What is the capital of France? Answer in one sentence."
  run tcp 200 "Explain the difference between TCP and UDP in three sentences."
  run zh 200 "用三句话介绍一下杭州。"
  run json 200 'Extract the name, city and age from this text as JSON: "Anna Petrova, 34, moved from Kazan to Berlin last year."'
  run py 300 "Write a Python function that returns the n-th Fibonacci number iteratively." ;;
esac
