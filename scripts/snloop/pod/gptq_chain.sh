#!/bin/bash
# gptq_chain.sh: chat calibration answers (q8_2f on CUDA) -> GPTQ q4tp (default profile) -> ppl of every file.
set -u
cd /root/sn
B=bin/cortiq-fast
export XDG_RUNTIME_DIR=/tmp
echo "== chat calib $(date)"
CMF_GPU=cuda $B serve out/sn9-q8_2f.cmf --port 8199 --host 127.0.0.1 > logs/calib_serve.log 2>&1 &
spid=$!
for i in $(seq 1 180); do grep -q -E "listening|API server" logs/calib_serve.log && break; sleep 1; done
python3 -I src/scripts/snloop/chat_calib_serve.py 8199 dl/SenseNova-Loop-9B src/scripts/snloop/prompts_calib.json data/chat_calib.txt 400 > logs/chat_calib.log 2>&1
python3 -I src/scripts/snloop/chat_calib_serve.py 8199 dl/SenseNova-Loop-9B src/scripts/snloop/prompts_eval.json data/chat_eval.txt 400 > logs/chat_eval.log 2>&1
kill $spid; wait $spid 2>/dev/null
tail -1 logs/chat_calib.log; tail -1 logs/chat_eval.log; wc -c data/chat_calib.txt data/chat_eval.txt
echo "== gptq $(date)"
$B quantize-gptq out/sn9-f16.cmf --codec q4tp --calib-model out/sn9-q8_2f.cmf \
  --calib data/wiki.train.raw=0.5 --calib data/chat_calib.txt=0.5 --tokens 16384 --window 2048 --lambda 0.01 \
  --calib-chunk-layers 8 --output out/sn9-q4tp-gptq.cmf > logs/gptq.log 2>&1
tail -3 logs/gptq.log; ls -la out/sn9-q4tp-gptq.cmf
echo "== ppl $(date)"
for f in sn9-q8_2f sn9-q4tp-rtn sn9-q4tp-gptq sn9-f16; do
  for w in "--tokens 2048" "--windows 8 --window-len 2048"; do
    r=$(CMF_GPU=cuda $B ppl out/$f.cmf --file data/wiki.test.raw $w 2>&1 | grep -i -E "ppl" | tail -1)
    echo "$f [$w] $r"
  done
  r=$(CMF_GPU=cuda $B ppl out/$f.cmf --file data/chat_eval.txt --tokens 4096 2>&1 | grep -i -E "ppl" | tail -1)
  echo "$f [chat_eval 4096] $r"
done
echo GPTQ_CHAIN_DONE
