#!/bin/bash
# quant_eval.sh: a larger held-out chat text (the q8_2f's answers to prompts_eval2.json), then wikitext-2 over
# 8 windows of 2048 tokens and both chat texts for the three candidate files, on CUDA.
set -u
cd /root/sn
B=bin/cortiq
export XDG_RUNTIME_DIR=/tmp
until grep -q PROF_CHAIN_DONE logs/prof_chain.log; do sleep 10; done
CMF_GPU=cuda $B serve out/sn9-q8_2f.cmf --port 8199 --host 127.0.0.1 > logs/calib_serve2.log 2>&1 &
spid=$!
for i in $(seq 1 180); do grep -q -E "listening|API server" logs/calib_serve2.log && break; sleep 1; done
python3 -I src/scripts/snloop/chat_calib_serve.py 8199 dl/SenseNova-Loop-9B src/scripts/snloop/prompts_eval2.json data/chat_eval2.txt 400 > logs/chat_eval2.log 2>&1
kill $spid; wait $spid 2>/dev/null
tail -1 logs/chat_eval2.log
for f in sn9-q4tp-gptq2 sn9-q4tp-rtn sn9-q8_2f; do
  echo "$f [chat_eval2 4096] $(CMF_GPU=cuda $B ppl out/$f.cmf --file data/chat_eval2.txt --tokens 4096 2>&1 | grep -i ppl | tail -1)"
done
echo QUANT_EVAL_DONE
