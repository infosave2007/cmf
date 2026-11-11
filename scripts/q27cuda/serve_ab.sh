#!/bin/bash
# serve_ab.sh TAG [ENV=VAL ...] — start `cortiq serve` on CUDA, send fixed
# greedy chats (no-think, think, a two-turn chat), keep every answer's text
# in out/serve_TAG/*.txt and the server's eval lines, then stop by PID.
tag=$1; shift
cd /root/cuda
d=out/serve_$tag; rm -rf $d; mkdir -p $d
env "$@" CMF_GPU=cuda CMF_GPU_PROBE=0 setsid bin/cortiq-cuda-fast serve /root/models/qwen38-27b-q4tp.cmf \
  --port 8090 --host 127.0.0.1 > $d/server.log 2>&1 < /dev/null &
pid=$!
for i in $(seq 1 90); do curl -s -m 2 127.0.0.1:8090/health >/dev/null 2>&1 && break; sleep 2; done
ask() { # name body
  curl -s -m 300 127.0.0.1:8090/v1/chat/completions -H "Content-Type: application/json" -d "$2" > $d/$1.json
  python3 -c "import json,sys;print(json.load(open('$d/$1.json'))['choices'][0]['message']['content'])" > $d/$1.txt 2>/dev/null
}
NT='"chat_template_kwargs":{"enable_thinking":false}'
ask warm  '{"model":"q","messages":[{"role":"user","content":"Hi"}],"max_tokens":8,"temperature":0,'"$NT"'}'
ask prime '{"model":"q","messages":[{"role":"user","content":"Write a Python function that checks whether a number is prime."}],"max_tokens":400,"temperature":0,'"$NT"'}'
ask essay '{"model":"q","messages":[{"role":"user","content":"Explain in a few paragraphs why the sky is blue."}],"max_tokens":400,"temperature":0,'"$NT"'}'
ask think '{"model":"q","messages":[{"role":"user","content":"Write a Python function that checks whether a number is prime."}],"max_tokens":400,"temperature":0}'
ask turn2 '{"model":"q","messages":[{"role":"user","content":"Write a Python function that checks whether a number is prime."},{"role":"assistant","content":"def is_prime(n):\n    if n < 2:\n        return False\n    for d in range(2, int(n ** 0.5) + 1):\n        if n % d == 0:\n            return False\n    return True"},{"role":"user","content":"Now rewrite it to skip even numbers and add a docstring."}],"max_tokens":400,"temperature":0,'"$NT"'}'
kill $pid; sleep 2
sed 's/\x1b\[[0-9;]*m//g' $d/server.log | grep -E "eval time|speculation trial|ERROR|WARN" > $d/evals.txt
cat $d/evals.txt
