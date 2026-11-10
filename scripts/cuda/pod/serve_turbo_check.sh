#!/bin/bash
# `cortiq serve` on the CUDA path, what docs/CUDA.md says about it: the
# health and model endpoints, a chat request and its follow-up (the K/V rows
# of the shared prefix reused), a document past the seal (the far field), a
# second question to the same document, a streamed answer, a plain
# completion, and two requests at once on two slots against the same two
# one after the other.
#   serve_turbo_check.sh [binary]
root=${CUDA_RUN_ROOT:-/root/cuda}
cd "$root" || exit 1
bin=${1:-bin/cortiq-cuda}
export XDG_RUNTIME_DIR=/tmp CMF_GPU_PROBE=0
m=models/spark-x2.6-4b-q4tp-turbo.cmf
port=8193
body() { # question [previous question, previous answer] -> a chat request
  python3 - "$@" <<'PY'
import json, sys
a = sys.argv[1:]
msgs = [{"role": "user", "content": a[0]}]
if len(a) == 3:
    msgs = [{"role": "user", "content": a[1]}, {"role": "assistant", "content": a[2]}, {"role": "user", "content": a[0]}]
print(json.dumps({"model": "spark", "messages": msgs, "temperature": 0, "max_tokens": 64, "enable_thinking": False}))
PY
}
ask() { # json body -> the answer's text
  curl -s --max-time 300 -H "Content-Type: application/json" -d @- http://127.0.0.1:$port/v1/chat/completions |
    python3 -c "import json,sys; j=json.load(sys.stdin); print(j['choices'][0]['message']['content'].strip().replace('\n',' ')[:300]); print('   usage:', j.get('usage'))"
}
start() { # log, env...
  log=$1; shift
  env "$@" "$bin" serve $m --port $port > "$log" 2>&1 &
  pid=$!
  for i in $(seq 1 90); do grep -q "Server listening\|API server" "$log" && break; sleep 1; done
}
stop() { kill $pid; wait $pid 2>/dev/null; }
times() { sed 's/\x1b\[[0-9;]*m//g' "$1" | grep -E "eval time|total time|kv-reuse|slot\(s\)|far field|sealed in" | sed 's/^.*openai: //' | cut -c1-150; }

echo "== one slot"
start out/serve_turbo.log CMF_GPU=cuda CMF_MAX_SEQ=131072
echo "healthz: $(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$port/healthz)  models: $(curl -s http://127.0.0.1:$port/v1/models | cut -c1-160)"
q1="A train leaves at 9:15 and the trip takes 2 hours 50 minutes. When does it arrive? Answer in one sentence."
a1=$(body "$q1" | ask | head -1); echo "1: $a1"
echo "2: $(body "And if it leaves 40 minutes later? One sentence." "$q1" "$a1" | ask | head -1)"
doc=$(cat prompts_long/p6_long_16k.txt)
echo "3 (document): $(body "$doc" | ask)"
echo "4 (document, another question): $(body "$doc

One more question: name one environment variable the documents mention. Answer in one sentence." | ask)"
n=$(python3 -c "import json,sys; print(json.dumps({'model':'spark','messages':[{'role':'user','content':'Count from one to ten in words.'}],'temperature':0,'max_tokens':48,'enable_thinking':False,'stream':True}))" |
  curl -s -N --max-time 120 -H "Content-Type: application/json" -d @- http://127.0.0.1:$port/v1/chat/completions | tee out/serve_turbo_stream.txt | grep -c '^data: ')
echo "5 (stream): $n data lines, last: $(tail -n 2 out/serve_turbo_stream.txt | tr '\n' ' ' | cut -c1-80)"
echo "6 (completion): $(curl -s --max-time 120 -H 'Content-Type: application/json' -d '{"model":"spark","prompt":"The capital of France is","max_tokens":8,"temperature":0}' http://127.0.0.1:$port/v1/completions | python3 -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['text'].strip()[:80])")"
stop
times out/serve_turbo.log

echo "== two slots, two requests at once"
qa="Explain in two sentences why the sky is blue."
qb="Write a haiku about a river in winter."
start out/serve_turbo2.log CMF_GPU=cuda CMF_SERVE_SLOTS=2
body "$qa" | ask | head -1 > out/serve_seq_a.txt
body "$qb" | ask | head -1 > out/serve_seq_b.txt
( body "$qa" | ask | head -1 > out/serve_par_a.txt ) &
pa=$!
( body "$qb" | ask | head -1 > out/serve_par_b.txt ) &
pb=$!
wait $pa $pb
stop
for x in a b; do
  if cmp -s out/serve_seq_$x.txt out/serve_par_$x.txt; then echo "request $x: the same text at once and alone"; else echo "request $x: DIFFERENT"; cat out/serve_seq_$x.txt out/serve_par_$x.txt; fi
done
cat out/serve_par_a.txt out/serve_par_b.txt | cut -c1-200
times out/serve_turbo2.log | tail -8
echo SERVE_TURBO_DONE
