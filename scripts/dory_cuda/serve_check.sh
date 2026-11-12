#!/bin/bash
# `cortiq serve` with Dory's turbo file, what docs/DORY.md says about it: the
# health and model endpoints, a chat request and its follow-up, a document past
# the seal (60 000 characters of wikitext: the exact prefix, the seal, the
# chunks past it and the far field built again), a streamed answer, a plain
# completion, and two requests at once on two slots against the same two one
# after the other. CMF_GPU=cuda by default; X=1 for the wgpu path.
#   serve_check.sh [binary] [ENV=VALUE ...]
root=${CUDA_RUN_ROOT:-/root/cuda}
cd "$root" || exit 1
bin=${1:-bin/cortiq-cuda}; shift
envs=("$@"); [ ${#envs[@]} -eq 0 ] && envs=(CMF_GPU=cuda)
export XDG_RUNTIME_DIR=/tmp CMF_GPU_PROBE=0
m=/root/models/dory/dory-turbo.cmf
port=8194
o=out/dory_serve; mkdir -p $o
body() { # question [previous question, previous answer] -> a chat request
  python3 - "$@" <<'PY'
import json, sys
a = sys.argv[1:]
msgs = [{"role": "user", "content": a[0]}]
if len(a) == 3:
    msgs = [{"role": "user", "content": a[1]}, {"role": "assistant", "content": a[2]}, {"role": "user", "content": a[0]}]
print(json.dumps({"model": "dory", "messages": msgs, "temperature": 0, "max_tokens": 96, "enable_thinking": False}))
PY
}
ask() { # json body -> the answer's text
  curl -s --max-time 900 -H "Content-Type: application/json" -d @- http://127.0.0.1:$port/v1/chat/completions |
    python3 -c "import json,sys; j=json.load(sys.stdin); print(j['choices'][0]['message']['content'].strip().replace('\n',' ')[:300]); print('   usage:', j.get('usage'))"
}
start() { # log, env...
  log=$1; shift
  env "$@" "$bin" serve $m --port $port --host 127.0.0.1 > "$log" 2>&1 &
  pid=$!
  for i in $(seq 1 120); do grep -q "Server listening\|API server\|listening" "$log" && break; sleep 1; done
}
stop() { kill $pid; wait $pid 2>/dev/null; }
times() { sed 's/\x1b\[[0-9;]*m//g' "$1" | grep -E "eval time|total time|kv-reuse|slot\(s\)|far field|rebuilt" | sed 's/^.*openai: //' | cut -c1-170; }

echo "== one slot (${envs[*]})"
( peak=0; while :; do m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ -n "$m" ] && [ "$m" -gt "$peak" ] && peak=$m && echo "$peak" > $o/peak; sleep 0.2; done ) &
poll=$!
start $o/serve.log "${envs[@]}" CMF_MAX_SEQ=131072
echo "healthz: $(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$port/healthz)  models: $(curl -s http://127.0.0.1:$port/v1/models | cut -c1-160)"
q1="A train leaves at 9:15 and the trip takes 2 hours 50 minutes. When does it arrive? Answer in one sentence."
a1=$(body "$q1" | ask | head -1); echo "1: $a1"
echo "2: $(body "And if it leaves 40 minutes later? One sentence." "$q1" "$a1" | ask | head -1)"
doc=$(python3 -c "t=open('data/wiki.test.raw',encoding='utf-8').read().replace(chr(13),''); print(t[5000:65000])")
echo "3 (document of 60 000 characters): $(body "$doc

In one paragraph, what is the last article above about?" | ask)"
n=$(python3 -c "import json; print(json.dumps({'model':'dory','messages':[{'role':'user','content':'Count from one to ten in words.'}],'temperature':0,'max_tokens':48,'enable_thinking':False,'stream':True}))" |
  curl -s -N --max-time 120 -H "Content-Type: application/json" -d @- http://127.0.0.1:$port/v1/chat/completions | tee $o/stream.txt | grep -c '^data: ')
echo "4 (stream): $n data lines, last: $(tail -n 2 $o/stream.txt | tr '\n' ' ' | cut -c1-80)"
echo "5 (completion): $(curl -s --max-time 120 -H 'Content-Type: application/json' -d '{"model":"dory","prompt":"The capital of France is","max_tokens":8,"temperature":0}' http://127.0.0.1:$port/v1/completions | python3 -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['text'].strip()[:80])")"
tool='{"model":"dory","temperature":0,"max_tokens":128,"enable_thinking":false,"messages":[{"role":"user","content":"What is the weather in Paris right now?"}],"tools":[{"type":"function","function":{"name":"get_weather","description":"Current weather in a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]}'
echo "6 (tools): $(curl -s --max-time 120 -H 'Content-Type: application/json' -d "$tool" http://127.0.0.1:$port/v1/chat/completions | python3 -c "import json,sys; m=json.load(sys.stdin)['choices'][0]; print(m.get('finish_reason'), json.dumps(m['message'].get('tool_calls'))[:200], (m['message'].get('content') or '')[:120])")"
stop
kill $poll 2>/dev/null; wait $poll 2>/dev/null
echo "the card's peak while the server ran: $(cat $o/peak) MiB"
times $o/serve.log

echo "== two slots, two requests at once"
qa="Explain in two sentences why the sky is blue."
qb="Write a haiku about a river in winter."
start $o/serve2.log "${envs[@]}" CMF_SERVE_SLOTS=2
body "$qa" | ask | head -1 > $o/seq_a.txt
body "$qb" | ask | head -1 > $o/seq_b.txt
( body "$qa" | ask | head -1 > $o/par_a.txt ) &
pa=$!
( body "$qb" | ask | head -1 > $o/par_b.txt ) &
pb=$!
wait $pa $pb
stop
for x in a b; do
  if cmp -s $o/seq_$x.txt $o/par_$x.txt; then echo "request $x: the same text at once and alone"; else echo "request $x: DIFFERENT"; cat $o/seq_$x.txt $o/par_$x.txt; fi
done
cat $o/par_a.txt $o/par_b.txt | cut -c1-200
times $o/serve2.log | tail -8
echo DORY_SERVE_DONE
