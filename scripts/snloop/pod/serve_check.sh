#!/bin/bash
# serve_check.sh BIN FILE BACKEND TAG: `cortiq serve` as the card describes it — health and models, a chat turn and
# its follow-up (prefix reuse), an image_url part, a long document, a streamed answer, a plain completion.
set -u
B=$1; M=$2; BK=$3; TAG=$4
cd /root/sn
export XDG_RUNTIME_DIR=/tmp CMF_GPU_PROBE=0
port=8193
E="CMF_MAX_SEQ=131072"; [ $BK = cuda ] && E="$E CMF_GPU=cuda"
L=logs/serve_$TAG.log
body() {
  python3 -I - "$@" <<'PY'
import json, sys
a = sys.argv[1:]
msgs = [{"role": "user", "content": a[0]}]
if len(a) == 3:
    msgs = [{"role": "user", "content": a[1]}, {"role": "assistant", "content": a[2]}, {"role": "user", "content": a[0]}]
print(json.dumps({"model": "sensenova", "messages": msgs, "temperature": 0, "max_tokens": 64, "enable_thinking": False}))
PY
}
ask() {
  curl -s --max-time 600 -H "Content-Type: application/json" -d @- http://127.0.0.1:$port/v1/chat/completions |
    python3 -I -c "import json,sys; j=json.load(sys.stdin); print(j['choices'][0]['message']['content'].strip().replace(chr(10),' ')[:300]); print('   usage:', j.get('usage'))"
}
env $E $B serve $M --port $port --host 127.0.0.1 > $L 2>&1 &
pid=$!
for i in $(seq 1 180); do grep -q -E "listening|API server" $L && break; sleep 1; done
echo "healthz: $(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$port/healthz)  models: $(curl -s http://127.0.0.1:$port/v1/models | cut -c1-160)"
q1="A train leaves at 9:15 and the trip takes 2 hours 50 minutes. When does it arrive? Answer in one sentence."
a1=$(body "$q1" | ask | head -1); echo "1: $a1"
echo "2: $(body "And if it leaves 40 minutes later? One sentence." "$q1" "$a1" | ask | head -1)"
python3 -I -c "import base64,json,sys; img=base64.b64encode(open(sys.argv[1],'rb').read()).decode(); print(json.dumps({'model':'sensenova','temperature':0,'max_tokens':64,'enable_thinking':False,'messages':[{'role':'user','content':[{'type':'image_url','image_url':{'url':'data:image/png;base64,'+img}},{'type':'text','text':'Describe this image in two sentences.'}]}]}))" data/img/hero1024.png > /tmp/serve_img.json
echo "3 (image): $(ask < /tmp/serve_img.json | head -1)"
python3 -I -c "import json; d=open('data/wiki.test.raw').read()[:60000]; print(json.dumps({'model':'sensenova','temperature':0,'max_tokens':64,'enable_thinking':False,'messages':[{'role':'user','content':d+chr(10)*2+'Summarize the first article above in one sentence.'}]}))" > /tmp/serve_doc.json
echo "4 (document 60000 chars): $(ask < /tmp/serve_doc.json)"
n=$(python3 -I -c "import json; print(json.dumps({'model':'sensenova','messages':[{'role':'user','content':'Count from one to ten in words.'}],'temperature':0,'max_tokens':48,'enable_thinking':False,'stream':True}))" |
  curl -s -N --max-time 120 -H "Content-Type: application/json" -d @- http://127.0.0.1:$port/v1/chat/completions | tee /tmp/serve_stream.txt | grep -c '^data: ')
echo "5 (stream): $n data lines, last: $(tail -n 2 /tmp/serve_stream.txt | tr '\n' ' ' | cut -c1-80)"
echo "6 (completion): $(curl -s --max-time 120 -H 'Content-Type: application/json' -d '{"model":"sensenova","prompt":"The capital of France is","max_tokens":8,"temperature":0}' http://127.0.0.1:$port/v1/completions | python3 -I -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['text'].strip()[:80])")"
kill $pid; wait $pid 2>/dev/null
sed 's/\x1b\[[0-9;]*m//g' $L | grep -E "eval time|total time|kv-reuse|reuse|far field|sealed|tok/s" | sed 's/^.*openai: //' | cut -c1-160 | tail -20
echo SERVE_DONE $TAG
