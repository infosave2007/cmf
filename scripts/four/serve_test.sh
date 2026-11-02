#!/bin/bash
# OpenAI-compatible server with image parts: Mage-VL and SenseNova-Loop.
export XDG_RUNTIME_DIR=/tmp
B=${B:-/root/cmf/target-dev/release/cortiq}
cd /root/four/out
IMG=$(base64 -w0 /root/four/img/assets/img01.png)
for m in ${MODELS:-mage-q8_2f sn-q4tp}; do
  $B serve $m.cmf --port 8089 > /tmp/serve_$m.log 2>&1 &
  SP=$!
  for i in $(seq 1 120); do curl -s localhost:8089/v1/models >/dev/null 2>&1 && break; sleep 2; done
  for round in 1 2; do
    printf '{"model":"x","max_tokens":64,"temperature":0,"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,%s"}},{"type":"text","text":"How many people are in this picture? Answer briefly."}]}],"chat_template_kwargs":{"enable_thinking":false}}' "$IMG" > /tmp/req.json
    t0=$(date +%s.%N)
    curl -s localhost:8089/v1/chat/completions -H "Content-Type: application/json" -d @/tmp/req.json | python3 -c "import json,sys; d=json.load(sys.stdin); print('$m', 'round $round:', repr(d['choices'][0]['message']['content'][:300]))" 2>&1
    echo "  $(echo "$(date +%s.%N) - $t0" | bc) s"
  done
  kill $SP; wait $SP 2>/dev/null
done
echo "== serve done"
