#!/bin/bash
# Two chat requests to one server, the second extending the first (the K/V
# rows of the shared prefix are reused between requests), on both paths.
cd /root/cuda
export XDG_RUNTIME_DIR=/tmp CMF_GPU_PROBE=0
m=models/spark-x2.6-4b-q4tp-turbo.cmf
ask() { # port, json body
  curl -s --max-time 120 -H "Content-Type: application/json" -d "$2" http://127.0.0.1:$1/v1/chat/completions |
    python3 -c "import json,sys; j=json.load(sys.stdin); print(j[\"choices\"][0][\"message\"][\"content\"].strip())"
}
run() { # tag, port, bin, env...
  tag=$1; port=$2; bin=$3; shift 3
  env "$@" RUST_LOG=info CMF_PREFILL_PROF=1 $bin serve $m --port $port > out/serve_$tag.log 2>&1 &
  pid=$!
  for i in $(seq 1 60); do grep -q "Server listening" out/serve_$tag.log && break; sleep 1; done
  q1="A train leaves at 9:15 and the trip takes 2 hours 50 minutes. When does it arrive? Answer in one sentence."
  b1=$(python3 -c "import json,sys; print(json.dumps({\"model\":\"spark\",\"messages\":[{\"role\":\"user\",\"content\":sys.argv[1]}],\"temperature\":0,\"max_tokens\":48,\"enable_thinking\":False}))" "$q1")
  a1=$(ask $port "$b1")
  b2=$(python3 -c "import json,sys; print(json.dumps({\"model\":\"spark\",\"messages\":[{\"role\":\"user\",\"content\":sys.argv[1]},{\"role\":\"assistant\",\"content\":sys.argv[2]},{\"role\":\"user\",\"content\":\"And if it leaves 40 minutes later? One sentence.\"}],\"temperature\":0,\"max_tokens\":48,\"enable_thinking\":False}))" "$q1" "$a1")
  a2=$(ask $port "$b2")
  printf "%s\n---\n%s\n" "$a1" "$a2" > out/serve_$tag.txt
  kill $pid; wait $pid 2>/dev/null
}
run vk 8191 ./bin/cortiq-vk CMF_O1=off
run cu 8192 ${CUDA_BIN:-./bin/cortiq-cuda} CMF_GPU=cuda
if cmp -s out/serve_vk.txt out/serve_cu.txt; then echo "serve: SAME"; else echo "serve: DIFFERENT"; diff out/serve_vk.txt out/serve_cu.txt; fi
cat out/serve_cu.txt
grep -i -E "kv-reuse|reus" out/serve_cu.log | head -5
