#!/bin/bash
# Final M4 numbers: prose/code, a 2048-token prompt, two server turns.
cd "$(dirname "$0")"
B=${BIN:-$HOME/dev/cmf-clover-runs/target-fast/release/cortiq}
M=$HOME/dev/cmf-clover-runs/mac-2026-11-01/hf/clover-35b-a3b-preview-q4tp-v2.cmf
MAXT=256 ./mac_bench.sh final
echo "== 2k prompt"; perl -e 'alarm 200; exec @ARGV' $B bench $M --ctx 2048 --tokens 32 --ignore-eos 2>&1 | grep -E "Prompt|Decode" | cut -c1-140
echo "== serve"
$B serve $M --port 8099 > $HOME/dev/cmf-clover-runs/mac-2026-11-01/serve.log 2>&1 &
S=$!
for i in $(seq 1 60); do curl -s -m 2 http://127.0.0.1:8099/v1/models >/dev/null && break; sleep 2; done
ask() { curl -s -m 120 http://127.0.0.1:8099/v1/chat/completions -H 'Content-Type: application/json' -d "$1" -w '\n%{time_total} s\n' | cut -c1-500; }
ask '{"model":"clover","enable_thinking":false,"max_tokens":48,"messages":[{"role":"user","content":"What is the capital of France? Answer in one sentence."}]}'
ask '{"model":"clover","enable_thinking":false,"max_tokens":48,"messages":[{"role":"user","content":"What is the capital of France? Answer in one sentence."},{"role":"assistant","content":"The capital of France is Paris."},{"role":"user","content":"And of Italy?"}]}'
kill $S
