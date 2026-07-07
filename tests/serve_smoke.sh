#!/usr/bin/env bash
# End-to-end smoke: tiny .cmf → server → real generation over HTTP.
# Checks: /v1/models, non-stream chat (real usage counts), SSE stream
# (content deltas + finish + [DONE]), task switch, loud 404 on unknown
# task, loud failure on a garbage file.

set -euo pipefail
cd "$(dirname "$0")/.."

PORT="${PORT:-18234}"
TMP="${TMPDIR:-/tmp}/cmf-smoke-$$"
mkdir -p "$TMP"
SERVER_PID=""
trap '[[ -n "$SERVER_PID" ]] && kill "$SERVER_PID" 2>/dev/null || true; rm -rf "$TMP"' EXIT

echo "── build tiny model"
python3 tests/make_tiny_model.py --out "$TMP/tiny" >/dev/null
python3 converter/convert_dtgma_to_cmf.py \
    --model "$TMP/tiny" --masks "$TMP/tiny/masks" \
    --quant Q8_ROW --output "$TMP/tiny.cmf" >/dev/null

echo "── garbage file must be rejected loudly"
echo "not a cmf, definitely not a 27B model" > "$TMP/garbage.cmf"
if cargo run -q -p cortiq-cli -- info "$TMP/garbage.cmf" >/dev/null 2>&1; then
    echo "  ✗ garbage file was accepted"; exit 1
fi
echo "  ✓ garbage rejected"

echo "── start server"
cargo build -q -p cortiq-cli
./target/debug/cortiq serve "$TMP/tiny.cmf" --port "$PORT" >"$TMP/server.log" 2>&1 &
SERVER_PID=$!
for _ in $(seq 1 50); do
    curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && break
    sleep 0.2
done

echo "── /v1/models"
curl -sf "http://localhost:$PORT/v1/models" | grep -q '"tiny-qwen-cortiq"'
echo "  ✓"

echo "── non-stream chat (real generation, real usage)"
RESP=$(curl -sf "http://localhost:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"tiny","messages":[{"role":"user","content":"hi"}],"max_tokens":8,"seed":7,"cortiq":{"task":"general"}}')
echo "$RESP" | python3 -c '
import json, sys
r = json.load(sys.stdin)
assert r["object"] == "chat.completion", r
assert r["usage"]["prompt_tokens"] > 0, "prompt_tokens must be real"
assert r["usage"]["completion_tokens"] > 0, "completion_tokens must be real"
assert "[Cortiq]" not in r["choices"][0]["message"]["content"], "placeholder leaked!"
assert r["cortiq"]["task_used"] == "general"
print("  ✓ usage:", r["usage"], "| finish:", r["choices"][0]["finish_reason"])
'

echo "── SSE stream: deltas + finish + [DONE]"
STREAM=$(curl -sf -N "http://localhost:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"tiny","messages":[{"role":"user","content":"hi"}],"max_tokens":6,"stream":true}')
echo "$STREAM" | grep -q '"role":"assistant"' || { echo "  ✗ no role chunk"; exit 1; }
echo "$STREAM" | grep -q '"content":' || { echo "  ✗ no content deltas"; exit 1; }
echo "$STREAM" | grep -q '"finish_reason":"' || { echo "  ✗ no finish chunk"; exit 1; }
echo "$STREAM" | tail -5 | grep -q '\[DONE\]' || { echo "  ✗ no [DONE] terminator"; exit 1; }
echo "  ✓ role → deltas → finish → [DONE]"

echo "── task switch + unknown task is 404"
curl -sf -X POST "http://localhost:$PORT/v1/cortiq/switch" \
    -H 'Content-Type: application/json' -d '{"task":"coding"}' | grep -q '"new_task":"coding"'
CODE=$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"t","messages":[{"role":"user","content":"x"}],"cortiq":{"task":"nope"}}')
[[ "$CODE" == "404" ]] || { echo "  ✗ unknown task returned $CODE, want 404"; exit 1; }
echo "  ✓ switch works, unknown task = 404"

echo "── masks endpoint carries the quality contract"
curl -sf "http://localhost:$PORT/v1/cortiq/masks" | python3 -c '
import json, sys
masks = {m["name"]: m for m in json.load(sys.stdin)["masks"]}
assert masks["general"]["quality_score"] is not None
assert masks["coding"]["quality_score"] is None, "unmeasured must be null"
print("  ✓ general measured, coding null (honest)")
'

echo "SMOKE OK"
