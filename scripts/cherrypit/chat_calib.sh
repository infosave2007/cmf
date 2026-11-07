#!/bin/bash
# Chat-format calibration text: q8_2f greedy answers (--no-think) to coding requests that are NOT in toolstart.sh,
# rendered through the model's own template by transformers. chat_calib.sh OUT_TXT
set -eu
OUT=$1
B=$HOME/dev/cherrypit/bin/cortiq
M=$HOME/dev/cherrypit/out/CherryPit-1.2-2B-q8_2f.cmf
D=$HOME/dev/cherrypit/out/chatcalib; mkdir -p "$D"
prompts=(
  "Write a React hook that fetches JSON from a URL and returns data, loading and error."
  "Create an HTML page with a contact form and client-side validation in plain JavaScript."
  "Build a React image gallery with a lightbox."
  "Write a Python class for a bank account with deposit, withdraw and a balance check."
  "Write a JavaScript function that deep-clones an object."
  "Create a Node.js script that reads a CSV file and prints the average of a column."
  "Write a Python function that returns the n-th Fibonacci number iteratively."
  "Build a Svelte component with a star rating input."
  "Write a SQL query that finds duplicate emails in a users table."
  "Create a React component that shows a live clock."
  "Write a Python decorator that times a function call."
  "Make a responsive navbar with a hamburger menu in HTML and CSS."
  "Write a Go function that reverses a slice of ints in place."
  "Create a REST API in FastAPI with CRUD endpoints for books."
  "Write a Bash script that backs up a directory into a dated tar.gz file."
  "Build a React form with a dynamic list of inputs that can be added and removed."
  "Explain what a closure is in JavaScript with a short example."
  "Write a Python generator that yields prime numbers."
  "Create a CSS animation of a bouncing ball."
  "Write a Rust function that counts vowels in a string."
)
: > "$D/pairs.jsonl"
for i in "${!prompts[@]}"; do
  ans=$("$B" run "$M" --no-think --greedy -n 400 --prompt "${prompts[$i]}" 2>/dev/null | sed -n '5,$p' | sed '$d')
  python3 -I -c 'import json,sys; print(json.dumps({"q": sys.argv[1], "a": sys.argv[2].strip()}, ensure_ascii=False))' \
    "${prompts[$i]}" "$ans" >> "$D/pairs.jsonl"
  echo "chat calib $i done"
done
"$HOME/dev/vision-test/venv/bin/python" -I -c '
import json, sys
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(sys.argv[1])
out = []
for line in open(sys.argv[2]):
    p = json.loads(line)
    msgs = [{"role": "user", "content": p["q"]}, {"role": "assistant", "content": p["a"]}]
    out.append(tok.apply_chat_template(msgs, tokenize=False))
open(sys.argv[3], "w").write("\n".join(out))
' "$HOME/dev/cherrypit/hf/CherryPit-1.2-2B" "$D/pairs.jsonl" "$OUT"
wc -c "$OUT"
