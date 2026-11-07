#!/bin/bash
# How often a file answers a coding request (no tools offered, --no-think) with a made-up tool call:
# toolstart.sh FILE TAG [SEEDS]  — greedy, plus SEEDS sampled runs (temp 1.0, top-p 0.95) per prompt.
set -u
FILE=$1; TAG=$2; SEEDS=${3:-0}
B=${BIN:-$HOME/dev/cherrypit/bin/cortiq}
OUT=$HOME/dev/cherrypit/out/toolstart/$TAG; mkdir -p "$OUT"
prompts=(
  "Build a React component with a search box and a list that loads on open."
  "Write a React todo list with add, delete and a filter for done items."
  "Create a login form in React with email and password validation."
  "Make an HTML page with a dark mode toggle that remembers the choice."
  "Write a Vue component that shows a paginated table of users fetched from an API."
  "Build a counter component in React with increment, decrement and reset buttons."
  "Write a JavaScript function that debounces another function."
  "Create a simple Express server with a GET /health endpoint and a POST /items endpoint."
  "Write a Python function that merges two sorted lists into one sorted list."
  "Write a Python script that counts word frequencies in a text file and prints the top 10."
  "Implement binary search in Python and explain its complexity."
  "Write a CSS grid layout with a header, a sidebar, main content and a footer."
  "Build a React modal component that closes on Escape and on a click outside."
  "Write a TypeScript function that groups an array of objects by a key."
  "Create a Flask app with a form that saves notes to a SQLite database."
  "Write a function that checks whether a string has balanced brackets."
)
tool=0; total=0
for i in "${!prompts[@]}"; do
  for s in $(seq 0 "$SEEDS"); do
    if [ "$s" = 0 ]; then flags=(--greedy); else flags=(--temperature 1.0 --top-p 0.95 --top-k 0 --min-p 0 --rep-penalty 1.0 --seed "$s"); fi
    "$B" run "$FILE" --no-think "${flags[@]}" -n 24 --prompt "${prompts[$i]}" 2>/dev/null | sed -n '5,$p' > "$OUT/p$i.s$s.txt"
    total=$((total+1))
    if grep -q "<function" "$OUT/p$i.s$s.txt"; then tool=$((tool+1)); echo "TOOL p$i s$s: $(head -c 90 "$OUT/p$i.s$s.txt" | tr '\n' ' ')"; fi
  done
done
echo "$TAG: $tool of $total answers start a tool call"
