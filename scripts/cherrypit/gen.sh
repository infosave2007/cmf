#!/bin/bash
# Greedy answers of one CMF file on a fixed prompt set: gen.sh FILE OUTDIR [extra cortiq run args]
set -u
FILE=$1; OUT=$2; shift 2
B=${BIN:-$HOME/dev/cherrypit/bin/cortiq}
mkdir -p "$OUT"
run() { # name max_tokens think(yes|no) prompt
  local name=$1 n=$2 think=$3 prompt=$4
  local flags=(--greedy -n "$n")
  [ "$think" = no ] && flags+=(--no-think)
  "$B" run "$FILE" "${flags[@]}" ${EXTRA:-} --prompt "$prompt" > "$OUT/$name.txt" 2> "$OUT/$name.err"
  echo "== $name: $(tail -1 "$OUT/$name.txt")"
}
run palindrome 400 no "Write a Python function that checks whether a string is a palindrome, ignoring case and punctuation. Include two asserts."
run react 900 no "Build a React component with a search box and a list that loads on open."
run tcp 200 no "Explain the difference between TCP and UDP in three sentences."
run who 60 no "Who are you?"
run math 1500 yes "A train leaves at 9:40 and arrives at 13:15. How long is the trip in minutes?"
run sql 300 no "Write an SQL query that returns the five customers with the highest total order amount from tables customers(id, name) and orders(id, customer_id, amount)."
