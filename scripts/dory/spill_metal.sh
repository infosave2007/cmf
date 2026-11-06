#!/bin/bash
# Dory's turbo file over a long prompt on Metal: a run with its peak footprint.
# spill_metal.sh NAME BIN MODEL PROMPT_FILE [ENV=VAL...]; passkey_prompt.py writes a prompt with a planted fact.
name=$1; bin=$2; m=$3; p=$4; shift 4
o=$(mktemp)
env "$@" RUST_LOG=cortiq_engine=info /usr/bin/time -l "$bin" run "$m" --prompt-file "$p" --greedy --no-think --max-tokens 24 > "$o" 2>&1
echo "$name: footprint $(grep 'peak memory footprint' "$o" | awk '{printf "%.2f", $1/1073741824}') GiB"
grep -a -E "sealed|rebuilt|first token|panicked" "$o" | sed -e 's/\x1b\[[0-9;]*m//g' -e 's/.*dory: /  /' | cut -c1-120
grep -a -B1 "first token" "$o" | head -1 | cut -c1-100
rm -f "$o"
