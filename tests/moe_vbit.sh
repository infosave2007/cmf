#!/usr/bin/env bash
# C4 gate: per-expert bit allocation (P15 клейм 12) + движок исполняет
# vbit-экспертов (Rust verify + greedy smoke).

set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-/Users/oleg/Documents/cortiq-bot/venv_heal/bin/python3}"
TMP="${TMPDIR:-/tmp}/cmf-moe-vbit-$$"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP"

echo "── C4: совместный бюджет битов через экспертов"
"$PY" tests/moe_vbit_check.py "$TMP"

echo "── Rust: verify + greedy smoke на vbit-MoE"
cargo run -q -p cortiq-cli --release -- verify "$TMP/moe-vbit.cmf" | tail -1
CMF_MAX_SEQ=32 cargo run -q -p cortiq-cli --release -- run "$TMP/moe-vbit.cmf" \
    -p "abc" --greedy >/dev/null
echo "  ✓ greedy прошёл"

echo "MOE VBIT OK"
