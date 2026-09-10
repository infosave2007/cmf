#!/usr/bin/env bash
# Tokenizer parity gate: bit-exact encode/decode vs HF `tokenizers`
# on an adversarial corpus (ru/CJK/emoji/code/whitespace/specials).
# Needs: a real tokenizer.json (default: the qwopus snapshot) and a
# python with `tokenizers` (default: venv_heal).
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-/Users/oleg/Documents/cortiq-bot/venv_heal/bin/python3}"
TOK_JSON="${TOK_JSON:-$HOME/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17/tokenizer.json}"

if [ ! -f "$TOK_JSON" ]; then
    echo "SKIP: no tokenizer.json at $TOK_JSON (set TOK_JSON)"
    exit 0
fi

CASES="$(mktemp -t tok_cases)"
trap 'rm -f "$CASES"' EXIT
"$PY" tests/gen_tokenizer_cases.py "$TOK_JSON" "$CASES"

CMF_TOK_JSON="$TOK_JSON" CMF_TOK_CASES="$CASES" \
    cargo test -p cortiq-engine --test tokenizer_parity -- --nocapture

echo "TOKENIZER PARITY OK"
