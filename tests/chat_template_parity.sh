#!/usr/bin/env bash
# Chat-template gate: minijinja render (runtime) must be byte-identical
# to the reference jinja2 render (transformers semantics) of the model's
# chat_template.jinja for several message sets.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-/Users/oleg/Documents/cortiq-bot/venv_heal/bin/python3}"
SNAP="${SNAP:-$HOME/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17}"

if [ ! -f "$SNAP/chat_template.jinja" ]; then
    echo "SKIP: no chat_template.jinja at $SNAP (set SNAP)"
    exit 0
fi

CASES="$(mktemp -t chat_cases)"
trap 'rm -f "$CASES"' EXIT
"$PY" tests/gen_chat_cases.py "$SNAP" "$CASES"

CMF_CHAT_CASES="$CASES" \
    cargo test -p cortiq-engine --test chat_template_parity -- --nocapture

echo "CHAT TEMPLATE PARITY OK"
