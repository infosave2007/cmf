#!/bin/bash
# facts.sh MODEL CHARS TRIALS [ENV=VALUE ...]: facts of four kinds planted in CHARS characters of
# Wikipedia and asked at the end (scripts/dory/passkey2.py), through `cortiq run`. The model reasons
# before it answers, so an answer gets 300 tokens.
set -uo pipefail
cd /root/rip
export XDG_RUNTIME_DIR=/tmp CMF_MAX_SEQ=70000 PASSKEY_TOKENS=300 PYTHONUNBUFFERED=1
m=$1; c=$2; t=$3; shift 3
python3 scripts/passkey2.py ./cortiq $m data/wiki.test.raw $c $t "$@" 2>&1
echo FACTS_DONE
