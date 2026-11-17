#!/usr/bin/env bash
# Measure $D/NAME.cmf: set R without examples (stage 0 runner; rule answers may
# come after a <think> block, so 96 new tokens) and the perplexity of general text.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
D="${D:-$HOME/embryo-data/hermes-core/skills56}"
S0="${STAGE0:-$HOME/embryo-data/hermes-core/stage0}"
C="$REPO/target/release/cortiq"
NAME="$1"
"$S0/venv/bin/python" -I "$HERE/../stage0/run_cortiq.py" --cortiq "$C" --model "$D/$NAME.cmf" --tasks "$S0/run/r.jsonl" \
  --out "$D/$NAME.outputs.jsonl" --max-tokens 96 --strip-think
CMF_GPU=0 CMF_GROWTH=off "$C" ppl --file "$HOME/embryo-data/heldout-en.txt" --windows 16 --window-len 512 "$D/$NAME.cmf" 2>&1 | tail -1 | tee "$D/$NAME.ppl"
