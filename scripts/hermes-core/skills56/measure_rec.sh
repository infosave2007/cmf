#!/usr/bin/env bash
# Measure a file that carries skill records, $D/NAME.cmf, on set R:
#   measure_rec.sh NAME ROUTE      ROUTE: auto (the file's router decides per request, records still in
#                                  quarantine included) | backbone | a skill id
# Writes $D/NAME.ROUTE.outputs.jsonl; with a skill id also the perplexity of general text under
# that skill ($D/NAME.ROUTE.ppl), the price of serving everything through it.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
D="${D:-$HOME/embryo-data/hermes-core/skills56}"
S0="${STAGE0:-$HOME/embryo-data/hermes-core/stage0}"
C="$REPO/target/release/cortiq"
NAME="$1"; ROUTE="$2"
"$S0/venv/bin/python" -I "$HERE/../stage0/run_cortiq.py" --cortiq "$C" --model "$D/$NAME.cmf" --tasks "$S0/run/r.jsonl" \
  --out "$D/$NAME.$ROUTE.outputs.jsonl" --max-tokens 96 --strip-think --route "$ROUTE" $([ "$ROUTE" = auto ] && echo --include-quarantine)
if [ "$ROUTE" != auto ] && [ "$ROUTE" != backbone ]; then
  CMF_GPU=0 CMF_GROWTH=off "$C" ppl --file "$HOME/embryo-data/heldout-en.txt" --windows 16 --window-len 512 --skill "$ROUTE" "$D/$NAME.cmf" 2>&1 | tail -1 | tee "$D/$NAME.$ROUTE.ppl"
fi
