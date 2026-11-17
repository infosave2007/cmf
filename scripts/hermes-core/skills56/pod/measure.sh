#!/bin/bash
# Answers of a file with skill records on set R, on this pod's CPU (grading runs in Docker on the Mac):
#   measure.sh NAME ROUTE      ROUTE: auto | backbone | a skill id
# Writes /root/sk/run/NAME.ROUTE.outputs.jsonl; with a skill id also the perplexity of general
# text under that skill (/root/sk/run/NAME.ROUTE.ppl).
D=/root/sk/data; R=/root/sk/run; C=/root/sk/cmf/target/release/cortiq
NAME="$1"; ROUTE="$2"
python3 -I /root/sk/cmf/scripts/hermes-core/stage0/run_cortiq.py --cortiq "$C" --model "$R/$NAME.cmf" --tasks "$D/r.jsonl" \
  --out "$R/$NAME.$ROUTE.outputs.jsonl" --max-tokens 96 --strip-think --route "$ROUTE" $([ "$ROUTE" = auto ] && echo --include-quarantine)
if [ "$ROUTE" != auto ] && [ "$ROUTE" != backbone ] && [ -f "$D/heldout-en.txt" ]; then
  CMF_GPU=0 CMF_GROWTH=off "$C" ppl --file "$D/heldout-en.txt" --windows 16 --window-len 512 --skill "$ROUTE" "$R/$NAME.cmf" 2>&1 | tail -1 | tee "$R/$NAME.$ROUTE.ppl"
fi
