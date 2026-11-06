#!/bin/bash
# Rounds of the DFlash block with the MTP chain drafting beside it, a file of rows per prompt, for
# `cortiq dflash-route-fit`: q27_dflash_rows.sh CORTIQ MODEL PROMPT_DIR OUT_DIR TOKENS
# MODEL has a `.dflash.cmf` beside it. Every round is the block's, seven deep; OUT_DIR/NAME.rows per PROMPT_DIR/NAME.txt.
# Prompt lookup stays on: a seven-deep round leaves it no room, and a record still says what it would have
# added to a round of the chain's own.
cortiq=$1; model=$2; d=$3; out=$4; n=$5
mkdir -p "$out"
for f in "$d"/*.txt; do
  name=$(basename "$f" .txt)
  [ -s "$out/$name.rows" ] && continue
  rm -f "$out/$name.rows"
  env CMF_GRAPH_SPEC=1 CMF_GRAPH_SPEC_PIN=1 CMF_GRAPH_SPEC_K=7 CMF_DFLASH=1 CMF_DFLASH_PICK=dflash \
    CMF_DFLASH_ROWS="$out/$name.rows" RUST_LOG=warn "$cortiq" run "$model" --prompt-file "$f" --greedy --no-think --max-tokens "$n" \
    > "$out/$name.out" 2> "$out/$name.err"
  echo "$name: $(grep -E '^\[' "$out/$name.out" | head -2 | cut -c1-90 | tr '\n' ' ') $(stat -f %z "$out/$name.rows" 2>/dev/null || stat -c %s "$out/$name.rows") bytes" >> "$out/rows.log"
done
echo DONE >> "$out/rows.log"
