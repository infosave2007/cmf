#!/usr/bin/env bash
# One step of the growth experiment on Embryo-56M: K new experts per layer are
# trained on a skill corpus over a frozen trunk (`cortiq-embryo grow`), the grown
# genome is exported in full (the baked form) and measured.
#
#   grow_skill.sh NAME START_CKPT SKILL K STEPS [more `grow` options]
#
# SKILL names the corpus in $D: SKILL.jsonl.gz and SKILL.held.jsonl.gz.
# Writes to $D: NAME.ckpt (the grown checkpoint, the start of the next step),
# NAME.grow.log (with the one-line JSON summary last), NAME.cmf (f16, chat),
# NAME.outputs.jsonl (set R, no examples, as in stage 0; grade.py grades it),
# NAME.ppl (perplexity of general text, 16 windows of 512 tokens).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
D="${D:-$HOME/embryo-data/hermes-core/skills56}"
O="${O1:-$HOME/embryo-data/o1}"
S0="${STAGE0:-$HOME/embryo-data/hermes-core/stage0}"
E="$REPO/target/release/cortiq-embryo"; C="$REPO/target/release/cortiq"
TOK="$O/sft-t27/gen1.tokenizer.json"
NAME="$1"; START="$2"; SKILL="$3"; K="$4"; STEPS="$5"; shift 5
echo "== $NAME: grow $K experts per layer on $SKILL for $STEPS steps from $START $(date -u +%T)"
"$E" grow --ckpt "$START" --tokenizer "$TOK" --corpus "$D/$SKILL.jsonl.gz" --held "$D/$SKILL.held.jsonl.gz" \
  --general "$O/sft-t27/rawval-tails-2m.u16" --experts "$K" --steps "$STEPS" --batch 4 --seq 512 \
  --trace-tokens 300000 --trace-docs 300 --out-ckpt "$D/$NAME.ckpt" "$@" > "$D/$NAME.grow.log" 2>&1 \
  || { echo "GROW FAILED, see $D/$NAME.grow.log"; tail -3 "$D/$NAME.grow.log" | cut -c1-400; exit 1; }
[ -f "$D/$NAME.ckpt" ] || { echo "the growth was rejected by its gate, see $D/$NAME.grow.log"; exit 1; }
"$E" export --ckpt "$D/$NAME.ckpt" --tokenizer "$TOK" --out "$D/$NAME.cmf" --dtype f16 --chat | tail -1
"$HERE/measure.sh" "$NAME"
echo "== $NAME done $(date -u +%T)"
