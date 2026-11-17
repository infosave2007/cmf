#!/usr/bin/env bash
# One skill baked onto Embryo-56M with `cortiq-embryo skill-bake` (DTG-MA mask over the
# shared FFN of the chosen layers, then the FCD polish under the hard mask; Patent 2),
# appended as a record to a copy of the base file. The genome stays frozen.
#
#   bake_skill.sh NAME BASE SET ID [more `skill-bake` options]
#
# BASE: $D/BASE.cmf, the genome file or a file that already carries records.
# SET: s1 | s2, the shards and prompts made by build_sft.py.
# Writes $D/NAME.cmf (BASE + the record ID) and $D/NAME.bake.log (the summary JSON last).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
D="${D:-$HOME/embryo-data/hermes-core/skills56}"
O="${O1:-$HOME/embryo-data/o1}"
E="$REPO/target/release/cortiq-embryo"
NAME="$1"; BASE="$2"; SET="$3"; ID="$4"; shift 4
echo "== $NAME: bake $ID from $SET onto $BASE $(date -u +%T)"
rm -f "$D/$NAME.cmf"
"$E" skill-bake --ckpt "${CKPT:-$O/chat-sft/keep-final-healed.ckpt}" --base "$D/$BASE.cmf" --out "$D/$NAME.cmf" --id "$ID" \
  --sft-train "$D/$SET-train.sft" --sft-dev "$D/$SET-dev.sft" --sft-final "$D/$SET-final.sft" --sft-manifest "$D/$SET.manifest.json" \
  --lm-train "$O/chat-sft/replay-16m.u16" --lm-dev "$O/sft-t27/rawval-tails-2m.u16" \
  --phi-prompts "$D/$SET.prompts.jsonl" --general-prompts "$D/general.prompts.jsonl" "$@" > "$D/$NAME.bake.log" 2>&1 \
  || { echo "BAKE FAILED, see $D/$NAME.bake.log"; tail -4 "$D/$NAME.bake.log" | cut -c1-500; exit 1; }
grep -E "phase|best|step .*dev|gate|route|published|appended" "$D/$NAME.bake.log" | tail -12 | cut -c1-240
echo "== $NAME done $(date -u +%T)"
