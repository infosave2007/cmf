#!/usr/bin/env bash
# The built-in tools check on the Mac: 500 fine-tune steps of the final 340M base on the v16 traces with the
# engine's calc/table (lookup_gen.py --builtin-tools, arith answered through table), then set R with the
# server running the built-in calls (run_lookup.py --builtin-tools, own store, ranked). One run at a time.
#   tools_check.sh NAME START_CKPT ORIGIN SET [LR] [STEPS]   (SET: the shards $STAGE0/ft/SET-{train,dev,final}.sft;
#                                                             LR 2e-5 and 500 steps by default)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; REPO="$(cd "$HERE/../../.." && pwd)"
B="${STAGE0:-$HOME/embryo-data/hermes-core/stage0}"; O="$HOME/embryo-data/o1"
NAME="$1"; START="$2"; ORIGIN="$3"; SET="$4"; LR="${5:-2e-5}"; STEPS="${6:-500}"
SET="$SET" SEQ=768 ORIGIN="$ORIGIN" "$HERE/ft_chain.sh" "$NAME" "$START" 5 "$LR" "" "$STEPS"
"$REPO/target/release/cortiq-embryo" export --ckpt "$B/ft/$NAME-$STEPS.ckpt" --tokenizer "$O/sft-t27/gen1.tokenizer.json" \
  --out "$B/ft/$NAME-$STEPS.cmf" --dtype f16 --chat | tail -1
rm -f "$B/ft/$NAME-$STEPS.ckpt"  # 4 GB with the moments; the export is what is measured
"$B/venv/bin/python" -I "$HERE/run_lookup.py" --cortiq "$REPO/target/release/cortiq" --model "$B/ft/$NAME-$STEPS.cmf" \
  --tasks "$B/run/r.jsonl" --out "$B/run/$NAME.own-ranked-bt.outputs.jsonl" --mode lookup --store own --ranked \
  --max-tokens 200 --builtin-tools
echo "TOOLS CHECK DONE $NAME"
