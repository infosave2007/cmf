#!/usr/bin/env bash
# A 56M file on the KLM-1.6 index tasks on the Mac's CPU (zero-shot, raw text, acc_norm; as
# scripts/embryo/bench): ARC-Easy, ARC-Challenge, HellaSwag, PIQA through lm-eval and ArithMark-3.
#   measure_klm.sh NAME FILE.cmf      runs under ~/embryo-data/bench-klm/runs/NAME, index on the last line
# CORTIQ_LM_JOBS engine processes (default 8) with one thread each.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
B="$HOME/embryo-data/bench-klm"; C="$REPO/target/release/cortiq"; H="$REPO/scripts/embryo/bench"
NAME="$1"; M="$2"; O="$B/runs/$NAME"
export HF_HOME="$B/hf-home" CMF_GPU=0 CMF_GROWTH=off
if [ ! -x "$B/venv/bin/python" ]; then
  uv venv --python python3.12 "$B/venv" >/dev/null && VIRTUAL_ENV="$B/venv" uv pip install --python "$B/venv/bin/python" -q "lm-eval==0.4.12"
fi
if [ ! -f "$B/dl-arithmark3/arithmark-3.jsonl" ]; then
  mkdir -p "$B/dl-arithmark3"
  for f in README.md bencharithmark-3.py arithmark-3.jsonl; do
    curl -sL -o "$B/dl-arithmark3/$f" "https://huggingface.co/datasets/AxiomicLabs/ArithMark-3.0/resolve/main/$f"
  done
fi
SECONDS=0
CORTIQ_LM_JOBS="${CORTIQ_LM_JOBS:-8}" CMF_THREADS=1 "$B/venv/bin/python" -I "$H/run_lmeval.py" --cortiq "$C" --model "$M" \
  --out-dir "$O" --tasks arc_easy,arc_challenge,hellaswag,piqa > "$O.lmeval.log" 2>&1
CMF_THREADS=8 "$B/venv/bin/python" -I "$H/arithmark3_cortiq.py" --cortiq "$C" --model "$M" \
  --script "$B/dl-arithmark3/bencharithmark-3.py" --data "$B/dl-arithmark3/arithmark-3.jsonl" --out-dir "$O" > "$O.arith.log" 2>&1
python3 -I - "$O" "$NAME" "$SECONDS" <<'PY'
import json, sys
o, name, secs = sys.argv[1], sys.argv[2], sys.argv[3]
acc = lambda t: 100 * json.load(open(f"{o}/{t}.raw.results.json"))["results"][t]["acc_norm,none"]
n = lambda s, c: 100 * (s - c) / (100 - c)
ae, ac, hs, pq = acc("arc_easy"), acc("arc_challenge"), acc("hellaswag"), acc("piqa")
am = json.load(open(f"{o}/arithmark-3.results.json"))["results"]["arithmark-3"]["acc_norm"]
idx = (n(hs, 25) + n((ae + ac) / 2, 25) + n(pq, 50) + 0.65 * n(am, 25)) / 3.65
print(f"{name}: ARC-E {ae:.2f} ARC-C {ac:.2f} HellaSwag {hs:.2f} PIQA {pq:.2f} ArithMark-3 {am:.2f} -> index {idx:.2f} ({secs} s)")
PY
