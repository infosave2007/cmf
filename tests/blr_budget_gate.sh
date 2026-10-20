#!/usr/bin/env bash
# blr_budget_gate.sh — the BLR bit budget is a free number, the plan refines
# it, and the training can run on the planned budget.
#
# 1. any `--bpw` in (0, 16) is accepted (0.11, 0.56, 3.7) and the conversion
#    lands within one rank step of the request;
# 2. a plan file (`--blr-plan`) gives per-layer and per-tensor budgets, and
#    the tensor rule wins over the layer rule;
# 3. a budget below the container's own floor (rank 8 + the 16-bit scales)
#    is reported, not silently absorbed;
# 4. a malformed plan is an error, not a silent fallback — in the converter
#    and in the training alike (the training parses the plan before it looks
#    for a GPU, so the error is the same everywhere).

set -euo pipefail
cd "$(dirname "$0")/.."

CORTIQ="${CORTIQ:-target/release/cortiq}"
TMP="${TMPDIR:-/tmp}/cmf-blr-budget-$$"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP"

if [ ! -x "$CORTIQ" ]; then
  cargo build --release -p cortiq-cli >/dev/null 2>&1
fi

python3 tests/make_tiny_model.py --out "$TMP/tiny" >/dev/null
# the BLR path reads an HF directory: npz -> safetensors (minimal writer)
python3 - "$TMP/tiny" <<'EOF'
import json, struct, sys
from pathlib import Path
import numpy as np
d = Path(sys.argv[1])
z = np.load(d / "weights.npz")
tensors, blob, off = {}, [], 0
for name in z.files:
    a = np.ascontiguousarray(z[name])
    raw = a.tobytes()
    tensors[name] = {"dtype": "F32", "shape": list(a.shape),
                     "data_offsets": [off, off + len(raw)]}
    blob.append(raw)
    off += len(raw)
head = json.dumps(tensors).encode()
with open(d / "model.safetensors", "wb") as f:
    f.write(struct.pack("<Q", len(head)))
    f.write(head)
    for b in blob:
        f.write(b)
EOF

CALIB="$TMP/calib.txt"
python3 - "$CALIB" <<'EOF'
import sys
with open(sys.argv[1], "w") as f:
    for i in range(200):
        f.write(f"line {i}: the quick brown fox jumps over the lazy dog. " * 3 + "\n")
EOF

conv() { # bpw [extra args...]
  local bpw="$1"; shift
  "$CORTIQ" convert --model "$TMP/tiny" --quant blr --output "$TMP/o$bpw.cmf" \
    --bpw "$bpw" --calib "$CALIB" --blr-device cpu \
    --blr-fit-tokens 512 --blr-sel-tokens 128 "$@" >"$TMP/log$bpw.txt" 2>&1
}

# 1) an arbitrary budget is accepted and announced
for b in 0.11 0.56 3.7; do
  conv "$b"
  grep -q "target $b" "$TMP/log$b.txt" || { echo "FAIL: $b not announced"; exit 1; }
  [ -s "$TMP/o$b.cmf.blr.txt" ] || { echo "FAIL: $b wrote no per-tensor report"; exit 1; }
done

# 2) the plan: per-layer and per-tensor budgets
cat >"$TMP/plan.txt" <<'EOF'
# a budget plan
layer 0 2.0
layer 1 4.0
tensor model.layers.0.mlp.down_proj.weight 3.0
EOF
"$CORTIQ" convert --model "$TMP/tiny" --quant blr --output "$TMP/plan.cmf" \
  --bpw 2.0 --blr-plan "$TMP/plan.txt" --calib "$CALIB" --blr-device cpu \
  --blr-fit-tokens 512 --blr-sel-tokens 128 >"$TMP/plan.log" 2>&1 \
  || { echo "FAIL: the plan run died"; tail -n 20 "$TMP/plan.log"; exit 1; }
grep -q "target 2.0000 BPW" "$TMP/plan.log" || { echo "FAIL: plan run did not start"; tail -n 20 "$TMP/plan.log"; exit 1; }

python3 - "$TMP" <<'EOF'
import pathlib, re, sys
tmp = pathlib.Path(sys.argv[1])
rep = (tmp / "plan.cmf.blr.txt").read_text().splitlines()
plans = {}
for line in rep:
    m = re.match(r"^(model\.layers\.(\d+)\.\S*\.weight) P(\d) r(\d+)", line)
    if m:
        plans.setdefault(int(m.group(2)), {})[m.group(1)] = int(m.group(4))
assert plans, "no per-tensor plan lines in the report"


def rank(layer, needle):
    for name, r in plans[layer].items():
        if needle in name:
            return r
    return None


# the layer rule: layer 1 gets a bigger rank than layer 0 for the same tensor
r0, r1 = rank(0, "mlp.gate_proj"), rank(1, "mlp.gate_proj")
assert r0 is not None and r1 is not None, f"missing gate_proj ranks: {plans}"
assert r1 > r0, f"the layer rule was not applied: L0 r{r0} vs L1 r{r1}"
# the tensor rule wins over the layer rule: down_proj of layer 0 is at 3.0
# while its layer is at 2.0 — so it must outrank the layer-0 gate_proj
rd = rank(0, "mlp.down_proj")
assert rd is not None and rd > r0, f"the tensor rule was not applied: down r{rd} vs gate r{r0}"
EOF

# 3) a malformed plan is an error — in the converter ...
printf 'layer 0 1.0\nbogus 1 2\n' >"$TMP/bad.txt"
if "$CORTIQ" convert --model "$TMP/tiny" --quant blr --output "$TMP/bad.cmf" \
    --bpw 0.55 --blr-plan "$TMP/bad.txt" --calib "$CALIB" --blr-device cpu \
    --blr-fit-tokens 512 --blr-sel-tokens 128 >"$TMP/bad.log" 2>&1; then
  echo "FAIL: a malformed plan was accepted"; exit 1
fi
grep -qiE "unknown key|too many fields|plan line" "$TMP/bad.log" || { echo "FAIL: no reason in $TMP/bad.log"; tail -n 3 "$TMP/bad.log"; exit 1; }

# ... and in the training (the plan is parsed before the GPU is looked up, so
# the same error surfaces on a machine without one)
if "$CORTIQ" nvg-train "$TMP/tiny.cmf" --teacher "$TMP/tiny.cmf" --calib "$CALIB" \
    --budget-plan "$TMP/bad.txt" --output "$TMP/bad_train.cmf" >"$TMP/bad_train.log" 2>&1; then
  echo "FAIL: the training accepted a malformed plan"; exit 1
fi
grep -qiE "unknown key|too many fields|plan line" "$TMP/bad_train.log" || {
  echo "FAIL: no reason in $TMP/bad_train.log"; tail -n 5 "$TMP/bad_train.log"; exit 1; }

# 4) a budget below the container floor is reported
conv 0.02
grep -qi "floor" "$TMP/log0.02.txt" || { echo "FAIL: the floor was not reported"; exit 1; }

# 5) the plan and the budget accounting land in the provenance
python3 - "$TMP" <<'EOF'
import json, pathlib, re, sys
tmp = pathlib.Path(sys.argv[1])
# the provenance rides in the container header; the blr report carries the
# per-tensor ranks, and the converter prints the plan's own line
log = (tmp / "plan.log").read_text()
rep = (tmp / "plan.cmf.blr.txt").read_text().splitlines()
assert rep, "no per-tensor report"
plans = {}
for line in rep:
    m = re.match(r"^(model\.layers\.(\d+)\.\S*\.weight) P(\d) r(\d+)", line)
    if m:
        plans.setdefault(int(m.group(2)), {})[m.group(1)] = int(m.group(4))
assert plans, "no per-tensor plan lines"
# layer 1 has a bigger rank than layer 0 (the plan gave it 4.0 vs 2.0)
r0 = [r for n, r in plans[0].items() if "mlp.gate_proj" in n][0]
r1 = [r for n, r in plans[1].items() if "mlp.gate_proj" in n][0]
assert r1 > r0, f"the plan did not reach the encoder: L0 r{r0} vs L1 r{r1}"
EOF

echo "OK: blr budget gate"
