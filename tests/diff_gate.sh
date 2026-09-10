#!/usr/bin/env bash
# `cortiq diff`: семантический дифф двух .cmf по идентичности тензора
# (name + hash64, spec §3). append-only skill_add обязан оставить все
# родительские тензоры verbatim (тот же hash) и добавить ровно новые —
# дифф это доказывает числом, без dequant и без ML-заявок.

set -euo pipefail
cd "$(dirname "$0")/.."

TMP="${TMPDIR:-/tmp}/cmf-diff-$$"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP"

python3 tests/make_tiny_model.py --out "$TMP/tiny" >/dev/null
python3 converter/convert_dtgma_to_cmf.py \
    --model "$TMP/tiny" --quant Q8_2F --output "$TMP/parent.cmf" >/dev/null

# Родитель до дозаписи — снимок для диффа.
cp "$TMP/parent.cmf" "$TMP/child.cmf"

# Синтетический скилл: layer-0 gate_proj ×2.
python3 - "$TMP" <<'EOF'
import json, sys
from pathlib import Path
import numpy as np
tmp = Path(sys.argv[1])
w = np.load(tmp / "tiny/weights.npz")["model.layers.0.mlp.gate_proj.weight"]
d = tmp / "skill/tensors"
d.mkdir(parents=True)
np.save(d / "model.layers.0.mlp.gate_proj.weight.npy", w * 2.0)
json.dump({"id": "boost", "layers": [0]}, open(tmp / "skill/skill.json", "w"))
EOF
python3 converter/skill_add.py "$TMP/child.cmf" "$TMP/skill" >/dev/null

OUT="$(cargo run -q -p cortiq-cli --release -- diff "$TMP/parent.cmf" "$TMP/child.cmf" 2>/dev/null)"
echo "$OUT" | grep -v '^$' | sed 's/\x1b\[[0-9;]*m//g' | tail -n +3

# Родительские тензоры не тронуты (verbatim), добавлен ровно 1 новый
# skill-тензор, ни один не удалён/не изменён, и рой прирос на «boost».
CLEAN="$(echo "$OUT" | sed 's/\x1b\[[0-9;]*m//g')"
echo "$CLEAN" | grep -qE '\+1 новых'     || { echo "FAIL: ожидался +1 новый тензор"; exit 1; }
echo "$CLEAN" | grep -qE '−0 удалено'     || { echo "FAIL: ничего не должно удаляться"; exit 1; }
echo "$CLEAN" | grep -qE '~0 изменено'    || { echo "FAIL: append-only не меняет тензоры"; exit 1; }
echo "$CLEAN" | grep -qE '\+\[boost\]'    || { echo "FAIL: рой должен прирасти 'boost'"; exit 1; }

# И обратный дифф самого-себя = всё идентично.
SELF="$(cargo run -q -p cortiq-cli --release -- diff "$TMP/child.cmf" "$TMP/child.cmf" 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g')"
echo "$SELF" | grep -qE 'идентичны'       || { echo "FAIL: self-diff не идентичен"; exit 1; }
echo "$SELF" | grep -qE '\+0 новых'       || { echo "FAIL: self-diff должен быть пуст"; exit 1; }

echo "DIFF OK"
