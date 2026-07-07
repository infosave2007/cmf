#!/usr/bin/env bash
# Компактация после append-only роста: tiny-контейнер + skill_add →
# cmf_compact → мёртвые зоны исчезли, тензоры байт-в-байт, verify
# зелёный, скилл жив. Плюс smoke на реальном qwen35-append.cmf.

set -euo pipefail
cd "$(dirname "$0")/.."

TMP="${TMPDIR:-/tmp}/cmf-compact-$$"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP"

python3 tests/make_tiny_model.py --out "$TMP/tiny" >/dev/null
python3 converter/convert_dtgma_to_cmf.py \
    --model "$TMP/tiny" --quant Q8_2F --output "$TMP/t.cmf" >/dev/null

# Синтетический скилл: layer-0 gate_proj ×2 (формат make_skill).
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
python3 converter/skill_add.py "$TMP/t.cmf" "$TMP/skill" >/dev/null

python3 converter/cmf_compact.py "$TMP/t.cmf" "$TMP/t-compact.cmf"

python3 - "$TMP" <<'EOF'
import sys
from pathlib import Path
sys.path.insert(0, "python")
from cmf_reader import CmfReader
tmp = Path(sys.argv[1])
a = CmfReader(tmp / "t.cmf")
b = CmfReader(tmp / "t-compact.cmf")
assert set(a.tensors) == set(b.tensors)
for n in a.tensors:
    assert bytes(a.tensor_bytes(n)) == bytes(b.tensor_bytes(n)), n
assert [s["id"] for s in b.skills] == ["boost"]
assert b.verify() == []
old, new = (tmp / "t.cmf").stat().st_size, (tmp / "t-compact.cmf").stat().st_size
assert new < old, f"компакт не меньше: {new} vs {old}"
print(f"  ✓ тензоры байт-в-байт, скилл жив, {old-new} байт мёртвых зон убрано")
EOF

cargo run -q -p cortiq-cli --release -- verify "$TMP/t-compact.cmf" | tail -1

# Реальный артефакт (если на месте): append-контейнер этапа B5.
if [ -f models/qwen35-append.cmf ]; then
    echo "── реальный append-контейнер"
    python3 converter/cmf_compact.py models/qwen35-append.cmf "$TMP/big.cmf"
    cargo run -q -p cortiq-cli --release -- verify "$TMP/big.cmf" | tail -1
fi

echo "COMPACT OK"
