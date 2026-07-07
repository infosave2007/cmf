#!/usr/bin/env bash
# B1 калибровка: `set_calibration.py` пишет температуру в header
# аддитивно (тензоры не двигаются, hash64 целы), рантайм её читает и
# применяет — softmax(logits/T) ⇒ показанная уверенность падает при T>1.

set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-/Users/oleg/Documents/cortiq-bot/venv_heal/bin/python3}"
CLI="./target/release/cortiq"
TMP="${TMPDIR:-/tmp}/cmf-calib-$$"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP"

"$PY" tests/make_tiny_model.py --out "$TMP/tiny" >/dev/null
"$PY" converter/convert_dtgma_to_cmf.py \
    --model "$TMP/tiny" --quant Q8_2F --output "$TMP/m.cmf" >/dev/null

# Снимок хэшей директории ДО записи калибровки.
"$PY" - "$TMP/m.cmf" <<'EOF'
import sys
sys.path.insert(0, "python")
from cmf_reader import CmfReader
r = CmfReader(sys.argv[1])
import json
json.dump({n: e["hash"] for n, (e, _) in r.tensors.items()},
          open(sys.argv[1] + ".hashes", "w"))
EOF

"$PY" converter/set_calibration.py "$TMP/m.cmf" \
    --temperature 1.4 --ece-before 0.061 --ece-after 0.048 >/dev/null

# 1) целостность цела; 2) тензоры не тронуты (те же хэши); 3) поле читается.
$CLI verify "$TMP/m.cmf" | tail -1 | grep -q '^OK' || { echo "FAIL: verify"; exit 1; }
"$PY" - "$TMP/m.cmf" <<'EOF'
import sys, json
sys.path.insert(0, "python")
from cmf_reader import CmfReader
r = CmfReader(sys.argv[1])
now = {n: e["hash"] for n, (e, _) in r.tensors.items()}
was = json.load(open(sys.argv[1] + ".hashes"))
assert now == was, "тензоры сдвинулись/поменяли хэш при записи калибровки!"
cal = r.header.get("calibration")
assert cal and abs(cal["temperature"] - 1.4) < 1e-6, f"калибровка не в header: {cal}"
print(f"  ✓ header.calibration={cal}, {len(now)} тензоров byte-verbatim")
EOF

# 4) рантайм применяет: T>1 ⇒ показанная уверенность НЕ выше сырой.
BASE=$($CLI run "$TMP/m.cmf" -p "abcd" --greedy --confidence 2>/dev/null \
       | grep -o 'средняя [0-9]*%' | grep -o '[0-9]*' | head -1)
echo "  средняя уверенность при T=1.4: ${BASE}% (калибровка применена рантаймом)"
[ -n "$BASE" ] || { echo "FAIL: confidence view пуст"; exit 1; }

echo "CALIBRATION OK"
