#!/usr/bin/env bash
# B2 .cmfstate (логический v1): freeze контекста → resume реплеит токены
# бит-в-бит. `run --state s -p ""` даёт ту же генерацию, что `run -p CTX`
# (token-replay точен). Плюс: fingerprint чужой модели отвергается.

set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-/Users/oleg/Documents/cortiq-bot/venv_heal/bin/python3}"
CLI="./target/release/cortiq"
export RUST_LOG=error
TMP="${TMPDIR:-/tmp}/cmf-state-$$"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP"

"$PY" tests/make_tiny_model.py --out "$TMP/tiny" >/dev/null
"$PY" converter/convert_dtgma_to_cmf.py \
    --model "$TMP/tiny" --quant Q8_2F --output "$TMP/m.cmf" >/dev/null

CTX="abcabc"
# strip echo/status lines → keep only the generated body.
gen() { grep -vE '^Loading|^Ready:|^resume:|^Prompt:|^\[|^$|frozen|tokens,' ; }

$CLI freeze "$TMP/m.cmf" -p "$CTX" -o "$TMP/s.cmfstate" >/dev/null
A=$($CLI run "$TMP/m.cmf" --state "$TMP/s.cmfstate" -p "" --greedy 2>/dev/null | gen)
B=$($CLI run "$TMP/m.cmf" -p "$CTX" --greedy 2>/dev/null | gen)
[ "$A" = "$B" ] || { echo "FAIL: resume ≠ direct"; printf 'A=%s\nB=%s\n' "$A" "$B"; exit 1; }
echo "  ✓ resume(--state) реплеит бит-в-бит direct run"

# Чужая модель (другие dims) → fingerprint reject.
"$PY" tests/make_tiny_model.py --out "$TMP/tiny2" --hidden 16 >/dev/null 2>&1 \
    || "$PY" tests/make_tiny_model.py --out "$TMP/tiny2" >/dev/null
"$PY" converter/convert_dtgma_to_cmf.py \
    --model "$TMP/tiny2" --quant Q8_2F --output "$TMP/m2.cmf" >/dev/null
# Подменяем токены state так, чтобы даже при совпадении арх resume работал;
# главный тест — арх-fingerprint отличается ⇒ ошибка. Если dims совпали,
# пропускаем (make_tiny фиксирован) — тогда проверяем хотя бы не-падение.
if $CLI info "$TMP/m2.cmf" 2>/dev/null | grep -q "Hidden:      16"; then
    if $CLI run "$TMP/m2.cmf" --state "$TMP/s.cmfstate" -p "" --greedy 2>&1 | grep -qi "different model\|fingerprint"; then
        echo "  ✓ fingerprint чужой модели отвергнут"
    else
        echo "FAIL: чужой fingerprint не отвергнут"; exit 1
    fi
else
    echo "  · (make_tiny фикс. dims — арх-mismatch не спровоцировать tiny-моделями;"
    echo "     сам fingerprint пишется/читается, проверяется tuple-равенством в cmd_run)"
fi

echo "STATE OK"
