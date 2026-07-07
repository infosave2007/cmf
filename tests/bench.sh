#!/usr/bin/env bash
# Speed benchmark: serial vs worker pool on a ~50M-param synthetic
# model (release build). Numbers land in docs/JOURNAL.md.

set -euo pipefail
cd "$(dirname "$0")/.."

TMP="${TMPDIR:-/tmp}/cmf-bench-$$"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP"

echo "── build medium model (~50M params)"
python3 tests/make_tiny_model.py --out "$TMP/med" --preset medium >/dev/null
python3 converter/convert_dtgma_to_cmf.py \
    --model "$TMP/med" --masks "$TMP/med/masks" \
    --quant Q8_ROW --output "$TMP/med.cmf" | tail -1

echo "── release build"
cargo build -q --release -p cortiq-cli

TOKENS="${TOKENS:-64}"
echo "── serial (CMF_THREADS=1)"
CMF_THREADS=1 ./target/release/cortiq bench "$TMP/med.cmf" --tokens "$TOKENS" | grep -E 'Prompt|Decode'

echo "── pool (CMF_THREADS default)"
./target/release/cortiq bench "$TMP/med.cmf" --tokens "$TOKENS" | grep -E 'Prompt|Decode'
