#!/usr/bin/env bash
# Cross-language integration test: Python writer ↔ Rust reader.
#
# 1. Generate a deterministic tiny checkpoint (numpy).
# 2. Convert it to .cmf with the Python converter.
# 3. Open it with the Rust reader: `cortiq info` (strict open) and
#    `cortiq verify` (recompute every tensor hash64 in Rust against
#    hashes written by numpy — the real cross-language contract).
#
# Practically every v1 bug was a writer/reader divergence; this script
# is the gate that class of bug has to pass.

set -euo pipefail
cd "$(dirname "$0")/.."

TMP="${TMPDIR:-/tmp}/cmf-py-roundtrip-$$"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP"

echo "── 1/3 generate tiny checkpoint"
python3 tests/make_tiny_model.py --out "$TMP/tiny"

echo "── 2/3 convert to .cmf (Q8_ROW)"
python3 converter/convert_dtgma_to_cmf.py \
    --model "$TMP/tiny" --masks "$TMP/tiny/masks" \
    --quant Q8_ROW --output "$TMP/tiny.cmf"

echo "── 3/3 open + verify with Rust reader"
cargo run -q -p cortiq-cli -- info "$TMP/tiny.cmf"
cargo run -q -p cortiq-cli -- verify "$TMP/tiny.cmf"

echo "── determinism: same inputs → byte-identical file"
python3 converter/convert_dtgma_to_cmf.py \
    --model "$TMP/tiny" --masks "$TMP/tiny/masks" \
    --quant Q8_ROW --output "$TMP/tiny2.cmf" >/dev/null
cmp "$TMP/tiny.cmf" "$TMP/tiny2.cmf" && echo "  ✓ deterministic output"

echo "── corruption: flipped weight byte must fail verify"
python3 - "$TMP/tiny.cmf" <<'EOF'
import struct, sys
path = sys.argv[1]
data = bytearray(open(path, 'rb').read())
data_off = struct.unpack('<Q', data[0x30:0x38])[0]
data[data_off + 128] ^= 0xFF
open(path, 'wb').write(bytes(data))
EOF
if cargo run -q -p cortiq-cli -- verify "$TMP/tiny.cmf" >/dev/null 2>&1; then
    echo "  ✗ corruption NOT detected"; exit 1
else
    echo "  ✓ corruption detected"
fi

echo "PY-ROUNDTRIP OK"
