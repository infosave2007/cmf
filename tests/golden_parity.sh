#!/usr/bin/env bash
# Golden parity gate: numpy reference ↔ Rust engine on the same .cmf,
# run for BOTH quantizations (F32 exact-path, Q8_ROW quantized path).

set -euo pipefail
cd "$(dirname "$0")/.."

TMP="${TMPDIR:-/tmp}/cmf-golden-$$"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP"

python3 tests/make_tiny_model.py --out "$TMP/tiny" >/dev/null

# Masked model. F32 = exact-path (strict). Q8_ROW/Q8_2F now run the
# QUANTIZED FFN even with masks (mask × mmap, this change) — so their
# logits carry the same A8W8 quant noise as the mmap group below → loose
# tolerance, greedy still token-for-token. (Historically masks forced the
# whole model to f32; that RSS blowup is exactly what we removed.)
for QUANT in F32 Q8_ROW Q8_2F; do
    echo "── golden parity: $QUANT (masked)"
    python3 converter/convert_dtgma_to_cmf.py \
        --model "$TMP/tiny" --masks "$TMP/tiny/masks" \
        --quant "$QUANT" --output "$TMP/tiny-$QUANT.cmf" >/dev/null
    python3 tests/gen_reference.py \
        --model "$TMP/tiny" --quant "$QUANT" --out "$TMP/ref-$QUANT.json"
    LOOSE=""
    [ "$QUANT" = "F32" ] || LOOSE="1"
    CMF_GOLDEN_LOOSE="$LOOSE" \
    CMF_GOLDEN_FILE="$TMP/tiny-$QUANT.cmf" \
    CMF_GOLDEN_REF="$TMP/ref-$QUANT.json" \
        cargo test -q -p cortiq-engine --test golden_parity -- --nocapture
done

# No masks → the loader keeps weights quantized in mmap and runs the
# fused int8 kernels. Two contracts:
#   CMF_SDOT=0 — exact fused kernels: strict logits tolerance;
#   CMF_SDOT=1 — A8W8: bounded activation noise, greedy still strict.
echo "── golden parity: Q8_2F (quantized mmap, exact kernels)"
python3 converter/convert_dtgma_to_cmf.py \
    --model "$TMP/tiny" \
    --quant Q8_2F --output "$TMP/tiny-mmap.cmf" >/dev/null
CMF_SDOT=0 \
CMF_GOLDEN_FILE="$TMP/tiny-mmap.cmf" \
CMF_GOLDEN_REF="$TMP/ref-Q8_2F.json" \
    cargo test -q -p cortiq-engine --test golden_parity -- --nocapture

echo "── golden parity: Q8_2F (quantized mmap, A8W8 SDOT)"
CMF_SDOT=1 CMF_GOLDEN_LOOSE=1 \
CMF_GOLDEN_FILE="$TMP/tiny-mmap.cmf" \
CMF_GOLDEN_REF="$TMP/ref-Q8_2F.json" \
    cargo test -q -p cortiq-engine --test golden_parity -- --nocapture

# q4_block fused-from-mmap kernel: reference dequantizes in numpy, the
# engine reads nibbles directly — same math, different f32 summation
# order → loose logits, greedy strict.
echo "── golden parity: Q4_BLOCK (quantized mmap, fused nibbles)"
python3 converter/convert_dtgma_to_cmf.py \
    --model "$TMP/tiny" \
    --quant Q4_BLOCK --output "$TMP/tiny-q4.cmf" >/dev/null
python3 tests/gen_reference.py \
    --model "$TMP/tiny" --quant Q4_BLOCK --out "$TMP/ref-Q4_BLOCK.json"
CMF_GOLDEN_LOOSE=1 \
CMF_GOLDEN_FILE="$TMP/tiny-q4.cmf" \
CMF_GOLDEN_REF="$TMP/ref-Q4_BLOCK.json" \
    cargo test -q -p cortiq-engine --test golden_parity -- --nocapture

# KV-quant 2f (CMF_KV=q8): int8-кэш + канальное поле — ограниченный
# шум активаций; greedy остаётся строгим (как A8W8-контракт).
echo "── golden parity: Q8_2F + CMF_KV=q8 (квантованный KV-кэш)"
CMF_KV=q8 CMF_GOLDEN_LOOSE=1 \
CMF_GOLDEN_FILE="$TMP/tiny-mmap.cmf" \
CMF_GOLDEN_REF="$TMP/ref-Q8_2F.json" \
    cargo test -q -p cortiq-engine --test golden_parity -- --nocapture

echo "GOLDEN PARITY OK"
