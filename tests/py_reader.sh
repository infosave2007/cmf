#!/usr/bin/env bash
# D4 gate: автономный Python-ридер (python/cmf_reader.py) читает всё,
# что пишет конвертер: каждый dtype, шарды, скиллы, целостность.

set -euo pipefail
cd "$(dirname "$0")/.."

TMP="${TMPDIR:-/tmp}/cmf-py-reader-$$"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP"

python3 tests/make_tiny_model.py --out "$TMP/tiny" >/dev/null

for QUANT in F32 F16 Q8_ROW Q8_2F Q4_BLOCK VBIT; do
    python3 converter/convert_dtgma_to_cmf.py \
        --model "$TMP/tiny" --masks "$TMP/tiny/masks" \
        --quant "$QUANT" --output "$TMP/tiny-$QUANT.cmf" >/dev/null
    python3 tests/py_reader_check.py quant \
        "$TMP/tiny-$QUANT.cmf" "$TMP/tiny/weights.npz" "$QUANT"
done

echo "── шардинг"
python3 converter/convert_dtgma_to_cmf.py \
    --model "$TMP/tiny" --quant Q8_2F \
    --shard-max-gb 0.00005 --output "$TMP/sh.cmf" >/dev/null
python3 tests/py_reader_check.py sharded \
    "$TMP/tiny-Q8_2F.cmf" "$TMP"/sh-00001-of-*.cmf

echo "── порча"
python3 tests/py_reader_check.py corruption "$TMP/tiny-Q8_ROW.cmf"

# Реальные артефакты, если лежат рядом (не обязательны для ворот).
if [ -f models/qwen35-swarm.cmf ]; then
    echo "── swarm smoke"
    python3 python/cmf_reader.py models/qwen35-swarm.cmf \
        --tensor model.embed_tokens.weight
fi

echo "PY-READER OK"
