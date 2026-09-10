#!/usr/bin/env bash
# MoE parity gate: transformers reference ↔ Rust engine on tiny random
# Qwen2-MoE (shared expert + qkv-bias + dense/MoE чередование) and
# Qwen3-MoE (qk-norm, norm_topk_prob). F32 = строгие логиты + greedy.

set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-/Users/oleg/Documents/cortiq-bot/venv_heal/bin/python3}"
TMP="${TMPDIR:-/tmp}/cmf-moe-$$"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP"

for FAM in qwen2 qwen3 qwen3_next; do
    echo "── moe parity: $FAM (F32)"
    "$PY" tests/gen_moe_case.py \
        --family "$FAM" --out "$TMP/$FAM" --ref "$TMP/ref-$FAM.json"
    python3 converter/convert_dtgma_to_cmf.py \
        --model "$TMP/$FAM" --quant F32 --output "$TMP/$FAM.cmf" >/dev/null
    CMF_GOLDEN_FILE="$TMP/$FAM.cmf" \
    CMF_GOLDEN_REF="$TMP/ref-$FAM.json" \
        cargo test -q -p cortiq-engine --test golden_parity -- --nocapture
done

# Fused-банки AgentWorld/qwen3_5_moe: та же tiny qwen3_next, эксперты
# перепакованы в gate_up_proj/down_proj [E,…] → parity против ТОГО ЖЕ
# эталона (раскрытие фьюза обязано быть чистой перестановкой).
echo "── moe parity: qwen3_next FUSED (раскладка AgentWorld)"
"$PY" tests/repack_fused_moe.py "$TMP/qwen3_next" "$TMP/qwen3_next_fused"
python3 converter/convert_dtgma_to_cmf.py \
    --model "$TMP/qwen3_next_fused" --quant F32 \
    --output "$TMP/qwen3_next_fused.cmf" >/dev/null
CMF_GOLDEN_FILE="$TMP/qwen3_next_fused.cmf" \
CMF_GOLDEN_REF="$TMP/ref-qwen3_next.json" \
    cargo test -q -p cortiq-engine --test golden_parity -- --nocapture

echo "MOE PARITY OK"
