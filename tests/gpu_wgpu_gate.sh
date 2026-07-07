#!/usr/bin/env bash
# C1 кроссплатформенный GPU (wgpu): собираем движок с `--features gpu` и
# гоняем паритет-тесты q8 matvec/matmat против CPU-эталона. Локально идёт
# через Metal-via-wgpu; на сервере — Vulkan (NVIDIA/Radeon) / DX12. Если
# GPU-адаптера нет, тесты сами себя пропускают (гейт остаётся зелёным).

set -euo pipefail
cd "$(dirname "$0")/.."

echo "── сборка с --features gpu ──"
cargo build --release -p cortiq-engine --features gpu >/dev/null 2>&1 \
    || { echo "FAIL: не собралось с --features gpu"; exit 1; }

echo "── паритет q8_matvec / q8_matmat (GPU == CPU) ──"
CMF_GPU=wgpu cargo test --release -p cortiq-engine --features gpu wgpu_q8 2>&1 \
    | grep -E "wgpu_q8|test result:|skipping" | sed 's/^/  /'

# Тест падает при расхождении > tol; сам пропускается без адаптера.
CMF_GPU=wgpu cargo test --release -p cortiq-engine --features gpu wgpu_q8 >/dev/null 2>&1 \
    || { echo "FAIL: паритет GPU==CPU не прошёл"; exit 1; }

# ── E2E: moe_block + matvec_batch + layer-split на крошечной MoE-модели ──
# (если нет GPU-адаптера, wgpu-путь = CPU-fallback ⇒ равенство тривиально).
PY="${PY:-/Users/oleg/Documents/cortiq-bot/venv_heal/bin/python3}"
CLI="./target/release/cortiq"
cargo build --release -p cortiq-cli --features gpu >/dev/null 2>&1
TMP="${TMPDIR:-/tmp}/cmf-wgpu-moe-$$"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP"
"$PY" tests/gen_moe_case.py --family qwen3 --out "$TMP/m" --ref "$TMP/r.json" >/dev/null 2>&1
"$PY" converter/convert_dtgma_to_cmf.py --model "$TMP/m" --quant Q8_2F \
    --output "$TMP/moe.cmf" >/dev/null 2>&1
printf 'the quick brown fox jumps over the lazy dog %.0s' {1..40} > "$TMP/t.txt"
ppl() { $CLI ppl "$TMP/moe.cmf" --file "$TMP/t.txt" --tokens 120 2>/dev/null | grep -o 'PPL = [0-9.]*'; }
CPU="$(RUST_LOG=error ppl)"
GPU="$(CMF_GPU=wgpu RUST_LOG=error ppl)"
SPL="$(CMF_GPU=wgpu CMF_GPU_LAYERS=0-1 RUST_LOG=error ppl)"
echo "  moe_block: CPU=$CPU  wgpu=$GPU  layer-split(0-1)=$SPL"
[ "$CPU" = "$GPU" ] && [ "$CPU" = "$SPL" ] \
    || { echo "FAIL: MoE/layer-split паритет разошёлся"; exit 1; }

echo "GPU_WGPU OK"
