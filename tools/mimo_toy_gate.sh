#!/bin/bash
# MiMo-V2 toy gate: converter + CPU engine against the oracle's golden
# references (tools/mk_mimo_toy.py). Run from the repo root on the pod after
# `cargo build --release -p cortiq-cli`.
#
#   TOY=/root/mimo/toy PY=/root/mimo/venv/bin/python tools/mimo_toy_gate.sh
#
# 1. positive toy, --quant f16: checker (f16 profile) + golden_parity must PASS
#    (first-step logits max|d| < 1e-3, 8 exact greedy tokens). reference.json
#    assumes the converter folds attention_value_scale into f16 V rows; a
#    converter that applies it at runtime must pass reference_exact.json.
# 2. negative toy (qkv stored contiguously), --quant f16: golden_parity must
#    FAIL -- otherwise the gate cannot see a qkv layout error.
# 3. positive toy with NO flags: the default mimo_v2 profile must be q4tp
#    (checker --profile q4tp) and `cortiq run` must work without flags.
# 4. with CMF_LAYER_DUMP in the engine: per-layer diff against ref/ dumps.
set -u
TOY=${TOY:-/root/mimo/toy}
PY=${PY:-python3}
CORTIQ=${CORTIQ:-./target/release/cortiq}
OUT=${OUT:-$TOY/gate}
CPU="CMF_GPU=0 CMF_SDOT=0 CMF_GPU_PROBE=0"
mkdir -p "$OUT"
fail=0

for t in pos neg_contig; do
  [ -f "$TOY/$t/reference.json" ] || { echo "GATE ABORT: no $TOY/$t (run tools/mk_mimo_toy.py)"; exit 1; }
  "$CORTIQ" convert --model "$TOY/$t" --quant f16 --output "$OUT/$t-f16.cmf" > "$OUT/$t-f16.convert.log" 2>&1 \
    || { echo "$t: convert --quant f16 FAILED (see $OUT/$t-f16.convert.log)"; fail=1; continue; }
  $PY scripts/check_mimo_cmf.py --model "$OUT/$t-f16.cmf" --source-config "$TOY/$t/config.json" \
      --source-index "$TOY/$t/model.safetensors.index.json" --profile f16 > "$OUT/$t-f16.check.log" 2>&1
  echo "$t f16 checker: $(head -1 "$OUT/$t-f16.check.log")"
  env $CPU CMF_GOLDEN_FILE="$OUT/$t-f16.cmf" CMF_GOLDEN_REF="$TOY/$t/reference.json" \
    cargo test --release -p cortiq-engine --test golden_parity -- --nocapture > "$OUT/$t-golden.log" 2>&1
  rc=$?
  if [ "$t" = pos ]; then
    [ $rc = 0 ] && echo "pos golden parity: PASS $(grep -o 'max|Δ|logits = [0-9.e-]*' "$OUT/$t-golden.log")" \
               || { echo "pos golden parity: FAIL (see $OUT/$t-golden.log)"; fail=1; }
  else
    [ $rc != 0 ] && echo "negative control: FAILS as it must" \
                 || { echo "negative control PASSED -- the gate is blind to the qkv layout"; fail=1; }
  fi
done

"$CORTIQ" convert --model "$TOY/pos" --output "$OUT/pos-default.cmf" > "$OUT/pos-default.convert.log" 2>&1 \
  || { echo "default convert FAILED"; fail=1; }
$PY scripts/check_mimo_cmf.py --model "$OUT/pos-default.cmf" --source-config "$TOY/pos/config.json" \
    --profile q4tp > "$OUT/pos-default.check.log" 2>&1 || fail=1
echo "default profile checker: $(head -1 "$OUT/pos-default.check.log")"
env $CPU "$CORTIQ" run "$OUT/pos-default.cmf" --prompt "Hello" --max-tokens 4 > "$OUT/pos-default.run.log" 2>&1 \
  && echo "default run: ok" || { echo "default run FAILED"; fail=1; }

# Per-layer diff, when the engine writes CMF_LAYER_DUMP. prompt_text
# re-encodes to exactly prompt_ids (checked by mk_mimo_toy.py); --raw keeps
# the chat template out, and MiMo has no BOS.
rm -rf "$OUT/dump" "$OUT/moe_trace.txt" && mkdir -p "$OUT/dump"
text=$($PY -c "import json;print(json.load(open('$TOY/pos/reference.json'))['prompt_text'])")
env $CPU CMF_PREFILL=seq CMF_TRACE_H=1 CMF_LAYER_DUMP="$OUT/dump" CMF_MOE_TRACE="$OUT/moe_trace.txt" \
  "$CORTIQ" run "$OUT/pos-f16.cmf" --raw --prompt "$text" --max-tokens 1 --greedy > "$OUT/dump.log" 2>&1
if ls "$OUT/dump"/p000000_l00*.f32 > /dev/null 2>&1; then
  $PY tools/mimo_cmp.py --engine "$OUT/dump" --ref "$TOY/pos/ref" --positions 0-23 \
      --engine-trace "$OUT/moe_trace.txt" --ref-trace "$TOY/pos/ref/moe_trace.txt" > "$OUT/cmp.log" 2>&1 || fail=1
  tail -1 "$OUT/cmp.log"
else
  echo "no CMF_LAYER_DUMP output (engine without the dump hook?) -- per-layer diff skipped"
fi
[ $fail = 0 ] && echo "MIMO TOY GATE: PASS" || echo "MIMO TOY GATE: FAIL"
exit $fail
