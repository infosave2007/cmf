#!/bin/bash
# MiMo-V2 GPU-graph toy gate: the wgpu whole-token graph and the batched
# prefill graph against the CPU walk on the toy checkpoints of
# tools/mk_mimo_toy.py (window 8, so a 24-token prompt plus 64 decode steps
# wraps every sliding layer's ring many times). Run on a wgpu machine from
# the repo root after `cargo build --release -p cortiq-cli --features gpu`:
#
#   TOY=/root/mimo/toy OUT=/tmp/gate tools/mimo_gpu_toy_gate.sh
#
# Reference: the CPU walk with exact f32 activations (CMF_GPU=0
# CMF_SDOT=0). The GPU arms also run CMF_SDOT=0, so whatever part of a
# token the host computes (the lm_head after a batched prefill, the host
# tail of a device prefix) is exact and the comparison sees the graphs:
#   tok      token graph only (CMF_BATCH_K=0: the prompt walks the graph
#            position by position), CMF_COOP=0
#   batch    batched prefill graph (k=32 chunks) + token-graph decode, COOP=0
#   prefix   a VRAM budget that holds only part of the stack: the batch and
#            token graphs run a device prefix, the host finishes each token
#   split    a budget that holds the layers but not the lm_head: the batched
#            prefill runs every layer, decode a prefix one layer shorter —
#            the host pulls that layer's prompt rows from the device mirror
#   coop     batch with the tf32-class coop GEMMs allowed (reported only)
#   default  no flags at all (int8 host activations, coop): reported only
# Each arm is compared step by step with tools/logit_steps_cmp.py; the
# gate needs top-1 equal at every step and max|d|/max|logit| < 1e-3 on
# tok/batch/prefix/split. GPU runs take /root/gpu.lock when it exists.
set -u
TOY=${TOY:-/root/mimo/toy}
OUT=${OUT:-$TOY/gpu_gate}
CORTIQ=${CORTIQ:-./target/release/cortiq}
PY=${PY:-python3}
STEPS=${STEPS:-64}
mkdir -p "$OUT"
LOCK=""
[ -e /root/gpu.lock ] && LOCK="flock /root/gpu.lock"
fail=0
text=$($PY -c "import json;print(json.load(open('$TOY/pos/reference.json'))['prompt_text'])")
run() { # name file env...
  local name=$1 file=$2; shift 2
  rm -rf "$OUT/$name"
  env RUST_LOG=info CMF_IGNORE_EOS=1 CMF_GPU_PROBE=0 CMF_LOGIT_DUMP_ALL="$OUT/$name" "$@" \
    $LOCK "$CORTIQ" run "$file" --raw --prompt "$text" --greedy --max-tokens "$STEPS" \
    > "$OUT/$name.log" 2>&1 || { echo "$name: run FAILED (see $OUT/$name.log)"; fail=1; }
}
cmp() { # ref test tol
  $PY tools/logit_steps_cmp.py --ref "$OUT/$1" --test "$OUT/$2" --tol "$3" --quiet > "$OUT/$2.cmp" 2>&1
  local rc=$?
  echo "$2 vs $1: $(tail -1 "$OUT/$2.cmp")"
  return $rc
}
for f in pos_q4tp pos_f16x; do
  F="$OUT/$f.cmf"
  case $f in
    pos_q4tp) "$CORTIQ" convert --model "$TOY/pos" --output "$F" > "$OUT/$f.convert.log" 2>&1 ;;
    pos_f16x) "$CORTIQ" convert --model "$TOY/pos" --quant f16 --tensor-quant '*.mlp.experts.*=q4tp' \
                --output "$F" > "$OUT/$f.convert.log" 2>&1 ;;
  esac || { echo "$f: convert FAILED"; fail=1; continue; }
  run "$f.cpu" "$F" CMF_GPU=0 CMF_SDOT=0
  run "$f.tok" "$F" CMF_SDOT=0 CMF_BATCH_K=0 CMF_COOP=0
  run "$f.batch" "$F" CMF_SDOT=0 CMF_COOP=0
  run "$f.prefix" "$F" CMF_SDOT=0 CMF_COOP=0 CMF_GPU_VRAM_MB="${PREFIX_MB:-1}"
  run "$f.split" "$F" CMF_SDOT=0 CMF_COOP=0 CMF_GPU_VRAM_MB="${SPLIT_MB:-20}"
  run "$f.coop" "$F" CMF_SDOT=0
  run "$f.default" "$F"
  for arm in tok batch prefix split; do cmp "$f.cpu" "$f.$arm" 1e-3 || fail=1; done
  cmp "$f.cpu" "$f.coop" 1e-2 || true
  cmp "$f.cpu" "$f.default" 1e-2 || true
  for arm in tok batch prefix split coop default; do
    grep -ah "declined\|refused\|device prefix\|batch-prefix\|batched prefill\|whole-token graph" "$OUT/$f.$arm.log" \
      | sed 's/\x1b\[[0-9;]*m//g' | sed 's/^.*\(INFO\|WARN\|ERROR\) //' | sort -u | sed "s/^/  [$arm] /" | head -8
  done
done
[ $fail = 0 ] && echo "MIMO GPU TOY GATE: PASS" || echo "MIMO GPU TOY GATE: FAIL"
exit $fail
