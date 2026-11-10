#!/bin/bash
# One `cortiq bench --core --json` run with a VRAM poller beside it.
#   bench_one.sh <tag> <binary> <model.cmf> <ctx> [ENV=VALUE ...]
# Writes out/<tag>.json (the bench object), out/<tag>.log (stderr),
# out/<tag>.vram (peak MiB of the card while the process ran) and
# out/<tag>.vram_trace (every sample). BENCH_TOKENS=<n> (environment): tokens
# to decode instead of 60; with a decode of four seconds or more the summary's
# `vram_decode` is what the card holds while it decodes (the median of the
# samples of the decode's second half).
set -u
tag=$1; bin=$2; model=$3; ctx=$4; shift 4
root=${CUDA_RUN_ROOT:-/root/cuda}
out=$root/out; mkdir -p "$out"
export XDG_RUNTIME_DIR=/tmp CMF_GPU_PROBE=0
for kv in "$@"; do export "$kv"; done
: > "$out/$tag.vram_trace"
( peak=0
  while :; do
    m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ -n "$m" ] && echo "$m" >> "$out/$tag.vram_trace"
    [ -n "$m" ] && [ "$m" -gt "$peak" ] && peak=$m && echo "$peak" > "$out/$tag.vram"
    sleep 0.2
  done ) &
poll=$!
"$bin" bench "$model" --ctx "$ctx" --tokens "${BENCH_TOKENS:-60}" --core --json > "$out/$tag.json" 2> "$out/$tag.log"
rc=$?
kill $poll 2>/dev/null; wait $poll 2>/dev/null
python3 - "$out/$tag.json" "$out/$tag.vram" "$tag" "$rc" "${BENCH_TOKENS:-60}" <<'PY'
import json, sys
path, vram, tag, rc = sys.argv[1:5]
try:
    txt = open(path).read()
    j = json.loads(txt[txt.index("{"):txt.rindex("}") + 1])
    peak = open(vram).read().strip()
    tr = [int(x) for x in open(vram + "_trace").read().split()]
    # samples are about 0.25 s apart; the last ones see the sequence cleared
    dec_s = float(sys.argv[5]) / max(j['decode_tok_s_steady'], 1e-6)
    n = int(dec_s / 0.25)
    mid = sorted(tr[-3 - n // 2:-3]) if dec_s >= 4 and len(tr) > n + 3 else []
    dec = f"{mid[len(mid) // 2]} MiB" if mid else "-"
    print(f"{tag}: rc={rc} prompt={j['prompt_tokens']} prefill={j['prefill_tok_s']:.1f} tok/s "
          f"ttft={j['ttft_s']:.2f} s decode={j['decode_tok_s_steady']:.1f} tok/s "
          f"vram_peak={peak} MiB vram_decode={dec} "
          f"graph_full={j.get('graph_full_tokens')} miss={j.get('graph_miss_tokens')} "
          f"path={j.get('prefill_measurement')}")
except Exception as e:
    print(f"{tag}: rc={rc} FAILED to parse ({e})")
PY
