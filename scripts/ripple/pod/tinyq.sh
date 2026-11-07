#!/bin/bash
# tinyq.sh [QUANT] [TOKENS]: the small random model in a codec the graphs take (default q4tp), scored
# over the same tokens: host chunks, host tokens, device chunks (the batch graph; chunks of 32 fit the
# small model's ring, so its sliding layers take the token-axis arm too), device tokens (the token graph). Prints each pass's ppl and its distance from the host chunks, position
# by position. The f16 file against PyTorch is tiny.sh; this one is device against host.
set -uo pipefail
cd /root/rip
export XDG_RUNTIME_DIR=/tmp
q=${1:-q4tp}; t=${2:-1400}
./cortiq convert --model tiny --output tiny-$q.cmf --quant $q 2>&1 | grep -a -E "rror|wrote" | tail -2
run() { # name, env...
  local n=$1; shift
  env "$@" CMF_PPL_TRACE=1 CMF_GRAPH_DEBUG=1 CMF_BATCH_DEBUG=1 ./cortiq ppl tiny-$q.cmf -f data/wiki.test.raw --tokens $t > logs/tinyq_$n.log 2>&1
  echo "$n: $(grep -a 'PPL =' logs/tinyq_$n.log | tail -1) | $(grep -a -c -i 'declin\|refus' logs/tinyq_$n.log) decline lines"
}
run host_chunks CMF_GPU=0
run host_tokens CMF_GPU=0 CMF_NLL_SERIAL=1
run dev_chunks
run dev_chunks32 CMF_PREFILL_CHUNK=32
run dev_tokens CMF_NLL_SERIAL=1
python3 -I - <<'PY'
import re
def load(n):
    d = {}
    for l in open(f"logs/tinyq_{n}.log", errors="ignore"):
        m = re.search(r"pos\s+(\d+)\s+(?:target|tgt)\s+(\d+)\s+nll\s+([-\d.]+)\s+(?:\|\s+)?top1?\s+(\d+)", l)
        if m:
            d[int(m.group(1))] = (float(m.group(3)), int(m.group(4)))
    return d
for n, base in (("host_tokens", "host_chunks"), ("dev_chunks", "host_chunks"), ("dev_chunks32", "host_chunks"), ("dev_tokens", "host_tokens")):
    ref, d = load(base), load(n)
    ks = sorted(set(ref) & set(d))
    if not ks:
        print(f"{n} vs {base}: no positions"); continue
    dn = [abs(ref[k][0] - d[k][0]) for k in ks]
    agree = sum(ref[k][1] == d[k][1] for k in ks) / len(ks)
    print(f"{n} vs {base}: {len(ks)} positions | top-1 agree {100*agree:.2f} % | mean |dNLL| {sum(dn)/len(dn):.5f} | max {max(dn):.5f}")
PY
