#!/bin/bash
# tinyhot.sh [FRAC]: the small random model as q2tp with a third of each layer's neurons in a q8_2f
# tube (hot-probe, hot-bake), scored over the same tokens on the host and on the device: chunks (the
# batch graph), chunks of 32, tokens (the token graph). Position by position against the host.
set -uo pipefail
cd /root/rip
export XDG_RUNTIME_DIR=/tmp
f=${1:-0.33}; t=1400
for q in q2tp q8_2f; do ./cortiq convert --model tiny --output tiny-$q.cmf --quant $q 2>&1 | grep -a -E "rror" | tail -2; done
CMF_GPU=0 ./cortiq hot-probe tiny-q2tp.cmf data/wiki.test.raw hot/tiny.mass --tokens 1024 --window 512 --skip 20000 2>&1 | grep -a "share\|rror" | cut -c1-200
./cortiq hot-bake tiny-q2tp.cmf tiny-q8_2f.cmf hot/tiny.mass tiny-hot.cmf --frac $f 2>&1 | grep -a "hot-bake: [0-9]* layers,\|rror" | cut -c1-200
run() { # name, env...
  local n=$1; shift
  env "$@" CMF_PPL_TRACE=1 CMF_GRAPH_DEBUG=1 CMF_BATCH_DEBUG=1 ./cortiq ppl tiny-hot.cmf -f data/wiki.test.raw --tokens $t > logs/tinyhot_$n.log 2>&1
  echo "$n: $(grep -a 'PPL =' logs/tinyhot_$n.log | tail -1) | $(grep -a -c -i 'declin\|refus' logs/tinyhot_$n.log) decline lines"
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
    for l in open(f"logs/tinyhot_{n}.log", errors="ignore"):
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
echo TINYHOT_DONE
