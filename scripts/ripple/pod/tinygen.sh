#!/bin/bash
# tinygen.sh [QUANT] [ENV=VALUE ...]: the small random model's greedy continuation of a long prompt,
# host against device (the prompt through the batch graph, the answer through the token graph).
# Prints the two token streams' agreement. Extra arguments are the device run's environment.
set -uo pipefail
cd /root/rip
export XDG_RUNTIME_DIR=/tmp
q=${1:-q4tp}; shift 2>/dev/null || true
[ -f tiny-$q.cmf ] || ./cortiq convert --model tiny --output tiny-$q.cmf --quant $q 2>&1 | grep -a -E "rror|wrote" | tail -2
head -c 2400 data/wiki.test.raw | tail -c 2000 > /tmp/tiny_prompt.txt
gen() { # name, env...
  local n=$1; shift
  env "$@" CMF_PREFILL_PROF=1 CMF_GRAPH_PROF=1 ./cortiq run tiny-$q.cmf --prompt "$(cat /tmp/tiny_prompt.txt)" --greedy --max-tokens 64 > logs/tinygen_$n.out 2> logs/tinygen_$n.err
  echo "$n: $(wc -c < logs/tinygen_$n.out) bytes | $(grep -a -c 'batch-chunk' logs/tinygen_$n.err) batch chunks | $(grep -a -i -c 'declin\|refus' logs/tinygen_$n.err) decline lines"
}
gen host CMF_GPU=0
gen dev "$@"
python3 -I - <<'PY'
a = open("logs/tinygen_host.out", "rb").read().split(b"\n[")[0]
b = open("logs/tinygen_dev.out", "rb").read().split(b"\n[")[0]
n = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
print(f"answers: host {len(a)} bytes, device {len(b)} bytes, same up to byte {n}{' (identical)' if a == b else ''}")
PY
grep -a -i "declin\|refus\|batch prefill" logs/tinygen_dev.err | cut -c1-200 | head -6
