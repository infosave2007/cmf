#!/bin/bash
# tinyspec.sh [FILE] [ENV=VALUE ...]: the verify and the rewind of a speculative round on the small
# random model (default tiny-q4tp.cmf), without a drafter: the plain greedy sequence is written out
# (CMF_GEN_IDS_OUT) and read back as the drafts (CMF_SPEC_SCRIPT), whole and with every seventh id
# changed, at a fixed depth and at the loop's own. Every run must give the plain run's ids.
set -uo pipefail
cd /root/rip
export XDG_RUNTIME_DIR=/tmp
f=${1:-tiny-q4tp.cmf}; shift 2>/dev/null || true
head -c ${TINY_CHARS:-2400} data/wiki.test.raw | tail -c $(( ${TINY_CHARS:-2400} - 400 )) > /tmp/tinyspec_prompt.txt
gen() { # name, env...
  local n=$1; shift
  env "$@" CMF_GRAPH_SPEC_TIME=1 CMF_GEN_IDS_OUT=/tmp/tinyspec_$n.ids ./cortiq run $f --prompt "$(cat /tmp/tinyspec_prompt.txt)" --greedy --max-tokens ${TINY_TOKENS:-96} > logs/tinyspec_$n.out 2> logs/tinyspec_$n.err
  echo "$n: $(wc -w < /tmp/tinyspec_$n.ids) ids | $(grep -a -c 'solo-round:' logs/tinyspec_$n.err) rounds | kept $(grep -a 'solo-round:' logs/tinyspec_$n.err | awk '{k += $(NF-2); d += $NF} END {print k+0 " of " d+0}') | $(grep -a -i -c 'declin\|refus\|failed' logs/tinyspec_$n.err) decline or failure lines"
}
gen plain CMF_GRAPH_SPEC=0 "$@"
python3 -I - <<'PY'
ids = open("/tmp/tinyspec_plain.ids").read().split()
open("/tmp/tinyspec_script.ids", "w").write(" ".join(ids))
bad = [str((int(t) + 1) % 1000) if i % 7 == 3 else t for i, t in enumerate(ids)]
open("/tmp/tinyspec_script_bad.ids", "w").write(" ".join(bad))
PY
gen all CMF_SPEC_SCRIPT=/tmp/tinyspec_script.ids "$@"
gen k4 CMF_SPEC_SCRIPT=/tmp/tinyspec_script.ids CMF_GRAPH_SPEC_K=4 "$@"
gen bad CMF_SPEC_SCRIPT=/tmp/tinyspec_script_bad.ids CMF_GRAPH_SPEC_K=5 "$@"
gen bad15 CMF_SPEC_SCRIPT=/tmp/tinyspec_script_bad.ids CMF_GRAPH_SPEC_K=15 "$@"
for n in all k4 bad bad15; do cmp -s /tmp/tinyspec_plain.ids /tmp/tinyspec_$n.ids && echo "$n: the plain run's ids" || echo "$n: DIFFERENT ids"; done
echo TINYSPEC_DONE
