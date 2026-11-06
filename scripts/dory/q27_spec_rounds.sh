#!/bin/bash
# What a round of K drafts costs: q27_spec_rounds.sh CORTIQ MODEL PROMPT TOKENS K [ENV=VAL ...]
# Medians of the draft chain, the verify, the commit and the round, tokens a round, and the text's hash.
cortiq=$1; model=$2; f=$3; n=$4; k=$5; shift 5
t=$(mktemp -d)
env "$@" CMF_GRAPH_SPEC=1 CMF_GRAPH_SPEC_K="$k" CMF_GRAPH_SPEC_TIME=1 RUST_LOG=info "$cortiq" run "$model" --prompt-file "$f" --greedy --no-think --max-tokens "$n" > "$t/out" 2> "$t/err"
python3 - "$k" "$*" "$t" <<'PY'
import re,sys,statistics as st,hashlib
e=open(sys.argv[3]+'/err',errors='replace').read()
d=[tuple(map(float,m)) for m in re.findall(r"spec-round: draft ([\d.]+) ms/\d+ sub \| verify ([\d.]+) ms/\d+ sub \| commit ([\d.]+) ms",e)]
w=[(float(a),int(b)) for a,b in re.findall(r"spec-round wall ([\d.]+) ms → (\d+) tokens",e)]
o=open(sys.argv[3]+'/out',errors='replace').read()
stat=" ".join(l for l in o.splitlines() if l.startswith('['))
txt="\n".join(l for l in o.splitlines() if not l.startswith('['))
med=lambda v: st.median(v) if v else 0
print(f"k={sys.argv[1]} {sys.argv[2]}: rounds {len(w)} draft {med([x[0] for x in d[2:]]):.1f} verify {med([x[1] for x in d[2:]]):.1f} commit {med([x[2] for x in d[2:]]):.1f} wall {med([x[0] for x in w[2:]]):.1f} tok/round {sum(x[1] for x in w)/max(1,len(w)):.2f} | {stat[:150]} | text {hashlib.md5(txt.encode()).hexdigest()[:8]}")
PY
rm -rf "$t"
