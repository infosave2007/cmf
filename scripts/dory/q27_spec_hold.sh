#!/bin/bash
# How deep a draft chain holds in real greedy rounds: q27_spec_hold.sh CORTIQ MODEL PROMPT_DIR K TOKENS [ENV…]
# Every p_*.txt of PROMPT_DIR is answered with drafts pinned K deep, rounds kept on whatever they cost
# (CMF_GRAPH_SPEC_PIN=1; pass CMF_GRAPH_SPEC_PIN=0 for the trial's own choice) and no prompt lookup; a round's line says
# how many of its K drafts the verify accepted. Prints per prompt and in total the share of rounds where draft
# j held among those where draft j − 1 did, and the hash of the text alone, without the lines that name the file
# and the speed (the same with any draft head).
bin=$1; model=$2; dir=$3; k=$4; n=$5; shift 5
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/tmp}
tmp=$(mktemp -d)
for p in "$dir"/p_*.txt; do
  name=$(basename "$p" .txt)
  env CMF_GRAPH_SPEC=1 CMF_GRAPH_SPEC_K="$k" CMF_GRAPH_SPEC_PIN=1 CMF_SPEC_LOOKUP=0 CMF_GRAPH_SPEC_TIME=1 "$@" \
    "$bin" run "$model" --prompt "$(cat "$p")" --greedy --no-think --max-tokens "$n" > "$tmp/$name.out" 2> "$tmp/$name.err"
  echo "$name $(grep -vE '^\[|^Loading model' "$tmp/$name.out" | shasum | cut -c1-12) $(grep -o 'accepted [0-9]* of [0-9]*' "$tmp/$name.err" | awk '{print $2}' | tr '\n' ' ')"
done | python3 -c '
import sys
k = int(sys.argv[1]); tot = [0] * (k + 1); rounds = 0
def line(name, acc, extra=""):
    ge = [sum(a >= j for a in acc) for j in range(k + 1)]
    rates = " / ".join(f"{ge[j] / max(ge[j - 1], 1):.3f}" for j in range(1, k + 1))
    print(f"{name:12s} rounds {len(acc):4d}  holds {rates}  tokens a round {1 + sum(acc) / max(len(acc), 1):.2f}  {extra}")
allacc, groups = [], {}
for l in sys.stdin:
    f = l.split()
    acc = [min(int(x), k) for x in f[2:]]
    allacc += acc
    groups.setdefault(f[0].rstrip("0123456789"), []).append(acc)
    line(f[0], acc, f[1])
for g, accs in groups.items():  # p_en01, p_en02, … together as p_en
    if len(accs) > 1:
        line(g, [a for acc in accs for a in acc], f"{len(accs)} prompts")
line("all", allacc)
' "$k"
rm -rf "$tmp"
