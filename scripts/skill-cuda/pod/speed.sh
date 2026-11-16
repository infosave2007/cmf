#!/bin/bash
# speed.sh OUT.tsv "MODEL:SKILL ..." [REPS]: CUDA bench with and without each skill, interleaved reps:
# decode 128 tokens (--core --ignore-eos, MTP default and CMF_MTP=0) and a 4096-token prompt.
set -u
out=$1; cfgs=$2; reps=${3:-3}
B=${B:-/root/h3i/bin/cortiq-skill3}
cd /root/h3i/skill
echo -e "model\tskill\tmode\trep\tprefill_tok_s\tdecode_tok_s\tmtp_drafted\tmtp_accepted" > $out
one() { # model skill mode rep
  local m=$1 s=$2 mode=$3 r=$4 sk=() env=() ex=()
  [ "$s" != none ] && sk=(--skill $s)
  case $mode in
    decode) ex=(--tokens 128 --core --ignore-eos);;
    decode_nomtp) ex=(--tokens 128 --core --ignore-eos); env=(CMF_MTP=0);;
    ctx4096) ex=(--ctx 4096 --tokens 32 --core --ignore-eos); env=(CMF_MTP=0);;
  esac
  env CMF_GPU=cuda "${env[@]}" timeout 600 $B bench $m "${sk[@]}" "${ex[@]}" --json 2>/dev/null > /tmp/sp.json
  python3 - "$m" "$s" "$mode" "$r" >> $out <<'PY'
import json, sys
try:
    d = json.load(open("/tmp/sp.json"))
except Exception:
    d = {}
m, s, mode, r = sys.argv[1:]
print("\t".join([m.split("/")[-1], s, mode, r] + [f"{d.get(k, float('nan')):.1f}" if isinstance(d.get(k), (int, float)) else "nan"
      for k in ("prefill_tok_s", "decode_tok_s_steady", "mtp_drafted", "mtp_accepted")]))
PY
}
for r in $(seq 1 $reps); do
  for c in $cfgs; do
    m=${c%%:*}; s=${c#*:}
    for mode in decode decode_nomtp ctx4096; do
      one $m none $mode $r
      one $m $s $mode $r
    done
  done
done
cat $out
