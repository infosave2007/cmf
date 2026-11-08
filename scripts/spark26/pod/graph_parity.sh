#!/bin/bash
# graph_parity.sh FILE CHARS TAG: greedy 64 tokens after a WikiText prompt of CHARS characters,
# the chunked host walk against the batched graph (scalar and coop GEMMs); prints the timing
# lines and whether the texts match. BIN= the binary (default ./cortiq.graph).
cd /root/spark26; export XDG_RUNTIME_DIR=/tmp CMF_MAX_SEQ=${CMF_MAX_SEQ:-140000}
F=$1; N=${2:-9000}; T=${3:-par}; B=${BIN:-./cortiq.graph}
venv/bin/python -c "t=open('data/wiki.test.raw').read(); open('logs/p_$T.txt','w').write(t[:$N]+'\n\nSummarize the text above.')"
run() { # run NAME ENV...
  local name=$1; shift
  env "$@" $B run $F --prompt-file logs/p_$T.txt --no-think --greedy -n 64 > logs/par_$T.$name.txt 2> logs/par_$T.$name.err
  echo "$name | $(tail -n 1 logs/par_$T.$name.txt | cut -c1-200)"
}
run walk CMF_DENSE_GRAPH_PREFILL=0
run graph CMF_DENSE_GRAPH_PREFILL=1 CMF_BATCH_DEBUG=1
run coop CMF_DENSE_GRAPH_PREFILL=1 CMF_BATCH_COOP=1
grep -i -E "batch graph|refus|declin" logs/par_$T.graph.err | sort | uniq -c | head -5
for v in graph coop; do
  if diff <(sed '$d' logs/par_$T.walk.txt) <(sed '$d' logs/par_$T.$v.txt) > /dev/null; then echo "$v: SAME_TEXT"; else echo "$v: DIFF_TEXT"; diff <(sed '$d' logs/par_$T.walk.txt) <(sed '$d' logs/par_$T.$v.txt) | head -6; fi
done
echo PARITY_DONE
