#!/bin/bash
# chainS.sh: after the zi3 build (parallel per-group seal, xc2 attend tile) and the standard rows
# (cb_x_std_q4): the seal A/B on the turbo file at 16 000 tokens (zi2 against zi3: the "far field
# rebuilt" line and the prompt tok/s), greedy text parity zi2 against zi3 on 60 000 characters, the
# turbo card rows with zi3, the turbo stage profile — each GPU step under quiet.sh. Output
# logs/seal_ab.txt, logs/seal_par.txt, logs/cb_x_turbo_q4.txt, logs/gS_turbo_prof_16000.log, chainS.done.
cd /root/spark26; export XDG_RUNTIME_DIR=/tmp CMF_MAX_SEQ=140000
until [ -f logs/build_zi3.done ]; do sleep 10; done
[ -x zi3 ] && mv zi3 cortiq.zi3
[ -x cortiq.zi3 ] || { echo "no cortiq.zi3"; touch logs/chainS.done; exit 1; }
until grep -q CB_DONE logs/cb_x_std_q4.txt 2>/dev/null; do sleep 20; done
T=out/spark-x2.6-4b-q4tp-turbo.cmf
: > logs/seal_ab.txt
for B in cortiq.zi2 cortiq.zi3; do
  bash quiet.sh bash -c "./$B bench $T --core --ignore-eos --tokens 4 --ctx 16000 > logs/seal_ab.$B.log 2>&1"
  echo "$B | $(grep -E 'far field rebuilt|o1 prefix on|Prompt:' logs/seal_ab.$B.log | sed 's/.*INFO.*pipeline.*: //' | tr '\n' ' ' | tr -s ' ' | cut -c1-320)" >> logs/seal_ab.txt
done
venv/bin/python -c "t=open('data/wiki.test.raw').read(); open('logs/p_seal.txt','w').write(t[:60000]+'\n\nSummarize the text above.')"
for B in cortiq.zi2 cortiq.zi3; do
  bash quiet.sh bash -c "./$B run $T --prompt-file logs/p_seal.txt --no-think --greedy -n 48 > logs/seal_par.$B.txt 2> logs/seal_par.$B.err"
done
if diff <(sed '$d' logs/seal_par.cortiq.zi2.txt) <(sed '$d' logs/seal_par.cortiq.zi3.txt) > /dev/null; then
  echo "seal: SAME_TEXT" > logs/seal_par.txt
else
  { echo "seal: DIFF_TEXT"; diff <(sed '$d' logs/seal_par.cortiq.zi2.txt) <(sed '$d' logs/seal_par.cortiq.zi3.txt) | head -6; } > logs/seal_par.txt
fi
BIN=./cortiq.zi3 bash quiet.sh bash ctxbench.sh $T x_turbo_q4 1000 16000 32000 64000
bash quiet.sh bash -c "CMF_GPU_TS=1 CMF_BATCH_TS=1 CMF_PREFILL_PROF=1 ./cortiq.zi3 bench $T --core --ignore-eos --tokens 4 --ctx 16000 > logs/gS_turbo_prof_16000.log 2>&1"
touch logs/chainS.done
