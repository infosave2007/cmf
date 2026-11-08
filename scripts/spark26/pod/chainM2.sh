#!/bin/bash
# chainM2.sh: the merged tree after fitting #99's 128-row cooperative tile to the grouped tile walk.
# Builds cortiq.zi5 (cores 8-47); the device test of the 128-row tile against the 64-row one; then
# under quiet.sh the prompt A/B zi3 vs zi5 at 4096/16000 and greedy text parity on 9 000 characters.
# Output logs/coop128_test.log, logs/gab_merge2.txt, logs/par_merge2.txt, chainM2.done.
cd /root/spark26; export XDG_RUNTIME_DIR=/tmp CMF_MAX_SEQ=140000
rm -f logs/chainM2.done
bash build_to.sh zi5 > logs/build_zi5.out 2>&1
[ -x zi5 ] && mv zi5 cortiq.zi5
. "$HOME/.cargo/env"; export CARGO_TARGET_DIR=/root/spark26/target
(cd cmf && CMF_GPU=wgpu CARGO_PROFILE_RELEASE_LTO=off CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 taskset -c 8-47 cargo test -p cortiq-engine --features gpu --release --lib -- coop128 --test-threads 1 > /root/spark26/logs/coop128_test.log 2>&1)
F=hf/cmf/spark-x2.6-4b-q4tp.cmf
: > logs/gab_merge2.txt
for n in 4096 16000; do
  for B in cortiq.zi3 cortiq.zi5; do
    bash quiet.sh bash -c "./$B bench $F --core --ignore-eos --tokens 16 --ctx $n > logs/gab_merge2.$B.$n.log 2>&1"
    echo "$n | $B | $(grep -E "Prompt:|Decode:|panicked" logs/gab_merge2.$B.$n.log | tr '\n' ' ' | tr -s ' ')" >> logs/gab_merge2.txt
  done
done
for B in cortiq.zi5; do
  bash quiet.sh bash -c "./$B run $F --prompt-file logs/p_merge.txt --no-think --greedy -n 48 > logs/par_merge.$B.txt 2> logs/par_merge.$B.err"
done
if diff <(sed '$d' logs/par_merge.cortiq.zi3.txt) <(sed '$d' logs/par_merge.cortiq.zi5.txt) > /dev/null; then
  echo "merge: SAME_TEXT" > logs/par_merge2.txt
else
  { echo "merge: DIFF_TEXT"; diff <(sed '$d' logs/par_merge.cortiq.zi3.txt) <(sed '$d' logs/par_merge.cortiq.zi5.txt) | head -6; } > logs/par_merge2.txt
fi
touch logs/chainM2.done
