#!/bin/bash
# mac_f1_check.sh DIR WORKTREE: the norm-fusion build (CMF_METAL_NORMFUSE) against the dense Metal
# decode it replaces — the greedy text of p2k with the fusion off and on (must agree), then the
# decode speed of both on `bench --ctx 1024` and on the p14k prompt, standard and turbo files.
# Waits for the release build and for the memory rows (one GPU job at a time).
D=$1; W=$2; cd "$D"
until grep -q BUILD_F1_RC build_f1.log && grep -q MEM_ROWS_DONE mem_rows.log; do sleep 10; done
grep -q "BUILD_F1_RC=0" build_f1.log || { echo "build failed"; tail -30 build_f1.log; exit 1; }
cp "$W/target/release/cortiq" cortiq.new && mv cortiq.new cortiq
M=hf/spark-x2.6-4b-q4tp.cmf; T=hf/spark-x2.6-4b-q4tp-turbo.cmf
for f in 0 1; do
  CMF_METAL_NORMFUSE=$f ./cortiq run $M --prompt-file p2k.txt --no-think --greedy -n 96 > f1_text_$f.out 2> f1_text_$f.err
  echo "normfuse=$f p2k: $(tail -1 f1_text_$f.out)"
done
if diff <(sed '$d' f1_text_0.out) <(sed '$d' f1_text_1.out) > /dev/null; then echo "TEXT PARITY: identical"; else echo "TEXT PARITY: differs"; diff <(sed '$d' f1_text_0.out) <(sed '$d' f1_text_1.out) | head -20; fi
for r in 1 2; do for f in 0 1; do
  echo "bench ctx1024 normfuse=$f: $(CMF_METAL_NORMFUSE=$f ./cortiq bench $M --ctx 1024 --tokens 128 --ignore-eos 2>/dev/null | grep -iE "decode|tok/s" | tr '\n' ' ')"
done; done
for f in 0 1; do
  echo "p14k std normfuse=$f: $(CMF_METAL_NORMFUSE=$f ./cortiq run $M --prompt-file p14k.txt --no-think --greedy -n 64 2>/dev/null | tail -1)"
  echo "p14k turbo normfuse=$f: $(CMF_METAL_NORMFUSE=$f ./cortiq run $T --prompt-file p14k.txt --no-think --greedy -n 64 2>/dev/null | tail -1)"
done
echo F1_CHECK_DONE
