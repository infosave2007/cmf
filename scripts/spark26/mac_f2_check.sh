#!/bin/bash
# mac_f2_check.sh DIR WORKTREE: the fused q|k|v + RoPE + append + gate build (CMF_METAL_QKVFUSE)
# against the dense Metal decode it replaces — the greedy text of p2k with the fusion off and on,
# then the decode speed of both on `bench --ctx 1024` (two rounds) and on the p14k prompt,
# standard and turbo files. Waits for the release build (one GPU job at a time).
D=$1; W=$2; cd "$D"
until grep -q BUILD_F2_RC build_f2.log; do sleep 10; done
grep -q "BUILD_F2_RC=0" build_f2.log || { echo "build failed"; grep -E "^error" -A6 build_f2.log | head -40; exit 1; }
cp "$W/target/release/cortiq" cortiq.new && mv cortiq.new cortiq
M=hf/spark-x2.6-4b-q4tp.cmf; T=hf/spark-x2.6-4b-q4tp-turbo.cmf
for f in 0 1; do
  CMF_METAL_QKVFUSE=$f ./cortiq run $M --prompt-file p2k.txt --no-think --greedy -n 96 > f2_text_$f.out 2> f2_text_$f.err
  echo "qkvfuse=$f p2k: $(tail -1 f2_text_$f.out)"
  grep -iE "error|panic|refus" f2_text_$f.err | head -3
done
if diff <(sed '$d' f2_text_0.out) <(sed '$d' f2_text_1.out) > /dev/null; then echo "TEXT PARITY: identical"; else echo "TEXT PARITY: differs"; diff <(sed '$d' f2_text_0.out) <(sed '$d' f2_text_1.out) | head -20; fi
for r in 1 2; do for f in 0 1; do
  echo "bench ctx1024 qkvfuse=$f: $(CMF_METAL_QKVFUSE=$f ./cortiq bench $M --ctx 1024 --tokens 128 --ignore-eos 2>/dev/null | grep -iE "decode|tok/s" | tr '\n' ' ')"
done; done
for f in 0 1; do
  echo "p14k std qkvfuse=$f: $(CMF_METAL_QKVFUSE=$f ./cortiq run $M --prompt-file p14k.txt --no-think --greedy -n 64 2>/dev/null | tail -1)"
  echo "p14k turbo qkvfuse=$f: $(CMF_METAL_QKVFUSE=$f ./cortiq run $T --prompt-file p14k.txt --no-think --greedy -n 64 2>/dev/null | tail -1)"
done
echo F2_CHECK_DONE
