#!/bin/bash
# mac_f6_check.sh DIR WORKTREE: the one-row far-field path (`o1c_tok_*`, CMF_O1C_TOK) against the
# eleven-stage path it replaces, on the turbo file — the greedy text of p14k with the path off and
# on (the sums run in another order, so the texts may part late; the first tokens must agree),
# then the decode speed of both on p2k and p14k, and the standard file once as the reference.
# Runs after the F2 check and the F6 build (one GPU job at a time).
D=$1; W=$2; cd "$D"
until grep -q F2_CHECK_DONE f2_check.log; do sleep 10; done
cd "$W" && cargo check -p cortiq-engine --features gpu > "$D/check_f6.log" 2>&1; echo CHECK_RC=$? >> "$D/check_f6.log"
grep -q CHECK_RC=0 "$D/check_f6.log" || { echo "check failed"; grep -E "^error" -A8 "$D/check_f6.log" | head -60; exit 1; }
cargo build --release -p cortiq-cli --features gpu > "$D/build_f6.log" 2>&1; echo BUILD_F6_RC=$? >> "$D/build_f6.log"
cd "$D"
grep -q "BUILD_F6_RC=0" build_f6.log || { echo "build failed"; grep -E "^error" -A6 build_f6.log | head -40; exit 1; }
cp "$W/target/release/cortiq" cortiq.new && mv cortiq.new cortiq
M=hf/spark-x2.6-4b-q4tp.cmf; T=hf/spark-x2.6-4b-q4tp-turbo.cmf
for f in 0 1; do
  CMF_O1C_TOK=$f ./cortiq run $T --prompt-file p14k.txt --no-think --greedy -n 96 > f6_text_$f.out 2> f6_text_$f.err
  echo "o1c_tok=$f turbo p14k: $(tail -1 f6_text_$f.out)"
  grep -iE "error|panic|refus" f6_text_$f.err | head -3
done
if diff <(sed '$d' f6_text_0.out) <(sed '$d' f6_text_1.out) > /dev/null; then echo "TEXT PARITY: identical"; else echo "TEXT PARITY: differs"; diff <(sed '$d' f6_text_0.out) <(sed '$d' f6_text_1.out) | head -20; fi
for f in 0 1; do
  echo "turbo p2k o1c_tok=$f: $(CMF_O1C_TOK=$f ./cortiq run $T --prompt-file p2k.txt --no-think --greedy -n 96 2>/dev/null | tail -1)"
done
echo "std p14k: $(./cortiq run $M --prompt-file p14k.txt --no-think --greedy -n 64 2>/dev/null | tail -1)"
echo F6_CHECK_DONE
