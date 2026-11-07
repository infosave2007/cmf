#!/bin/bash
# One q4tp GPTQ variant (the chosen wiki+code calibration) with extra q8_2f tensors, then its checks:
# q4variant.sh TAG [--tensor-quant PATTERN=q8_2f ...]
set -eu
TAG=$1; shift
cd "$HOME/dev/cherrypit"
B=./bin/cortiq
OUTF=out/CherryPit-1.2-2B-q4tp-$TAG.cmf
nice -n 10 $B quantize-gptq out/CherryPit-1.2-2B-f16.cmf --codec q4tp --calib-model out/CherryPit-1.2-2B-q8_2f.cmf \
  ${CALIB:---calib data/wiki.train.raw=0.5 --calib data/code_calib.txt=0.5} --tokens ${TOKENS:-2048} --window 2048 --lambda 0.01 \
  --calib-chunk-layers 8 --tensor-quant lm_head.weight=q8_2f --tensor-quant model.embed_tokens.weight=q8_2f "$@" \
  --output "$OUTF" > "out/gptq_$TAG.log" 2>&1
echo "$TAG size $(stat -f %z "$OUTF")"
for t in "wiki $HOME/hf/wikitext2_test.txt" "code data/code_eval.txt" "py data/python_eval.txt"; do
  set -- $t
  echo "$TAG $1 $(nice -n 5 $B ppl "$OUTF" --file "$2" --windows 12 --window-len 512 2>/dev/null | grep PPL)"
done
scripts/toolstart.sh "$OUTF" "$TAG" 2 | tail -1
P="Build a React component with a search box and a list that loads on open."; n=0
for s in $(seq 1 20); do
  $B run "$OUTF" --no-think --temperature 1.0 --top-p 0.95 --top-k 0 --min-p 0 --rep-penalty 1.0 --seed $s -n 6 --prompt "$P" 2>/dev/null | sed -n '5,6p' | grep -q "<function" && n=$((n+1)) || true
done
echo "$TAG react: $n/20 tool-call starts"
