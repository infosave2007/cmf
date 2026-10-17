#!/bin/bash
# par.sh: cortiq GPTQ-q2tp vs the q2tp study's torch arms on IDENTICAL Hessians (the study's FP-stream
# 32k gate/up Hessians, converted by hess2cmf.py). cortiq rule -> arm N1, placed rule -> arm W0.
# Only gate/up have Hessians in that cache, so every other linear is converter RTN (not compared).
# The study's N1/W0 arms have no output debias, so it is turned off here (the CLI default is on).
set -u
E=/root/eng_q2tp
B=$E/bin/cortiq
Q=/root/q4x/src/target/release/cortiq
export CMF_GPU=0 CMF_THREADS=4 RAYON_NUM_THREADS=4
HQ="--tensor-quant lm_head.weight=q8_2f --tensor-quant model.embed_tokens.weight=q8_2f"
for rule in ${RULES:-cortiq placed}; do
  tag=par_$rule
  ref=N1; [ "$rule" = placed ] && ref=W0
  echo "=== $(date -u +%FT%TZ) $tag" >> $E/logs/par.log
  python3 $E/scripts/timerun.py $B quantize-gptq /root/q4x/models/f16.cmf --codec q2tp --calib /root/q4x/data/wiki.train.raw \
    --hessians $E/q2x_cal.hess --q2tp-rule $rule --q2tp-debias off $HQ --threads 4 --output $E/models/$tag.cmf > $E/logs/$tag.log 2>&1
  echo "gptq rc=$? $(date -u +%FT%TZ)" >> $E/logs/par.log
  mkdir -p $E/dumps/$tag
  $Q dequant $E/models/$tag.cmf --name model.layers. --all --dtype raw --out $E/dumps/$tag 2> $E/dumps/$tag/listing.raw
  /root/q4x/venv/bin/python $E/scripts/compare_q2.py /root/q2x/arms/$ref $E/dumps/$tag $E/logs/cmp_$tag.json --hess \
    > $E/logs/cmp_$tag.log 2>&1
  echo "compare rc=$? $(date -u +%FT%TZ)" >> $E/logs/par.log
  rm -rf $E/dumps/$tag
done
