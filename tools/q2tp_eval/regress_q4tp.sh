#!/bin/bash
# regress_q4tp.sh: the reworked `quantize-gptq --codec q4tp` path against the q4tp study's A1 file — same
# command, same Hessian cache (A1.hess, read lazily now, read-only), 3 threads instead of 24. The output must
# be byte-identical to /root/q4x/models/A1_gptq.cmf (written by the master build's old runner).
set -u
E=/root/eng_q2tp
M=/root/q4x/models
export CMF_GPU=0 CMF_THREADS=3 RAYON_NUM_THREADS=3
echo "=== $(date -u +%FT%TZ) regress_q4tp" >> $E/logs/regress.log
python3 $E/scripts/timerun.py $E/bin/cortiq quantize-gptq $M/f16.cmf --codec q4tp --calib-model $M/q8_2f.cmf \
  --calib /root/q4x/data/wiki.train.raw --tokens 2048 --window 2048 --lambda 0.01 \
  --tensor-quant lm_head.weight=q8_2f --tensor-quant model.embed_tokens.weight=q8_2f --threads 3 \
  --hessians $M/A1.hess --output $E/models/A1_regress.cmf > $E/logs/regress_q4tp.log 2>&1
echo "rc=$? $(date -u +%FT%TZ)" >> $E/logs/regress.log
sha256sum $E/models/A1_regress.cmf $M/A1_gptq.cmf >> $E/logs/regress.log
cmp $E/models/A1_regress.cmf $M/A1_gptq.cmf >> $E/logs/regress.log 2>&1 && echo "BYTE-IDENTICAL" >> $E/logs/regress.log
