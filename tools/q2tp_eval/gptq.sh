#!/bin/bash
# gptq.sh TAG [extra quantize-gptq args...]: one MiniCPM5-2B quantize-gptq run on the 5090 pod (CPU),
# 32k WikiText-2 train tokens in 2048-token windows, q8_2f calibration model, head/embed q8_2f,
# Hessian cache hess/H32k.hess (2 calibration passes of 21 layers when it is not there yet).
# Peak RSS and wall time from /usr/bin/time -v. Serialised by our own lock.
set -u
W=/root/cmfbuild/q2tp_work
L=/root/cmfbuild/q2tp_logs
TAG=$1; shift
export CMF_GPU=0 CMF_THREADS=${THREADS:-13} RAYON_NUM_THREADS=${THREADS:-13}
mkdir -p $W/hess $W/out
exec 9> $W/gptq.lock
flock 9
cd $W
CAL=()
for c in ${CALIB:-data/wiki.train.raw}; do CAL+=(--calib "$c"); done
echo "=== $(date -u +%FT%TZ) $TAG ${CAL[*]} $*" >> $L/gptq_runs.log
python3 /root/cmfbuild/q2tp_scripts/timerun.py ${BIN:-./bin/cortiq} quantize-gptq f16.cmf --calib-model q8_2f.cmf "${CAL[@]}" \
  --tokens ${TOKENS:-32768} --window 2048 --threads ${THREADS:-13} \
  --tensor-quant lm_head.weight=q8_2f --tensor-quant model.embed_tokens.weight=q8_2f \
  --hessians ${HESS:-hess/H32k.hess} --calib-chunk-layers ${CHUNK:-21} \
  "$@" --output out/$TAG.cmf > $L/gptq_$TAG.log 2>&1
rc=$?
echo "rc=$rc $(date -u +%FT%TZ) $(grep -E 'Maximum resident|Elapsed' $L/gptq_$TAG.log | tr -s ' ' | tr '\n' ' ')" >> $L/gptq_runs.log
[ $rc = 0 ] && sha256sum out/$TAG.cmf >> $L/gptq_runs.log
