#!/bin/bash
# Quality of the three Apex files against the PyTorch bf16 reference:
# wikitext-2 test and Python standard-library code (first 4096 tokens each,
# `cortiq ppl` with CMF_PPL_TRACE for per-position NLL/top-1), the
# reference on the same ids, and the per-position comparison.
# Expects ids4096.txt + ref_ppl.npz (wiki) from the first pass.
set -u
cd /root/apex
B=${CORTIQ:-/root/apex/bin/cortiq}
export XDG_RUNTIME_DIR=/tmp
O=out/apex-35b-v1.0
echo "== wiki, q4tp-v2"
CMF_PPL_TRACE=1 "$B" ppl $O-q4tp-v2.cmf -f data/wiki.test.raw --tokens 4096 > trace_q4tp-v2.log 2>&1
grep -a "PPL =" trace_q4tp-v2.log
"$B" ppl $O-q4tp-v2.cmf -f data/wiki.test.raw --windows 12 --window-len 512 2>&1 | grep -a "PPL ="
for q in q8_2f q4tp q4tp-v2; do
  echo "== code, $q"
  CMF_PPL_IDS_OUT=/root/apex/ids_code.txt CMF_PPL_TRACE=1 "$B" ppl $O-$q.cmf -f data/pycode.txt --tokens 4096 \
    > trace_code_$q.log 2>&1
  grep -a "PPL =" trace_code_$q.log
done
echo "== reference, code"
python3 -I apex_ref.py /root/models/apex ppl ids_code.txt ref_code.npz 2>&1 | grep -a "PPL ="
echo "== per position, wiki"
python3 -I ppl_cmp.py ref_ppl.npz trace_q8_2f.log trace_q4tp.log trace_q4tp-v2.log
echo "== per position, code"
python3 -I ppl_cmp.py ref_code.npz trace_code_q8_2f.log trace_code_q4tp.log trace_code_q4tp-v2.log
echo EVAL_DONE
