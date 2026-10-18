#!/bin/bash
# det.sh: determinism and calibration-plumbing checks on a short budget (4096 tokens = 2 windows), placed rule.
#   detA  calibrate in one pass (chunk 0)            -> hess/H4k_a.hess
#   detB  calibrate in 21-layer passes (chunk 21)     -> hess/H4k_b.hess.L0-20, .L21-41
#   detC  calibrate again in one pass                 -> hess/H4k_c.hess (must equal H4k_a byte for byte)
#   detD  GPTQ from H4k_a with 5 threads instead of 13
#   detM  mixed calibration: 75 % WikiText-2 train + 25 % repo docs (a functional check; differs by design)
# A, B, C, D must be byte-identical files. down_proj is left to the converter (CMF_GPTQ_SKIP) to keep each
# GPTQ phase short: its 6144-wide folds are 90 % of the GPTQ compute and exercise nothing the others don't.
set -u
export CMF_GPTQ_SKIP=embed_tokens,lm_head,down_proj
W=/root/cmfbuild/q2tp_work
S=/root/cmfbuild/q2tp_scripts
L=/root/cmfbuild/q2tp_logs
cat /root/cmfbuild/q2tp/docs/*.md > $W/data/docs.txt
TOKENS=4096 CHUNK=0  HESS=hess/H4k_a.hess bash $S/gptq.sh detA --codec q2tp
TOKENS=4096 CHUNK=21 HESS=hess/H4k_b.hess bash $S/gptq.sh detB --codec q2tp
TOKENS=4096 CHUNK=0  HESS=hess/H4k_c.hess bash $S/gptq.sh detC --codec q2tp
TOKENS=4096 CHUNK=0  HESS=hess/H4k_a.hess THREADS=5 bash $S/gptq.sh detD --codec q2tp
TOKENS=4096 CHUNK=0  HESS=hess/H4k_m.hess CALIB="data/wiki.train.raw=0.75 data/docs.txt=0.25" bash $S/gptq.sh detM --codec q2tp
cd $W
{ echo "=== det $(date -u +%FT%TZ)"; sha256sum out/detA.cmf out/detB.cmf out/detC.cmf out/detD.cmf out/detM.cmf hess/H4k_a.hess hess/H4k_c.hess; } >> $L/det.log
