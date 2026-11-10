#!/bin/bash
# dl2.sh: the private fcaq256 and Strata Coder IQ1_M (setup2's first try got no token). HF_TOKEN on stdin.
cd /root/fc; read -r HF_TOKEN; export HF_TOKEN
HF=/root/fc/venv/bin/hf
( cd out && $HF download infosave/Qwen3.8-Flash-Next-Fast-Code-cmf qwen38-flash-next-fast-code.cmf fast-code.profile qwen38-flash-next-fast-code.cmf.sha256 --local-dir . ; echo "DL_FC_DONE rc=$?" ) > logs/dl_fc.log 2>&1 &
( cd strata && $HF download ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF --include "IQ1_M/*" --include "tensor-allocation/*" --local-dir . ; echo "DL_ST_DONE rc=$?" ) > logs/dl_strata.log 2>&1 &
wait
echo DL2_DONE
