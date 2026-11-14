#!/usr/bin/env bash
# TF32 against fp32 cuBLAS on the real recipe: 300 steps each from the same checkpoint, same seed.
cd /root/hc
EVAL=100 WARMUP=30 ./birth.sh ab-tf32 8 300 --dropless > run/ab-tf32.out 2>&1
rm -f run/ab-tf32.ckpt
EVAL=100 WARMUP=30 CMF_CUDA_TF32=0 ./birth.sh ab-fp32 8 300 --dropless > run/ab-fp32.out 2>&1
rm -f run/ab-fp32.ckpt
echo done > run/ab.done
