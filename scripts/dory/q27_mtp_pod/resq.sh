#!/bin/bash
# the trained-resonance queue: one run at a time, each under its own deadline
cd /root/q27
export SAVE=/root/q27/res
timeout 2400 python3 q27_res_train.py /root/q27 head 16384 0 6 > res-train-head-k0.log 2>&1
timeout 3000 python3 q27_res_train.py /root/q27 head 16384 4 6 > res-train-head-k4.log 2>&1
timeout 2400 python3 q27_res_train.py /root/q27 free 16384 0 8 > res-train-free-k0.log 2>&1
timeout 3000 python3 q27_res_train.py /root/q27 free 16384 4 8 > res-train-free-k4.log 2>&1
echo done > resq.done
