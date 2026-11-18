#!/bin/bash
# nsys.sh TAG: 10 steps of the real recipe under nsys (CUDA trace only), then the kernel summary;
# per-step numbers: python3 nsys_steps.py run/nsys-TAG.sqlite [N]
cd /root/hc/data
EMBRYO_LOG_EVERY=1 nsys profile -t cuda -s none --cpuctxsw=none -o /root/hc/run/nsys-$1 -f true /root/hc/cmf/target/release/cortiq-embryo birth --resume base1-step426200.ckpt \
  --shard en.u16:0.07 ru.u16:0.06 code.u16:0.02 math.u16:0.015 se.u16:0.07 docs.u16:0.03 b_hermes_code.u16:0.04 b_harness_code.u16:0.07 \
  b_ultrachat.u16:0.05 b_magpie.u16:0.06 b_chat_misc.u16:0.03 b_ot_answers.u16:0.03 b_multilingual.u16:0.04 b_multilingual_think.u16:0.05 \
  b_ot_think.u16:0.03 b_think_misc.u16:0.035 b_toolmind.u16:0.08 b_toucan.u16:0.08 b_nemotron.u16:0.045 b_tools_misc.u16:0.035 \
  b_lookup16.u16:0.04 b_q27_answers.u16:0.006 b_q27_chat.u16:0.007 b_q27_code.u16:0.007 --val rawval.u16 \
  --batch 8 --seq 1024 --steps 426210 --warmup 4 --lr 1e-4 --lr-restart --eval-every 1000000 --save-every 1000000 \
  --carry --carry-reset-every 16 --carry-eot 32760 --dropless --seed 31 --out /root/hc/run/nsys-$1.ckpt > /root/hc/run/nsys-$1.log 2>&1
rm -f /root/hc/run/nsys-$1.ckpt
nsys stats -r cuda_gpu_kern_sum -f csv -o /root/hc/run/nsys-$1 /root/hc/run/nsys-$1.nsys-rep > /dev/null 2>&1
echo done > /root/hc/run/nsys-$1.done
