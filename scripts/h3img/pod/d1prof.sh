#!/bin/bash
# d1-Omni system-one with a clip: the forward's own clock (CMF_D1_DEBUG) against
# the whole call, CUDA / CPU / Vulkan, two runs each, after the NVG round (its
# builds share the CPU).
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
end=$(( $(date +%s) + 7200 ))
until grep -q -E 'NVG_DONE|FAILED|TIMEOUT' logs/nvg_cuda.out 2>/dev/null; do
  [ "$(date +%s)" -gt $end ] && { echo NVG_WAIT_TIMEOUT; exit 1; }
  sleep 30
done
mkdir -p cmf/dl logs/d1p
f=cmf/dl/d1-omni-2b-q4tp.cmf
[ -s $f ] || timeout 3600 venv/bin/hf download infosave/d1-Omni-2B-cmf d1-omni-2b-q4tp.cmf --local-dir cmf/dl > logs/d1p/dl.log 2>&1
clip=/root/qt/mac_g/g_q8_2f_cpux_en.wav
for be in cuda 0 1; do
  for i in 1 2; do
    flock gpu.lock env CMF_GPU=$be CMF_D1_DEBUG=1 CMF_CUDA_PROF=1 CMF_CUDA_DEBUG=1 timeout 1200 bin/cortiq system-one $f --state "A customer voice message." --questions '{"animal": {"type": "noul", "instructions": "Does the clip mention an animal?"}}' --audio $clip > logs/d1p/s1_${be}_$i.log 2>&1
    echo "$be run$i: $(grep -h -oE 'system-one: [0-9]+ ms|d1: forward [^\n]*ms' logs/d1p/s1_${be}_$i.log | tr '\n' ' ')"
  done
  grep -h -E "cuda h3 product|built in|cuda hybrid graph|  (gemm|copy up|rows down|host planes)" logs/d1p/s1_${be}_2.log | sed -E 's/\x1b\[[0-9;]*m//g; s/.*INFO [^ ]* //' | head -6 | sed 's/^/    /'
done
rm -f $f
echo D1P_DONE
