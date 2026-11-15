#!/bin/bash
# d1's set forward on CUDA (bin/cortiq-fast) against the CPU: d1-Omni with a
# clip and d1-2B (q4tp, q8_2f) with the card's question, two runs a backend,
# the answers and the forward's clock (CMF_D1_DEBUG).
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
end=$(( $(date +%s) + 3600 ))
until grep -q -E 'BUILD_FAST_DONE|BUILD_FAST_FAILED' logs/${D1_BUILD_LOG:-build_d1.log} 2>/dev/null; do
  [ "$(date +%s)" -gt $end ] && { echo BUILD_WAIT_TIMEOUT; exit 1; }
  sleep 20
done
grep -q BUILD_FAST_DONE logs/${D1_BUILD_LOG:-build_d1.log} || { echo BUILD_FAILED; tail -40 logs/${D1_BUILD_LOG:-build_d1.log}; exit 1; }
mkdir -p cmf/dl logs/d1c
clip=/root/qt/mac_g/g_q8_2f_cpux_en.wav
q2='{"refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"}, "team": {"type": "choice", "instructions": "Which team should handle this?", "criteria": {"billing": "Charges, refunds, invoices", "shipping": "Delivery and parcels", "technical": "App or site faults"}}}'
while read -r repo file kind; do
  [ -z "$repo" ] && continue
  f=cmf/dl/$file
  [ -s $f ] || timeout 3600 venv/bin/hf download $repo $file --local-dir cmf/dl > logs/d1c/${file%.cmf}_dl.log 2>&1
  for be in cuda 0; do
    for i in 1 2; do
      log=logs/d1c/${file%.cmf}_${be}_$i.log
      if [ $kind = omni ]; then
        flock gpu.lock env CMF_GPU=$be CMF_D1_DEBUG=1 CMF_CUDA_DEBUG=1 timeout 900 bin/cortiq-fast system-one $f --state "A customer voice message." --questions '{"animal": {"type": "noul", "instructions": "Does the clip mention an animal?"}}' --audio $clip > $log 2>&1
      else
        flock gpu.lock env CMF_GPU=$be CMF_D1_DEBUG=1 CMF_CUDA_DEBUG=1 timeout 900 bin/cortiq-fast system-one $f --state "I was charged twice and my parcel never arrived. Please refund one charge." --questions "$q2" > $log 2>&1
      fi
      echo "${file%.cmf} $be run$i: exit $? | $(grep -h -oE 'd1: forward [0-9]+ tokens.* ms|system-one: [0-9]+ ms' $log | tr '\n' ' ')"
    done
    grep -h -oE '"(noul|billing|shipping|technical)": [0-9.e-]+' logs/d1c/${file%.cmf}_${be}_2.log | tr '\n' ' ' | sed 's/^/    /'; echo
    grep -h -iE "declin|refused" logs/d1c/${file%.cmf}_${be}_2.log | sed -E 's/\x1b\[[0-9;]*m//g' | sort -u | head -3 | sed 's/^/    /'
  done
  rm -f $f
done <<'LIST'
infosave/d1-Omni-2B-cmf d1-omni-2b-q4tp.cmf omni
infosave/d1-2B-cmf d1-2b-q4tp.cmf text
infosave/d1-2B-cmf d1-2b-q8_2f.cmf text
LIST
echo D1C_DONE
