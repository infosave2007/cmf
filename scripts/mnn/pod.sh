#!/bin/bash
# MNN-Pocket-0.3B on the pod (Vulkan): pod.sh  — runs after the other job's chain is done and the GPU is free.
# Per file: ppl (WikiText-2, 12x512), bench --core x3, 2k and 8k prompts, greedy chat answers.
set -u
R=/root/mnn; B=$R/cortiq; J=$R/JOURNAL.md
export XDG_RUNTIME_DIR=/tmp
log() { echo "$(date +%T) $*" | tee -a $J; }
# wait (deadline 10 h) for the other session's chain to finish and the GPU to have no compute apps
for i in $(seq 1 3600); do
  grep -q ${WAIT_MARK:-CHAINQ_DONE} /root/fc/JOURNAL.md 2>/dev/null && \
    [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -eq 0 ] && break
  sleep 10
done
sleep 30  # a chain that follows starts within seconds: check the GPU is still free
[ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -eq 0 ] || { log "GPU taken again, not measuring"; log "MNN_POD_DONE"; exit 1; }
log "start: gpu apps $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l), util $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader)"
for f in q8_2f q4tp; do
  F=$R/out/MNN-Pocket-0.3B-$f.cmf; [ -f $F ] || continue
  log "$f ppl wiki $($B ppl $F --file $R/wikitext2_test.txt --windows 12 --window-len 512 2>&1 | grep PPL | tail -1)"
  log "$f ppl chat $($B ppl $F --file $R/chat_eval.txt --windows 16 --window-len 512 2>&1 | grep PPL | tail -1)"
  for i in 1 2 3; do
    log "$f bench$i $($B bench $F --core --ignore-eos --tokens 256 2>&1 | grep -E 'tok/s' | grep -v INFO | tr '\n' ' ')"
  done
  for c in 2048 8192; do
    ( $B bench $F --core --ignore-eos --ctx $c --tokens 128 > $R/logs/bench_${f}_ctx$c.log 2>&1 ) &
    p=$!; peak=0
    while kill -0 $p 2>/dev/null; do
      m=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -v p=$p '{s+=$2} END {print s+0}')
      [ "$m" -gt "$peak" ] && peak=$m; sleep 0.5
    done
    log "$f ctx$c $(grep -E 'tok/s|TTFT' $R/logs/bench_${f}_ctx$c.log | grep -v INFO | tr '\n' ' ') vram_peak_mib $peak"
  done
  mkdir -p $R/logs/chat_$f
  for q in "Who are you?" "What is the capital of France? Answer in one sentence." "用三句话介绍一下杭州。"; do
    $B run $F --greedy -n 120 --prompt "$q" >> $R/logs/chat_$f/answers.txt 2>/dev/null
    echo "-----" >> $R/logs/chat_$f/answers.txt
  done
done
log "MNN_POD_DONE"
