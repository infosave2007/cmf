#!/bin/bash
# MNN-Pocket collection on a pod (Vulkan): pod_all.sh — every file once, the GPU to itself.
#   ppl (WikiText-2 12x512, chat_eval 16x512), bench --core x3, 2k / 8k prompts with peak VRAM,
#   DFlash2 sidecar against plain greedy decode, Dec decide-pointer timing, greedy chat answers.
set -u
R=/root/mnn; B=$R/cortiq; J=$R/JOURNAL.md; O=$R/out
export XDG_RUNTIME_DIR=/tmp
log() { echo "$(date +%T) $*" | tee -a $J; }
vram() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
log "start: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader), gpu apps $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)"
for f in q8_2f q4tp Base-q8_2f SFT-q8_2f; do
  F=$O/MNN-Pocket-0.3B-$f.cmf; [ -f $F ] || continue
  log "$f ppl wiki $($B ppl $F --file $R/wikitext2_test.txt --windows 12 --window-len 512 2>&1 | grep PPL | tail -1)"
  log "$f ppl chat $($B ppl $F --file $R/chat_eval.txt --windows 16 --window-len 512 2>&1 | grep PPL | tail -1)"
  for i in 1 2 3; do
    log "$f bench$i $($B bench $F --core --ignore-eos --tokens 256 2>&1 | grep -E 'tok/s' | grep -v INFO | tr '\n' ' ')"
  done
  for c in 2048 8192; do
    base=$(vram); ( $B bench $F --core --ignore-eos --ctx $c --tokens 128 > $R/logs/bench_${f}_ctx$c.log 2>&1 ) &
    p=$!; peak=0
    while kill -0 $p 2>/dev/null; do m=$(vram); [ "$m" -gt "$peak" ] && peak=$m; sleep 0.3; done
    log "$f ctx$c $(grep -E 'tok/s|TTFT' $R/logs/bench_${f}_ctx$c.log | grep -v INFO | tr '\n' ' ') vram_peak_mib $((peak - base))"
  done
done
# DFlash2: greedy, plain against the sidecar, on four prompts; the same text is the check
P1="Write a Python function that returns the n-th Fibonacci number iteratively."
P2="A train leaves at 9:40 and arrives at 13:15. How long is the trip in minutes? Explain step by step."
P3="Explain the difference between TCP and UDP in three sentences."
P4="写一个Python函数，判断一个字符串是否是回文。"
for f in q8_2f q4tp; do
  F=$O/MNN-Pocket-0.3B-$f.cmf; mkdir -p $R/logs/dflash_$f
  n=0
  for p in "$P1" "$P2" "$P3" "$P4"; do
    n=$((n+1))
    for arm in plain dflash; do
      e="CMF_DFLASH=0"; [ $arm = dflash ] && e="CMF_DFLASH=1"
      for rep in 1 2; do
        env $e RUST_LOG=info $B run $F --greedy -n 256 --prompt "$p" > $R/logs/dflash_$f/$n.$arm.txt 2> $R/logs/dflash_$f/$n.$arm.err
        log "$f dflash p$n $arm rep$rep $(tail -1 $R/logs/dflash_$f/$n.$arm.txt) $(grep -aoE 'DFlash drafter[^|]*|accepted[^,]*' $R/logs/dflash_$f/$n.$arm.err | head -2 | tr '\n' ' ')"
      done
    done
    cmp -s <(head -n -1 $R/logs/dflash_$f/$n.plain.txt) <(head -n -1 $R/logs/dflash_$f/$n.dflash.txt) && s=same || s=DIFFERENT
    log "$f dflash p$n text $s"
  done
done
# Dec: decide-pointer on the 12 requests (3 questions each), twice
for q in q8_2f f16; do
  F=$O/MNN-Pocket-0.3B-Dec-$q.cmf; [ -f $F ] || continue
  for rep in 1 2; do
    t0=$(date +%s.%N); $B decide-pointer $F --input $R/requests2.json > $R/logs/dec2_$q.json 2> $R/logs/dec2_$q.err; rc=$?
    log "Dec $q rep$rep rc=$rc $(python3 -c "import time,sys; print(round(time.time()-float(sys.argv[1]),2))" $t0) s for 12 requests x 3 questions"
  done
done
for f in q8_2f q4tp SFT-q8_2f; do
  F=$O/MNN-Pocket-0.3B-$f.cmf; : > $R/logs/chat_$f.txt
  for q in "Who are you?" "What is the capital of France? Answer in one sentence." "用三句话介绍一下杭州。"; do
    $B run $F --greedy -n 120 --prompt "$q" >> $R/logs/chat_$f.txt 2>/dev/null; echo "-----" >> $R/logs/chat_$f.txt
  done
done
$B run $O/MNN-Pocket-0.3B-Base-q8_2f.cmf --raw --greedy -n 60 --prompt "The capital of France is" > $R/logs/chat_Base.txt 2>&1
log "MNN_POD_DONE"
