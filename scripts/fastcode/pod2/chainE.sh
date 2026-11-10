#!/bin/bash
# chainE.sh: expert packs by task. Each task's text through fcq2 on the whole card (cortiq.s2): its routing profile
# and its 128-token routing windows (φ) — fit and held-out texts apart; then the patent-1 router fitted on the fit
# windows (cortiq qwen-packs), its accuracy on the held-out texts, and the packs' coverage of held-out routing.
set -uo pipefail
cd /root/fc; J=/root/fc/JOURNAL.md; export XDG_RUNTIME_DIR=/tmp
log() { echo "$(date +%T) $*" | tee -a $J; }
for i in $(seq 1 720); do grep -q CHAINF_DONE $J && grep -q CORPUS_DONE logs/corpus.log && break; sleep 10; done
B=/root/fc/cortiq.s2
mkdir -p packs packs_held
for d in python rust c web php go java ts; do
  for part in fit held; do
    dir=packs; [ $part = held ] && dir=packs_held
    f=corpus/${d}_$part.txt
    [ -s $f ] || { log "pack $d $part: no text"; continue; }
    rm -f $dir/$d.phi $dir/$d.profile
    t0=$(date +%s)
    env CMF_QWEN_PROFILE=out/fcq2.profile CMF_QWEN_PROFILE_SAVE=$dir/$d.profile CMF_QWEN_PHI_DUMP=$dir/$d.phi CMF_QWEN_PHI_WINDOW=128 CMF_QWEN_PREFILL_CHUNK=128 \
      timeout 1500 $B run out/fcq2.cmf --prompt-file $f -n 1 --temperature 0 --no-think > logs/pack_${d}_$part.out 2> logs/pack_${d}_$part.err
    log "pack $d $part rc=$? $(( $(date +%s) - t0 )) s, $(stat -c %s $f) chars, phi $(stat -c %s $dir/$d.phi 2>/dev/null) B, profile $(stat -c %s $dir/$d.profile 2>/dev/null) B"
  done
done
log "qwen-packs: $($B qwen-packs packs --layers 8 --rank 16 --blocks 1,2,4 2>&1 | tr '\n' ' ' | cut -c1-1500)"
cp packs/router.json packs_held/router.json 2>/dev/null
log "$(/root/fc/venv/bin/python packs_eval.py packs packs_held out/fcq2.profile 2>&1 | tr '\n' ' ' | cut -c1-2500)"
log CHAINE_DONE
