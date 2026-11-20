#!/usr/bin/env bash
# bench2.sh FILE TAG BACKEND CTX... : decode (3 runs) + prompt at each CTX, with the card's peak memory
# NODEC=1: the prompt lengths only. BENCH_ENV: extra environment of the runs.
source /root/mx/scripts/env.sh
F=$1; TAG=$2; be=$3; shift 3
mkdir -p bench
case $be in cuda) E="CMF_GPU=cuda";; wgpu) E="CMF_GPU=1";; cpu) E="CMF_GPU=0";; esac
vram() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 200 > /tmp/vram_$$.log 2>/dev/null & echo $!; }
one() { # label args...
  local label=$1; shift
  local vp=$(vram)
  env $E ${BENCH_ENV:-} $B bench $F --core --ignore-eos "$@" > bench/$TAG.$be.$label.log 2>&1
  kill $vp 2>/dev/null; wait $vp 2>/dev/null
  local peak=$(sort -n /tmp/vram_$$.log | tail -1)
  echo "$TAG $be $label: $(grep -aE 'Prompt:|Decode:' bench/$TAG.$be.$label.log | sed 's/  */ /g' | tr '\n' ' ') vram_peak ${peak} MiB"
}
n=128; [ $be = cpu ] && n=32
[ -n "${NODEC:-}" ] || for i in 1 2 3; do one d$i --tokens $n; done
for c in "$@"; do m=64; [ $be = cpu ] && m=16; one c$c --ctx $c --tokens $m; done
