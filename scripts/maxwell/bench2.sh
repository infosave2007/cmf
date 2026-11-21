#!/usr/bin/env bash
# bench2.sh FILE TAG BACKEND CTX... : decode (3 runs) + prompt at each CTX, with the card's peak memory and the
# process' peak resident memory
# NODEC=1: the prompt lengths only; DECN=n: n decode runs (3). BENCH_ENV: extra environment of the runs.
source /root/mx/scripts/env.sh
F=$1; TAG=$2; be=$3; shift 3
mkdir -p bench
case $be in cuda) E="CMF_GPU=cuda";; wgpu) E="CMF_GPU=1";; cpu) E="CMF_GPU=0";; esac
vram() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 200 > /tmp/vram_$$.log 2>/dev/null & echo $!; }
one() { # label args...
  local label=$1; shift
  local vp=$(vram)
  env $E ${BENCH_ENV:-} $B bench $F --core --ignore-eos "$@" > bench/$TAG.$be.$label.log 2>&1 &
  local bp=$! rss=0 r
  # the process' peak resident memory beside the card's
  while kill -0 $bp 2>/dev/null; do
    r=$(ps -o rss= -p $bp 2>/dev/null | tr -d ' ')
    [ -n "$r" ] && [ "$r" -gt "$rss" ] && rss=$r
    sleep 0.5
  done
  wait $bp
  kill $vp 2>/dev/null; wait $vp 2>/dev/null
  local peak=$(sort -n /tmp/vram_$$.log | tail -1)
  echo "$TAG $be $label: $(grep -aE 'Prompt:|Decode:' bench/$TAG.$be.$label.log | sed 's/  */ /g' | tr '\n' ' ') vram_peak ${peak} MiB rss_peak $(( rss / 1024 )) MiB"
}
n=128; [ $be = cpu ] && n=32
[ -n "${NODEC:-}" ] || for i in $(seq 1 ${DECN:-3}); do one d$i --tokens $n; done
for c in "$@"; do m=64; [ $be = cpu ] && m=16; one c$c --ctx $c --tokens $m; done
