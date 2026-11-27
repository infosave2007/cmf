#!/usr/bin/env bash
# bench.sh FILE TAG BACKEND SECONDS... : the time to the transcript of the benchmark recordings.
#
#   CORTIQ=/path/to/cortiq CLIPS=/dir/with/bench_<seconds>s.wav bench.sh model.cmf rnnt-q8_2f cuda 10 30 60 300 1800
#
# BACKEND: cuda | vulkan | metal | cpu. Each recording is transcribed twice in one process: the first pass is
# the cold one (weights paged in and sent to the card), the second the warm one. One line a recording:
#   TAG BACKEND s<SECONDS>: cold <total s> warm <total s> | vad .. features .. encoder .. decode .. | <x> real time | vram_peak <MiB> | rss_peak <MiB>
# vram_peak is the card's memory above what was in use before the run (nvidia-smi, 100 ms samples): nothing
# else may START on the card meanwhile.
set -u
F=$1; TAG=$2; be=$3; shift 3
B=${CORTIQ:-cortiq}; C=${CLIPS:-data/bench}
case $be in cuda) E="CMF_GPU=cuda";; vulkan) E="CMF_GPU=1";; metal) E="CMF_GPU=1";; cpu) E="CMF_GPU=0";; *) echo "backend $be?"; exit 1;; esac
for s in "$@"; do
  clip=$C/bench_${s}s.wav
  log=$(mktemp); vlog=$(mktemp); vp=""; base=0
  if command -v nvidia-smi >/dev/null && [ $be != cpu ]; then
    # the card's memory before the run is taken off the peak (the desktop, a paused job)
    base=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 100 > $vlog 2>/dev/null & vp=$!
  fi
  if [ "$(uname)" = Darwin ]; then TIMEV="/usr/bin/time -l"; else TIMEV="/usr/bin/time -v"; fi
  env $E ${BENCH_ENV:-} $TIMEV $B transcribe $F $clip $clip --stats > /dev/null 2> $log
  [ -n "$vp" ] && { kill $vp 2>/dev/null; wait $vp 2>/dev/null; }
  peak=$(sort -n $vlog | tail -1); [ -n "$peak" ] && peak=$((peak - base))
  if [ "$(uname)" = Darwin ]; then rss=$(awk '/maximum resident set size/ {printf "%d", $1 / 1048576}' $log)
  else rss=$(awk -F: '/Maximum resident set size/ {printf "%d", $2 / 1024}' $log); fi
  cold=$(grep -a "real time" $log | sed -n 1p | sed -E 's/.*total ([0-9.]+)s.*/\1/')
  warm=$(grep -a "real time" $log | sed -n 2p)
  echo "$TAG $be s$s: cold $cold warm $(echo "$warm" | sed -E 's/.*total ([0-9.]+)s.*/\1/') | $(echo "$warm" | sed -E 's/.*\| (vad [0-9.]+s \| features [0-9.]+s \| encoder [0-9.]+s \| decode [0-9.]+s).*/\1/' | tr -d '|' ) | $(echo "$warm" | sed -E 's/.*= ([0-9.]+)x real time.*/\1/')x real time | vram_peak ${peak:-0} MiB | rss_peak ${rss:-0} MiB"
  rm -f $log $vlog
done
