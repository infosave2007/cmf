#!/bin/bash
# One LTX-2.5 clip on the Mac (Metal), with the process's peak memory.
#   mac_ltx.sh TAG [WIDTH HEIGHT FRAMES]
#   env: RUNS (default ~/dev/h3img-runs), BIN, MODEL, PROMPT, EXTRA, CMF_*.
# Leaves $RUNS/ltx/TAG.{y4m,wav,log} and a summary line in $RUNS/ltx/TAG.sum.
set -u
here=$(cd "$(dirname "$0")" && pwd)
runs=${RUNS:-$HOME/dev/h3img-runs}
tag=$1; w=${2:-384}; h=${3:-256}; frames=${4:-49}
model=${MODEL:-$runs/cmf/ltx25-q4tp.cmf}
prompt=${PROMPT:-A corgi in a chef hat flips a pancake in a sunlit kitchen. Warm morning light, static camera.}
bin=${BIN:-$here/../../target-fast/release/cortiq}
out=$runs/ltx/$tag
mkdir -p "$runs/ltx"
t0=$(date +%s)
/usr/bin/time -l "$bin" ltx-video --model "$model" --prompt "$prompt" --height "$h" --width "$w" --frames "$frames" --fps 24 --seed 42 \
  --out-audio "$out.wav" ${EXTRA:-} --out "$out.y4m" > "$out.log" 2>&1
rc=$?
t1=$(date +%s)
rss=$(awk '/maximum resident set size/ {printf "%.0f", $1 / 1048576}' "$out.log")
echo "tag=$tag size=${w}x${h}x${frames} rc=$rc wall_s=$((t1 - t0)) peak_rss_mib=$rss" | tee "$out.sum"
grep -E "prompt:|denoised|decoded|stereo|total|error|panicked" "$out.log" | tail -8
