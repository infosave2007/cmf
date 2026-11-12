#!/bin/bash
# One clip on the Mac (Metal), with the process's peak memory and the stage
# times of the run.
#   mac_anim.sh TAG [WIDTH HEIGHT FRAMES STEPS]
#   env: RUNS (the run directory, default ~/dev/h3img-runs), BIN, MODEL,
#        PROMPT (text), plus any CMF_* knob.
# Leaves $RUNS/turbo/TAG.{avi,wav,log}, $RUNS/turbo/TAG.frames/ and one
# summary line in $RUNS/turbo/TAG.sum.
set -u
here=$(cd "$(dirname "$0")" && pwd)
runs=${RUNS:-$HOME/dev/h3img-runs}
tag=$1; w=${2:-512}; h=${3:-256}; frames=${4:-39}; steps=${5:-4}
model=${MODEL:-$runs/cmf/mmh3-turbo-clipproj4b-q4tp.cmf}
prompt=${PROMPT:-A corgi in a chef hat flipping a pancake, sizzling sounds and a cheerful bark.}
bin=${BIN:-$here/../../target-fast/release/cortiq}
out=$runs/turbo/$tag
mkdir -p "$runs/turbo" "$out.frames"
t0=$(date +%s)
RUST_LOG=${RUST_LOG:-info} /usr/bin/time -l "$bin" animate "$model" --prompt "$prompt" --width "$w" --height "$h" \
  --frames "$frames" --steps "$steps" --seed 42 --frames-dir "$out.frames" --out "$out.avi" > "$out.log" 2>&1
rc=$?
t1=$(date +%s)
rss=$(awk '/maximum resident set size/ {printf "%.0f", $1 / 1048576}' "$out.log")
echo "tag=$tag size=${w}x${h}x${frames} steps=$steps rc=$rc wall_s=$((t1 - t0)) peak_rss_mib=$rss" | tee "$out.sum"
grep -E "stages:|error|Error|panicked|declined" "$out.log" | tail -5
