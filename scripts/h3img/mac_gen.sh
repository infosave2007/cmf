#!/bin/bash
# One image on the Mac (Metal), with the process's peak memory.
#   mac_gen.sh TAG WIDTH HEIGHT [SEED]
#   env: RUNS (the run directory, default ~/dev/h3img-runs), BIN, MODEL,
#        PROMPT (file), IMAGE (a reference picture: an edit), STEPS,
#        MIN_AREA (--min-area), plus any CMF_* knob.
# Leaves $RUNS/out/TAG.{png,log} and one summary line in $RUNS/out/TAG.sum.
set -u
here=$(cd "$(dirname "$0")" && pwd)
runs=${RUNS:-$HOME/dev/h3img-runs}
tag=$1; w=$2; h=$3; seed=${4:-42}
model=${MODEL:-$runs/cmf/minimax-h3-image-q4tp.cmf}
prompt=${PROMPT:-$here/prompts/lighthouse.txt}
bin=${BIN:-$here/../../target-fast/release/cortiq}
out=$runs/out/$tag
mkdir -p "$runs/out"
t0=$(date +%s)
/usr/bin/time -l "$bin" imagine "$model" --prompt "@$prompt" ${IMAGE:+--image "$IMAGE"} --width "$w" --height "$h" --seed "$seed" ${STEPS:+--steps $STEPS} ${MIN_AREA:+--min-area $MIN_AREA} --out "$out.png" > "$out.log" 2>&1
rc=$?
t1=$(date +%s)
rss=$(awk '/maximum resident set size/ {printf "%.0f", $1 / 1048576}' "$out.log")
echo "tag=$tag size=${w}x${h} seed=$seed rc=$rc wall_s=$((t1 - t0)) peak_rss_mib=$rss" | tee "$out.sum"
grep -E "image\(s\), |error|Error|panicked|declined" "$out.log" | tail -5
