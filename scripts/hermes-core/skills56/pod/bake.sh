#!/bin/bash
# One skill baked onto Embryo-56M on this pod (the CUDA build of the trainer), the same call as
# ../bake_skill.sh on the Mac:
#   bake.sh NAME BASE SET ID [more `skill-bake` options]
# Reads /root/sk/data, writes /root/sk/run/NAME.cmf and /root/sk/run/NAME.bake.log.
# BASE: f0 (the genome file in data/) or the NAME of an earlier bake in run/.
# CKPT: the trainer checkpoint the base was exported from (default: the release 56M).
D=/root/sk/data; R=/root/sk/run; E="${E:-/root/sk/cmf/target/release/cortiq-embryo}"
export LD_LIBRARY_PATH=/usr/local/cuda/lib64 XDG_RUNTIME_DIR=/tmp
NAME="$1"; BASE="$2"; SET="$3"; ID="$4"; shift 4
B="$R/$BASE.cmf"; [ -f "$B" ] || B="$D/$BASE.cmf"
mkdir -p "$R"; rm -f "$R/$NAME.cmf"
echo "== $NAME: bake $ID from $SET onto $B $(date -u +%T)"
SECONDS=0
"$E" skill-bake --ckpt "${CKPT:-$D/keep-final-healed.ckpt}" --base "$B" --out "$R/$NAME.cmf" --id "$ID" \
  --sft-train "$D/$SET-train.sft" --sft-dev "$D/$SET-dev.sft" --sft-final "$D/$SET-final.sft" --sft-manifest "$D/$SET.manifest.json" \
  --lm-train "$D/replay-16m.u16" --lm-dev "$D/rawval-tails-2m.u16" \
  --phi-prompts "$D/$SET.prompts.jsonl" --general-prompts "$D/general.prompts.jsonl" "$@" > "$R/$NAME.bake.log" 2>&1
echo "BAKE_EXIT $? WALL $SECONDS s" >> "$R/$NAME.bake.log"
grep -E "phase|best|step .*dev|gate|route|published|appended|WALL|BAKE_EXIT" "$R/$NAME.bake.log" | tail -14 | cut -c1-240
echo "== $NAME done $(date -u +%T)"
