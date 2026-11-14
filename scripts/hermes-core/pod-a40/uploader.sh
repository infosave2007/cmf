#!/usr/bin/env bash
# The uploader of one run of the campaign: a report and a backup on the Hub per stage
# (stage_sync.py, on the pod as stage_sync2.py). chain.sh ends in it; started by hand it
# replaces an uploader that died or was stopped: it reads what is on the Hub and goes on.
# HF_TOKEN comes in the environment, read from stdin by the caller, never from a file:
#
#   IFS= read -r HF_TOKEN; export HF_TOKEN; setsid nohup ./uploader.sh base1.conf "<report title>" > /dev/null 2>&1 < /dev/null &
set -u
CONF="$1"; TITLE="$2"
cd /root/hc
source "$CONF"
R=/root/hc/run
exec venv/bin/python stage_sync2.py --run "$R" --name "$NAME" --repo infosave/cmf-experiments \
  --prefix runs/hermes_core_a40_20261114 --subdir "$NAME" --start-step "${ORIGIN:-0}" --steps "$STEPS" --batch "${BATCH:-8}" --seq 1024 \
  --ckpt-every 1 --keep 3 --title "$TITLE" --about "/root/hc/$NAME.about.md" --base "$R/$NAME.base.log" \
  --also /root/hc/train2.sh "$CONF" /root/hc/stage_sync2.py /root/hc/convert.py /root/hc/shard_all.sh /root/hc/data/shards.tsv
