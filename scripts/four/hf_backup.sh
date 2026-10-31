#!/bin/bash
# Back up the run's references, logs, scripts and journal to the private
# experiments repo (HF_TOKEN from the environment, piped in over stdin).
set -e
D=/root/four/backup/four-models-2026-10-31
rm -rf /root/four/backup; mkdir -p $D
cp -r /root/four/ref $D/ref
cp -r /root/four/logs $D/logs
cp -r /root/cmf/scripts/four $D/scripts
cp /root/cmf/scripts/four/JOURNAL_FOUR_MODELS_2026-10-31.md $D/JOURNAL.md 2>/dev/null || true
(cd /root/four/out && sha256sum *.cmf.sha256 >/dev/null 2>&1; cat *.sha256 > $D/cmf_sha256.txt 2>/dev/null || true; ls -la > $D/out_listing.txt)
hf repo create infosave/cmf-experiments --repo-type model --private >/dev/null 2>&1 || true
hf upload infosave/cmf-experiments /root/four/backup . --commit-message "four models campaign backup (31.10)" 2>&1 | tail -2
echo "== backup done $(date)"
