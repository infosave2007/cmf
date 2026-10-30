#!/bin/bash
# Upload the Apex CMF files to infosave/Apex-35B-V1.0-cmf (created private
# when absent): each file whole, in one call, with its .sha256 beside it.
# HF_TOKEN comes from the environment only — never a file. A failed upload
# stops the script: no later step can print the done marker.
#
# usage: HF_TOKEN=... pod_upload.sh FILE [FILE ...]
set -euo pipefail
: "${HF_TOKEN:?HF_TOKEN must be set in the environment}"
R=infosave/Apex-35B-V1.0-cmf
cd /root/apex/out
for f in "$@"; do
  echo "== $f $(date -u +%T)"
  hf upload "$R" "$f" "$f" --private --commit-message "Add $f"
  hf upload "$R" "$f.sha256" "$f.sha256" --private --commit-message "Add $f.sha256"
done
echo "UPLOAD_DONE $(date -u +%T)"
