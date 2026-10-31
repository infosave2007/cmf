#!/bin/bash
# Upload LFM3 CMF files to a Hugging Face repo (created private when absent):
# each file whole, in one call, with its .sha256 beside it. HF_TOKEN comes
# from the environment only — never a file. A failed upload stops the
# script: no later step can print the done marker.
#
# usage: HF_TOKEN=... pod_upload.sh REPO DIR FILE [FILE ...]
set -euo pipefail
: "${HF_TOKEN:?HF_TOKEN must be set in the environment}"
R=$1; cd "$2"; shift 2
for f in "$@"; do
  echo "== $f $(date -u +%T)"
  case "$f" in
    *.cmf)
      [ -f "$f.sha256" ] || sha256sum "$f" > "$f.sha256"
      hf upload "$R" "$f" "$f" --private --commit-message "Add $f"
      hf upload "$R" "$f.sha256" "$f.sha256" --private --commit-message "Add $f.sha256" ;;
    *) hf upload "$R" "$f" "$f" --private --commit-message "Add $f" ;;
  esac
done
echo "UPLOAD_DONE $(date -u +%T)"
