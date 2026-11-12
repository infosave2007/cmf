#!/bin/bash
# fetch_private.sh REPO FILE SHA256 WAIT_GONE WAIT_FOR
# One file of a private repository into src/, started once WAIT_GONE has left
# the disk and WAIT_FOR exists (the pod's disk holds one large source at a
# time). HF_TOKEN comes in the environment and never touches the disk.
# Markers for dl.sh: src/FILE.fetching while pending, src/FILE.fetched once
# the sha256 matches.
set -u
root=${H3I_ROOT:-/root/h3i}
cd "$root" || exit 1
repo=$1; file=$2; sha=$3; gone=$4; need=$5
rm -f "src/$file.fetched"
touch "src/$file.fetching"
deadline=$(( $(date +%s) + 5400 ))
while [ -e "$gone" ] || [ ! -s "$need" ]; do
  [ "$(date +%s)" -gt "$deadline" ] && { echo "FETCH_TIMEOUT waiting for $gone to go"; exit 1; }
  sleep 5
done
echo "fetch start $(date -u +%H:%M:%S)"
venv/bin/python - "$repo" "$file" <<'PY'
import sys
from huggingface_hub import hf_hub_download
print(hf_hub_download(sys.argv[1], sys.argv[2], local_dir="/root/h3i/src"))
PY
echo "$sha  src/$file" | sha256sum -c - && touch "src/$file.fetched" && echo "FETCH_DONE $(date -u +%H:%M:%S)"
