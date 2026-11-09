#!/bin/bash
# upload.sh LOCAL REMOTE_PATH: into the private infosave/cmf-experiments, then size + sha256 checked by the API.
# HF_TOKEN comes from the caller's environment only (never a file).
set -euo pipefail
f=$1; dst=$2
[ -n "${HF_TOKEN:-}" ] || { echo "UPLOAD_FAILED no HF_TOKEN"; exit 1; }
sha=$(sha256sum "$f" | cut -d' ' -f1); sz=$(stat -c %s "$f")
echo "$sha  $(basename "$f")" > "$f.sha256"
nice -n 10 /root/fc/venv/bin/hf upload infosave/cmf-experiments "$f" "$dst" --repo-type model --private > /root/fc/logs/up_$(basename "$f").log 2>&1 || { echo "UPLOAD_FAILED hf upload $f"; exit 1; }
/root/fc/venv/bin/python -I - "$dst" "$sha" "$sz" <<'PY'
import sys
from huggingface_hub import HfApi
dst, sha, sz = sys.argv[1], sys.argv[2], int(sys.argv[3])
info = HfApi().get_paths_info("infosave/cmf-experiments", [dst], expand=True)
ok = bool(info) and info[0].size == sz and info[0].lfs is not None and info[0].lfs.sha256 == sha
print(("UPLOAD_OK " if ok else "UPLOAD_FAILED mismatch ") + f"{dst} {sz} {sha[:16]} remote={(info[0].size, info[0].lfs.sha256[:16] if info and info[0].lfs else None) if info else None}")
PY
