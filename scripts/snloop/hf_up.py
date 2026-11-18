#!/usr/bin/env python3
"""Upload files to a HF model repo and check each one's size and sha256 (LFS oid) by the API.
Token from HF_TOKEN in the environment only. A repo that does not exist is created PRIVATE.

usage: hf_up.py REPO DEST_DIR FILE [FILE ...]   (DEST_DIR "." for the repo root)
"""
import hashlib, os, sys
from huggingface_hub import HfApi

repo, dest, files = sys.argv[1], sys.argv[2], sys.argv[3:]
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(repo, private=True, exist_ok=True)
info = api.model_info(repo)
print(f"{repo}: private={info.private}", flush=True)
for f in files:
    name = os.path.basename(f)
    path = name if dest == "." else f"{dest}/{name}"
    h = hashlib.sha256()
    with open(f, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    sha, size = h.hexdigest(), os.path.getsize(f)
    api.upload_file(path_or_fileobj=f, path_in_repo=path, repo_id=repo, commit_message=f"Add {path}")
    meta = api.get_paths_info(repo, [path], expand=True)[0]
    lfs = getattr(meta, "lfs", None)
    remote_sha = lfs.sha256 if lfs else None
    ok = meta.size == size and (remote_sha == sha or (lfs is None and size < 10 << 20))
    print(f"{'VERIFIED' if ok else 'MISMATCH'} {path} size {size} remote {meta.size} sha {sha} remote {remote_sha}", flush=True)
    if not ok:
        sys.exit(1)
print("HF_UP_DONE", flush=True)
