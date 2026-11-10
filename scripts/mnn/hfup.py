#!/usr/bin/env python3
"""Upload files into a PRIVATE HF model repo, then check each one through the tree API:
size and LFS sha256 must equal the local file. The token comes from the HF cache (or HF_TOKEN).

hfup.py REPO FILE [FILE ...]          upload + verify
hfup.py --verify REPO FILE [FILE ...]  verify only
"""
import hashlib
import os
import sys

from huggingface_hub import HfApi

args = sys.argv[1:]
verify_only = args and args[0] == "--verify"
if verify_only:
    args = args[1:]
repo, files = args[0], args[1:]
api = HfApi()
assert api.model_info(repo).private, f"{repo} is not private"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


if not verify_only:
    for f in files:
        api.upload_file(path_or_fileobj=f, path_in_repo=os.path.basename(f), repo_id=repo,
                        commit_message=f"Add {os.path.basename(f)}")
        print("uploaded", f, flush=True)

tree = {e.path: e for e in api.list_repo_tree(repo, recursive=True)}
bad = 0
for f in files:
    name = os.path.basename(f)
    e = tree.get(name)
    size, digest = os.path.getsize(f), sha256(f)
    if e is None:
        print(f"MISSING {name}")
        bad += 1
        continue
    remote_sha = e.lfs.sha256 if getattr(e, "lfs", None) else None
    ok = e.size == size and (remote_sha is None or remote_sha == digest)
    print(f"{'OK' if ok else 'MISMATCH'} {name} local {size} {digest} remote {e.size} {remote_sha}")
    bad += not ok
sys.exit(1 if bad else 0)
