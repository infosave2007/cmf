#!/usr/bin/env python3
"""hf_upload.py REPO LOCAL DEST [LOCAL DEST ...]: files into a PRIVATE repo of the Hub, each verified
by size and sha256 against the tree API. The repo is created private when it does not exist (an
`hf upload` into a missing repo makes it public). The token comes from HF_TOKEN in the environment
and never touches the disk."""
import hashlib
import os
import sys

from huggingface_hub import HfApi

repo = sys.argv[1]
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(repo, private=True, exist_ok=True)
assert api.repo_info(repo).private, "the repo is not private"
bad = 0
for local, dest in zip(sys.argv[2::2], sys.argv[3::2]):
    h = hashlib.sha256()
    with open(local, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    api.upload_file(path_or_fileobj=local, path_in_repo=dest, repo_id=repo)
    info = api.get_paths_info(repo, [dest])[0]
    lfs = getattr(info, "lfs", None)
    remote = lfs.sha256 if lfs else None
    size_ok = info.size == os.path.getsize(local)
    sha_ok = remote == h.hexdigest() if lfs else True
    bad += not (size_ok and sha_ok)
    print(f"{dest}: {os.path.getsize(local)} bytes local, {info.size} on the Hub, sha256 {'OK' if sha_ok else 'DIFFERS: ' + str(remote)}", flush=True)
print("UPLOAD_DONE" if bad == 0 else f"UPLOAD_BAD {bad}", flush=True)
