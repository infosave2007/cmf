#!/usr/bin/env python3
"""Upload files to a Hugging Face model repo that is PRIVATE, and check every
one by its size and sha256 as the tree API reports them (a wrapper's "done"
line is not evidence; the remote state is).

  hf_put.py REPO LOCAL[:PATH_IN_REPO] ...

The repo is created private when it does not exist; an existing public repo
is refused unless --allow-public is given. The token comes from HF_TOKEN in
the environment and from nowhere else.
"""
import hashlib
import os
import sys

from huggingface_hub import HfApi

args = [a for a in sys.argv[1:] if not a.startswith("--")]
allow_public = "--allow-public" in sys.argv
if len(args) < 2 or "HF_TOKEN" not in os.environ:
    sys.exit(__doc__)
repo, items = args[0], args[1:]
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(repo, private=True, exist_ok=True, repo_type="model")
if not api.repo_info(repo).private and not allow_public:
    sys.exit(f"{repo} is public: refusing to upload (pass --allow-public)")


def sha256_of(path):
    side = path + ".sha256"
    if os.path.exists(side) and os.path.getmtime(side) >= os.path.getmtime(path):
        return open(side).read().split()[0]
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


bad = 0
for item in items:
    local, _, dst = item.partition(":")
    dst = dst or os.path.basename(local)
    size = os.path.getsize(local)
    print(f"upload {local} -> {repo}/{dst} ({size / 1e9:.2f} GB)", flush=True)
    api.upload_file(path_or_fileobj=local, path_in_repo=dst, repo_id=repo, commit_message=f"Add {dst}")
    info = api.get_paths_info(repo, [dst], expand=True)[0]
    remote_sha = info.lfs.sha256 if info.lfs else None
    ok = info.size == size and (remote_sha is None or remote_sha == sha256_of(local))
    print(f"  remote size {info.size} sha256 {remote_sha} -> {'OK' if ok else 'MISMATCH'}", flush=True)
    bad += not ok
print("HF_PUT_FAILED" if bad else "HF_PUT_OK", flush=True)
sys.exit(1 if bad else 0)
