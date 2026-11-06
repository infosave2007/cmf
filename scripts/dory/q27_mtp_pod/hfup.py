# hfup.py LOCAL REPO_PATH …: files into the private experiments repo; the token comes from the environment only
import os, sys, hashlib
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
repo = "infosave/cmf-experiments"
assert api.repo_info(repo).private, "the repo is not private"
for local, dest in zip(sys.argv[1::2], sys.argv[2::2]):
    h = hashlib.sha256()
    with open(local, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    api.upload_file(path_or_fileobj=local, path_in_repo=dest, repo_id=repo)
    info = api.get_paths_info(repo, [dest])[0]
    lfs = getattr(info, "lfs", None)
    remote = lfs.sha256 if lfs else None
    print(dest, os.path.getsize(local), info.size, h.hexdigest(), "sha256 OK" if remote == h.hexdigest() else f"remote {remote}", flush=True)
