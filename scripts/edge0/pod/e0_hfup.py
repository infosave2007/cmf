# Upload one file to a (private) HF repo and verify it by size + LFS sha256.
# Usage: HF_TOKEN read from stdin; python e0_hfup.py <repo> <local file> <path in repo> [--public]
import hashlib, os, sys
from huggingface_hub import HfApi
tok = sys.stdin.readline().strip()
repo, local, inrepo = sys.argv[1], sys.argv[2], sys.argv[3]
api = HfApi(token=tok)
api.create_repo(repo, private=("--public" not in sys.argv), exist_ok=True, repo_type="model")
h = hashlib.sha256()
with open(local, "rb") as f:
    while True:
        b = f.read(64 << 20)
        if not b:
            break
        h.update(b)
sha = h.hexdigest()
size = os.path.getsize(local)
print("local", size, sha, flush=True)
api.upload_file(path_or_fileobj=local, path_in_repo=inrepo, repo_id=repo, commit_message=f"Add {inrepo}")
i = api.get_paths_info(repo, [inrepo], expand=True)[0]
ok = i.size == size and i.lfs is not None and i.lfs.sha256 == sha
print("remote", i.size, i.lfs.sha256 if i.lfs else None, "OK" if ok else "MISMATCH", flush=True)
with open(local + ".sha256", "w") as f:
    f.write(f"{sha}  {os.path.basename(inrepo)}\n")
sys.exit(0 if ok else 1)
