#!/bin/bash
# Final names on the private model repo, logs and samples on the experiments repo.
[ -n "$HF_TOKEN" ] || { read -r HF_TOKEN; export HF_TOKEN; }
cd /workspace/s2v
. venv/bin/activate
python - <<'PY'
import os
from huggingface_hub import HfApi, CommitOperationDelete
api = HfApi(token=os.environ["HF_TOKEN"])
repo = "infosave/Kandinsky-S2V-1.0-Pro-distill-cmf"
assert api.model_info(repo).private is True
exp = "infosave/cmf-experiments"
assert api.model_info(exp).private is True
for old, new in (("s2v-q4tp", "kandinsky-s2v-pro-q4tp"), ("s2v-q8", "kandinsky-s2v-pro-q8")):
    sha = open(f"cmf/{old}.cmf.sha256").read().split()[0]
    print("up", new, flush=True)
    api.upload_file(path_or_fileobj=f"cmf/{old}.cmf", path_in_repo=f"{new}.cmf", repo_id=repo, repo_type="model")
    api.upload_file(path_or_fileobj=f"{sha}  {new}.cmf\n".encode(), path_in_repo=f"{new}.cmf.sha256", repo_id=repo, repo_type="model")
tree = {f.path: f for f in api.list_repo_tree(repo)}
for old, new in (("s2v-q4tp", "kandinsky-s2v-pro-q4tp"), ("s2v-q8", "kandinsky-s2v-pro-q8")):
    assert tree[f"{new}.cmf"].lfs.sha256 == tree[f"{old}.cmf"].lfs.sha256
    assert tree[f"{new}.cmf"].size == tree[f"{old}.cmf"].size
api.create_commit(
    repo_id=repo,
    commit_message="Final file names",
    operations=[CommitOperationDelete(path_in_repo=p) for p in tree if p.startswith("s2v-")],
)
stamp = "kandinsky-s2v-2026-11-03"
for name in sorted(os.listdir("out")):
    if name.endswith((".log", ".txt")):
        api.upload_file(path_or_fileobj=f"out/{name}", path_in_repo=f"{stamp}/{name}", repo_id=exp, repo_type="model")
for name in ("sample.avi", "small.avi", "portrait.avi", "q4tp.avi", "q8.avi"):
    api.upload_file(path_or_fileobj=f"out/{name}", path_in_repo=f"{stamp}/{name}", repo_id=exp, repo_type="model")
for name in ("lat_q4tp2.safetensors", "lat_q4tp.safetensors", "lat_q8.safetensors"):
    api.upload_file(path_or_fileobj=f"out/{name}", path_in_repo=f"{stamp}/{name}", repo_id=exp, repo_type="model")
for name in ("sample.png", "small.png", "q4tp_tile.png", "q8_tile.png"):
    api.upload_file(path_or_fileobj=f"out/fr/{name}", path_in_repo=f"{stamp}/{name}", repo_id=exp, repo_type="model")
for name in ("oracle.py", "pod_chain.sh", "pod_setup.sh", "pod_final.sh", "compare.py"):
    api.upload_file(path_or_fileobj=f"src/scripts/s2v/{name}", path_in_repo=f"{stamp}/scripts/{name}", repo_id=exp, repo_type="model")
assert api.model_info(repo).private is True
print("BACKUP2_DONE", flush=True)
PY
