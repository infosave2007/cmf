#!/bin/bash
# Reference dumps, logs and scripts to the private experiments repo, and the
# containers to the private model repo. The token comes on stdin, never to disk.
read -r HF_TOKEN
export HF_TOKEN
cd /workspace/s2v
. venv/bin/activate
python - <<'PY'
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
repo = "infosave/Kandinsky-S2V-1.0-Pro-distill-cmf"
api.create_repo(repo, private=True, exist_ok=True)
assert api.model_info(repo).private is True
exp = "infosave/cmf-experiments"
assert api.model_info(exp).private is True
stamp = "kandinsky-s2v-2026-11-03"
for name in sorted(os.listdir("out")):
    if name.endswith((".safetensors", ".log")):
        print("up", name, flush=True)
        api.upload_file(path_or_fileobj=f"out/{name}", path_in_repo=f"{stamp}/{name}", repo_id=exp, repo_type="model")
for name in ("s2v-q4tp.cmf", "s2v-q8.cmf"):
    print("up", name, flush=True)
    api.upload_file(path_or_fileobj=f"cmf/{name}", path_in_repo=name, repo_id=repo, repo_type="model")
    api.upload_file(path_or_fileobj=f"cmf/{name}.sha256", path_in_repo=f"{name}.sha256", repo_id=repo, repo_type="model")
assert api.model_info(repo).private is True
print("BACKUP_DONE", flush=True)
PY
