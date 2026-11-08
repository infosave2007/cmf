#!/usr/bin/env python3
"""Upload a local folder into a PRIVATE HF repo path; refuses a public repo. Token on stdin.
hfup_dir.py REPO LOCAL_DIR PATH_IN_REPO"""
import sys
from huggingface_hub import HfApi
repo, local, dest = sys.argv[1:4]
api = HfApi(token=sys.stdin.readline().strip())
assert api.model_info(repo).private, f"{repo} is not private"
api.upload_folder(folder_path=local, path_in_repo=dest, repo_id=repo, commit_message=f"Backup {dest}")
print("uploaded", local, "->", repo, dest, flush=True)
