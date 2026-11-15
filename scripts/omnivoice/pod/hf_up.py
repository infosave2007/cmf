#!/usr/bin/env python3
"""Upload ov/hf to infosave/OmniVoice-cmf (created PRIVATE; made public after
the release) and print every file's size and sha256 as the Hub reports them.
The token comes from the HF_TOKEN environment variable only."""
import os
import sys

from huggingface_hub import HfApi

repo, folder = "infosave/OmniVoice-cmf", sys.argv[1]
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
info = api.upload_folder(repo_id=repo, folder_path=folder, commit_message="OmniVoice in CMF: q8_2f file, card, samples, licenses")
print("COMMIT", getattr(info, "oid", info), flush=True)
for f in api.list_repo_tree(repo, recursive=True, expand=True):
    lfs = getattr(f, "lfs", None)
    print(getattr(f, "path", ""), getattr(f, "size", ""), (lfs.sha256 if lfs else ""), flush=True)
print("HF_UP_DONE", flush=True)
