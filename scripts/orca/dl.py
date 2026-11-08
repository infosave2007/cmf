#!/usr/bin/env python3
# Download files of a (gated) HF repo into DIR; token only from the HF_TOKEN env. dl.py REPO DIR FILE...
import os, sys, time
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
from huggingface_hub import hf_hub_download
repo, d = sys.argv[1], sys.argv[2]
for f in sys.argv[3:]:
    t0 = time.time()
    p = hf_hub_download(repo, f, local_dir=d, token=os.environ.get("HF_TOKEN"))
    print(f"{f} -> {p} {os.path.getsize(p)} B {time.time()-t0:.0f} s", flush=True)
print("DL_DONE", flush=True)
