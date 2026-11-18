"""Download files of the private experiments repo into a flat folder and verify size + sha256."""
import hashlib, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

token = os.environ.pop("HF_TOKEN")
api = HfApi(token=token)
r = "infosave/cmf-experiments"
assert api.model_info(r).private
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
paths = sys.argv[2:]
info = {f.path: f for f in api.get_paths_info(r, paths, expand=True)}

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()

def one(p):
    t = time.time()
    dst = out / Path(p).name
    f = info[p]
    if not (dst.exists() and dst.stat().st_size == f.size):
        tmp = hf_hub_download(r, p, token=token, local_dir=str(out / ".dl"))
        os.replace(tmp, dst)
    ok = dst.stat().st_size == f.size and (f.lfs is None or sha256(dst) == f.lfs.sha256)
    return f"{'OK ' if ok else 'BAD'} {dst.name} {f.size} {time.time()-t:.0f}s"

with ThreadPoolExecutor(6) as ex:
    for line in ex.map(one, paths):
        print(line, flush=True)
print("GET_DONE", flush=True)
