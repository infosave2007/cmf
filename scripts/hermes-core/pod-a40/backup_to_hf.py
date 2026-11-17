"""Back up a folder tree to the PRIVATE experiments repo and verify every file on the Hub.

    HF_TOKEN=<from stdin, never on disk> python -I backup_to_hf.py --folder /root/hc/hfstage \
        --repo infosave/cmf-experiments --prefix runs/hermes_core_a40_20261114/pod

`--folder` mirrors the repo layout (`<folder>/<prefix>/...`, hard links to the real files). Every entry of
`<folder>/<prefix>` goes up as its own commit (`upload_folder` for a folder, `upload_file` for a file), retried
up to 3 times (LFS objects already on the Hub are not sent again). Afterwards every local file under `<folder>/<prefix>` is
checked against the Hub tree: the size, and the sha256 of an LFS file or the git blob sha1 of a plain one.
Refuses a repo that is not private. Prints one line per mismatch and a summary; exit 1 on any mismatch.
"""

import argparse
import hashlib
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import HfApi


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def blob_sha1(p: Path) -> str:
    data = p.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--verify-only", action="store_true")
    a = ap.parse_args()
    token = os.environ.pop("HF_TOKEN", None)
    if not token:
        sys.exit("no HF_TOKEN in the environment")
    api = HfApi(token=token)
    if not api.model_info(a.repo).private:
        sys.exit(f"{a.repo} is not private: refusing to upload")
    root = Path(a.folder)
    local = {str(p.relative_to(root)): p for p in (root / a.prefix).rglob("*") if p.is_file() and ".cache" not in p.parts}
    total = sum(p.stat().st_size for p in local.values())
    print(f"{len(local)} files, {total / 1e9:.2f} GB under {a.prefix}", flush=True)
    if not a.verify_only:
        import time
        for child in sorted((root / a.prefix).iterdir()):
            remote = f"{a.prefix}/{child.name}"
            for attempt in range(1, 4):
                t = time.time()
                try:
                    if child.is_dir():
                        api.upload_folder(repo_id=a.repo, repo_type="model", folder_path=str(child),
                                          path_in_repo=remote, commit_message=f"backup {remote}")
                    else:
                        api.upload_file(path_or_fileobj=str(child), path_in_repo=remote, repo_id=a.repo,
                                        repo_type="model", commit_message=f"backup {remote}")
                    print(f"uploaded {remote} in {time.time() - t:.0f} s", flush=True)
                    break
                except Exception as e:  # network or Hub errors: retry
                    print(f"attempt {attempt} of {remote} failed: {type(e).__name__}: {str(e)[:200]}", flush=True)
                    time.sleep(30)
            else:
                print(f"GAVE UP {remote}", flush=True)
    hub = {f.path: f for f in api.list_repo_tree(a.repo, path_in_repo=a.prefix, recursive=True, expand=True) if hasattr(f, "size")}

    def check(item):
        rel, p = item
        f = hub.get(rel)
        if f is None:
            return f"MISSING {rel}"
        if f.size != p.stat().st_size:
            return f"SIZE {rel}: hub {f.size}, local {p.stat().st_size}"
        if f.lfs is not None:
            if f.lfs.sha256 != sha256(p):
                return f"SHA256 {rel}"
        elif f.blob_id != blob_sha1(p):
            return f"BLOB {rel}"
        return None

    with ThreadPoolExecutor(8) as ex:
        bad = [r for r in ex.map(check, sorted(local.items())) if r]
    for b in bad:
        print(b, flush=True)
    print(f"VERIFIED {len(local) - len(bad)} of {len(local)} files ({total / 1e9:.2f} GB), {len(bad)} mismatches", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
