"""Survey of text we could train the Hermes core on: how much there is.

    python3 -I survey.py datasets --card qvac/BotPsy-0.8B --out hf-sizes.json
    python3 -I survey.py repos --list repos.txt --out repos.tsv

`datasets` reads a model card on the Hugging Face Hub, takes every dataset
it names (`huggingface.co/datasets/<org>/<name>`), and asks the Hub's public
API for the licence in the dataset's own metadata, whether access is gated,
and the size: the in-memory bytes of the parquet conversion, or, where the
dataset has none, the bytes of its files. Tokens are estimated as bytes / 4.

`repos` asks GitHub (through `gh api`) for the licence and the bytes of
source per language of each repository in the list. Tokens are estimated
as bytes / 3.46: the ratio measured on the Hermes sources with the Embryo
tokenizer (69.5 MB of text = 20.07 M tokens).

Nothing is downloaded but metadata; no token is sent to the Hub.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
import urllib.parse
import urllib.request

HUB = "https://huggingface.co"


def get(url: str):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
        with urllib.request.urlopen(req, timeout=40) as f:
            return json.load(f)
    except Exception as e:  # a gated or missing dataset answers 401 / 404
        return {"_err": str(e)[:80]}


def card_datasets(model: str) -> list[str]:
    req = urllib.request.Request(f"{HUB}/{model}/raw/main/README.md", headers={"User-Agent": "curl/8"})
    with urllib.request.urlopen(req, timeout=40) as f:
        text = f.read().decode("utf-8").replace("\\", "")
    seen: list[str] = []
    for name in re.findall(r"huggingface\.co/datasets/([\w.\-]+/[\w.\-]+)", text):
        if name not in seen and not name.startswith("choosealicense/"):
            seen.append(name)
    return seen


def datasets(args) -> None:
    out = {}
    for name in card_datasets(args.card):
        meta = get(f"{HUB}/api/datasets/{name}")
        size = get("https://datasets-server.huggingface.co/size?dataset=" + urllib.parse.quote(name, safe=""))
        whole = (size.get("size") or {}).get("dataset") or {}
        nbytes = whole.get("num_bytes_memory")
        source = "parquet in memory"
        if not nbytes and "_err" not in meta:
            tree = get(f"{HUB}/api/datasets/{name}/tree/main?recursive=true&expand=false")
            if isinstance(tree, list):
                nbytes = sum(x.get("size", 0) for x in tree if x.get("type") == "file")
                source = "files"
        out[name] = {
            "license": (meta.get("cardData") or {}).get("license"),
            "gated": meta.get("gated"),
            "error": meta.get("_err"),
            "rows": whole.get("num_rows"),
            "bytes": nbytes,
            "bytes_source": source if nbytes else None,
            "tokens_est": round(nbytes / 4) if nbytes else None,
        }
        v = out[name]
        print(
            f"{name:<62} {str(v['license'])[:24]:<24} gated={str(v['gated']):<5} "
            f"{(v['bytes'] or 0) / 1e9:8.2f} GB  ~{(v['tokens_est'] or 0) / 1e9:6.2f} B tokens  {v['error'] or ''}"
        )
        time.sleep(0.3)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)


def gh(path: str, jq: str) -> str:
    p = subprocess.run(["gh", "api", path, "--jq", jq], capture_output=True, text=True)
    return p.stdout.strip() if p.returncode == 0 else ""


def repos(args) -> None:
    rows = []
    for name in (l.strip() for l in open(args.list) if l.strip() and not l.startswith("#")):
        info = gh(f"repos/{name}", '[.full_name, (.license.spdx_id // "none"), (.archived|tostring)] | @tsv').split("\t")
        code = gh(f"repos/{name}/languages", "[to_entries[] | .value] | add")
        nbytes = int(code) if code.isdigit() else 0
        full, lic, archived = (info + ["?", "?", "?"])[:3] if info != [""] else (name, "?", "?")
        rows.append((full, lic, archived, nbytes, round(nbytes / 3.46)))
        print(f"{full:<36} {lic:<12} archived={archived:<5} {nbytes / 1e6:8.1f} MB  ~{nbytes / 3.46 / 1e6:6.1f} M tokens")
    with open(args.out, "w") as f:
        f.write("repo\tlicense\tarchived\tcode_bytes\ttokens_est\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("datasets")
    d.add_argument("--card", required=True)
    d.add_argument("--out", required=True)
    d.set_defaults(fn=datasets)
    r = sub.add_parser("repos")
    r.add_argument("--list", required=True)
    r.add_argument("--out", required=True)
    r.set_defaults(fn=repos)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
