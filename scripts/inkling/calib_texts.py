#!/usr/bin/env python3
"""The calibration texts of the 4-bit file's dense trunk (`cortiq requant --quant q4tp-dense --calib`): prose, Rust
and Python that the evaluation texts (WikiText-2 test, the cortiq CLI's sources, transformers' sources) do not hold.

calib_texts.py OUT_DIR   ->  OUT_DIR/wiki.txt, rust.txt, py.txt (120 KB each)

The published 4-bit file, step by step (SRC the checkpoint, T = OUT_DIR):

    cortiq convert --model SRC --output a.cmf --quant q4tp          # routed experts q4tp, the rest q8_2f
    CMF_GPU=0 cortiq requant a.cmf --output Inkling-2-Mini-q4tp.cmf --quant q4tp-dense \\
        --calib T/wiki.txt --calib T/rust.txt --calib T/py.txt --tokens 3072 --calib-window 1024   # the dense trunk, GPTQ through the host stack

Needs pandas + huggingface_hub (WikiText-2 train), a cargo registry with serde_json and regex, a CPython stdlib."""
import glob, os, sys

from huggingface_hub import hf_hub_download
import pandas as pd

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
LIMIT = 120000
p = hf_hub_download(repo_id="Salesforce/wikitext", filename="wikitext-2-raw-v1/train-00000-of-00001.parquet", repo_type="dataset")
open(os.path.join(out, "wiki.txt"), "w").write("".join(pd.read_parquet(p).text[200:4000])[:LIMIT])


def cat(patterns, name):
    s = ""
    for pat in patterns:
        for f in sorted(glob.glob(os.path.expanduser(pat), recursive=True)):
            try:
                s += open(f, encoding="utf-8").read() + "\n"
            except Exception:
                pass
            if len(s) > LIMIT:
                break
        if len(s) > LIMIT:
            break
    open(os.path.join(out, name), "w").write(s[:LIMIT])
    print(name, len(s[:LIMIT]))


cat(["~/.cargo/registry/src/*/serde_json-*/src/*.rs", "~/.cargo/registry/src/*/regex-1*/src/**/*.rs"], "rust.txt")
cat(["/usr/lib/python3*/json/*.py", "/usr/lib/python3*/argparse.py", "/usr/lib/python3*/asyncio/tasks.py", "/usr/lib/python3*/collections/__init__.py"], "py.txt")
