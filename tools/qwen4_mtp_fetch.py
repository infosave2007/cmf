#!/usr/bin/env python3
"""Fetch only the Qwen3.8-Flash-Next MTP head from the BF16 checkpoint.

The 31 `mtp.*` tensors sit in 28 of the 131 shards (~50 GB of shards for
~5.5 GB of tensors). Safetensors keeps a JSON header (name -> dtype, shape,
byte range) at the start of every shard, so the headers are read with small
HTTP range requests and only the MTP byte ranges follow. The result is a
minimal HF-style directory the converter accepts:

    config.json                      (copied)
    model.safetensors                (the 31 tensors, bf16, one file)
    model.safetensors.index.json     (weight_map -> that file)

    python tools/qwen4_mtp_fetch.py --out /workspace/flashnext/mtp-src
    cortiq convert --model /workspace/flashnext/mtp-src --quant q2tp \
        --output qwen38-flash-next-q2tp.cmf --mtp-sidecar

Resumable: tensors already present and complete in `tensors/` are skipped.
No model runs here.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
import urllib.request

REPO = "Qwen/Qwen3.8-Flash-Next"
DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4}


def get(url: str, headers: dict | None = None, retries: int = 6) -> bytes:
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers or {})
            with urllib.request.urlopen(req, timeout=120) as r:
                status = r.status
                data = r.read()
                if headers and "Range" in headers and status != 206:
                    raise RuntimeError(f"{url}: expected 206 for a range, got {status}")
                return data
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"{url}: {last}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--revision", default="main")
    ap.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", "https://huggingface.co"))
    args = ap.parse_args()
    base = f"{args.endpoint.rstrip('/')}/{REPO}/resolve/{args.revision}"
    out = args.out
    os.makedirs(os.path.join(out, "tensors"), exist_ok=True)

    cfg = get(f"{base}/config.json")
    with open(os.path.join(out, "config.json"), "wb") as f:
        f.write(cfg)
    index = json.loads(get(f"{base}/model.safetensors.index.json"))
    wm = index["weight_map"]
    mtp = {k: v for k, v in wm.items() if k.startswith("mtp.")}
    shards = sorted(set(mtp.values()))
    print(f"{len(mtp)} MTP tensors in {len(shards)} shards", flush=True)

    manifest = {}
    for shard in shards:
        url = f"{base}/{shard}"
        head8 = get(url, {"Range": "bytes=0-7"})
        (hlen,) = struct.unpack("<Q", head8)
        header = json.loads(get(url, {"Range": f"bytes=8-{8 + hlen - 1}"}))
        for name, meta in header.items():
            if not name.startswith("mtp."):
                continue
            dtype, shape, (a, b) = meta["dtype"], meta["shape"], meta["data_offsets"]
            nbytes = b - a
            dst = os.path.join(out, "tensors", name)
            if os.path.exists(dst) and os.path.getsize(dst) == nbytes:
                print(f"  have {name} {shape} {dtype}", flush=True)
            else:
                t0 = time.time()
                # chunked: a 3.4 GB tensor in 64 MB ranges, appended
                with open(dst + ".part", "wb") as f:
                    off = a
                    while off < b:
                        end = min(b, off + (64 << 20)) - 1
                        f.write(get(url, {"Range": f"bytes={8 + hlen + off}-{8 + hlen + end}"}))
                        off = end + 1
                os.replace(dst + ".part", dst)
                mb = nbytes / 1e6
                print(f"  got  {name} {shape} {dtype} {mb:.0f} MB in {time.time() - t0:.0f}s", flush=True)
            manifest[name] = {"dtype": dtype, "shape": shape, "nbytes": nbytes}

    # one safetensors file from the raw tensors
    names = sorted(manifest)
    hdr = {}
    off = 0
    for n in names:
        m = manifest[n]
        hdr[n] = {"dtype": m["dtype"], "shape": m["shape"], "data_offsets": [off, off + m["nbytes"]]}
        off += m["nbytes"]
    hdr_bytes = json.dumps(hdr, separators=(",", ":")).encode()
    pad = (8 - len(hdr_bytes) % 8) % 8
    hdr_bytes += b" " * pad
    st_path = os.path.join(out, "model.safetensors")
    with open(st_path + ".part", "wb") as f:
        f.write(struct.pack("<Q", len(hdr_bytes)))
        f.write(hdr_bytes)
        for n in names:
            with open(os.path.join(out, "tensors", n), "rb") as src:
                while True:
                    chunk = src.read(64 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
    os.replace(st_path + ".part", st_path)
    with open(os.path.join(out, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": off}, "weight_map": {n: "model.safetensors" for n in names}}, f)
    print(f"wrote {st_path} ({off / 1e9:.2f} GB, {len(names)} tensors)")


if __name__ == "__main__":
    sys.exit(main())
