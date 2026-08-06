#!/usr/bin/env python3
"""Pull a few named tensors out of a remote safetensors file.

`cortiq animate-pack --lora` on a pruned MiniMax-H3 base needs the FULL
checkpoint's `time_embedder.*` — four tensors, 64 MB — because the Turbo
LoRA's adaLN update is written against the full 2688-d timestep
embedding, which the pruned checkpoint replaced with a rank-8 table.
Downloading 66 GB for 64 MB is not on, so read the header, then the
spans, with HTTP range requests, and write a small safetensors file.

    python3 tools/mmh3_fetch.py time-embedder --out temb.safetensors

`--check` additionally pulls block 0's full adaLN weight (520 MB) and
measures what the rank-8 curve actually costs, which is the one claim
the packer makes that is not arithmetic.
"""

import argparse
import json
import struct
import sys
import urllib.request

FULL = ("https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/"
        "diffusion_models/minimax_h3_fl2va_bf16.safetensors")
PRUNED = ("https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/"
          "diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors")

TIME_EMBEDDER = [
    "time_embedder.proj_in.weight",
    "time_embedder.proj_in.bias",
    "time_embedder.proj_out.weight",
    "time_embedder.proj_out.bias",
]

DTYPE_SIZE = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "U8": 1}


def _range(url, start, end):
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    return urllib.request.urlopen(req, timeout=600).read()


def header(url):
    n = struct.unpack("<Q", _range(url, 0, 7))[0]
    return json.loads(_range(url, 8, 8 + n - 1)), 8 + n


def fetch(url, names):
    """[(name, dtype, shape, raw bytes)] for `names`, one range GET each."""
    hdr, base = header(url)
    out = []
    for name in names:
        meta = hdr.get(name)
        if meta is None:
            raise SystemExit(f"{name}: not in the remote header")
        a, b = meta["data_offsets"]
        raw = _range(url, base + a, base + b - 1)
        if len(raw) != b - a:
            raise SystemExit(f"{name}: short read {len(raw)} of {b - a}")
        out.append((name, meta["dtype"], meta["shape"], raw))
        print(f"  {name} {meta['dtype']}{meta['shape']} {len(raw) / 1e6:.1f} MB",
              file=sys.stderr)
    return out


def write_safetensors(path, tensors):
    hdr, off = {}, 0
    for name, dtype, shape, raw in tensors:
        hdr[name] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + len(raw)]}
        off += len(raw)
    blob = json.dumps(hdr, separators=(",", ":")).encode()
    pad = (-len(blob)) % 8
    blob += b" " * pad
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for _, _, _, raw in tensors:
            f.write(raw)


def bf16(raw):
    import numpy as np
    return (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def to_f32(dtype, raw):
    import numpy as np
    if dtype == "BF16":
        return bf16(raw)
    if dtype == "F16":
        return np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    return np.frombuffer(raw, dtype=np.float32)


def check():
    """What the rank-8 adaLN curve costs, measured on block 0.

    Full: adaln(t) = W_full · silu(e(t)) + b. Pruned: W_p · u(t) + b.
    They are two spellings of the same map only to the extent that the
    time-embedding curve is rank 8; this prints the extent.
    """
    import numpy as np
    print("full checkpoint:", file=sys.stderr)
    te = {n: (d, s, r) for n, d, s, r in fetch(FULL, TIME_EMBEDDER + [
        "blocks.0.adaln_proj.linear.weight", "blocks.0.adaln_proj.linear.bias"])}
    print("pruned checkpoint:", file=sys.stderr)
    pr = {n: (d, s, r) for n, d, s, r in fetch(PRUNED, [
        "adaln_t_table", "blocks.0.adaln_proj.linear.weight",
        "blocks.0.adaln_proj.linear.bias"])}

    def arr(src, name):
        d, s, r = src[name]
        return to_f32(d, r).reshape(s)

    w_in, b_in = arr(te, "time_embedder.proj_in.weight"), arr(te, "time_embedder.proj_in.bias")
    w_out, b_out = arr(te, "time_embedder.proj_out.weight"), arr(te, "time_embedder.proj_out.bias")
    grid = arr(pr, "adaln_t_table").shape[0]
    t = np.linspace(0.0, 1.0, grid, dtype=np.float32)
    half = w_in.shape[1] // 2
    freqs = np.exp(-np.log(10000.0) * np.arange(half, dtype=np.float32) / half)
    args = t[:, None] * freqs[None]
    emb = np.concatenate([np.cos(args), np.sin(args)], axis=-1)
    silu = lambda x: x / (1.0 + np.exp(-x))
    e = silu(emb @ w_in.T + b_in) @ w_out.T + b_out          # [grid, 2688]
    e = silu(e)                                              # AdalnProj's own silu

    w_full, b_full = arr(te, "blocks.0.adaln_proj.linear.weight"), arr(te, "blocks.0.adaln_proj.linear.bias")
    w_p, b_p = arr(pr, "blocks.0.adaln_proj.linear.weight"), arr(pr, "blocks.0.adaln_proj.linear.bias")
    u = arr(pr, "adaln_t_table")                             # [grid, 8]

    ref = e @ w_full.T + b_full                              # [grid, 96768]
    got = u @ w_p.T + b_p
    err = np.abs(ref - got)
    print(f"adaln_t_table grid {grid}, rank {u.shape[1]}")
    print(f"bias  max|Δ| {np.abs(b_full - b_p).max():.3e}")
    print(f"adaln max|Δ| {err.max():.3e}  rel {err.max() / np.abs(ref).max():.3e}  "
          f"rms {np.sqrt((err ** 2).mean()):.3e} over |ref| rms {np.sqrt((ref ** 2).mean()):.3e}")
    s = np.linalg.svd(e - e.mean(0), compute_uv=False)
    print("time-curve singular values 1..12:",
          " ".join(f"{v / s[0]:.2e}" for v in s[:12]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["time-embedder", "check"])
    ap.add_argument("--out", default="minimax_h3_time_embedder.safetensors")
    ap.add_argument("--url", default=FULL)
    a = ap.parse_args()
    if a.what == "check":
        check()
        return
    write_safetensors(a.out, fetch(a.url, TIME_EMBEDDER))
    print(a.out)


if __name__ == "__main__":
    main()
