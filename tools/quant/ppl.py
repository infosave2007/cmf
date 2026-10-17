#!/usr/bin/env python3
"""Exact perplexity of a REAL CMF file on cortiq's window protocol, in torch on the GPU.

Windows are placed exactly like `cortiq ppl --windows N --window-len WL` (offsets
k * ((n - WL - 1) // N), fresh context, no BOS); every payload is decoded from the file with
the decoder's arithmetic (qmodel.py), matmuls in exact f32 (no TF32). Per-window NLLs are
saved so files can be compared with paired standard errors (--ref). Parity with the engine:
`cortiq ppl FILE --file TEXT --windows 1 --window-len WL` prints window 0's NLL.

  ppl.py --model G-C2.cmf --ids test.ids --windows 160 --also 12 --out pw/G-C2 [--ref pw/C2]
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402
import torch  # noqa: E402

import calib  # noqa: E402
import cmfio  # noqa: E402
import qmodel  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--ids", required=True, help="token ids of the evaluation text (cmfio.load_ids)")
    ap.add_argument("--windows", type=int, default=160)
    ap.add_argument("--also", type=int, nargs="*", default=[], help="extra window counts, e.g. 12")
    ap.add_argument("--wl", type=int, default=512)
    ap.add_argument("--mb", type=int, default=4)
    ap.add_argument("--block", type=int, default=32, help="windows per streamed pass")
    ap.add_argument("--residency", choices=("resident", "stream"), default="stream")
    ap.add_argument("--vchunk", type=int, default=16384)
    ap.add_argument("--out", help="prefix for per-window NLLs (<out>_w<N>.npy) and <out>.json")
    ap.add_argument("--ref", help="prefix of a reference run: print the paired change vs it")
    ap.add_argument("--gpu-cap-gb", type=float, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    qmodel.setup_torch(0, tf32=False, deterministic=True)
    dev = qmodel.resolve_device(args.device, args.gpu_cap_gb)
    t0 = time.time()
    ids = cmfio.load_ids(args.ids)
    cmf = cmfio.CmfFile(args.model)
    model = qmodel.Model(cmf, dev, residency=args.residency, vchunk=args.vchunk)
    res = {"model": os.path.abspath(args.model), "ids": os.path.abspath(args.ids), "wl": args.wl}
    for n in [args.windows] + list(args.also):
        w = calib.gather(ids, calib.even_starts(len(ids), n, args.wl), args.wl)
        nll = model.evaluate(w, mb=args.mb, block=args.block)["nll"]
        tok = n * (args.wl - 1)
        ppl = float(np.exp(nll.sum() / tok))
        res[f"w{n}"] = {"ppl": ppl, "tokens": tok, "window0_nll": float(nll[0])}
        line = f"w{n}: ppl {ppl:.4f} ({tok} tokens), window-0 NLL {nll[0]:.6f}"
        if args.out:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            np.save(f"{args.out}_w{n}.npy", nll)
        if args.ref and os.path.exists(f"{args.ref}_w{n}.npy"):
            ref = np.load(f"{args.ref}_w{n}.npy")
            lr = (nll - ref) / (args.wl - 1)                       # per-window log-ratio of ppl
            d = float(np.expm1(lr.mean()) * 100)
            se = float(lr.std(ddof=1) / np.sqrt(len(lr)) * 100)
            res[f"w{n}"]["vs_ref_pct"], res[f"w{n}"]["vs_ref_se_pct"] = d, se
            line += f", vs ref {d:+.3f} % +- {se:.3f}"
        print(line, flush=True)
    res["seconds"] = time.time() - t0
    if args.out:
        with open(f"{args.out}.json", "w") as f:
            json.dump(res, f, indent=1)
    return res


if __name__ == "__main__":
    main()
