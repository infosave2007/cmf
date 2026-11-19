#!/usr/bin/env python3
"""Layer-by-layer relative error of two dumps (`CMF_HY4_DUMP` / `HY4_DUMP`).
usage: dump_diff.py REF_DIR OTHER_DIR POS [POS...]"""
import os, sys
import numpy as np

ref, oth, poss = sys.argv[1], sys.argv[2], [int(p) for p in sys.argv[3:]]
for kind in ("attn", "streams"):
    print(kind, "layer", *[f"p{p:>6}" for p in poss])
    for li in range(64):
        row = []
        for p in poss:
            a, b = f"{ref}/{kind}_l{li}_p{p}.f32", f"{oth}/{kind}_l{li}_p{p}.f32"
            if not (os.path.exists(a) and os.path.exists(b)):
                row = None
                break
            x, y = np.fromfile(a, np.float32), np.fromfile(b, np.float32)
            row.append(np.linalg.norm(x - y) / max(np.linalg.norm(x), 1e-30))
        if row is None:
            if li > 0:
                break
            continue
        print(f"{kind:7} {li:3}", *[f"{v:8.4f}" for v in row])
