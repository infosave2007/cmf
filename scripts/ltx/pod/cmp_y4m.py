#!/usr/bin/env python3
"""PSNR of the luma plane between two .y4m clips of one size, frame by frame.

  cmp_y4m.py A.y4m B.y4m
"""
import sys

import numpy as np


def frames(path):
    with open(path, "rb") as f:
        head = f.readline().decode().split()
        w = int(next(t[1:] for t in head if t.startswith("W")))
        h = int(next(t[1:] for t in head if t.startswith("H")))
        cs = next((t[1:] for t in head if t.startswith("C")), "420")
        chroma = {"420": w * h // 2, "422": w * h, "444": 2 * w * h, "mono": 0}[cs[:3] if cs[:3] in ("420", "422", "444") else cs]
        out = []
        while True:
            line = f.readline()
            if not line:
                break
            y = np.frombuffer(f.read(w * h), dtype=np.uint8).astype(np.float64)
            f.read(chroma)
            out.append(y)
        return out


a, b = frames(sys.argv[1]), frames(sys.argv[2])
if len(a) != len(b) or not a or a[0].size != b[0].size:
    sys.exit(f"clips differ in shape: {len(a)} vs {len(b)} frames")
v = []
for x, y in zip(a, b):
    mse = ((x - y) ** 2).mean()
    v.append(float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse))
v = np.array(v)
fin = v[np.isfinite(v)]
same = int(np.isinf(v).sum())
if len(fin):
    print(f"{len(v)} frames  luma psnr min {fin.min():.2f}  median {np.median(fin):.2f}  max {fin.max():.2f} dB  identical {same}")
else:
    print(f"{len(v)} frames  identical")
