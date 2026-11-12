#!/usr/bin/env python3
"""Compare one f32 tensor of two safetensors files: relative rms of the
difference, the largest difference, and — for frames in [-1, 1] — the PSNR.

  cmp_st.py A.safetensors B.safetensors [NAME]
"""
import json
import struct
import sys

import numpy as np


def load(path, name):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        head = json.loads(f.read(n))
        keys = [k for k in head if k != "__metadata__"]
        k = name if name in head else keys[0]
        lo, hi = head[k]["data_offsets"]
        f.seek(8 + n + lo)
        return k, np.frombuffer(f.read(hi - lo), dtype="<f4").astype(np.float64), head[k]["shape"]


name = sys.argv[3] if len(sys.argv) > 3 else "frames"
ka, a, sa = load(sys.argv[1], name)
kb, b, sb = load(sys.argv[2], name)
if a.size != b.size:
    sys.exit(f"sizes differ: {ka} {sa} vs {kb} {sb}")
d = a - b
mse = (d * d).mean()
psnr = float("inf") if mse == 0 else 10 * np.log10(4.0 / mse)
print(f"{ka} {sa}: rel rms diff {np.sqrt(mse) / max(np.sqrt((a * a).mean()), 1e-30):.3e}  "
      f"max abs diff {np.abs(d).max():.3e}  psnr {psnr:.2f} dB (range 2)")
