#!/usr/bin/env python3
"""Compare a raw little-endian f32 dump with a reference .npy, row-wise.
usage: cmp_rows.py <dump.f32> <ref.npy> [row_width]"""
import sys
import numpy as np
a = np.fromfile(sys.argv[1], dtype="<f4")
b = np.load(sys.argv[2]).astype(np.float32).reshape(-1)
w = int(sys.argv[3]) if len(sys.argv) > 3 else None
if a.size != b.size:
    print(f"size mismatch {a.size} vs {b.size}"); sys.exit(1)
if w:
    A, B = a.reshape(-1, w), b.reshape(-1, w)
    cos = (A * B).sum(1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-30)
    print(f"rows {A.shape[0]}: cosine mean {cos.mean():.6f} min {cos.min():.6f}; max|d| {np.abs(a-b).max():.4g} rel {np.abs(a-b).max()/np.abs(b).max():.3g}")
else:
    print(f"max|d| {np.abs(a-b).max():.4g} rel {np.abs(a-b).max()/np.abs(b).max():.3g}")
