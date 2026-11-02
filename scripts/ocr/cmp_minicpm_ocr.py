#!/usr/bin/env python3
"""Compare a CMF_OCR_DUMP directory with ref_minicpm_ocr.py output."""
import json, sys, glob, re
import numpy as np
ref, dump = sys.argv[1:3]
inp = json.load(open(f"{ref}/inputs.json"))
sizes = inp["target_sizes"]
px = np.fromfile(f"{ref}/pixels.f32", "<f4").reshape(3, 14, -1)
rows = np.fromfile(f"{ref}/rows.f32", "<f4").reshape(-1, 1024)
at = 0; r0 = 0
for i, (h, w) in enumerate(sizes):
    n = h * w
    strip = px[:, :, at * 14:(at + n) * 14]
    # to raster rows (c, py, px)
    ref_p = np.stack([strip[:, :, k * 14:(k + 1) * 14].reshape(-1) for k in range(n)])
    got_p = np.fromfile(f"{dump}/pixels_0_{i}.f32", "<f4")
    got_r = np.fromfile(f"{dump}/rows_0_{i}.f32", "<f4").reshape(-1, 1024)
    ref_r = rows[r0:r0 + n // 4]
    dp = np.abs(ref_p.reshape(-1) - got_p).max() if got_p.size == ref_p.size else float("nan")
    cos = (ref_r * got_r).sum(1) / (np.linalg.norm(ref_r, axis=1) * np.linalg.norm(got_r, axis=1))
    print(f"view {i} grid {h}x{w}: pixels max|d| {dp:.4f}, rows max|d| {np.abs(ref_r-got_r).max():.5f}, cos mean {cos.mean():.7f} min {cos.min():.7f}")
    at += n; r0 += n // 4
