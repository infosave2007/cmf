#!/usr/bin/env python3
"""Logits of `CMF_DORY_DUMP` against the reference: max and rms difference, the top tokens."""
import json, sys
import numpy as np

dump, ref_npy = sys.argv[1], sys.argv[2]
a = np.fromfile(dump, dtype=np.float32)
b = np.load(ref_npy)
d = a - b
print(f"max|d| {np.abs(d).max():.4f}  rms {np.sqrt((d*d).mean()):.5f}  top1 {a.argmax()} vs {b.argmax()}  top5 {list(np.argsort(-a)[:5])} vs {list(np.argsort(-b)[:5])}")
