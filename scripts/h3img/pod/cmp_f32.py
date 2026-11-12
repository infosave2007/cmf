#!/usr/bin/env python3
"""Compare two raw little-endian f32 dumps (CMF_LATENT_OUT, CMF_TE_DUMP bodies):
relative rms of the difference, the largest difference, and the rms of each.

  cmp_f32.py A B [--skip BYTES]
"""
import sys

import numpy as np

skip = int(sys.argv[sys.argv.index("--skip") + 1]) if "--skip" in sys.argv else 0
a = np.fromfile(sys.argv[1], dtype="<f4", offset=skip).astype(np.float64)
b = np.fromfile(sys.argv[2], dtype="<f4", offset=skip).astype(np.float64)
if a.size != b.size:
    sys.exit(f"sizes differ: {a.size} vs {b.size}")
d = a - b
ra, rb = np.sqrt((a * a).mean()), np.sqrt((b * b).mean())
print(f"n {a.size}  rms {ra:.5f} / {rb:.5f}  rel rms diff {np.sqrt((d * d).mean()) / max(ra, 1e-30):.3e}  "
      f"max abs diff {np.abs(d).max():.3e}  nonfinite {np.count_nonzero(~np.isfinite(a))} / {np.count_nonzero(~np.isfinite(b))}")
