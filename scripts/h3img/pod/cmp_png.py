#!/usr/bin/env python3
"""PSNR between two images of one size (the same latent decoded two ways, or
the same seed on two backends).

  cmp_png.py A.png B.png
"""
import sys

import numpy as np
from PIL import Image

a = np.asarray(Image.open(sys.argv[1]).convert("RGB"), dtype=np.float64)
b = np.asarray(Image.open(sys.argv[2]).convert("RGB"), dtype=np.float64)
if a.shape != b.shape:
    sys.exit(f"shapes differ: {a.shape} vs {b.shape}")
mse = ((a - b) ** 2).mean()
psnr = float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse)
print(f"{a.shape[1]}x{a.shape[0]}  psnr {psnr:.2f} dB  mean abs diff {np.abs(a - b).mean():.3f}  "
      f"pixels off by > 8: {100 * (np.abs(a - b).max(axis=2) > 8).mean():.2f}%")
