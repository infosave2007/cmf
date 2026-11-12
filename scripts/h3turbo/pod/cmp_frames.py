#!/usr/bin/env python3
"""PSNR between two renders of one clip, frame by frame, from the frame
dumps of `cortiq animate --frames-dir` (frame_%04d.ppm), and the waveform
difference of the two .wav files beside them when both exist.

  cmp_frames.py A.frames B.frames [A.wav B.wav]
"""
import glob
import os
import sys
import wave

import numpy as np
from PIL import Image


def psnr(a, b):
    mse = ((a - b) ** 2).mean()
    return float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse)


fa = sorted(glob.glob(os.path.join(sys.argv[1], "frame_*.ppm")))
fb = sorted(glob.glob(os.path.join(sys.argv[2], "frame_*.ppm")))
if not fa or len(fa) != len(fb):
    sys.exit(f"frame counts differ: {len(fa)} vs {len(fb)}")
vals = []
for x, y in zip(fa, fb):
    a = np.asarray(Image.open(x).convert("RGB"), dtype=np.float64)
    b = np.asarray(Image.open(y).convert("RGB"), dtype=np.float64)
    if a.shape != b.shape:
        sys.exit(f"shapes differ: {a.shape} vs {b.shape}")
    vals.append(psnr(a, b))
v = np.array(vals)
same = int(np.isinf(v).sum())
fin = v[np.isfinite(v)]
if len(fin):
    print(f"{len(v)} frames  psnr min {fin.min():.2f}  median {np.median(fin):.2f}  max {fin.max():.2f} dB  identical {same}")
else:
    print(f"{len(v)} frames  identical")
if len(sys.argv) >= 5:
    def pcm(p):
        with wave.open(p) as w:
            return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float64)
    a, b = pcm(sys.argv[3]), pcm(sys.argv[4])
    n = min(len(a), len(b))
    d = a[:n] - b[:n]
    sig = (a[:n] ** 2).mean()
    snr = float("inf") if (d ** 2).mean() == 0 else 10 * np.log10(sig / (d ** 2).mean())
    print(f"audio {n} samples  snr {snr:.2f} dB  max abs diff {np.abs(d).max():.0f}")
