#!/usr/bin/env python3
"""parity_cmp.py DIR: CUDA vs the float host per prompt — top-1 agreement along the CUDA ids, the largest and
mean |logit difference|, and where the free greedy texts part (the host's top-1/top-2 margin there)."""
import glob, os, sys
import numpy as np
d = sys.argv[1]
print(f"== {os.path.basename(d)}")
for f in sorted(glob.glob(f"{d}/cuda/*.gen")):
    n = os.path.basename(f)[:-4]
    same = steps = 0; mx = 0.0; mean = []
    i = 0
    while True:
        a, b = f"{d}/cuda/{n}.logits/step{i:05d}.f32", f"{d}/cpu_forced/{n}.logits/step{i:05d}.f32"
        if not (os.path.exists(a) and os.path.exists(b)):
            break
        x, y = np.fromfile(a, np.float32), np.fromfile(b, np.float32)
        k = min(len(x), len(y)); x, y = x[:k], y[:k]
        diff = np.abs(x - y); mx = max(mx, float(diff.max())); mean.append(float(diff.mean()))
        same += int(x.argmax() == y.argmax()); steps += 1; i += 1
    ga = open(f).read().split()
    fi = f"{d}/cpu_free/{n}.ids"
    gb = open(fi).read().split()[-len(ga):] if os.path.exists(fi) else []
    part = next((j for j, (p, q) in enumerate(zip(ga, gb)) if p != q), None)
    margin = ""
    if part is not None:
        y = np.fromfile(f"{d}/cpu_forced/{n}.logits/step{part:05d}.f32", np.float32)
        t = np.sort(y)[-2:]; margin = f", host margin there {t[1]-t[0]:.3f}"
    print(f"{n:14s} top-1 {same}/{steps} | max |dlogit| {mx:.3f} mean {np.mean(mean) if mean else 0:.4f} | "
          f"free texts {'identical' if part is None else f'part at step {part}{margin}'}")
