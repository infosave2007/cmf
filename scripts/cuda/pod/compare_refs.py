#!/usr/bin/env python3
"""Compare two backends' outputs kept by ref_outputs.sh.

  compare_refs.py <ref tag> <other tag> [--root /root/cuda]

For every prompt: how many of the generated ids agree (the common prefix and
the total), and over the decode steps both kept: the largest and the mean
absolute logit difference, and how often the top-1 id is the same. When
<other tag> was teacher-forced along <ref tag> (FORCE=), every step compares;
a free run compares up to the first differing id.
"""
import glob
import os
import sys

import numpy as np


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    root = "/root/cuda"
    if "--root" in sys.argv:
        root = sys.argv[sys.argv.index("--root") + 1]
    ref, oth = args[0], args[1]
    rd, od = f"{root}/out/ref/{ref}", f"{root}/out/ref/{oth}"
    tot_steps = tot_top1 = 0
    worst = 0.0
    print(f"{oth} against {ref}")
    print(f"{'prompt':<16} {'ids same':>9} {'prefix':>7} {'steps':>6} {'top-1 same':>10} {'max |d|':>9} {'mean |d|':>9}")
    for gen in sorted(glob.glob(f"{rd}/*.gen")):
        name = os.path.basename(gen)[:-4]
        a = open(gen).read().split()
        try:
            b = open(f"{od}/{name}.gen").read().split()
        except FileNotFoundError:
            print(f"{name:<16} missing in {oth}")
            continue
        n = min(len(a), len(b))
        same = sum(1 for i in range(n) if a[i] == b[i])
        prefix = next((i for i in range(n) if a[i] != b[i]), n)
        steps = top1 = 0
        mx = mean = 0.0
        # the logits of step i led to id i: comparable while the ids before agree
        for i in range(n):
            if a[:i] != b[:i]:
                break
            fa, fb = f"{rd}/{name}.logits/step{i:05d}.f32", f"{od}/{name}.logits/step{i:05d}.f32"
            if not (os.path.exists(fa) and os.path.exists(fb)):
                break
            la, lb = np.fromfile(fa, dtype=np.float32), np.fromfile(fb, dtype=np.float32)
            if la.shape != lb.shape:
                break
            d = np.abs(la - lb)
            mx = max(mx, float(d.max()))
            mean += float(d.mean())
            top1 += int(la.argmax() == lb.argmax())
            steps += 1
        mean = mean / steps if steps else 0.0
        tot_steps += steps
        tot_top1 += top1
        worst = max(worst, mx)
        print(f"{name:<16} {same:>4}/{n:<4} {prefix:>7} {steps:>6} {top1:>5}/{steps:<4} {mx:>9.4f} {mean:>9.5f}")
    if tot_steps:
        print(f"total: top-1 same {tot_top1}/{tot_steps} ({100.0 * tot_top1 / tot_steps:.1f} %), largest logit difference {worst:.4f}")


if __name__ == "__main__":
    main()
