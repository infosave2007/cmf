#!/usr/bin/env python3
"""The confidence gate with the patent-1 router in front of it: tokens the router calls
hard skip the first read-out and go straight to the second pass. Cost in layers: 72 for
input, a pass and output, 40 for a pass, 16 for a read-out, the head as `head` layers."""
import numpy as np, sys

d = np.load("router/probe.npz")
t = np.load("router/topo.phi_1.npz")
S, ans, kinds = d["sample"], d["answer"], d["kinds"]
test = (S % 2 == 1) & ans & (kinds[S] != "wiki")
fit = (S % 2 == 0) & ans & (kinds[S] != "wiki")
x = d["phi_1"].astype(np.float32)
def err(x, mu, b):
    y = x - mu
    return (y * y).sum(1) - ((y @ b) ** 2).sum(1)
score = err(x, t["mh"], t["bh"]) - err(x, t["me"], t["be"])
p1, t1, t2, n1, n2 = d["p1"], d["t1"], d["t2"], d["n1"], d["n2"]
head = float(sys.argv[1]) if len(sys.argv) > 1 else 5.5
fence = float(sys.argv[2]) if len(sys.argv) > 2 else 1.6
base = 112 + head
fs = np.sort(score[fit])
for th in (0.5, 0.7, 0.9):
    conf = p1[test] >= th
    one = conf
    cost = np.where(conf, 72 + head, 72 + head + 56 + head).mean()
    print(f"conf {th}: one pass {conf.mean():.3f}  x{base/cost:.3f}  nll {np.where(conf,n1[test],n2[test]).mean():.4f}  top1 {np.where(conf,t1[test]==t2[test],True).mean():.4f}")
    for q in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5):
        tau = fs[int(len(fs) * q)]
        hard = score[test] < tau
        c = np.where(hard, 112 + head + fence, np.where(conf, 72 + head + fence, 72 + head + 56 + head + fence)).mean()
        keep = conf & ~hard
        print(f"   + router sends {hard.mean():.3f} straight to two passes: one pass {keep.mean():.3f}  x{base/c:.3f}  nll {np.where(keep,n1[test],n2[test]).mean():.4f}  top1 {np.where(keep,t1[test]==t2[test],True).mean():.4f}")
