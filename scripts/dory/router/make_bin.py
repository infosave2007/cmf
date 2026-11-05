#!/usr/bin/env python3
"""Topologies from router_fit.py as the engine's flat file, and the bias of a few
shares of one-pass tokens on the fit split. make_bin.py topo.npz probe.npz SIG OUT"""
import sys
import numpy as np
topo, probe, sig, out = sys.argv[1:5]
t = np.load(topo)
d = np.load(probe)
S, ans, kinds = d["sample"], d["answer"], d["kinds"]
fit = (S % 2 == 0) & ans & (kinds[S] != "wiki")
me, be, mh, bh = t["me"], t["be"], t["mh"], t["bh"]
r, dim = be.shape[1], be.shape[0]
head = np.array([r, dim], dtype=np.float32)
np.concatenate([head, me.astype(np.float32), be.T.astype(np.float32).ravel(), mh.astype(np.float32), bh.T.astype(np.float32).ravel()]).astype("<f4").tofile(out)
x = d[sig][fit].astype(np.float32)
def err(x, mu, b):
    y = x - mu
    return (y * y).sum(1) - ((y @ b) ** 2).sum(1)
score = err(x, mh, bh) - err(x, me, be)
fs = np.sort(score)
for q in (0.3, 0.4, 0.5, 0.6, 0.7):
    print(f"share {q}: bias {fs[int(len(fs) * (1 - q))]:.6f}")
