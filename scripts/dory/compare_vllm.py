#!/usr/bin/env python3
"""Greedy tokens and first-step log-probabilities of vLLM against ref_dory.py."""
import json, math, sys
import numpy as np

v = json.load(open(sys.argv[1]))
r = json.load(open(sys.argv[2]))
d = sys.argv[3]
for name, cfgs in v.items():
    for tag, o in cfgs.items():
        a, b = o["greedy"], r[name][tag]["greedy"]
        n = min(len(a), len(b))
        same = next((i for i in range(n) if a[i] != b[i]), n)
        lg = np.load(f"{d}/{name}.{tag}.logits.npy").astype(np.float64)
        lp = lg - (lg.max() + np.log(np.exp(lg - lg.max()).sum()))
        err = max(abs(lp[int(t)] - p) for t, p in o["lp0"].items())
        print(f"{name:14s} {tag:14s} tokens equal {same}/{n}  max|dlogp| first step {err:.4f}")
