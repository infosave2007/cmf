"""Paired comparison of per-window NLLs (160 windows x 511 scored tokens), the q2tp study's convention:
delta = exp(mean per-window log-ppl difference) - 1, SE = SE of that mean x 100 (a relative SE).
usage: paired.py A B [A B ...]   each name is TAG (ours, /root/eng_q2tp/pw) or q2x:TAG (study, read-only)"""
import sys
import numpy as np

def load(name):
    if name.startswith("q2x:"):
        return np.load(f"/root/q2x/pw/{name[4:]}_w160.npy")
    if name.startswith("q4x:"):
        return np.load(f"/root/q4x/pw/{name[4:]}_w160.npy")
    return np.load(f"/root/eng_q2tp/pw/{name}_w160.npy")

args = sys.argv[1:]
for a, b in zip(args[0::2], args[1::2]):
    x, y = load(a) / 511.0, load(b) / 511.0
    d = x - y
    print(f"{a:>14} {np.exp(x.mean()):9.3f}  vs {b:>14} {np.exp(y.mean()):9.3f}   "
          f"delta {100 * (np.exp(d.mean()) - 1):+7.2f} % +- {100 * d.std(ddof=1) / np.sqrt(len(d)):.2f}")
