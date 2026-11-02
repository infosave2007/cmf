#!/usr/bin/env python3
"""Compare a CMF_OCR_DUMP directory and a text with ref_unlimited_ocr.py output."""
import json, sys, difflib, os
import numpy as np
ref, dump = sys.argv[1:3]
txt = sys.argv[3] if len(sys.argv) > 3 else None
g = np.fromfile(f"{ref}/pixels_global.f32", "<f4")
rows = np.fromfile(f"{ref}/projector.f32", "<f4")
print("ref global", g.shape, "projector tail", rows.shape)
d = np.fromfile(f"{dump}/global_0.f32", "<f4")
if d.size == g.size or g.size % d.size == 0:
    print("global pixels max|d|", np.abs(g[:d.size] - d).max())
c = np.fromfile(f"{ref}/pixels_crop.f32", "<f4")
n = 3 * 640 * 640
if c.size % n == 0 and os.path.exists(f"{dump}/tile_0_0.f32"):
    for i in range(c.size // n):
        t = np.fromfile(f"{dump}/tile_0_{i}.f32", "<f4")
        print("tile", i, "max|d|", np.abs(c[i * n:(i + 1) * n] - t).max())
ids = json.load(open(f"{ref}/gen.json"))["prompt_token_ids"]
e = np.fromfile(f"{ref}/embeds.f32", "<f4").reshape(-1, 1280)
mask = np.array(ids) == 128815
at = 0; pages = []
for f in sorted(x for x in os.listdir(dump) if x.startswith("rows_")):
    pages.append(np.fromfile(f"{dump}/{f}", "<f4").reshape(-1, 1280))
r = np.concatenate(pages)
ref_r = e[mask]
print("image rows", r.shape, ref_r.shape)
if r.shape == ref_r.shape:
    cos = (r * ref_r).sum(1) / (np.linalg.norm(r, axis=1) * np.linalg.norm(ref_r, axis=1))
    print("image rows max|d|", np.abs(r - ref_r).max(), "cos mean", cos.mean(), "min", cos.min())
if txt:
    a = open(f"{ref}/gen.txt").read(); b = open(txt).read()
    k = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), min(len(a), len(b)))
    print("text ref", len(a), "got", len(b), "common prefix", k, "ratio", round(difflib.SequenceMatcher(None, a, b, autojunk=False).ratio(), 4))
    print("ref:", repr(a[max(0,k-60):k+80])); print("got:", repr(b[max(0,k-60):k+80]))
