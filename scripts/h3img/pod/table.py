#!/usr/bin/env python3
"""The medians of suite.sh's runs: a Markdown table, and with --json the same
numbers for the page's figures.

  table.py [--root /root/h3i] [--prefix s] [--json FILE]
"""
import glob
import json
import os
import re
import statistics
import sys

root = sys.argv[sys.argv.index("--root") + 1] if "--root" in sys.argv else "/root/h3i"
prefix = sys.argv[sys.argv.index("--prefix") + 1] if "--prefix" in sys.argv else "s"
line = re.compile(r"in ([\d.]+)s \(text ([\d.]+)s for (\d+) tokens, dit load ([\d.]+)s, denoise ([\d.]+)s, "
                  r"median step ([\d.]+)s over (\d+) image tokens, vae ([\d.]+)s\)")
rows = {}
for path in glob.glob(f"{root}/out/{prefix}_*_r*.log"):
    m = re.match(rf"{prefix}_(cuda|vulkan)_(t2i|edit)_(\d+)_c(\d+)_r(\d+)\.log", os.path.basename(path))
    if not m:
        continue
    backend, kind, size, cap, _ = m.groups()
    txt = open(path, errors="replace").read()
    t = line.search(txt)
    s = re.search(r"peak_vram_mib=(\d+)", open(path[:-4] + ".sum").read()) if os.path.exists(path[:-4] + ".sum") else None
    if not t or not s:
        continue
    total, text, toks, load, den, step, itoks, vae = (float(x) for x in t.groups())
    rows.setdefault((backend, kind, int(size), int(cap)), []).append((total, text, den, step, vae, float(s.group(1))))
print("| backend | mode | size | ceiling, MiB | runs | total, s | text, s | denoise, s | step, s | decode, s | peak VRAM, MiB |")
print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
out = {}
for (backend, kind, size, cap), v in sorted(rows.items()):
    med = [statistics.median(x[i] for x in v) for i in range(6)]
    lo, hi = min(x[0] for x in v), max(x[0] for x in v)
    print(f"| {backend} | {kind} | {size} | {cap or '-'} | {len(v)} | {med[0]:.1f} ({lo:.1f}–{hi:.1f}) | {med[1]:.1f} | "
          f"{med[2]:.1f} | {med[3]:.2f} | {med[4]:.1f} | {med[5]:.0f} |")
    out[f"{backend}_{kind}_{size}_c{cap}"] = dict(total=med[0], text=med[1], denoise=med[2], step=med[3], vae=med[4],
                                                 peak=med[5], runs=len(v))
if "--json" in sys.argv:
    json.dump(out, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)
