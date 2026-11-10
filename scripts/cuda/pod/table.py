#!/usr/bin/env python3
"""The medians of chain_final.sh's runs as a Markdown table.

  table.py [--root /root/cuda] [--prefix fin]
"""
import glob
import json
import os
import re
import statistics
import sys

root = "/root/cuda"
prefix = "fin"
if "--root" in sys.argv:
    root = sys.argv[sys.argv.index("--root") + 1]
if "--prefix" in sys.argv:
    prefix = sys.argv[sys.argv.index("--prefix") + 1]
rows = {}
for path in glob.glob(f"{root}/out/{prefix}_*_r*.json"):
    m = re.match(rf"{prefix}_(\w+?)_(q4|q8)_(\d+)_r(\d+)\.json", os.path.basename(path))
    if not m:
        continue
    path_tag, quant, ctx, _ = m.groups()
    try:
        txt = open(path).read()
        j = json.loads(txt[txt.index("{"):txt.rindex("}") + 1])
        vram = int(open(path[:-5] + ".vram").read().strip())
    except Exception:
        continue
    rows.setdefault((quant, int(ctx), path_tag), []).append(
        (j["prefill_tok_s"], j["ttft_s"], j["decode_tok_s_steady"], vram))
names = {"vk": "Vulkan, exact attention", "cu": "CUDA", "cux": "CUDA, exact f32 prompt", "vkturbo": "Vulkan, turbo as shipped"}
print("| file | prompt | path | runs | prompt, tok/s | first token, s | decode, tok/s | peak VRAM, MiB |")
print("|---|---:|---|---:|---:|---:|---:|---:|")
for (quant, ctx, tag), v in sorted(rows.items(), key=lambda kv: (kv[0][0], kv[0][1], ["vk", "cu", "cux", "vkturbo"].index(kv[0][2]))):
    med = [statistics.median(x[i] for x in v) for i in range(4)]
    lo = [min(x[i] for x in v) for i in range(4)]
    hi = [max(x[i] for x in v) for i in range(4)]
    print(f"| {'q4tp' if quant == 'q4' else 'q8_2f'} | {ctx} | {names[tag]} | {len(v)} | {med[0]:.0f} ({lo[0]:.0f}–{hi[0]:.0f}) | "
          f"{med[1]:.2f} | {med[2]:.1f} ({lo[2]:.1f}–{hi[2]:.1f}) | {med[3]:.0f} |")
