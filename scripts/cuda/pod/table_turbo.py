#!/usr/bin/env python3
"""The medians of chain_final_turbo.sh's runs: a Markdown table, and with
--json the same numbers for the page's figures (card_figures.py).

  table_turbo.py [--root /root/cuda] [--prefix ft] [--json FILE]
"""
import glob
import json
import os
import re
import statistics
import sys

root = "/root/cuda"
prefix = "ft"
if "--root" in sys.argv:
    root = sys.argv[sys.argv.index("--root") + 1]
if "--prefix" in sys.argv:
    prefix = sys.argv[sys.argv.index("--prefix") + 1]


def load(path):
    txt = open(path).read()
    j = json.loads(txt[txt.index("{"):txt.rindex("}") + 1])
    return j, int(open(path[:-5] + ".vram").read().strip())


def decode_mem(path, j, tokens):
    """What the card holds in the second half of a long decode (bench_one.sh's rule)."""
    tr = [int(x) for x in open(path[:-5] + ".vram_trace").read().split()]
    n = int(tokens / max(j["decode_tok_s_steady"], 1e-6) / 0.25)
    mid = sorted(tr[-3 - n // 2:-3]) if n >= 16 and len(tr) > n + 3 else []
    return mid[len(mid) // 2] if mid else None


rows = {}
for path in glob.glob(f"{root}/out/{prefix}_*_r*.json"):
    m = re.match(rf"{prefix}_(vkt|cut|cux)_(q4|q8)_(\d+)_r(\d+)\.json", os.path.basename(path))
    if not m:
        continue
    tag, quant, ctx, _ = m.groups()
    try:
        j, vram = load(path)
    except Exception:
        continue
    rows.setdefault((quant, int(ctx), tag), []).append((j["prefill_tok_s"], j["ttft_s"], j["decode_tok_s_steady"], vram))
long = {}
for path in glob.glob(f"{root}/out/{prefix}_*L_q4_*.json"):
    m = re.match(rf"{prefix}_(vkt|cut|cux)L_(q4|q8)_(\d+)\.json", os.path.basename(path))
    if not m:
        continue
    tag, quant, ctx = m.groups()
    try:
        j, _ = load(path)
        long[(quant, int(ctx), tag)] = (decode_mem(path, j, 500), j["decode_tok_s_steady"])
    except Exception:
        continue
names = {"vkt": "Vulkan, far field", "cut": "CUDA, far field", "cux": "CUDA, exact attention"}
order = ["vkt", "cut", "cux"]
print("| file | prompt | path | runs | prompt, tok/s | first token, s | decode, tok/s | peak VRAM, MiB | VRAM while decoding, MiB | decode over 500 tokens, tok/s |")
print("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|")
out = {}
for (quant, ctx, tag), v in sorted(rows.items(), key=lambda kv: (kv[0][0], kv[0][1], order.index(kv[0][2]))):
    med = [statistics.median(x[i] for x in v) for i in range(4)]
    lo = [min(x[i] for x in v) for i in range(4)]
    hi = [max(x[i] for x in v) for i in range(4)]
    dm, dl = long.get((quant, ctx, tag), (None, None))
    print(f"| {'q4tp' if quant == 'q4' else 'q8_2f'} | {ctx} | {names[tag]} | {len(v)} | {med[0]:.0f} ({lo[0]:.0f}–{hi[0]:.0f}) | "
          f"{med[1]:.2f} ({lo[1]:.2f}–{hi[1]:.2f}) | {med[2]:.1f} ({lo[2]:.1f}–{hi[2]:.1f}) | {med[3]:.0f} | "
          f"{dm if dm is not None else '-'} | {f'{dl:.1f}' if dl is not None else '-'} |")
    out[f"{quant}_{ctx}_{tag}"] = {"prompt": med[0], "first": med[1], "decode": med[2], "peak": med[3], "decode_mem": dm, "decode_long": dl}
if "--json" in sys.argv:
    json.dump(out, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)
