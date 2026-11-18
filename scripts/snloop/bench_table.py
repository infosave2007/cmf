#!/usr/bin/env python3
"""The rows of bench.sh logs as JSON and a markdown table.
usage: bench_table.py LOG_DIR TAG [TAG ...]   (reads LOG_DIR/bench_TAG.txt)"""
import json, re, sys

d, tags = sys.argv[1], sys.argv[2:]
out = {}
for tag in tags:
    rows = []
    for line in open(f"{d}/bench_{tag}.txt"):
        if "|" not in line or line.startswith("BENCH_DONE"):
            continue
        n = int(line.split("|")[0])
        vram = int(re.search(r"vram_mib=(\d+)", line).group(1))
        rss = int(re.search(r"rss_kb=(\d+)", line).group(1))
        pp = re.search(r"Prompt:\s*(\d+) tokens \|[^;]*?([\d.]+) tok/s", line)
        dec = re.search(r"Decode:\s*(\d+) tokens \| ([\d.]+) tok/s steady \(TTFT ([\d.]+)s", line)
        rows.append({"ctx": n, "prompt_tokens": int(pp.group(1)) if pp else None,
                     "prompt_tps": float(pp.group(2)) if pp else None,
                     "decode": float(dec.group(2)) if dec else None, "ttft": float(dec.group(3)) if dec else None,
                     "vram_gb": round(vram / 1024, 2), "rss_gb": round(rss / 1024 / 1024, 2)})
    out[tag] = rows
    print(f"\n{tag}\n| prompt | first token, s | prompt, tok/s | decode, tok/s | peak VRAM, GB | host RSS, GB |")
    print("|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        print(f"| {r['ctx']} | {r['ttft']} | {r['prompt_tps']} | {r['decode']} | {r['vram_gb']} | {r['rss_gb']} |")
json.dump(out, open(f"{d}/bench_table.json", "w"), indent=1)
