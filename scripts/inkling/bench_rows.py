#!/usr/bin/env python3
"""The lines of scripts/inkling/bench2.sh ("TAG backend cCTX: Prompt: ... Decode: ... vram_peak N MiB") as the rows
card_figures.py reads. bench_rows.py LOG [EXTRA.json] > figures_data.json
TAG chart-q4tp; EXTRA.json adds rows measured elsewhere."""
import json, re, sys

FILES = {"chart-q4tp": "q4tp"}
BACKENDS = {"cuda": "cuda", "wgpu": "vulkan", "metal": "metal"}
data = {}
for line in open(sys.argv[1], errors="replace"):
    m = re.match(r"\s*(chart-\S+) (\S+) c(\d+): .*?Decode: \d+ tokens \| ([\d.]+) tok/s steady \(TTFT ([\d.]+)s.*?vram_peak (\d+) MiB", line)
    if not m or m.group(1) not in FILES or m.group(2) not in BACKENDS:
        continue
    row = {"ctx": int(m.group(3)), "ttft": float(m.group(5)), "decode": float(m.group(4)), "vram_gb": round(int(m.group(6)) / 1024, 2)}
    data.setdefault(BACKENDS[m.group(2)], {}).setdefault(FILES[m.group(1)], []).append(row)
if len(sys.argv) > 2:
    data.update(json.load(open(sys.argv[2])))
json.dump(data, sys.stdout, indent=1)
