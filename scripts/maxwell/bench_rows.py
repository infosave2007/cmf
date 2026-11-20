#!/usr/bin/env python3
"""The lines of scripts/maxwell/bench2.sh ("TAG backend cCTX: Prompt: ... Decode: ... vram_peak N MiB") as the rows
card_figures.py reads. bench_rows.py LOG [EXTRA.json] > figures_data.json
TAG chart-q8 | chart-q4tp | chart-q2tp; EXTRA.json adds stands measured elsewhere (the Mac)."""
import json, re, sys

FILES = {"chart-q8": "q8_2f", "chart-q4tp": "q4tp", "chart-q2tp": "q2tp"}
BACKENDS = {"cuda": "cuda", "wgpu": "vulkan"}
data = {}
for line in open(sys.argv[1], errors="replace"):
    m = re.match(r"\s*(chart-\S+) (\S+) c(\d+): .*?Decode: \d+ tokens \| ([\d.]+) tok/s steady \(TTFT ([\d.]+)s.*?vram_peak (\d+) MiB", line)
    if not m or m.group(1) not in FILES or m.group(2) not in BACKENDS:
        continue
    row = {"ctx": int(m.group(3)), "ttft": float(m.group(5)), "decode": float(m.group(4)), "vram_gb": round(int(m.group(6)) / 1024, 2)}
    rows = data.setdefault(BACKENDS[m.group(2)], {}).setdefault(FILES[m.group(1)], [])
    # a length measured again replaces its earlier row
    rows[:] = [r for r in rows if r["ctx"] != row["ctx"]] + [row]
    rows.sort(key=lambda r: r["ctx"])
if len(sys.argv) > 2:
    data.update(json.load(open(sys.argv[2])))
json.dump(data, sys.stdout, indent=1)
