#!/usr/bin/env python3
"""The lines of scripts/gigaam/bench.sh as the rows card_figures.py reads.

    bench_rows.py LOG [LOG ...] > figures_data.json

A line (a Mac's CPU log is renamed `cpu` → `cpu_m4` before it comes here): `TAG backend sSECONDS: cold X warm Y | vad As  features Bs  encoder Cs  decode Ds | Nx real time |
vram_peak M MiB | rss_peak R MiB`. Output: {backend: {TAG: [{"secs", "time" (warm, s), "cold", "x" (seconds of
audio a second), "vad", "features", "encoder", "decode", "vram_gb", "rss_gb"}]}}; memory is MiB / 1024. A
backend measured twice keeps the later line.
"""
import json, re, sys

pat = re.compile(
    r"\s*(\S+) (cuda|vulkan|metal|cpu_m4|cpu) s(\d+): cold ([\d.]+) warm ([\d.]+) \| vad ([\d.]+)s\s+features ([\d.]+)s\s+"
    r"encoder ([\d.]+)s\s+decode ([\d.]+)s\s*\| ([\d.]+)x real time \| vram_peak (\d+) MiB \| rss_peak (\d+) MiB"
)
data = {}
for path in sys.argv[1:]:
    for line in open(path, errors="replace"):
        m = pat.match(line)
        if not m:
            continue
        tag, be, secs = m.group(1), m.group(2), int(m.group(3))
        row = {
            "secs": secs, "cold": float(m.group(4)), "time": float(m.group(5)), "x": round(secs / float(m.group(5)), 1),
            "vad": float(m.group(6)), "features": float(m.group(7)), "encoder": float(m.group(8)), "decode": float(m.group(9)),
            "rss_gb": round(int(m.group(12)) / 1024, 2),
        }
        if int(m.group(11)) > 0:
            row["vram_gb"] = round(int(m.group(11)) / 1024, 2)
        rows = data.setdefault(be, {}).setdefault(tag, [])
        rows[:] = [r for r in rows if r["secs"] != secs] + [row]
        rows.sort(key=lambda r: r["secs"])
json.dump(data, sys.stdout, indent=1)
