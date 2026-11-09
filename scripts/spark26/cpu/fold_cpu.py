#!/usr/bin/env python3
"""fold_cpu.py FINAL_ROWS.txt: fold the CPU server's back-to-back rows (chain6.sh logs/final_rows.txt) into
figures_data.json as the "cpu" block: prompt and decode tok/s and peak resident GB of cortiq q4tp and
llama.cpp Q4_K_M at 1000, 4000 and 16 000 tokens. Prints the block."""
import json
import os
import re
import sys

rows = open(sys.argv[1]).read().splitlines()
ctx = [1000, 4000, 16000]
cq = {"prefill": [None] * 3, "dec": [None] * 3, "ram": [None] * 3}
ll = {"prefill": [None] * 3, "dec": [None] * 3, "ram": [None] * 3}
for line in rows:
    m = re.match(r"cortiq (\d+) \| rss_kb=(\d+) \|.*prefill ([\d.]+) tok/s.*\| ([\d.]+) tok/s steady", line)
    if m:
        i = ctx.index(int(m.group(1)))
        cq["prefill"][i] = float(m.group(3))
        cq["dec"][i] = float(m.group(4))
        cq["ram"][i] = round(int(m.group(2)) / 1024 / 1024, 1)
        continue
    m = re.match(r"llama pp (\d+) \| rss_kb=(\d+) \|.*\|\s+([\d.]+) ± [\d.]+ \|", line)
    if m:
        i = ctx.index(int(m.group(1)))
        ll["prefill"][i] = float(m.group(3))
        ll["ram"][i] = round(int(m.group(2)) / 1024 / 1024, 1)
        continue
    m = re.match(r"llama tg@(\d+) \| rss_kb=(\d+) \|.*\|\s+([\d.]+) ± [\d.]+ \|", line)
    if m:
        i = ctx.index(int(m.group(1)))
        ll["dec"][i] = float(m.group(3))
        ll["ram"][i] = max(ll["ram"][i] or 0, round(int(m.group(2)) / 1024 / 1024, 1))
p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures_data.json")
d = json.load(open(p))
d["cpu"] = {"ctx": ctx, "cortiq": cq, "llama": ll}
d["_note_cpu"] = ("CPU server (AMD EPYC Zen 1 VM, 8 vCPU at 2.0 GHz, 23 GB, no GPU): scripts/spark26/cpu/chain6.sh, "
                  "cortiq bench --core with 60 tokens after the prompt and llama.cpp b12202 llama-bench -t 8 (-p N, "
                  "then -d N -n 60), run back to back at each length because this shared VM drifted by a third over "
                  "an evening; ram = peak resident set (VmHWM), GiB like the VRAM figures.")
json.dump(d, open(p, "w"), indent=2, ensure_ascii=False)
open(p, "a").write("\n")
print(json.dumps(d["cpu"]))
