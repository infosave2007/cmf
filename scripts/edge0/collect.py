#!/usr/bin/env python3
"""The figures/cards data from the pod's logs.

    python3 scripts/edge0/collect.py LOG_DIR DATA.json

LOG_DIR holds, copied from the pod's /root/e0/logs: bench_c{145,120}h_r{1,2}.json (speedtest.py, the CUDA full
card), chainH.log (card memory, HumanEval), tier_c{145,120}_{32,24,16,12}_r1.json with tier_*.serve.log and
tier_*.log (the emulated smaller cards), and optionally tier_v{145,120}_{GB}_r1.json (Vulkan on them).
DATA.json is updated in place: its Vulkan full-card block (measured earlier) is kept.
"""
import json, os, re, sys

log_dir, out = sys.argv[1], sys.argv[2]
data = json.load(open(out))
labels = {1024: "1K", 4096: "4K", 16384: "16K", 32768: "32K"}
chain = open(f"{log_dir}/chainH.log").read()
cur, mem = None, {}
for line in chain.splitlines():
    m = re.search(r"\] (145|120) cuda full", line)
    if m:
        cur = m.group(1)
    m = re.search(r"card memory used after 32K: (\d+) MiB; server RSS: ([\d.]+) GB", line)
    if m and cur:
        mem[cur] = (int(m.group(1)), float(m.group(2)))
    m = re.search(r"HUMANEVAL edge0: pass@1 (\d+)/(\d+)", line)
    if m and cur:
        data["humaneval"]["m" + cur] = round(100.0 * int(m.group(1)) / int(m.group(2)), 2)
d = {"prompts": [labels[c] for c in sorted(labels)], "prefill": {}, "decode": {}, "card_mib": {}, "rss_gb": {}}
for m in ("145", "120"):
    runs = [json.load(open(f"{log_dir}/bench_c{m}h_r{r}.json")) for r in (1, 2)]
    pre, dec = [], []
    for c in sorted(labels):
        rows = [x for run in runs for x in run if x["ctx"] == c]
        pre.append(round(sum(x["prefill_tok_s"] for x in rows) / len(rows), 1))
        dec.append(round(sum(x["decode_tok_s"] for x in rows) / len(rows), 1))
    d["prefill"]["m" + m], d["decode"]["m" + m] = pre, dec
    d["card_mib"]["m" + m], d["rss_gb"]["m" + m] = mem.get(m, (0, 0.0))
data["cuda"] = d


def tier_rss():
    """RSS of each tier's server from the chain log: {(prefix, model, gb): GB}."""
    out, cur = {}, None
    p = f"{log_dir}/chainJ.log"
    if not os.path.exists(p):
        return out
    for line in open(p, errors="replace"):
        m = re.search(r"\] (145|120) (cuda|vulkan) (\d+) GB", line)
        if m:
            cur = ("c" if m.group(2) == "cuda" else "v", m.group(1), int(m.group(3)))
        m = re.search(r"([\d.]+) GB RSS", line)
        if m and cur and float(m.group(1)) > 1:
            out[cur] = max(out.get(cur, 0.0), float(m.group(1)))
        m = re.search(r"HUMANEVAL edge0: pass@1 (\d+)/(\d+)", line)
        if m:
            data.setdefault("humaneval_24gb", {})["m145"] = round(100.0 * int(m.group(1)) / int(m.group(2)), 2)
    return out


def tiers(prefix):
    res = {}
    rss = tier_rss()
    for m in ("145", "120"):
        rows = []
        for gb in (32, 24, 16, 12):
            p = f"{log_dir}/tier_{prefix}{m}_{gb}_r1.json"
            if not os.path.exists(p):
                continue
            # a context the card had no memory for leaves a row without tokens
            run = {x["ctx"]: x for x in json.load(open(p)) if x.get("prompt_tokens", 1) and x["prefill_tok_s"] > 0 and x["decode_tok_s"] > 0}
            if not run:
                continue
            srv = open(f"{log_dir}/tier_{prefix}{m}_{gb}.serve.log", errors="replace").read()
            row = {"gb": gb, "ctx": [labels[c] for c in sorted(run)],
                   "prefill": [round(run[c]["prefill_tok_s"], 1) for c in sorted(run)],
                   "decode": [round(run[c]["decode_tok_s"], 1) for c in sorted(run)]}
            x = re.search(r"(\d+) of (\d+) latent experts on the card", srv)
            if x:
                row["experts_pct"] = round(100.0 * int(x.group(1)) / int(x.group(2)), 1)
            x = re.search(r"latent experts: (\d+) MB on the card, (\d+) MB in system memory", srv)
            if x:
                a, b = int(x.group(1)), int(x.group(2))
                row["experts_pct"] = round(100.0 * a / max(1, a + b), 1)
            if (prefix, m, gb) in rss:
                row["rss_gb"] = rss[(prefix, m, gb)]
            row["mtp"] = "MTP head stays off" not in srv
            rows.append(row)
        if rows:
            res["m" + m] = rows
    return res


# HumanEval of the 120B on the final build (the last pod step)
fin = f"{log_dir}/final.log"
if os.path.exists(fin):
    m = re.search(r"HUMANEVAL edge0: pass@1 (\d+)/(\d+)", open(fin).read())
    if m:
        data["humaneval"]["m120"] = round(100.0 * int(m.group(1)) / int(m.group(2)), 2)
data["cuda_tiers"] = tiers("c")
vt = tiers("v")
if vt:
    data["vulkan_tiers"] = vt
json.dump(data, open(out, "w"), indent=1)
print(json.dumps({k: data[k] for k in ("cuda", "cuda_tiers", "humaneval")}, indent=1))
