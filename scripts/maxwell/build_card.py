#!/usr/bin/env python3
"""The Maxwell-V1-Flash page (EN/RU/ZH) from the measured numbers.
build_card.py FIGURES_DATA.json NUMBERS.json OUT.md"""
import json, sys

data = json.load(open(sys.argv[1]))
N = json.load(open(sys.argv[2]))
REPO = "infosave/Maxwell-V1-Flash-cmf"
IMG = f"https://huggingface.co/{REPO}/resolve/main/assets"
Q8, Q4 = "Maxwell-V1-Flash-q8_2f.cmf", "Maxwell-V1-Flash-q4tp.cmf"
SRC = "https://huggingface.co/maxwell-model/Maxwell-V1-Flash"


def n(lang, v, d=1):
    s = f"{v:.{d}f}"
    return s.replace(".", ",") if lang == "ru" else s


def tt(lang, v):  # a first-token time
    return n(lang, v, 2 if v < 10 else 1)


def toks(c):
    return f"{c:,}".replace(",", " ")


def rows(bk, f):
    return {r["ctx"]: r for r in data.get(bk, {}).get(f, [])}


def perf_table(lang, bk, head, head1=None):
    q4, q8 = rows(bk, "q4tp"), rows(bk, "q8_2f")
    if not q8 and head1:  # one file measured on this backend
        out = [head1, "|---:|---:|---:|---:|"]
        for c in sorted(q4):
            r = q4[c]
            out.append(f"| {toks(c)} | {tt(lang, r['ttft'])} | {n(lang, r['decode'])} | {n(lang, r['vram_gb'])} |")
        return "\n".join(out)
    out = [head, "|---:|---:|---:|---:|---:|---:|---:|"]
    for c in sorted(set(q4) | set(q8)):
        a, b = q4.get(c), q8.get(c)
        cell = lambda r, k, f: f(r[k]) if r else "—"
        out.append(
            f"| {toks(c)} | {cell(a, 'ttft', lambda v: tt(lang, v))} | {cell(b, 'ttft', lambda v: tt(lang, v))} | "
            f"{cell(a, 'decode', lambda v: n(lang, v))} | {cell(b, 'decode', lambda v: n(lang, v))} | "
            f"{cell(a, 'vram_gb', lambda v: n(lang, v))} | {cell(b, 'vram_gb', lambda v: n(lang, v))} |"
        )
    return "\n".join(out)


P = N["ppl"]  # {"bf16": {wiki, code, py, wiki6k}, "q8_2f": {...}, "q4tp": {...}}
D = N["decode"]  # {"cuda": {"q4tp": x, "q8_2f": y}, "vulkan": {...}, "cpu_vm": {"q4tp": x}, "cpu_mac": {...}}
S = N["size_gb"]
c4, c8 = rows("cuda", "q4tp"), rows("cuda", "q8_2f")
v4 = rows("vulkan", "q4tp")
last = max(c4)


def pct(a, b):
    return (a / b - 1) * 100


def ppl_rows(lang):
    f = lambda k, d=2: (n(lang, P[k]["wiki"], d), n(lang, P[k]["code"], d), n(lang, P[k]["py"], d), n(lang, P[k]["wiki6k"], d))
    return f("q8_2f"), f("q4tp"), f("bf16")


def card(lang):
    T = TEXT[lang]
    r8, r4, rb = ppl_rows(lang)
    g = lambda v, d=1: n(lang, v, d)
    fill = dict(
        src=SRC, repo=REPO, q8=Q8, q4=Q4, img=IMG, pr=N["pr"], ver=N["version"],
        d_c4=g(D["cuda"]["q4tp"]), d_c32k=g(D["cuda"]["q4tp_cap32k"]), d_c8=g(D["cuda"]["q8_2f"]), fitrows=toks(N["fit_rows"]), vk6k=g(N["vk_ppl6k"], 2), vklast=toks(max(v4)), vktlast=tt(lang, v4[max(v4)]["ttft"]), vkdlast=g(v4[max(v4)]["decode"]), calib=N["calib_note"][lang], mtp=N["mtp_note"][lang], d8_2k=g(c8[2048]["decode"]),  d_v4=g(D["vulkan"]["q4tp"]),
        t2k=tt(lang, c4[2048]["ttft"]), tlast=tt(lang, c4[last]["ttft"]), last=toks(last), dlast=g(c4[last]["decode"]), mlast=g(c4[last]["vram_gb"]),
        m2k=g(c4[2048]["vram_gb"]), s8=g(S["q8_2f"], 2), s4=g(S["q4tp"], 2), sb=g(S["bf16"], 2),
        p8w=r8[0], p8c=r8[1], p8p=r8[2], p8l=r8[3], p4w=r4[0], p4c=r4[1], p4p=r4[2], p4l=r4[3], pbw=rb[0], pbc=rb[1], pbp=rb[2], pbl=rb[3],
        cuda_table=perf_table(lang, "cuda", T["thead"]), vk_table=perf_table(lang, "vulkan", T["thead"], T["thead1"]), vkw=g(N["vk_weights_gb"]), vkrows=toks(N["vk_rows"]),
        cpu_vm=g(D["cpu_vm"]["q4tp"]), greedy=N["greedy"][lang], ram8=g(N["rss_gb"]["q8_2f"], 0), ram4=g(N["rss_gb"]["q4tp"], 0),
        sha8=N["sha256"]["q8_2f"], sha4=N["sha256"]["q4tp"], kv=g(N["kv_kb_per_token"], 0), kv30=g(N["kv_kb_per_token"] * 30000 / 1024 / 1024),
        mac=N.get("mac", {}).get(lang, ""),
    )
    return T["body"].format(**fill)


exec(open(__file__.replace("build_card.py", "card_text.py")).read())  # TEXT

front = """---
license: apache-2.0
library_name: cortiq
base_model: maxwell-model/Maxwell-V1-Flash
base_model_relation: quantized
pipeline_tag: text-generation
tags:
  - cmf
  - cortiq
  - quantized
  - moe
  - q4tp
  - reasoning
  - tool-calling
  - long-context
  - cuda
  - vulkan
language:
  - en
  - zh
  - ru
  - th
  - vi
  - id
---

"""
out = front + card("en") + "\n---\n\n" + card("ru") + "\n---\n\n" + card("zh")
open(sys.argv[3], "w").write(out)
print("card", len(out), "bytes")
