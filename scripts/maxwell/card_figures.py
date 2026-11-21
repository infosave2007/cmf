#!/usr/bin/env python3
"""The figures of the Maxwell-V1-Flash page, in English, Russian and Chinese, from the measured numbers.
card_figures.py DATA.json OUT_DIR   (rsvg-convert on PATH)

DATA.json: {"cuda"|"vulkan"|"cpu": {"q8_2f": [rows], "q4tp": [rows], "q2tp": [rows]}} with a row
{"ctx": prompt tokens, "ttft": s, "decode": tok/s, "vram_gb": peak card memory} as bench_rows.py writes them.
files_<backend>_<lang>.png: the files against the prompt's length on one backend of one stand."""
import json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "dory"))
import card_figures as cf  # scripts/dory: the SVG writer every card figure shares
from card_figures import Svg, INK, INK2, LINE, BLUE, ORANGE, GREY, TINT, head

STAND = {
    "cuda": {"en": "NVIDIA L4 (24 GB), Google Colab", "ru": "NVIDIA L4 (24 ГБ), Google Colab", "zh": "NVIDIA L4（24 GB），Google Colab"},
    "vulkan": {"en": "NVIDIA L4 (24 GB), Google Colab", "ru": "NVIDIA L4 (24 ГБ), Google Colab", "zh": "NVIDIA L4（24 GB），Google Colab"},
    "cpu": {"en": "MacBook Air M4 (24 GB), fanless", "ru": "MacBook Air M4 (24 ГБ), без вентилятора", "zh": "MacBook Air M4（24 GB），无风扇"},
}
BACKEND = {"cuda": "CUDA", "vulkan": "Vulkan", "cpu": "CPU"}
# the build the numbers come from: a release ("0.9.11"), or the pull request before one
BUILD = os.environ.get("CORTIQ_BUILD", "PR #182")
VERSION = {"en": f"cortiq ({BUILD})", "ru": f"cortiq ({BUILD})", "zh": f"cortiq（{BUILD}）"} if BUILD.startswith("PR") else {k: f"cortiq {BUILD}" for k in ("en", "ru", "zh")}
T = {
    "en": {
        "fl_t": "Speed and card memory against the prompt's length",
        "sub": f"Maxwell-V1-Flash, {VERSION['en']}, cortiq bench --core, 64 tokens after the prompt.",
        "first": "First token, s", "dec": "Decode, tok/s", "mem": "Peak card memory, GB", "ctx": "prompt, tokens",
        "q4tp": "q4tp", "q8_2f": "q8_2f", "q2tp": "q2tp", "stand": "Stand",
    },
    "ru": {
        "fl_t": "Скорость и память карты в зависимости от длины промпта",
        "sub": f"Maxwell-V1-Flash, {VERSION['ru']}, cortiq bench --core, 64 токена после промпта.",
        "first": "Первый токен, с", "dec": "Генерация, ток/с", "mem": "Пик памяти карты, ГБ", "ctx": "промпт, токенов",
        "q4tp": "q4tp", "q8_2f": "q8_2f", "q2tp": "q2tp", "stand": "Стенд",
    },
    "zh": {
        "fl_t": "速度与显存随提示词长度的变化",
        "sub": f"Maxwell-V1-Flash，{VERSION['zh']}，cortiq bench --core，提示词之后 64 个 token。",
        "first": "首个 token，秒", "dec": "生成，tok/s", "mem": "显存峰值，GB", "ctx": "提示词，token 数",
        "q4tp": "q4tp", "q8_2f": "q8_2f", "q2tp": "q2tp", "stand": "测试平台",
    },
}


def num(lang, v, d=1):
    s = f"{v:.{d}f}"
    return s.replace(".", ",") if lang == "ru" else s


def pill(s, x, y, text):
    w = 18 + 13 * len(text)
    s.rect(x, y - 28, w, 38, TINT, 19, BLUE)
    s.text(x + w / 2, y - 2, text, 21, BLUE, "700", "middle")
    return w


def ctx_label(c):
    if c < 1000:
        return str(c)
    return f"{round(c / 1024)}K" if c % 1024 == 0 else f"{c // 1000}K"


def fig(lang, bk, title, name, series, panels):
    """series: [(rows, colour, stroke, label)], the headline one last; panels: [(title key, row key)].
    Each line is named at its end, so the figure has no legend."""
    t = T[lang]
    s = Svg(1600, 700)
    head(s, t, title, t["sub"])
    w = pill(s, 50, 158, BACKEND[bk])
    s.text(50 + w + 16, 156, f"{t['stand']}: {STAND[bk][lang]}", 22, INK, "700")
    ctxs = sorted({r["ctx"] for rows, *_ in series for r in rows})
    gap = 40
    pw = (1500 - gap * (len(panels) - 1)) / len(panels)
    for pi, (tkey, key) in enumerate(panels):
        x0 = 50 + pi * (pw + gap)
        x1 = x0 + pw
        y0, y1 = 580, 250
        s.text(x0, 222, t[tkey], 25, INK, "700")
        vals = [r[key] for rows, *_ in series for r in rows if r.get(key) is not None]
        top = max(vals) * 1.2
        base = 10 ** math.floor(math.log10(top / 4))
        step = next(base * m for m in (1, 2, 5, 10) if top / (base * m) <= 8)
        Y = lambda v: y0 + (y1 - y0) * v / top
        # the prompt's length on a linear axis: memory that grows with the prompt is a straight line
        xr = x1 - 135
        X = lambda c: x0 + 50 + (xr - x0 - 50) * (c - ctxs[0]) / max(ctxs[-1] - ctxs[0], 1)
        v = 0
        while v <= top:
            s.line(x0 + 30, Y(v), xr + 12, Y(v), LINE, 1.5)
            s.text(x0 + 22, Y(v) + 7, f"{v:g}", 18, INK2, "400", "end")
            v += step
        shown = -1e9
        for c in ctxs:
            if X(c) - shown >= 34:
                s.text(X(c), y0 + 30, ctx_label(c), 18, INK2, "400", "middle")
                shown = X(c)
        s.text((x0 + 50 + xr) / 2, y0 + 60, t["ctx"], 18, INK2, "400", "middle")
        ends = []
        # where the two lines coincide (the card's memory on CUDA, say) the headline one is drawn first and the
        # other over it, dashed with small dots, so both show
        by_ctx = [{r["ctx"]: r[key] for r in rows if r.get(key) is not None} for rows, *_ in series]
        shared = set(by_ctx[0]) & set(by_ctx[-1])
        same = len(series) == 2 and shared and all(abs(by_ctx[0][c] - by_ctx[1][c]) <= 0.04 * max(by_ctx[0][c], by_ctx[1][c]) for c in shared)
        for rows, col, sw, label in (series[::-1] if same else series):
            dashed = same and sw <= 3
            pts = [(X(r["ctx"]), Y(r[key]), r[key], r["ctx"]) for r in rows if r.get(key) is not None]
            if not pts:
                continue
            s.path("M " + " L ".join(f"{p[0]:.1f} {p[1]:.1f}" for p in pts), INK2 if dashed else col, sw, dash="9 9" if dashed else None)
            for p in pts:
                s.dot(p[0], p[1], 4 if dashed else 7 if sw > 3 else 6, INK2 if dashed else col)
            ends.append([pts[-1], col, label])
        # a line that runs to the last length is named to the right of its end; names closer than a line of
        # text move apart. A line that stops early is named above its last point.
        full = sorted((e for e in ends if e[0][3] == ctxs[-1]), key=lambda e: e[0][1])
        ys = [e[0][1] + 6 for e in full]
        for i in range(1, len(ys)):
            if ys[i] - ys[i - 1] < 24:
                mid = (ys[i] + ys[i - 1]) / 2
                ys[i - 1], ys[i] = mid - 12, mid + 12
        for (last, col, label), y in zip(full, ys):
            s.text(last[0] + 14, y, f"{label} {num(lang, last[2], 1 if last[2] < 100 else 0)}", 19, INK if col == BLUE else INK2, "700", "start")
        for last, col, label in (e for e in ends if e[0][3] != ctxs[-1]):
            others = [Y(r[key]) for rows, *_ in series for r in rows if r["ctx"] == last[3] and r.get(key) is not None and Y(r[key]) < last[1] - 1]
            below = any(last[1] - o < 44 for o in others)
            s.text(last[0], last[1] + 32 if below else last[1] - 16, f"{label} {num(lang, last[2], 1 if last[2] < 100 else 0)}", 19, INK if col == BLUE else INK2, "700", "middle")
    s.done(f"{name}_{bk}_{lang}")


if __name__ == "__main__":
    data = json.load(open(sys.argv[1]))
    cf.out = sys.argv[2]
    os.makedirs(cf.out, exist_ok=True)
    # the headline file last (drawn over the others, in the accent colour)
    order = [("q8_2f", ORANGE, 3), ("q2tp", GREY, 3), ("q4tp", BLUE, 5)]
    for lang, t in T.items():
        for bk, d in data.items():
            mem = [("mem", "vram_gb")] if any(r.get("vram_gb") is not None for rows in d.values() for r in rows) else []
            panels = [("first", "ttft"), ("dec", "decode")] + mem
            files = [(d[k], col, sw, t[k]) for k, col, sw in order if d.get(k)]
            if files:
                fig(lang, bk, t["fl_t"], "files", files, panels)
    print("ok")
