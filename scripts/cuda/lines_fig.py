#!/usr/bin/env python3
"""A card figure of measured lines against the prompt's length, in English, Russian and Chinese:
first token (log scale), decode and peak device memory, one line a series.
lines_fig.py SPEC.json OUT_DIR      (rsvg-convert on PATH)

SPEC: {"name": "cuda", "ctx": [1000, ...], "labels": ["1 000", ...],
       "series": [{"key": "cuda", "color": "blue", "first": [...], "decode": [...], "mem": [...]}, ...],
       "text": {"en": {"t": ..., "sub": ..., "first": ..., "decode": ..., "mem": ..., "ctx": ..., "res": ...,
                       "cuda": "<series name>", ...}, "ru": {...}, "zh": {...}},
       "mem_max": 24, "decode_max": 250}
A value of null is not drawn (a run that did not finish). A series with "dash": true is drawn dashed with
small dots: list it after the series it may coincide with, and both show."""
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dory"))
import card_figures as cf
from card_figures import Svg, INK, INK2, LINE, BLUE, ORANGE, AQUA, MUTE, head

COLORS = {"blue": BLUE, "orange": ORANGE, "aqua": AQUA, "grey": MUTE}


def num(lang, v, d):
    s = f"{v:.{d}f}"
    return s.replace(".", ",") if lang == "ru" else s


def fig(spec, lang):
    t = spec["text"][lang]
    ctx, labels, series = spec["ctx"], spec["labels"], spec["series"]
    s = Svg(1600, 760)
    head(s, t, t["t"], t["sub"])
    x = 50
    for se in series:
        s.rect(x, 138, 22, 22, COLORS[se["color"]], 5)
        s.text(x + 32, 156, t[se["key"]], 22, INK)
        x += 60 + 13 * len(t[se["key"]]) * (1.6 if lang == "zh" else 1.0)
    lg = [math.log2(c) for c in ctx]
    y0, y1 = 590, 250
    panels = (
        (50, 520, "first", True, None),
        (570, 1040, "decode", False, spec.get("decode_max")),
        (1090, 1550, "mem", False, spec.get("mem_max")),
    )
    for x0, x1, key, logy, vmax in panels:
        s.text(x0, 222, t[key], 24, INK, "700")
        X = lambda i: x0 + 70 + (x1 - x0 - 110) * (lg[i] - lg[0]) / (lg[-1] - lg[0])
        vals = [v for se in series for v in se[key] if v is not None]
        if logy:
            lo = 10 ** math.floor(math.log10(min(vals)))
            hi = 10 ** math.ceil(math.log10(max(vals)))
            L = lambda v: math.log10(v)
            Y = lambda v: y0 + (y1 - y0) * (L(v) - L(lo)) / (L(hi) - L(lo))
            v = lo
            while v <= hi * 1.0001:
                s.line(x0 + 60, Y(v), x1, Y(v), LINE, 1.5)
                s.text(x0 + 50, Y(v) + 7, f"{v:g}", 19, INK2, "400", "end")
                v *= 10
        else:
            top = vmax or max(vals) * 1.15
            step = 10 ** math.floor(math.log10(top / 2))
            while top / step > 6:
                step *= 2
            Y = lambda v: y0 + (y1 - y0) * v / top
            v = 0.0
            while v <= top + 1e-9:
                s.line(x0 + 60, Y(v), x1, Y(v), LINE, 2 if v == 0 else 1.5)
                s.text(x0 + 50, Y(v) + 7, f"{v:g}", 19, INK2, "400", "end")
                v += step
        for i, lb in enumerate(labels):
            s.text(X(i), y0 + 32, lb, 19, INK2, "400", "middle")
        s.text((x0 + x1) / 2, y0 + 66, t["ctx"], 19, INK2, "400", "middle")
        placed = []
        for se in series:
            c = COLORS[se["color"]]
            # a dashed series (an earlier version, say) has small dots, so a
            # line it coincides with shows under it
            dashed = se.get("dash")
            pts = [(X(i), Y(v)) for i, v in enumerate(se[key]) if v is not None]
            if len(pts) > 1:
                s.path("M " + " L ".join(f"{p[0]:.1f} {p[1]:.1f}" for p in pts), c, 3, dash="9 9" if dashed else None)
            for p in pts:
                s.dot(p[0], p[1], 4 if dashed else 7, c)
            last = [(i, v) for i, v in enumerate(se[key]) if v is not None][-1]
            d = 1 if key == "mem" or (key == "first" and last[1] < 100) else 0
            # a line that stops early is labelled under its last point, where
            # the lines that go on do not run
            ly = Y(last[1]) - 12 if last[0] == len(ctx) - 1 else Y(last[1]) + 30
            # labels that would touch one already placed move apart
            while any(abs(ly - p) < 21 for p in placed):
                ly += 22 if ly >= Y(last[1]) - 12 else -22
            placed.append(ly)
            s.text(X(last[0]) + 10, ly, num(lang, last[1], d), 19, c, "700", "start")
    s.text(50, 716, t["res"], 23, INK, "600")
    s.done(f"{spec['name']}_{lang}")


if __name__ == "__main__":
    spec = json.load(open(sys.argv[1], encoding="utf-8"))
    cf.out = sys.argv[2]
    os.makedirs(cf.out, exist_ok=True)
    for lang in spec["text"]:
        fig(spec, lang)
    print("ok")
