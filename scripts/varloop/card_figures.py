#!/usr/bin/env python3
"""The figures of the TML-d24-varloop-research page, in English, Russian and Chinese, from the measured numbers.
card_figures.py DATA.json OUT_DIR   (rsvg-convert on PATH)

DATA.json: {"cuda"|"vulkan"|"metal": {"q4tp": [rows]}, "passes": [rows]}; a speed row is {"ctx", "ttft", "decode",
"vram_gb"} as in scripts/inkling/card_figures.py, a passes row {"passes", "ppl", "ref" (the f32 reference, may be
absent), "decode", "ttft"} with "ttft" the first token after 2 048 prompt tokens.
speed_<backend>_<lang>.png: first token, decode and card memory against the prompt's length (3 passes);
passes_<lang>.png: perplexity, decode and first token against the passes through the looped layers."""
import json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "dory"))
import card_figures as cf  # scripts/dory: the SVG writer every card figure shares
from card_figures import Svg, INK, INK2, LINE, BLUE, GREY, ORANGE, head
import importlib.util

_spec = importlib.util.spec_from_file_location("ink_fig", os.path.join(HERE, "..", "inkling", "card_figures.py"))
ink = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ink)

FILE = "TML-d24-varloop-research-q4tp.cmf"
VER = os.environ.get("CORTIQ_VER", "0.9.14")
# the second CUDA stand
ink.STAND["cuda_g4"] = {"en": "NVIDIA RTX PRO 6000 Blackwell (96 GB), Google Colab", "ru": "NVIDIA RTX PRO 6000 Blackwell (96 ГБ), Google Colab",
                        "zh": "NVIDIA RTX PRO 6000 Blackwell（96 GB），Google Colab"}
ink.BACKEND["cuda_g4"] = "CUDA"
SUB = {
    "en": f"{FILE}, cortiq {VER}, cortiq bench --core, 3 passes, 64 tokens after the prompt.",
    "ru": f"{FILE}, cortiq {VER}, cortiq bench --core, 3 прохода, 64 токена после промпта.",
    "zh": f"{FILE}，cortiq {VER}，cortiq bench --core，3 遍，提示词之后 64 个 token。",
}
P = {
    "en": {
        "title": "More passes: better perplexity, slower decoding",
        "sub": f"{FILE}, cortiq {VER}, CUDA. Trained at 1 to 5 passes, mostly 3.",
        "ppl": "WikiText-2 perplexity", "dec": "Decode, tok/s", "first": "First token after 2 048 tokens, s",
        "x": "passes", "ours": "q4tp", "ref": "f32 reference", "l4": "L4", "g4": "RTX PRO 6000",
    },
    "ru": {
        "title": "Больше проходов: ниже перплексия, медленнее генерация",
        "sub": f"{FILE}, cortiq {VER}, CUDA. Обучена на 1–5 проходах, в основном на 3.",
        "ppl": "Перплексия WikiText-2", "dec": "Генерация, ток/с", "first": "Первый токен после 2 048 токенов, с",
        "x": "проходов", "ours": "q4tp", "ref": "эталон f32", "l4": "L4", "g4": "RTX PRO 6000",
    },
    "zh": {
        "title": "遍数越多：困惑度越低，生成越慢",
        "sub": f"{FILE}，cortiq {VER}，CUDA。在 1 到 5 遍上训练，以 3 遍为主。",
        "ppl": "WikiText-2 困惑度", "dec": "生成，tok/s", "first": "2 048 token 后的首 token，秒",
        "x": "遍数", "ours": "q4tp", "ref": "f32 参考", "l4": "L4", "g4": "RTX PRO 6000",
    },
}


def passes_fig(lang, rows):
    t = P[lang]
    s = Svg(1600, 640)
    head(s, {"res": ""}, t["title"], t["sub"])
    panels = [("ppl", [("ppl", BLUE, 5, t["ours"]), ("ref", GREY, 3, t["ref"])]), ("dec", [("decode", BLUE, 5, t["l4"]), ("g4_decode", ORANGE, 3, t["g4"])]), ("first", [("ttft", BLUE, 5, t["l4"])])]
    gap = 40
    pw = (1500 - gap * (len(panels) - 1)) / len(panels)
    xs = [r["passes"] for r in rows]
    for pi, (tkey, series) in enumerate(panels):
        x0 = 50 + pi * (pw + gap)
        x1 = x0 + pw
        y0, y1 = 520, 200
        s.text(x0, 172, t[tkey], 25, INK, "700")
        vals = [r[k] for k, *_ in series for r in rows if r.get(k) is not None]
        lo, hi = min(vals), max(vals)
        if tkey == "ppl":
            # perplexity moves by a few percent: the axis starts near the values
            pad = (hi - lo) * 0.25
            bot, top = lo - pad, hi + pad
            step = next(m for m in (0.05, 0.1, 0.2, 0.25, 0.5, 1.0) if (top - bot) / m <= 7)
            bot = math.floor(bot / step) * step
        else:
            bot, top = 0.0, hi * 1.2
            base = 10 ** math.floor(math.log10(top / 4))
            step = next(base * m for m in (1, 2, 5, 10) if top / (base * m) <= 8)
        Y = lambda v: y0 + (y1 - y0) * (v - bot) / (top - bot)
        xr = x1 - 150
        X = lambda p: x0 + 60 + (xr - x0 - 60) * (p - xs[0]) / max(xs[-1] - xs[0], 1)
        v = bot
        while v <= top + 1e-9:
            s.line(x0 + 40, Y(v), xr + 12, Y(v), LINE, 1.5)
            lab = f"{v:.2f}".rstrip("0").rstrip(".") if step < 1 else f"{v:g}"
            s.text(x0 + 32, Y(v) + 7, ink.num(lang, float(lab), len(lab.split(".")[1]) if "." in lab else 0) if lang == "ru" else lab, 18, INK2, "400", "end")
            v += step
        for p in xs:
            s.text(X(p), y0 + 30, str(p), 18, INK2, "400", "middle")
        s.text((x0 + 60 + xr) / 2, y0 + 60, t["x"], 18, INK2, "400", "middle")
        ends = []
        for key, col, sw, label in series[::-1]:
            pts = [(X(r["passes"]), Y(r[key]), r[key]) for r in rows if r.get(key) is not None]
            if not pts:
                continue
            s.path("M " + " L ".join(f"{a:.1f} {b:.1f}" for a, b, _ in pts), col, sw, dash="9 9" if col == GREY else None)
            for a, b, _ in pts:
                s.dot(a, b, 7 if sw > 3 else 5, col)
            ends.append((pts[-1], col, label, key))
        ys = sorted(((e[0][1] + 6, e) for e in ends), key=lambda z: z[0])
        yy = [y for y, _ in ys]
        for i in range(1, len(yy)):
            if yy[i] - yy[i - 1] < 24:
                mid = (yy[i] + yy[i - 1]) / 2
                yy[i - 1], yy[i] = mid - 12, mid + 12
        for y, (_, (last, col, label, key)) in zip(yy, ys):
            d = 3 if key in ("ppl", "ref") else (2 if last[2] < 10 else 1)
            s.text(last[0] + 14, y, f"{label} {ink.num(lang, last[2], d)}", 19, INK if col != GREY else INK2, "700", "start")
    s.done(f"passes_{lang}")


if __name__ == "__main__":
    data = json.load(open(sys.argv[1]))
    cf.out = sys.argv[2]
    os.makedirs(cf.out, exist_ok=True)
    for lang, t in ink.T.items():
        t["sub"] = SUB[lang]
        for bk, d in data.items():
            if bk == "passes":
                continue
            mem = [("mem", "vram_gb")] if any(r.get("vram_gb") for r in d["q4tp"]) else []
            ink.fig(lang, bk, t["fl_t"], "speed", [(d["q4tp"], BLUE, 5, t["q4tp"])], [("first", "ttft"), ("dec", "decode")] + mem)
        if "passes" in data:
            passes_fig(lang, data["passes"])
    print("ok")
