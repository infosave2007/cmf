#!/usr/bin/env python3
"""The figures of the SenseNova-Loop-9B and SenseNova-Loop-9B-Turbo pages, in English, Russian and Chinese,
from the measured numbers. card_figures.py DATA.json OUT_DIR   (rsvg-convert on PATH)

DATA.json: {"cuda"|"vulkan": {"std": [bench rows], "turbo": [bench rows], "survey": {kind: [spec, plain]}}}
with bench rows as bench_table.py writes them (ctx, ttft, decode, vram_gb)."""
import json, math, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dory"))
import card_figures as cf
from card_figures import Svg, INK, INK2, LINE, BLUE, ORANGE, GREY, TINT, head

STAND = {"cuda": "NVIDIA RTX A5000 (24 GB)", "vulkan": "NVIDIA RTX A5000 (24 GB)", "metal": "Apple M4, MacBook Air (24 GB)"}
T = {
    "en": {
        "tb_t": "A long prompt, standard and turbo",
        "tb_sub": "SenseNova-Loop-9B q4tp, cortiq bench --core without drafts, 128 tokens after the prompt.",
        "first": "First token, s", "dec": "Decode, tok/s", "mem": "Peak card memory, GB", "ctx": "prompt, tokens",
        "std": "standard", "tur": "turbo", "oom": "standard: does not fit",
        "sv_t": "Draft tokens on real answers",
        "sv_sub": "SenseNova-Loop-9B q4tp, cortiq run --greedy --no-think, 256 tokens; the same text with and without drafts.",
        "spec": "with drafts", "plain": "without", "kinds": {"code": "code", "general": "general", "instruct": "instructions"},
        "stand": "Stand", "res": "",
    },
    "ru": {
        "tb_t": "Длинный промпт: обычный файл и турбо",
        "tb_sub": "SenseNova-Loop-9B q4tp, cortiq bench --core без черновиков, 128 токенов после промпта.",
        "first": "Первый токен, с", "dec": "Декод, ток/с", "mem": "Пик памяти карты, ГБ", "ctx": "промпт, токенов",
        "std": "обычный", "tur": "турбо", "oom": "обычный: не помещается",
        "sv_t": "Черновые токены на реальных ответах",
        "sv_sub": "SenseNova-Loop-9B q4tp, cortiq run --greedy --no-think, 256 токенов; текст с черновиками и без одинаковый.",
        "spec": "с черновиками", "plain": "без", "kinds": {"code": "код", "general": "общие", "instruct": "инструкции"},
        "stand": "Стенд", "res": "",
    },
    "zh": {
        "tb_t": "长提示词：普通文件与 Turbo",
        "tb_sub": "SenseNova-Loop-9B q4tp，cortiq bench --core，不用草稿，提示词之后 128 个 token。",
        "first": "首个 token，秒", "dec": "解码，tok/s", "mem": "显卡内存峰值，GB", "ctx": "提示词，token 数",
        "std": "普通", "tur": "Turbo", "oom": "普通：放不下",
        "sv_t": "真实回答上的草稿 token",
        "sv_sub": "SenseNova-Loop-9B q4tp，cortiq run --greedy --no-think，256 个 token；有无草稿文本相同。",
        "spec": "用草稿", "plain": "不用", "kinds": {"code": "代码", "general": "通用", "instruct": "指令"},
        "stand": "测试平台", "res": "",
    },
}
BACKEND = {"cuda": "CUDA", "vulkan": "Vulkan", "metal": "Metal"}


def num(lang, v, d=1):
    s = f"{v:.{d}f}"
    return s.replace(".", ",") if lang == "ru" else s


def pill(s, x, y, text):
    w = 18 + 13 * len(text)
    s.rect(x, y - 28, w, 38, TINT, 19, BLUE)
    s.text(x + w / 2, y - 2, text, 21, BLUE, "700", "middle")
    return w


def ctx_label(c):
    return f"{c // 1000}K" if c >= 1000 else str(c)


def turbo_fig(lang, bk, d):
    t = T[lang]
    s = Svg(1600, 700)
    head(s, t, t["tb_t"], t["tb_sub"])
    w = pill(s, 50, 158, BACKEND[bk])
    s.text(50 + w + 16, 156, f"{t['stand']}: {STAND[bk]}", 22, INK, "700")
    std, tur = d["std"], d["turbo"]
    ctxs = sorted({r["ctx"] for r in std + tur if r["ctx"] >= 4000})
    lg = [math.log2(c) for c in ctxs]

    def panel(x0, x1, title, key, scale=1.0):
        y0, y1 = 580, 250
        s.text(x0, 222, title, 25, INK, "700")
        vals = [r[key] * scale for r in std + tur if r.get(key) is not None and r["ctx"] in ctxs]
        top = max(vals) * 1.15
        base = 10 ** math.floor(math.log10(top / 4))
        step = next(base * m for m in (1, 2, 5, 10) if top / (base * m) <= 8)
        Y = lambda v: y0 + (y1 - y0) * v / top
        X = lambda c: x0 + 40 + (x1 - x0 - 80) * (math.log2(c) - lg[0]) / max(lg[-1] - lg[0], 1e-9)
        v = 0
        while v <= top:
            s.line(x0 + 30, Y(v), x1, Y(v), LINE, 1.5)
            s.text(x0 + 22, Y(v) + 7, f"{v:g}", 18, INK2, "400", "end")
            v += step
        for c in ctxs:
            s.text(X(c), y0 + 30, ctx_label(c), 18, INK2, "400", "middle")
        s.text((x0 + x1) / 2, y0 + 60, t["ctx"], 18, INK2, "400", "middle")
        for rows, col, sw, name in ((std, GREY, 3, t["std"]), (tur, BLUE, 5, t["tur"])):
            pts = [(X(r["ctx"]), Y(r[key] * scale), r[key] * scale) for r in rows if r.get(key) is not None and r["ctx"] in ctxs]
            if not pts:
                continue
            s.path("M " + " L ".join(f"{p[0]:.1f} {p[1]:.1f}" for p in pts), col, sw)
            for p in pts:
                s.dot(p[0], p[1], 7 if col == BLUE else 6, col)
            last = pts[-1]
            label = f"{name} {num(lang, last[2], 1 if last[2] < 100 else 0)}"
            if col == BLUE:
                s.text(last[0] - 6, last[1] - 16, label, 19, INK, "700", "end")
            else:
                s.text(last[0] + 12, last[1] + 34, label, 19, INK2, "700", "start")

    panel(50, 540, t["first"], "ttft")
    panel(580, 1070, t["dec"], "decode")
    panel(1110, 1550, t["mem"], "vram_gb")
    s.done(f"turbo_{bk}_{lang}")


def survey_fig(lang, bk, d):
    t = T[lang]
    sv = d["survey"]
    s = Svg(1600, 560)
    head(s, t, t["sv_t"], t["sv_sub"])
    w = pill(s, 50, 158, BACKEND[bk])
    s.text(50 + w + 16, 156, f"{t['stand']}: {STAND[bk]}", 22, INK, "700")
    kinds = list(sv)
    top = max(max(v) for v in sv.values()) * 1.2
    y0, y1 = 480, 220
    Y = lambda v: y0 + (y1 - y0) * v / top
    gw = 1400 / len(kinds)
    for i, k in enumerate(kinds):
        spec, plain = sv[k]
        gx = 120 + i * gw
        for j, (v, col, name) in enumerate(((plain, GREY, t["plain"]), (spec, BLUE, t["spec"]))):
            x = gx + 40 + j * 150
            s.rect(x, Y(v), 120, y0 - Y(v), col, 6)
            s.text(x + 60, Y(v) - 12, num(lang, v), 22, INK, "700", "middle")
            s.text(x + 60, y0 + 28, name, 18, INK2, "400", "middle")
        s.text(gx + 175, y0 + 62, t["kinds"].get(k, k), 22, INK, "700", "middle")
    s.line(100, y0, 1550, y0, INK2, 2)
    s.done(f"drafts_{bk}_{lang}")


if __name__ == "__main__":
    data = json.load(open(sys.argv[1]))
    cf.out = sys.argv[2]
    os.makedirs(cf.out, exist_ok=True)
    for lang in T:
        for bk, d in data.items():
            if d.get("std") and d.get("turbo"):
                turbo_fig(lang, bk, d)
            if d.get("survey"):
                survey_fig(lang, bk, d)
    print("ok")
