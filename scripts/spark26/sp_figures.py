#!/usr/bin/env python3
"""The figures of the Spark-X2.6-4B Turbo page, in English, Russian and Chinese, from measured numbers.
sp_figures.py SCORES.bin FACT_ROW_FIRST FACT_ROW_LAST OUT_DIR
(SCORES.bin: `CMF_DORY_EXACT_LOG` of a prompt with a fact; rsvg-convert on PATH)"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dory"))
import card_figures as cf
from card_figures import Svg, INK, INK2, LINE, BLUE, ORANGE, AQUA, GREY, TINT, TINT2, head

score_file, f0, f1, cf.out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
os.makedirs(cf.out, exist_ok=True)
DATA = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures_data.json")))
CTX = DATA["ctx"]
LABELS = [f"{c:,}".replace(",", " ") for c in CTX]
STD, TUR = DATA["std"], DATA["turbo"]

T = {
    "en": {
        "t": "Longer prompts, the same speed and memory",
        "sub": "Spark-X2.6-4B q4tp, RTX 2000 Ada (16 GB), Vulkan. Prompts below 8192 rows run exact in both files.",
        "dec": "Decode, tok/s", "first": "First token, s", "vram": "Peak device memory, GB", "ctx": "prompt, tokens",
        "std": "Standard", "tur": "Turbo",
        "c_t": "What the turbo file keeps of a long prompt",
        "c_sub": "27 of the 36 layers already keep 512 rows. The 9 full-attention layers get a far field of fixed size.",
        "a": "first 8192 rows", "a2": "read exactly",
        "b": "the rest of the prompt", "b2": "keys and values wait in RAM for the far field",
        "c": "last 512", "c2": "read against all of it",
        "s1": "Choose", "s1t": "Among all the stored rows, those holding passwords, codes and names are found.",
        "s2": "Build", "s2t": "The far field is built anew from the stored rows, one layer at a time: 32 landmarks per head.",
        "s3": "Keep exact", "s3t": "The 248 chosen rows, the first 4 and the last 1800 stay exact on the card.",
        "s4": "Decode", "s4t": "A token reads the exact rows and the far field, so its cost stops growing with the text.",
        "res": "{tv} GB on the card and {td} tok/s at 64 000 tokens; the standard file {sv} GB and {sd} tok/s",
        "x_t": "Passwords and codes stay exact",
        "x_sub": "Each row is scored by what the layers do with it, with no word lists.",
        "x_y": "row’s score", "x_x": "prompt rows read before the choice", "x_fact": "“Remember this pass key.”",
        "x_rank": "its rows: places {} of {}", "x_line": "the rows kept lie above this line",
        "x_1": "For each KV head of the 9 full layers, two numbers per row: how far its key lies from the landmarks’ span, and its value’s norm.",
        "x_2": "Each becomes a rank within the prompt; a linear fit over 72 ranks gives the score. The weights are in the file.",
        "x_3": "The choice comes before the prompt’s last 512 rows: a question reads the exact fact and carries it to the answer.",
        "facts": "Planted facts in 60 000 characters (about 14 000 tokens): {f}",
        "e_t": "Spark-X2.6-4B in four engines",
        "e_sub": "The same prompts, 128 greedy tokens. 4-bit files: cortiq q4tp 2.16 GB, MLX 4-bit 2.2 GB, llama.cpp Q4_K_M 3.5 GB.",
        "e_mac": "MacBook Air M4, 24 GB", "e_rtx": "RTX 2000 Ada, 16 GB",
        "e_p": "Prompt of 14 000 tokens, tok/s", "e_d": "Decode after it, tok/s", "e_p16": "Prompt of 16 000 tokens, tok/s",
        "e_d16": "Decode after it, tok/s", "e_d64": "Decode at 64 000 tokens, tok/s",
        "e_ane": "cortiq + Neural Engine", "e_cq": "cortiq q4tp", "e_turbo": "cortiq Turbo", "e_mlx": "MLX 4-bit", "e_lcpp": "llama.cpp Q4_K_M",
        "e_lcppv": "llama.cpp Q4_K_M, Vulkan", "e_lcppc": "llama.cpp Q4_K_M, CUDA", "e_oom": "does not fit",
    },
    "ru": {
        "t": "Длинный промпт, та же скорость и память",
        "sub": "Spark-X2.6-4B q4tp, RTX 2000 Ada (16 ГБ), Vulkan. Промпты короче 8192 строк в обоих файлах идут точно.",
        "dec": "Декод, ток/с", "first": "Первый токен, с", "vram": "Пик памяти карты, ГБ", "ctx": "промпт, токенов",
        "std": "Обычный", "tur": "Турбо",
        "c_t": "Что турбо-файл хранит от длинного промпта",
        "c_sub": "27 из 36 слоёв и так держат 512 строк. 9 слоёв полного внимания получают дальнее поле постоянного размера.",
        "a": "первые 8192 строки", "a2": "читаются точно",
        "b": "остальной промпт", "b2": "ключи и значения ждут дальнего поля в RAM",
        "c": "последние 512", "c2": "читают всё остальное",
        "s1": "Выбор", "s1t": "Среди всех сохранённых строк находятся те, где пароли, коды и имена.",
        "s2": "Сборка", "s2t": "Дальнее поле строится заново из сохранённых строк, слой за слоем: 32 опорные точки на голову.",
        "s3": "Точные", "s3t": "248 выбранных строк, первые 4 и последние 1800 остаются на карте точными.",
        "s4": "Декод", "s4t": "Токен читает точные строки и дальнее поле, и его цена перестаёт расти вместе с текстом.",
        "res": "{tv} ГБ карты и {td} ток/с на 64 000 токенов; у обычного файла {sv} ГБ и {sd} ток/с",
        "x_t": "Пароли и коды остаются точными",
        "x_sub": "Строка получает оценку по тому, что с ней делают слои. Списков слов нет.",
        "x_y": "оценка строки", "x_x": "строки промпта, прочитанные до выбора", "x_fact": "«Запомни этот ключ.»",
        "x_rank": "его строки: места {} из {}", "x_line": "оставленные строки лежат выше этой линии",
        "x_1": "На каждую KV-голову 9 полных слоёв два числа на строку: насколько ключ далёк от опорных точек и норма значения.",
        "x_2": "Каждое становится рангом внутри промпта, и линейная модель по 72 рангам даёт оценку. Веса лежат в файле.",
        "x_3": "Выбор идёт до последних 512 строк промпта: вопрос читает точный факт и несёт его к ответу.",
        "facts": "Факты в 60 000 символов (около 14 000 токенов): {f}",
        "e_t": "Spark-X2.6-4B в четырёх движках",
        "e_sub": "Одни и те же промпты, 128 токенов жадно. 4-битные файлы: cortiq q4tp 2,16 ГБ, MLX 4-bit 2,2 ГБ, llama.cpp Q4_K_M 3,5 ГБ.",
        "e_mac": "MacBook Air M4, 24 ГБ", "e_rtx": "RTX 2000 Ada, 16 ГБ",
        "e_p": "Промпт 14 000 токенов, ток/с", "e_d": "Декод после него, ток/с", "e_p16": "Промпт 16 000 токенов, ток/с",
        "e_d16": "Декод после него, ток/с", "e_d64": "Декод на 64 000 токенов, ток/с",
        "e_ane": "cortiq + Neural Engine", "e_cq": "cortiq q4tp", "e_turbo": "cortiq Турбо", "e_mlx": "MLX 4-bit", "e_lcpp": "llama.cpp Q4_K_M",
        "e_lcppv": "llama.cpp Q4_K_M, Vulkan", "e_lcppc": "llama.cpp Q4_K_M, CUDA", "e_oom": "не помещается",
    },
    "zh": {
        "t": "提示更长，速度和显存不变",
        "sub": "Spark-X2.6-4B q4tp，RTX 2000 Ada（16 GB），Vulkan。短于 8192 行的提示在两个文件中都走精确模式。",
        "dec": "解码，tok/s", "first": "首个 token，秒", "vram": "显存峰值，GB", "ctx": "提示，token 数",
        "std": "普通", "tur": "Turbo",
        "c_t": "Turbo 文件如何保存长提示",
        "c_sub": "36 层中的 27 层本来只保留 512 行；9 个全注意力层使用固定大小的远场。",
        "a": "前 8192 行", "a2": "精确读取",
        "b": "提示的其余部分", "b2": "键和值暂存在内存中，等待构建远场",
        "c": "最后 512 行", "c2": "读取前面所有内容",
        "s1": "选择", "s1t": "在所有保存的行中找出含有密码、编码和名字的行。",
        "s2": "构建", "s2t": "用保存的行逐层重建远场：每个头 32 个地标。",
        "s3": "保持精确", "s3t": "选出的 248 行、前 4 行和后 1800 行以精确形式留在显卡上。",
        "s4": "解码", "s4t": "每个 token 读取精确行和远场，成本不再随文本增长。",
        "res": "64 000 token 时显存 {tv} GB、解码 {td} tok/s；普通文件为 {sv} GB、{sd} tok/s",
        "x_t": "密码和编码保持精确",
        "x_sub": "根据各层对每一行的处理来打分，不用词表。",
        "x_y": "行的分数", "x_x": "选择之前读取的提示行", "x_fact": "“记住这个密钥。”",
        "x_rank": "它的行：第 {} 位，共 {} 行", "x_line": "保留的行在这条线之上",
        "x_1": "9 个全注意力层的每个 KV 头，每行两个数：键与地标张成空间的距离，以及值的范数。",
        "x_2": "两者都转为提示内的排名，再由 72 个排名的线性拟合给出分数，权重存放在文件中。",
        "x_3": "选择发生在提示最后 512 行之前：问题读到精确的事实，并把它带到答案中。",
        "facts": "60 000 字符（约 14 000 token）中的事实：{f}",
        "e_t": "Spark-X2.6-4B 在四个引擎上",
        "e_sub": "相同的提示，贪心生成 128 个 token。4 位文件：cortiq q4tp 2.16 GB，MLX 4-bit 2.2 GB，llama.cpp Q4_K_M 3.5 GB。",
        "e_mac": "MacBook Air M4，24 GB", "e_rtx": "RTX 2000 Ada，16 GB",
        "e_p": "14 000 token 提示，tok/s", "e_d": "随后解码，tok/s", "e_p16": "16 000 token 提示，tok/s",
        "e_d16": "随后解码，tok/s", "e_d64": "64 000 token 时解码，tok/s",
        "e_ane": "cortiq + 神经网络引擎", "e_cq": "cortiq q4tp", "e_turbo": "cortiq Turbo", "e_mlx": "MLX 4-bit", "e_lcpp": "llama.cpp Q4_K_M",
        "e_lcppv": "llama.cpp Q4_K_M，Vulkan", "e_lcppc": "llama.cpp Q4_K_M，CUDA", "e_oom": "放不下",
    },
}


def num(lang, v, d=1):
    s = f"{v:.{d}f}"
    return s.replace(".", ",") if lang == "ru" else s


def compare(lang):
    t = T[lang]
    s = Svg(1600, 680)
    head(s, t, t["t"], t["sub"])
    s.path("M 50 148 L 90 148", ORANGE, 3)
    s.dot(70, 148, 8, ORANGE)
    s.text(102, 156, t["std"], 22, INK)
    s.path("M 250 148 L 290 148", BLUE, 3)
    s.dot(270, 148, 8, BLUE)
    s.text(302, 156, t["tur"], 22, INK)
    last = len(CTX) - 1

    def panel(x0, x1, title, a, b, ymax, step, ends, d):
        y0, y1 = 570, 240
        s.text(x0, 215, title, 25, INK, "700")
        Y = lambda v: y0 + (y1 - y0) * v / ymax
        lg = [math.log2(c) for c in CTX]
        X = lambda i: x0 + 56 + (x1 - x0 - 76) * (lg[i] - lg[0]) / (lg[-1] - lg[0])
        v = 0
        while v <= ymax + 1e-9:
            s.line(x0 + 44, Y(v), x1, Y(v), LINE, 1.5)
            s.text(x0 + 34, Y(v) + 7, f"{v:.0f}", 19, INK2, "400", "end")
            v += step
        for i, lb in enumerate(LABELS):
            s.text(X(i), y0 + 30, lb, 19, INK2, "400", "middle")
        s.text((x0 + x1) / 2, y0 + 62, t["ctx"], 19, INK2, "400", "middle")
        for ser, c in ((a, ORANGE), (b, BLUE)):
            pts = [(X(i), Y(v)) for i, v in enumerate(ser)]
            s.path("M " + " L ".join(f"{p[0]:.1f} {p[1]:.1f}" for p in pts), c, 3)
            for p in pts:
                s.dot(p[0], p[1], 7, c)
        s.text(X(last), Y(a[last]) + ends[0], num(lang, a[last], d), 22, INK, "700", "end")
        s.text(X(last), Y(b[last]) + ends[1], num(lang, b[last], d), 22, INK, "700", "end")

    panel(50, 540, t["dec"], STD["dec"], TUR["dec"], 60, 10, (34, -16), 1)
    panel(570, 1060, t["first"], STD["first"], TUR["first"], 450, 50, (-16, 34), 0)
    panel(1090, 1550, t["vram"], STD["vram"], TUR["vram"], 10, 2, (-16, 34), 1)
    s.done(f"compare_{lang}")


def ceiling(lang):
    t = T[lang]
    s = Svg(1600, 640)
    head(s, t, t["c_t"], t["c_sub"])
    y, h = 160, 96
    x0, xa, xb, x1 = 50, 480, 1300, 1550
    s.rect(x0, y, xa - x0 - 6, h, TINT, 10, BLUE)
    s.rect(xa, y, xb - xa - 6, h, TINT2, 10, ORANGE)
    s.rect(xb, y, x1 - xb, h, TINT, 10, BLUE)
    for xl, xr, a, b in ((x0, xa - 6, t["a"], t["a2"]), (xa, xb - 6, t["b"], t["b2"]), (xb, x1, t["c"], t["c2"])):
        s.text((xl + xr) / 2, y + 40, a, 25, INK, "700", "middle")
        s.text((xl + xr) / 2, y + 72, b, 19, INK2, "400", "middle")
    ww = 362
    yy = 300
    for i, k in enumerate(("s1", "s2", "s3", "s4")):
        x = 50 + i * (ww + 17)
        s.rect(x, yy, ww, 225, TINT if i != 3 else TINT2, 12)
        s.dot(x + 32, yy + 36, 19, BLUE if i != 3 else ORANGE)
        s.text(x + 32, yy + 44, str(i + 1), 22, "#ffffff", "700", "middle")
        s.text(x + 62, yy + 45, t[k], 27, INK, "700")
        for j, ln in enumerate(cf.wrap(t[k + "t"], lang, 29 if lang != "zh" else 15)):
            s.text(x + 22, yy + 92 + j * 30, ln, 21, INK)
    s.rect(50, 548, 1500, 62, TINT, 10)
    last = len(CTX) - 1
    res = t["res"].format(tv=num(lang, TUR["vram"][last]), td=num(lang, TUR["dec"][last]), sv=num(lang, STD["vram"][last]), sd=num(lang, STD["dec"][last]))
    s.text(800, 588, res, 24 if lang != "ru" else 23, INK, "700", "middle")
    s.done(f"ceiling_{lang}")


def exact(lang, sc, rows):
    t = T[lang]
    s = Svg(1600, 800)
    head(s, t, t["x_t"], t["x_sub"])
    px0, px1, py0, py1 = 110, 1550, 470, 150
    n = len(sc)
    lo, hi = float(np.percentile(sc, 50)), float(sc.max())
    bins = int(px1 - px0) // 2
    top = np.full(bins, -1e9)
    for i, v in enumerate(sc):
        b = min(bins - 1, i * bins // n)
        top[b] = max(top[b], v)
    Y = lambda v: py0 + (py1 - py0) * (v - lo) / (hi - lo)
    fb = {min(bins - 1, r * bins // n) for r in rows}
    for b in range(bins):
        if b not in fb and top[b] > lo:
            s.line(px0 + b * 2, py0, px0 + b * 2, Y(top[b]), "#b9c9dc", 1.6)
    th = float(np.sort(sc)[-248])
    s.line(px0, Y(th), px1, Y(th), INK2, 2, "8 6")
    s.text(px0 + 10, Y(th) - 10, t["x_line"], 20, INK2)
    for b in fb:
        s.line(px0 + b * 2, py0, px0 + b * 2, Y(top[b]), ORANGE, 5)
    order = np.argsort(-sc)
    place = {int(r): int(i) for i, r in enumerate(order)}
    ranks = sorted(place[r] for r in rows)
    fx = px0 + min(fb) * 2
    anchor = "end" if fx > 900 else "start"
    dx = -16 if fx > 900 else 16
    s.text(fx + dx, py1 + 26, t["x_fact"], 24, INK, "700", anchor)
    s.text(fx + dx, py1 + 56, t["x_rank"].format(f"{ranks[0]}–{ranks[-1]}", f"{n:,}".replace(",", " ")), 21, ORANGE, "600", anchor)
    s.line(px0, py0, px1, py0, LINE, 2)
    s.text(px0, py0 + 34, "0", 20, INK2)
    s.text(px1, py0 + 34, f"{n:,}".replace(",", " "), 20, INK2, "400", "end")
    s.text((px0 + px1) / 2, py0 + 34, t["x_x"], 21, INK2, "400", "middle")
    s.o.append(f'<text transform="translate(66 {(py0 + py1) / 2}) rotate(-90)" font-size="21" fill="{INK2}" text-anchor="middle">{t["x_y"]}</text>')
    yy, ww = 545, 494
    for i, k in enumerate(("x_1", "x_2", "x_3")):
        x = 50 + i * (ww + 9)
        s.rect(x, yy, ww, 140, TINT if i != 2 else TINT2, 12)
        for j, ln in enumerate(cf.wrap(t[k], lang, 43 if lang != "zh" else 22)):
            s.text(x + 22, yy + 44 + j * 30, ln, 21, INK)
    s.rect(50, 708, 1500, 62, TINT, 10)
    s.text(800, 748, t["facts"].format(f=DATA["facts"][lang]), 24 if lang != "ru" else 23, INK, "700", "middle")
    s.done(f"exact_rows_{lang}")


def bars(s, lang, x0, y0, w, title, rows, unit_max):
    """A titled group of horizontal bars: rows = [(label, value or None, colour, bold)]."""
    t = T[lang]
    s.text(x0, y0, title, 23, INK, "700")
    lw = 300
    bh, gap = 34, 14
    for i, (label, v, col, bold) in enumerate(rows):
        y = y0 + 22 + i * (bh + gap)
        s.text(x0 + lw - 14, y + 24, label, 20, INK if bold else INK2, "700" if bold else "400", "end")
        if v is None:
            s.text(x0 + lw, y + 24, t["e_oom"], 20, INK2)
            continue
        bw = (w - lw - 90) * v / unit_max
        s.rect(x0 + lw, y, max(bw, 3), bh, col, 6)
        s.text(x0 + lw + bw + 10, y + 24, num(lang, v, 0 if v >= 100 else 1), 21, INK, "700" if bold else "400")


def engines(lang):
    t = T[lang]
    e = DATA["engines"]
    s = Svg(1600, 980)
    head(s, t, t["e_t"], t["e_sub"])
    m, r = e["mac"], e["rtx"]
    s.text(50, 170, t["e_mac"], 27, BLUE, "700")
    mac_p = [(t["e_ane"], m["ane"][0], ORANGE, True), (t["e_cq"], m["cortiq"][0], BLUE, True), (t["e_mlx"], m["mlx"][0], GREY, False), (t["e_lcpp"], m["llama"][0], GREY, False)]
    mac_d = [(t["e_ane"], m["ane"][1], ORANGE, True), (t["e_cq"], m["cortiq"][1], BLUE, True), (t["e_mlx"], m["mlx"][1], GREY, False), (t["e_lcpp"], m["llama"][1], GREY, False)]
    bars(s, lang, 50, 220, 760, t["e_p"], mac_p, max(v for _, v, _, _ in mac_p if v))
    bars(s, lang, 820, 220, 730, t["e_d"], mac_d, max(v for _, v, _, _ in mac_d if v))
    s.line(50, 470, 1550, 470, LINE, 2)
    s.text(50, 530, t["e_rtx"], 27, BLUE, "700")
    rtx_p = [(t["e_cq"], r["cortiq"][0], BLUE, True), (t["e_turbo"], r["turbo"][0], AQUA, True), (t["e_lcppv"], r["llama_vk"][0], GREY, False), (t["e_lcppc"], r["llama_cuda"][0], GREY, False)]
    rtx_d = [(t["e_cq"], r["cortiq"][1], BLUE, True), (t["e_turbo"], r["turbo"][1], AQUA, True), (t["e_lcppv"], r["llama_vk"][1], GREY, False), (t["e_lcppc"], r["llama_cuda"][1], GREY, False)]
    rtx_64 = [(t["e_cq"], r["cortiq"][2], BLUE, True), (t["e_turbo"], r["turbo"][2], AQUA, True), (t["e_lcppv"], r["llama_vk"][2], GREY, False), (t["e_lcppc"], r["llama_cuda"][2], GREY, False)]
    bars(s, lang, 50, 580, 520, t["e_p16"], rtx_p, max(v for _, v, _, _ in rtx_p if v))
    bars(s, lang, 560, 580, 500, t["e_d16"], rtx_d, max(v for _, v, _, _ in rtx_d if v))
    bars(s, lang, 1060, 580, 490, t["e_d64"], rtx_64, max(v for _, v, _, _ in rtx_64 if v))
    s.done(f"engines_{lang}")


T2 = {
    "en": {
        "m_t": "Four engines on a MacBook Air M4, prompt by prompt",
        "m_sub": "Spark-X2.6-4B at 4 bits on Metal, 96 greedy tokens after the prompt. One run at a time, a minute of rest between runs.",
        "m_dec": "Decode, tok/s", "m_first": "First token, s",
        "mm_t": "Memory after the prompt",
        "mm_sub": "Peak of the process while it answered, GB. Solid: what Activity Monitor shows (phys_footprint). Outline: resident set.",
        "mm_p": "Prompt of {} tokens",
        "mm_1": "cortiq and llama.cpp map the weights file: those pages are the system’s file cache, counted in the resident set and not in the footprint, and leave when memory is short. MLX copies the weights into its own buffers.",
        "mm_2": "So the footprint is the memory an engine holds for itself: cortiq {cq} GB against MLX {mlx} GB after 14 000 tokens.",
    },
    "ru": {
        "m_t": "Четыре движка на MacBook Air M4, промпт за промптом",
        "m_sub": "Spark-X2.6-4B в 4 бита на Metal, 96 токенов жадно после промпта. По одному запуску, минута отдыха между запусками.",
        "m_dec": "Декод, ток/с", "m_first": "Первый токен, с",
        "mm_t": "Память после промпта",
        "mm_sub": "Пик процесса за время ответа, ГБ. Сплошная полоса — что показывает Мониторинг системы (phys_footprint), контур — резидентная память.",
        "mm_p": "Промпт {} токенов",
        "mm_1": "cortiq и llama.cpp отображают файл весов в память: эти страницы — файловый кэш системы, они входят в резидентную память, не входят в footprint и уходят при нехватке памяти. MLX копирует веса в свои буферы.",
        "mm_2": "Поэтому footprint — это память, которую движок держит сам: cortiq {cq} ГБ против MLX {mlx} ГБ после 14 000 токенов.",
    },
    "zh": {
        "m_t": "MacBook Air M4 上的四个引擎，逐个提示比较",
        "m_sub": "Spark-X2.6-4B 4 位，Metal，提示之后贪心生成 96 个 token。每次只跑一个，两次之间休息一分钟。",
        "m_dec": "解码，tok/s", "m_first": "首个 token，秒",
        "mm_t": "提示之后的内存",
        "mm_sub": "回答期间进程的峰值，GB。实心：活动监视器显示的值（phys_footprint）；空心：常驻内存。",
        "mm_p": "{} token 的提示",
        "mm_1": "cortiq 和 llama.cpp 以内存映射方式读取权重文件：这些页面属于系统的文件缓存，计入常驻内存而不计入 footprint，内存紧张时会被释放。MLX 把权重复制到自己的缓冲区。",
        "mm_2": "因此 footprint 是引擎自己持有的内存：14 000 token 之后 cortiq {cq} GB，MLX {mlx} GB。",
    },
}
for _l in T:
    T[_l].update(T2[_l])
LLAMA_INK = "#6b6a66"


def mac_depth(lang):
    """Decode and first token against the prompt length, five rows on the Mac."""
    t = T[lang]
    d = DATA["mac_depth"]
    if any(v is None for v in d["turbo"]["dec"]):
        return
    s = Svg(1600, 660)
    head(s, t, t["m_t"], t["m_sub"])
    ctx = d["ctx"]
    labels = [f"{c:,}".replace(",", " ") for c in ctx]

    def panel(x0, x1, title, series, ymax, step):
        y0, y1 = 540, 230
        s.text(x0, 170, title, 26, INK, "700")
        Y = lambda v: y0 + (y1 - y0) * v / ymax
        lg = [math.log2(c) for c in ctx]
        X = lambda i: x0 + 50 + (x1 - x0 - 150) * (lg[i] - lg[0]) / (lg[-1] - lg[0])
        v = 0
        while v <= ymax + 1e-9:
            s.line(x0 + 40, Y(v), x1, Y(v), LINE, 1.5)
            s.text(x0 + 30, Y(v) + 7, f"{v:.0f}", 20, INK2, "400", "end")
            v += step
        for i, lb in enumerate(labels):
            s.text(X(i), y0 + 32, lb, 20, INK2, "400", "middle")
        s.text((x0 + x1) / 2, y0 + 66, t["ctx"], 20, INK2, "400", "middle")
        lx = x0 + 40
        for label, ser, c, bold, dash in series:
            s.dot(lx + 8, 201, 7, c)
            s.text(lx + 24, 208, label, 19, INK if bold else INK2, "700" if bold else "400")
            lx += 24 + 10.5 * len(label) + 30
        for label, ser, c, bold, dash in series:
            pts = [(X(i), Y(v)) for i, v in enumerate(ser) if v is not None]
            if not pts:
                continue
            s.path("M " + " L ".join(f"{p[0]:.1f} {p[1]:.1f}" for p in pts), c, 3.5 if bold else 2.5, dash=dash)
            for p in pts:
                s.dot(p[0], p[1], 8 if bold else 6, c)
            px, py = pts[-1]
            s.text(px + 14, py + 7, num(lang, [v for v in ser if v is not None][-1]), 20, INK if bold else INK2, "700" if bold else "400")

    dec = [(t["e_turbo"], d["turbo"]["dec"], AQUA, True, None), (t["e_cq"], d["cortiq"]["dec"], BLUE, True, None),
           (t["e_mlx"], d["mlx"]["dec"], GREY, False, None), (t["e_lcpp"], d["llama"]["dec"], LLAMA_INK, False, "8 6")]
    first = [(t["e_ane"], d["ane"]["first"], ORANGE, True, None), (t["e_cq"], d["cortiq"]["first"], BLUE, True, None),
             (t["e_mlx"], d["mlx"]["first"], GREY, False, None), (t["e_lcpp"], d["llama"]["first"], LLAMA_INK, False, "8 6")]
    dmax = max(v for _, ser, _, _, _ in dec for v in ser if v is not None)
    fmax = max(v for _, ser, _, _, _ in first for v in ser if v is not None)
    panel(50, 780, t["m_dec"], dec, 10 * math.ceil(dmax / 10), 10)
    panel(830, 1550, t["m_first"], first, 50 * math.ceil(fmax / 50), 50)
    s.done(f"mac_depth_{lang}")


def memory(lang):
    """Peak footprint (solid) and resident set (outline) of each engine after the 2k and 14k prompts."""
    t = T[lang]
    d = DATA["memory"]
    s = Svg(1600, 640)
    head(s, t, t["mm_t"], t["mm_sub"])
    rows = [("cortiq", t["e_cq"], BLUE, True), ("turbo", t["e_turbo"], AQUA, True), ("mlx", t["e_mlx"], GREY, False), ("llama", t["e_lcpp"], LLAMA_INK, False)]
    umax = max(v for k, _, _, _ in rows for pr in d[k] for v in pr if v is not None)
    for pi, p in enumerate(d["prompts"]):
        x0, w, lw = 50 + pi * 770, 730, 290
        s.text(x0, 195, t["mm_p"].format(f"{p:,}".replace(",", " ")), 23, INK, "700")
        y = 220
        for k, label, col, bold in rows:
            fp, res = d[k][pi]
            if fp is None:
                continue
            s.text(x0 + lw - 14, y + 22, label, 20, INK if bold else INK2, "700" if bold else "400", "end")
            bw = (w - lw - 80) * fp / umax
            s.rect(x0 + lw, y, max(bw, 3), 28, col, 6)
            s.text(x0 + lw + bw + 10, y + 22, num(lang, fp), 21, INK, "700" if bold else "400")
            bw2 = (w - lw - 80) * res / umax
            s.rect(x0 + lw, y + 33, max(bw2, 3), 12, "none", 4, stroke=col, sw=1.5)
            s.text(x0 + lw + bw2 + 10, y + 44, num(lang, res), 16, INK2)
            y += 70
    s.rect(50, 510, 1500, 110, TINT, 10)
    cq = num(lang, d["cortiq"][1][0])
    mlx = num(lang, d["mlx"][1][0])
    lines = cf.wrap(t["mm_1"], lang, 150 if lang != "zh" else 70) + cf.wrap(t["mm_2"].format(cq=cq, mlx=mlx), lang, 150 if lang != "zh" else 70)
    for j, ln in enumerate(lines[:3]):
        s.text(70, 548 + j * 30, ln, 21, INK)
    s.done(f"memory_{lang}")


T3 = {
    "en": {
        "q_t": "Perplexity against file size",
        "q_sub": "WikiText-2 test, 64 windows of 512 tokens, the second half of each scored; the six files on a MacBook Air M4. Lower is better.",
        "q_x": "file size, GB", "q_y": "perplexity, lower is better",
        "q_4": "4-bit files", "q_8": "8-bit files",
        "q_cq4": "cortiq q4tp", "q_cq8": "cortiq q8_2f", "q_mlx4": "MLX 4-bit", "q_mlx8": "MLX 8-bit", "q_l4": "llama.cpp Q4_K_M", "q_l8": "llama.cpp Q8_0",
        "q_1": "At the same 2.2 GB the q4tp file sits 3 % above the MLX one. llama.cpp’s Q4_K_M keeps attention and part of the FFNs at 6 and 8 bits: it is 1.4 GB larger and lands where the 8-bit files are.",
        "q_2": "The turbo header changes none of this: a 512-token window is read exactly, so the turbo file scores as the standard one.",
    },
    "ru": {
        "q_t": "Перплексия и размер файла",
        "q_sub": "Тест WikiText-2, 64 окна по 512 токенов, оценивается вторая половина каждого; шесть файлов на MacBook Air M4. Ниже — лучше.",
        "q_x": "размер файла, ГБ", "q_y": "перплексия, ниже — лучше",
        "q_4": "4-битные файлы", "q_8": "8-битные файлы",
        "q_cq4": "cortiq q4tp", "q_cq8": "cortiq q8_2f", "q_mlx4": "MLX 4-bit", "q_mlx8": "MLX 8-bit", "q_l4": "llama.cpp Q4_K_M", "q_l8": "llama.cpp Q8_0",
        "q_1": "При одних 2,2 ГБ файл q4tp на 3 % выше MLX. Q4_K_M у llama.cpp держит внимание и часть FFN в 6 и 8 бит: он на 1,4 ГБ больше и стоит там, где 8-битные файлы.",
        "q_2": "Турбо-заголовок ничего здесь не меняет: окно в 512 токенов читается точно, и турбо-файл даёт те же числа, что обычный.",
    },
    "zh": {
        "q_t": "困惑度与文件大小",
        "q_sub": "WikiText-2 测试，64 个 512 词元窗口，各窗口后半段计分；六个文件均在 MacBook Air M4 上测得。越低越好。",
        "q_x": "文件大小，GB", "q_y": "困惑度，越低越好",
        "q_4": "4 位文件", "q_8": "8 位文件",
        "q_cq4": "cortiq q4tp", "q_cq8": "cortiq q8_2f", "q_mlx4": "MLX 4 位", "q_mlx8": "MLX 8 位", "q_l4": "llama.cpp Q4_K_M", "q_l8": "llama.cpp Q8_0",
        "q_1": "同为 2.2 GB 时，q4tp 文件比 MLX 高 3%。llama.cpp 的 Q4_K_M 把注意力和部分 FFN 保留在 6 位和 8 位：大 1.4 GB，落在 8 位文件所在的位置。",
        "q_2": "turbo 文件头不改变这些数字：512 词元的窗口是精确读取的，turbo 文件与标准文件得分相同。",
    },
}
for _l in T:
    T[_l].update(T3[_l])


def quality(lang):
    """Perplexity against file size, six files of three engines; lower is better."""
    t = T[lang]
    d, sz = DATA["ppl"], DATA["ppl_size"]
    s = Svg(1600, 900)
    head(s, t, t["q_t"], t["q_sub"])
    x0, x1, y0, y1 = 170, 1500, 170, 660
    xr, yr = (1.9, 4.7), (24.6, 34.0)
    X = lambda g: x0 + (g - xr[0]) / (xr[1] - xr[0]) * (x1 - x0)
    Y = lambda p: y1 - (p - yr[0]) / (yr[1] - yr[0]) * (y1 - y0)
    # the two size bands
    for a, b, label in ((2.02, 2.36, t["q_4"]), (3.95, 4.52, t["q_8"])):
        s.rect(X(a), y0 - 6, X(b) - X(a), y1 - y0 + 12, TINT, 10)
        s.text((X(a) + X(b)) / 2, y0 + 24, label, 21, INK2, "600", "middle")
    for p in range(26, 35, 2):
        s.line(x0, Y(p), x1, Y(p), LINE, 1.5, "6 8")
        s.text(x0 - 14, Y(p) + 7, num(lang, p, 0), 20, INK2, "400", "end")
    for g in (2.0, 2.5, 3.0, 3.5, 4.0, 4.5):
        s.line(X(g), y1, X(g), y1 + 8, LINE, 2)
        s.text(X(g), y1 + 34, num(lang, g), 20, INK2, "400", "middle")
    s.line(x0, y1, x1, y1, INK2, 2)
    s.text((x0 + x1) / 2, y1 + 70, t["q_x"], 22, INK2, "400", "middle")
    s.o.append(f'<text transform="translate(60 {(y0 + y1) / 2:.0f}) rotate(-90)" font-size="22" fill="{INK2}" text-anchor="middle">{t["q_y"]}</text>')
    # points: (key, label, colour, bold, label dx, dy, anchor)
    pts = [
        ("cortiq_q4tp", t["q_cq4"], BLUE, True, 22, 8, "start"),
        ("mlx4", t["q_mlx4"], GREY, False, 22, 8, "start"),
        ("llama_q4", t["q_l4"], LLAMA_INK, False, -22, 8, "end"),
        ("cortiq_q8", t["q_cq8"], BLUE, True, 22, 30, "start"),
        ("mlx8", t["q_mlx8"], GREY, False, -22, 8, "end"),
        ("llama_q8", t["q_l8"], LLAMA_INK, False, 22, -10, "start"),
    ]
    for k, label, col, bold, dx, dy, anchor in pts:
        x, y = X(sz[k]), Y(d[k])
        s.dot(x, y, 13 if bold else 11, col)
        s.text(x + dx, y + dy, f"{label}  {num(lang, d[k], 2)}", 21, INK if bold else INK2, "700" if bold else "400", anchor)
    s.rect(50, 745, 1500, 130, TINT, 10)
    lines = cf.wrap(t["q_1"], lang, 150 if lang != "zh" else 70) + cf.wrap(t["q_2"], lang, 150 if lang != "zh" else 70)
    for j, ln in enumerate(lines[:4]):
        s.text(70, 783 + j * 30, ln, 21, INK)
    s.done(f"quality_{lang}")


T4 = {
    "en": {
        "u_t": "Spark-X2.6-4B on a CPU server",
        "u_sub": "AMD EPYC (Zen 1), 8 cores at 2.0 GHz, 23 GB of RAM, no GPU. 8 threads; cortiq bench --core and llama-bench, 60 tokens after the prompt.",
        "u_p": "Prompt, tok/s", "u_d": "Decode after it, tok/s", "u_n": "{} tokens",
        "u_cq": "cortiq q4tp", "u_tu": "cortiq Turbo", "u_l": "llama.cpp Q4_K_M",
        "u_1": "Peak resident memory at 16 000 tokens: cortiq q4tp {cq} GB, Turbo {tu} GB, llama.cpp {ll} GB. Below 8192 tokens the turbo file reads exactly, as the standard one.",
        "u_2": "Peak resident memory: cortiq q4tp {cq1} GB at 1 000 tokens and {cq} GB at 16 000, llama.cpp {ll1} and {ll} GB. On the CPU the turbo file runs as the standard one.",
    },
    "ru": {
        "u_t": "Spark-X2.6-4B на CPU-сервере",
        "u_sub": "AMD EPYC (Zen 1), 8 ядер по 2,0 ГГц, 23 ГБ RAM, без GPU. 8 потоков; cortiq bench --core и llama-bench, 60 токенов после промпта.",
        "u_p": "Промпт, ток/с", "u_d": "Декод после него, ток/с", "u_n": "{} токенов",
        "u_cq": "cortiq q4tp", "u_tu": "cortiq Турбо", "u_l": "llama.cpp Q4_K_M",
        "u_1": "Пик резидентной памяти на 16 000 токенов: cortiq q4tp {cq} ГБ, Турбо {tu} ГБ, llama.cpp {ll} ГБ. До 8192 токенов турбо-файл читает точно, как обычный.",
        "u_2": "Пик резидентной памяти: cortiq q4tp {cq1} ГБ на 1 000 токенов и {cq} ГБ на 16 000, llama.cpp {ll1} и {ll} ГБ. На CPU турбо-файл работает как обычный.",
    },
    "zh": {
        "u_t": "Spark-X2.6-4B 在 CPU 服务器上",
        "u_sub": "AMD EPYC（Zen 1），8 核 2.0 GHz，23 GB 内存，无 GPU。8 线程；cortiq bench --core 与 llama-bench，提示之后生成 60 个词元。",
        "u_p": "提示，tok/s", "u_d": "随后解码，tok/s", "u_n": "{} 词元",
        "u_cq": "cortiq q4tp", "u_tu": "cortiq Turbo", "u_l": "llama.cpp Q4_K_M",
        "u_1": "16 000 词元时的常驻内存峰值：cortiq q4tp {cq} GB，Turbo {tu} GB，llama.cpp {ll} GB。8192 词元以内 turbo 文件与标准文件一样精确读取。",
        "u_2": "常驻内存峰值：cortiq q4tp 在 1 000 词元时 {cq1} GB、16 000 词元时 {cq} GB，llama.cpp 为 {ll1} 和 {ll} GB。在 CPU 上 turbo 文件按标准文件运行。",
    },
}
for _l in T:
    T[_l].update(T4[_l])


def cpu(lang):
    """Prompt and decode speed on the CPU server, cortiq against llama.cpp, by prompt length."""
    t = T[lang]
    d = DATA["cpu"]
    s = Svg(1600, 900)
    head(s, t, t["u_t"], t["u_sub"])
    keys = [("cortiq", t["u_cq"], BLUE, True), ("turbo", t["u_tu"], AQUA, True), ("llama", t["u_l"], GREY, False)]
    keys = [k for k in keys if k[0] in d]
    for pi, (metric, title, x0, w) in enumerate((("prefill", t["u_p"], 50, 760), ("dec", t["u_d"], 820, 730))):
        s.text(x0, 185, title, 26, BLUE, "700")
        umax = max(v for k, _, _, _ in keys for v in d[k][metric] if v is not None)
        for ci, c in enumerate(d["ctx"]):
            rows = [(label, d[k][metric][ci], col, bold) for k, label, col, bold in keys]
            bars(s, lang, x0, 235 + ci * 185, w, t["u_n"].format(f"{c:,}".replace(",", " ")), rows, umax)
    i16 = d["ctx"].index(16000)
    s.rect(50, 800, 1500, 80, TINT, 10)
    note = t["u_1" if "turbo" in d else "u_2"].format(
        cq=num(lang, d["cortiq"]["ram"][i16]), tu=num(lang, d["turbo"]["ram"][i16]) if "turbo" in d else "", ll=num(lang, d["llama"]["ram"][i16]),
        cq1=num(lang, d["cortiq"]["ram"][0]), ll1=num(lang, d["llama"]["ram"][0]))
    for j, ln in enumerate(cf.wrap(note, lang, 140 if lang != "zh" else 66)[:2]):
        s.text(70, 835 + j * 30, ln, 21, INK)
    s.done(f"cpu_{lang}")


sc = np.fromfile(score_file, dtype=np.float32)
rows = list(range(f0, f1 + 1))
for lang in T:
    compare(lang)
    ceiling(lang)
    exact(lang, sc, rows)
    engines(lang)
    mac_depth(lang)
    memory(lang)
    quality(lang)
    if "cpu" in DATA:
        cpu(lang)
print("ok", cf.out, "places of the fact:", sorted(int(np.sum(sc > sc[r])) for r in rows))
