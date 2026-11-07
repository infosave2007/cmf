#!/usr/bin/env python3
"""The figures of the Minico M1 1105 card, in English, Russian and Chinese, from measured numbers.
card_figures.py SCORES.bin TOKENS.bin TOKENIZER.json OUT_DIR

SCORES.bin: `CMF_DORY_EXACT_LOG` of the turbo file on a prompt with a planted pass key;
TOKENS.bin: `CMF_O1_TOKENS_DUMP` of the same run. Needs numpy, tokenizers and rsvg-convert."""
import math
import os
import subprocess
import sys

import numpy as np
from tokenizers import Tokenizer

SURF, INK, INK2, MUTE, LINE = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8983", "#dedcd5"
BLUE, ORANGE, AQUA, GREY, TINT, TINT2 = "#2a78d6", "#eb6834", "#1baf7a", "#c9c7bf", "#eaf2fc", "#fdeee7"
SUMMARY = "#efeee9"
FONT = "Helvetica Neue, Arial, Hiragino Sans GB, Arial Unicode MS, sans-serif"
out = "."

# Measured on a MacBook Air M4, Metal: `cortiq run --greedy --no-think`, 128 tokens after the prompt.
CTX = [1000, 4000, 16000, 32000, 64000]
DEC = ([155.3, 136.9, 92.2, 65.1, 38.6], [152.8, 137.5, 121.5, 104.2, 106.2])
TTFT = ([0.9, 3.6, 19.8, 56.5, 183.1], [0.9, 3.6, 24.5, 61.2, 193.3])
MEM = ([0.32, 0.59, 1.30, 1.95, 3.69], [0.32, 0.59, 1.05, 1.75, 3.38])

T = {
    "en": {
        "h_title": "How the turbo file reads a long prompt",
        "h_sub": "Prompts under 8192 tokens run exact, as in the standard file. A longer one takes four steps.",
        "h_a": "the prompt up to its last 512 rows", "h_a2": "read exactly, as in the standard file",
        "h_b": "last 512 rows", "h_b2": "the question",
        "s1": "Read", "s1t": "Every row's keys and values are kept for now, in the 6 attention layers.",
        "s2": "Choose", "s2t": "Each row gets a score from what the layers do with it. The 248 best stay exact.",
        "s3": "Fold", "s3t": "All other rows fold into 32 landmarks per head; the first 4 and the last 1800 stay exact.",
        "s4": "Answer", "s4t": "The question and every new token read 2052 exact rows and the landmarks.",
        "h_res": "After 64 000 tokens: decode 106 tok/s against 38.6, peak memory 3.38 GB against 3.69",
        "f_title": "What a new token reads",
        "f_sub": "One attention layer of the six, after a prompt of 64 000 tokens.",
        "f_std": "Standard", "f_std2": "every row: the cost of a token grows with the text",
        "f_tur": "Turbo", "f_near1": "first 4 rows", "f_far": "everything else: 32 landmarks per head",
        "f_near2": "last 1800 rows", "f_keep": "248 chosen rows: exact",
        "f_q": "a token reads", "f_ex": "exact", "f_ch": "exact, chosen", "f_sk": "summary",
        "f_res1": "38.6 tok/s", "f_res2": "2052 exact rows and the landmarks at any length: 106 tok/s",
        "x_title": "Passwords and codes stay exact",
        "x_sub": "Each row of a 23 465-token prompt is scored by what the layers do with it, with no word lists.",
        "x_line": "the rows kept lie above this line", "x_win": "last 1800: exact anyway",
        "x_fact": "“Remember this pass key: 653269.”", "x_rank": "its digits: places 0 and 2 of 23 453",
        "x_x": "prompt rows read before the choice", "x_y": "row's score",
        "b1": "Per KV head of each attention layer, two numbers per row: how far its key lies from the span of the landmarks, and its value's norm.",
        "b2": "Each becomes a rank within the prompt, and a linear fit over the 96 ranks gives the score, stored in the file.",
        "b3": "The choice comes before the prompt's last 512 rows: the question reads the exact fact and carries it on to the answer.",
        "x_res": "A planted fact in 100 000 characters: turbo 4 of 8, standard 3 of 8, far field without exact rows 0 of 8",
        "c_title": "Long prompts decode faster",
        "c_sub": "Minico M1 1105 q8_2f, MacBook Air M4, Metal. Prompts below 8192 tokens run exact in both files.",
        "c_std": "Standard", "c_tur": "Turbo", "c_dec": "Decode, tok/s", "c_ttft": "First token, s",
        "c_mem": "Peak memory, GB", "c_ctx": "prompt, tokens",
    },
    "ru": {
        "h_title": "Как турбо-файл читает длинный промпт",
        "h_sub": "Промпты короче 8192 токенов идут точно, как в обычном файле. Длинный — в четыре шага.",
        "h_a": "промпт до последних 512 строк", "h_a2": "читается точно, как в обычном файле",
        "h_b": "последние 512 строк", "h_b2": "вопрос",
        "s1": "Чтение", "s1t": "Ключи и значения всех строк пока хранятся в 6 слоях внимания.",
        "s2": "Выбор", "s2t": "Каждая строка получает оценку по тому, что с ней делают слои. 248 лучших остаются точными.",
        "s3": "Свёртка", "s3t": "Остальные строки сворачиваются в 32 опорные точки на голову; первые 4 и последние 1800 точны.",
        "s4": "Ответ", "s4t": "Вопрос и каждый новый токен читают 2052 точные строки и опорные точки.",
        "h_res": "После 64 000 токенов: генерация 106 ток/с вместо 38,6, пик памяти 3,38 ГБ вместо 3,69",
        "f_title": "Что читает новый токен",
        "f_sub": "Один слой внимания из шести после промпта в 64 000 токенов.",
        "f_std": "Обычный", "f_std2": "каждая строка: цена токена растёт вместе с текстом",
        "f_tur": "Турбо", "f_near1": "первые 4 строки", "f_far": "всё остальное: 32 опорные точки на голову",
        "f_near2": "последние 1800 строк", "f_keep": "248 выбранных строк: точно",
        "f_q": "токен читает", "f_ex": "точно", "f_ch": "точно, выбранные", "f_sk": "сводка",
        "f_res1": "38,6 ток/с", "f_res2": "2052 точные строки и опорные точки при любой длине: 106 ток/с",
        "x_title": "Пароли и коды остаются точными",
        "x_sub": "Каждая строка промпта в 23 465 токенов получает оценку по тому, что с ней делают слои, без списков слов.",
        "x_line": "сохранённые строки лежат выше этой линии", "x_win": "последние 1800: точны и так",
        "x_fact": "«Remember this pass key: 653269.»", "x_rank": "его цифры: места 0 и 2 из 23 453",
        "x_x": "строки промпта, прочитанные до выбора", "x_y": "оценка строки",
        "b1": "Для каждой KV-головы каждого слоя внимания два числа на строку: насколько ключ далёк от опорных точек и норма значения.",
        "b2": "Каждое становится рангом внутри промпта, а линейная модель по 96 рангам даёт оценку; её веса лежат в файле.",
        "b3": "Выбор делается до последних 512 строк промпта: вопрос читает точный факт и доносит его до ответа.",
        "x_res": "Факт в 100 000 символов: турбо 4 из 8, обычный файл 3 из 8, дальнее поле без точных строк 0 из 8",
        "c_title": "Длинные промпты генерируются быстрее",
        "c_sub": "Minico M1 1105 q8_2f, MacBook Air M4, Metal. Промпты короче 8192 токенов идут точно в обоих файлах.",
        "c_std": "Обычный", "c_tur": "Турбо", "c_dec": "Генерация, ток/с", "c_ttft": "Первый токен, с",
        "c_mem": "Пик памяти, ГБ", "c_ctx": "промпт, токенов",
    },
    "zh": {
        "h_title": "turbo 文件如何读取长提示词",
        "h_sub": "8192 token 以下的提示词与标准文件一样精确运行。更长的提示词分四步。",
        "h_a": "提示词，直到最后 512 行", "h_a2": "与标准文件一样精确读取",
        "h_b": "最后 512 行", "h_b2": "问题",
        "s1": "读取", "s1t": "6 个注意力层暂时保存每一行的键和值。",
        "s2": "选择", "s2t": "按各层对每一行的作用打分，得分最高的 248 行保持精确。",
        "s3": "折叠", "s3t": "其余行折叠为每个头 32 个地标；前 4 行和最后 1800 行保持精确。",
        "s4": "回答", "s4t": "问题和每个新 token 读取 2052 个精确行以及地标。",
        "h_res": "64 000 token 之后：生成 106 token/s（标准 38.6），峰值内存 3.38 GB（标准 3.69）",
        "f_title": "新 token 读取什么",
        "f_sub": "六个注意力层之一，64 000 token 提示词之后。",
        "f_std": "标准", "f_std2": "每一行：每个 token 的开销随文本增长",
        "f_tur": "turbo", "f_near1": "前 4 行", "f_far": "其余：每个头 32 个地标",
        "f_near2": "最后 1800 行", "f_keep": "选出的 248 行：精确",
        "f_q": "token 读取", "f_ex": "精确", "f_ch": "精确，选出的行", "f_sk": "摘要",
        "f_res1": "38.6 token/s", "f_res2": "任意长度下 2052 个精确行加地标：106 token/s",
        "x_title": "密码和编码保持精确",
        "x_sub": "23 465 token 提示词的每一行按各层对它的作用打分，不使用词表。",
        "x_line": "保留的行位于此线之上", "x_win": "最后 1800 行：本来就精确",
        "x_fact": "“Remember this pass key: 653269.”", "x_rank": "其数字：在 23 453 行中排第 0 和第 2",
        "x_x": "选择之前读取的提示词行", "x_y": "行得分",
        "b1": "每个注意力层的每个 KV 头，每行两个数：键离地标张成空间多远，以及值的范数。",
        "b2": "每个数转为提示词内的排名，对 96 个排名做线性拟合得到得分，权重存放在文件中。",
        "b3": "选择发生在提示词最后 512 行之前：问题读取精确的事实并带到回答中。",
        "x_res": "100 000 字符中植入的事实：turbo 8 个中 4 个，标准 3 个，无精确行的远场 0 个",
        "c_title": "长提示词之后生成更快",
        "c_sub": "Minico M1 1105 q8_2f，MacBook Air M4，Metal。8192 token 以下两个文件都精确运行。",
        "c_std": "标准", "c_tur": "turbo", "c_dec": "生成，token/s", "c_ttft": "首 token，秒",
        "c_mem": "峰值内存，GB", "c_ctx": "提示词，token",
    },
}


class Svg:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.o = []

    def rect(self, x, y, w, h, fill, r=10, stroke=None, sw=2):
        s = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
        self.o.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{r}" fill="{fill}"{s}/>')

    def text(self, x, y, s, size=24, fill=INK, weight="400", anchor="start"):
        s = s.replace("&", "&amp;").replace("<", "&lt;")
        self.o.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" font-weight="{weight}" text-anchor="{anchor}">{s}</text>')

    def line(self, x1, y1, x2, y2, stroke=LINE, sw=2, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.o.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{sw}" stroke-linecap="round"{d}/>')

    def path(self, d, stroke, sw=3):
        self.o.append(f'<path d="{d}" stroke="{stroke}" stroke-width="{sw}" fill="none" stroke-linecap="round" stroke-linejoin="round"/>')

    def dot(self, x, y, r, fill):
        self.o.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}" stroke="{SURF}" stroke-width="2"/>')

    def done(self, name):
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" viewBox="0 0 {self.w} {self.h}" '
               f'font-family="{FONT}"><rect width="{self.w}" height="{self.h}" fill="{SURF}"/>{"".join(self.o)}</svg>')
        p = os.path.join(out, name)
        open(p + ".svg", "w", encoding="utf-8").write(svg)
        subprocess.run(["rsvg-convert", "-w", str(self.w), p + ".svg", "-o", p + ".png"], check=True)
        os.remove(p + ".svg")


def head(s, title, sub):
    s.text(50, 66, title, 40, INK, "700")
    s.text(50, 108, sub, 23, INK2)


def wrap(text, lang, n):
    if lang == "zh":
        lines, cur = [], ""
        for ch in text:
            if len(cur) >= n and ch not in "，。；：、）":
                lines.append(cur)
                cur = ""
            cur += ch
        return lines + [cur] if cur else lines
    lines, cur = [], ""
    for word in text.split():
        if cur and len(cur) + len(word) + 1 > n:
            lines.append(cur)
            cur = word
        else:
            cur = (cur + " " + word).strip()
    return lines + [cur] if cur else lines


def steps(s, lang, keys, y, h, hi):
    w = (1500 - 3 * 17) / len(keys)
    for i, (k, kt) in enumerate(keys):
        x = 50 + i * (w + 17)
        warm = i == hi
        s.rect(x, y, w, h, TINT2 if warm else TINT, 12)
        s.dot(x + 32, y + 36, 19, ORANGE if warm else BLUE)
        s.text(x + 32, y + 44, str(i + 1), 22, "#ffffff", "700", "middle")
        s.text(x + 62, y + 45, T[lang][k], 27, INK, "700")
        for j, ln in enumerate(wrap(T[lang][kt], lang, {"en": 30, "ru": 28, "zh": 15}[lang])):
            s.text(x + 22, y + 90 + j * 30, ln, 21, INK)


def result(s, y, text, size=26):
    s.rect(50, y, 1500, 62, TINT, 10)
    s.text(800, y + 40, text, size, INK, "700", "middle")


def how_fig(lang):
    t = T[lang]
    s = Svg(1600, 650)
    head(s, t["h_title"], t["h_sub"])
    y, h = 150, 96
    s.rect(50, y, 1180, h, TINT, 12, BLUE)
    s.text(640, y + 42, t["h_a"], 26, INK, "700", "middle")
    s.text(640, y + 74, t["h_a2"], 20, INK2, "400", "middle")
    s.rect(1246, y, 304, h, TINT2, 12, ORANGE)
    s.text(1398, y + 42, t["h_b"], 26, INK, "700", "middle")
    s.text(1398, y + 74, t["h_b2"], 20, INK2, "400", "middle")
    steps(s, lang, [("s1", "s1t"), ("s2", "s2t"), ("s3", "s3t"), ("s4", "s4t")], 290, 236, 3)
    result(s, 562, t["h_res"], 25 if lang != "ru" else 24)
    s.done(f"how_{lang}")


def far_fig(lang):
    t = T[lang]
    s = Svg(1600, 690)
    head(s, t["f_title"], t["f_sub"])
    x0, x1 = 50, 1550
    s.text(x0, 180, t["f_std"], 28, INK, "700")
    s.text(x0 + {"en": 160, "ru": 160, "zh": 90}[lang], 180, t["f_std2"], 21, INK2)
    n = 150
    w = (x1 - x0) / n
    for i in range(n):
        s.rect(x0 + i * w, 196, w - 2, 52, AQUA, 1)
    s.text(x1, 286, t["f_res1"], 26, INK, "700", "end")
    s.text(x0, 340, t["f_tur"], 28, INK, "700")
    y, h = 372, 76
    a, c = 60, 300
    b = x1 - x0 - a - c - 16
    s.rect(x0, y, a, h, AQUA, 6)
    s.rect(x0 + a + 8, y, b, h, SUMMARY, 6, GREY)
    s.rect(x1 - c, y, c, h, AQUA, 6)
    s.text(x0, y + 112, t["f_near1"], 20, INK2)
    s.text(x0 + a + 8 + b / 2, y + h / 2 + 8, t["f_far"], 24, INK2, "500", "middle")
    s.text(x1 - c / 2, y + h / 2 + 8, t["f_near2"], 24, "#ffffff", "700", "middle")
    for p in (0.02, 0.07, 0.13, 0.19, 0.81, 0.87, 0.93, 0.98):
        xx = x0 + a + 8 + 20 + p * (b - 40)
        s.rect(xx, y - 8, 8, h + 16, ORANGE, 2)
    s.text(x0 + a + 8 + b / 2, y + 112, t["f_keep"], 24, ORANGE, "700", "middle")
    ly = 570
    s.text(x0, ly + 8, t["f_q"] + ":", 23, INK2)
    lx = x0 + {"en": 190, "ru": 180, "zh": 140}[lang]
    for col, lab, stroke in ((AQUA, t["f_ex"], None), (ORANGE, t["f_ch"], None), (SUMMARY, t["f_sk"], GREY)):
        s.rect(lx, ly - 14, 34, 26, col, 4, stroke)
        s.text(lx + 44, ly + 8, lab, 23, INK)
        lx += 44 + 13 * len(lab) * (2 if lang == "zh" else 1) + 60
    result(s, 610, t["f_res2"], 26 if lang != "ru" else 25)
    s.done(f"far_field_{lang}")


def exact_fig(lang, sc, frows):
    t = T[lang]
    s = Svg(1600, 810)
    head(s, t["x_title"], t["x_sub"])
    px0, px1, py0, py1 = 110, 1550, 470, 150
    n = len(sc)
    lo_r, hi_r = 4, n - 1800
    # The rows the file keeps: a chosen row with two neighbours each side, 248 in all.
    order = lo_r + np.argsort(-sc[lo_r:hi_r])
    kept, last_seed = np.zeros(n, bool), None
    for r in order:
        if kept.sum() >= 248:
            break
        last_seed = r
        for j in range(max(lo_r, r - 2), min(hi_r, r + 3)):
            if kept.sum() < 248:
                kept[j] = True
    lo, hi = float(np.percentile(sc, 50)), float(sc.max())
    bins = (px1 - px0) // 2
    top = np.full(bins, -1e9)
    for i, v in enumerate(sc):
        b = min(bins - 1, i * bins // n)
        top[b] = max(top[b], v)
    Y = lambda v: py0 + (py1 - py0) * (v - lo) / (hi - lo)
    fb = {min(bins - 1, r * bins // n) for r in frows}
    wx = px0 + (hi_r * bins // n) * 2
    s.rect(wx, py1 - 10, px1 - wx, py0 - py1 + 10, TINT, 0)
    s.text(px1, py1 - 20, t["x_win"], 18, BLUE, "600", "end")
    for b in range(bins):
        if b not in fb and top[b] > lo:
            x = px0 + b * 2
            s.line(x, py0, x, Y(top[b]), "#b9c9dc", 1.6)
    th = float(sc[last_seed])
    s.line(px0, Y(th), wx - 8, Y(th), INK2, 2, "8 6")
    s.text(px0 + 10, Y(th) - 10, t["x_line"], 20, INK2)
    for b in fb:
        x = px0 + b * 2
        s.line(x, py0, x, Y(top[b]), ORANGE, 5)
    fx = px0 + max(fb) * 2
    s.text(fx + 16, py1 + 26, t["x_fact"], 24, INK, "700")
    s.text(fx + 16, py1 + 56, t["x_rank"], 21, ORANGE, "600")
    s.line(px0, py0, px1, py0, LINE, 2)
    s.text(px0, py0 + 34, "0", 20, INK2)
    s.text(px1, py0 + 34, f"{n:,}".replace(",", " "), 20, INK2, "400", "end")
    s.text((px0 + px1) / 2, py0 + 34, t["x_x"], 21, INK2, "400", "middle")
    s.o.append(f'<text transform="translate(66 {(py0 + py1) / 2}) rotate(-90)" font-size="21" fill="{INK2}" text-anchor="middle">{t["x_y"]}</text>')
    y, w = 545, (1500 - 2 * 17) / 3
    for i, k in enumerate(("b1", "b2", "b3")):
        x = 50 + i * (w + 17)
        s.rect(x, y, w, 150, TINT2 if i == 2 else TINT, 12)
        for j, ln in enumerate(wrap(t[k], lang, {"en": 44, "ru": 40, "zh": 21}[lang])):
            s.text(x + 22, y + 38 + j * 30, ln, 21, INK)
    result(s, 718, t["x_res"], 24 if lang == "en" else 23)
    s.done(f"exact_rows_{lang}")


def compare_fig(lang):
    t = T[lang]
    s = Svg(1600, 680)
    head(s, t["c_title"], t["c_sub"])
    for k, (lab, col) in enumerate(((t["c_std"], ORANGE), (t["c_tur"], BLUE))):
        x = 50 + k * 200
        s.line(x, 148, x + 40, 148, col, 3)
        s.dot(x + 20, 148, 8, col)
        s.text(x + 52, 156, lab, 22, INK)
    num = (lambda v: f"{v:g}".replace(".", ",")) if lang == "ru" else (lambda v: f"{v:g}")
    lg = [math.log2(c) for c in CTX]

    def panel(x0, x1, title, a, b, ymax, step, ends):
        y0, y1 = 570, 240
        s.text(x0, 215, title, 26, INK, "700")
        Y = lambda v: y0 + (y1 - y0) * v / ymax
        X = lambda i: x0 + 50 + (x1 - x0 - 90) * (lg[i] - lg[0]) / (lg[-1] - lg[0])
        v = 0
        while v <= ymax + 1e-9:
            s.line(x0 + 40, Y(v), x1, Y(v), LINE, 1.5)
            s.text(x0 + 30, Y(v) + 7, num(v), 20, INK2, "400", "end")
            v += step
        for i in (0, 2, 4):
            s.text(X(i), y0 + 32, f"{CTX[i]:,}".replace(",", " "), 20, INK2, "400", "middle")
        s.text((x0 + x1) / 2, y0 + 66, t["c_ctx"], 20, INK2, "400", "middle")
        for ser, col in ((a, ORANGE), (b, BLUE)):
            pts = [(X(i), Y(v)) for i, v in enumerate(ser)]
            s.path("M " + " L ".join(f"{p[0]:.1f} {p[1]:.1f}" for p in pts), col, 3)
            for p in pts:
                s.dot(p[0], p[1], 8, col)
        for ser, (dx, dy, anc) in zip((a, b), ends):
            s.text(X(4) + dx, Y(ser[-1]) + dy, num(ser[-1]), 23, INK, "700", anc)

    panel(50, 530, t["c_dec"], *DEC, 180, 30, ((-6, 34, "end"), (-6, -16, "end")))
    panel(570, 1050, t["c_ttft"], *TTFT, 210, 30, ((14, 8, "start"), (-6, -16, "end")))
    panel(1090, 1570, t["c_mem"], *MEM, 4, 1, ((-6, -16, "end"), (14, 8, "start")))
    s.done(f"compare_{lang}")


if __name__ == "__main__":
    scores_p, toks_p, tokj, out = sys.argv[1:5]
    os.makedirs(out, exist_ok=True)
    sc = np.fromfile(scores_p, dtype=np.float32)
    ids = np.fromfile(toks_p, dtype=np.uint32).tolist()
    tok = Tokenizer.from_file(tokj)
    frows = [i for i in range(len(sc)) if tok.decode([ids[i]]) in ("653", "269")]
    for lang in T:
        how_fig(lang)
        far_fig(lang)
        exact_fig(lang, sc, frows)
        compare_fig(lang)
    print("figures written to", out, "fact rows", frows)
