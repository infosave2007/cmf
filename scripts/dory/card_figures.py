#!/usr/bin/env python3
"""The figures of the Dory card, in English, Russian and Chinese, from measured numbers.
card_figures.py SCORE.bin OUT_DIR   (SCORE.bin: `CMF_DORY_EXACT_LOG` of a prompt with a fact)"""
import json, math, os, subprocess, sys
import numpy as np

out = "."
SURF, INK, INK2, MUTE, LINE = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8983", "#dedcd5"
BLUE, ORANGE, AQUA, GREY, TINT, TINT2 = "#2a78d6", "#eb6834", "#1baf7a", "#c9c7bf", "#eaf2fc", "#fdeee7"
FONT = "Helvetica Neue, Arial, Hiragino Sans GB, Arial Unicode MS, sans-serif"

# ── Text ─────────────────────────────────────────────────────────────────────────────
T = {
    "en": {
        "r_title": "One pass or two, token by token",
        "r_sub": "One checkpoint runs any loop count. The router decides after the first pass, before any output is read.",
        "token": "token", "input": "Input block", "sub16": "16 sub-layers", "rec": "Recurrent block", "sub40": "40 sub-layers",
        "router": "Router", "state": "state after the pass", "easy": "easy tokens", "hard": "hard tokens",
        "err": "which subspace rebuilds the state better?", "output": "Output block", "next": "next token",
        "leave": "easy: leaves the loop", "again": "hard: second pass",
        "r_n1": "Two PCA subspaces of rank 16, fitted on the model’s own answers.",
        "r_n2": "No gradient training, no read-out through the output block and the head.",
        "r_n3": "One cache per layer: each pass overwrites this token’s row, so a shorter token leaves no hole.",
        "r_res": "Real prompts: 49.6 → 60.7 tok/s on Vulkan, 25.7 → 32.9 on an M4",
        "f_title": "A long prompt stays small on the card",
        "f_sub": "What one full-attention layer keeps from a prompt. Sliding layers already hold only 1024 rows.",
        "f_std": "Two loops, exact", "f_std2": "every row, for every loop: memory and the cost of a token grow with the text",
        "f_tur": "Turbo", "f_near1": "first rows", "f_far": "everything else: 32 landmarks per head", "f_near2": "last 1536 rows",
        "f_keep": "512 chosen rows: exact", "f_q": "a token reads", "f_ex": "exact", "f_sk": "summary", "f_res1": "4.3 GB at 32 000 rows", "f_res2": "0.5 GB at any length",
        "x_title": "Passwords and codes stay exact",
        "x_sub": "The rows the far field would lose are found by what the layers do with them, with no word lists.",
        "x_y": "row’s score", "x_x": "prompt rows read before the choice", "x_fact": "“Remember this pass key: 653269.”", "x_rank": "its six rows: places 0 to 7 of 22 976", "x_line": "the rows kept lie above this line",
        "s1": "Read", "s1t": "Keys and values of the prompt’s rows are kept for now.",
        "s2": "Score", "s2t": "Per KV head: how far a key lies from the span of the landmarks, and the value’s norm. A linear fit over 72 ranks.",
        "s3": "Keep", "s3t": "Before the last 512 rows, the top rows and their neighbours leave the far field, which is built anew without them.",
        "s4": "Answer", "s4t": "The question is read against the exact fact and carries it on to the answer.",
        "x_res": "Facts in 100 000 characters: 8 of 8, at a tenth of the memory of the exact model",
        "c_title": "Standard and turbo, the same weights",
        "c_sub": "q4tp, 2.09 GB, NVIDIA L4 (24 GB) over Vulkan. Exact is the standard file at two loops.",
        "c_speed": "Decode, tok/s", "c_mem": "Device KV memory, GB", "c_ctx": "context, tokens", "c_std": "standard", "c_tur": "turbo", "c_dnf": "prompt not read in 25 min",
    },
    "ru": {
        "r_title": "Один проход или два, для каждого токена",
        "r_sub": "Один чекпоинт работает с любым числом циклов. Роутер решает после первого прохода, ещё до чтения выхода.",
        "token": "токен", "input": "Входной блок", "sub16": "16 подслоёв", "rec": "Рекуррентный блок", "sub40": "40 подслоёв",
        "router": "Роутер", "state": "состояние после прохода", "easy": "лёгкие токены", "hard": "трудные токены",
        "err": "какое лучше восстановит состояние?", "output": "Выходной блок", "next": "следующий токен",
        "leave": "лёгкий: выходит из цикла", "again": "трудный: второй проход",
        "r_n1": "Два подпространства PCA ранга 16, построены по ответам самой модели.",
        "r_n2": "Без обучения градиентом и без чтения через выходной блок и голову.",
        "r_n3": "Один кеш на слой: каждый проход перезаписывает строку токена, поэтому дыр не остаётся.",
        "r_res": "Реальные запросы: 49,6 → 60,7 ток/с на Vulkan, 25,7 → 32,9 на M4",
        "f_title": "Длинный промпт занимает на карте мало места",
        "f_sub": "Что слой полного внимания хранит от промпта. Скользящие слои и так держат только 1024 строки.",
        "f_std": "Два цикла, точно", "f_std2": "каждая строка для каждого цикла: память и цена токена растут вместе с текстом",
        "f_tur": "Турбо", "f_near1": "первые строки", "f_far": "всё остальное: 32 опорные точки на голову", "f_near2": "последние 1536 строк",
        "f_keep": "512 выбранных строк: точно", "f_q": "токен читает", "f_ex": "точно", "f_sk": "сводка", "f_res1": "4,3 ГБ при 32 000 строк", "f_res2": "0,5 ГБ при любой длине",
        "x_title": "Пароли и коды остаются точными",
        "x_sub": "Строки, которые дальнее поле потеряло бы, находятся по работе самих слоёв, без списков слов.",
        "x_y": "оценка строки", "x_x": "строки промпта, прочитанные до отбора", "x_fact": "«Remember this pass key: 653269.»", "x_rank": "шесть его строк: места с 0 по 7 из 22 976", "x_line": "сохранённые строки лежат выше этой линии",
        "s1": "Чтение", "s1t": "Ключи и значения строк промпта пока сохраняются.",
        "s2": "Оценка", "s2t": "На каждую KV-голову: насколько ключ вне опорных точек и норма значения. Линейная модель по 72 рангам.",
        "s3": "Отбор", "s3t": "Перед последними 512 строками лучшие строки с соседями уходят из дальнего поля, а оно собирается заново без них.",
        "s4": "Ответ", "s4t": "Вопрос читается уже при точном факте и несёт его дальше к ответу.",
        "x_res": "Факты в 100 000 символов: 8 из 8, при десятой части памяти точной модели",
        "c_title": "Обычная и турбо: одни и те же веса",
        "c_sub": "q4tp, 2,09 ГБ, NVIDIA L4 (24 ГБ), Vulkan. Точный режим: обычный файл, два цикла.",
        "c_speed": "Декод, ток/с", "c_mem": "Память KV на устройстве, ГБ", "c_ctx": "контекст, токенов", "c_std": "обычная", "c_tur": "турбо", "c_dnf": "промпт не прочитан за 25 мин",
    },
    "zh": {
        "r_title": "每个 token 一次或两次循环",
        "r_sub": "同一个权重文件可以跑任意循环次数。路由器在第一次循环后决定,此时还没有读取输出。",
        "token": "token", "input": "输入块", "sub16": "16 个子层", "rec": "循环块", "sub40": "40 个子层",
        "router": "路由器", "state": "循环后的状态", "easy": "简单 token", "hard": "困难 token",
        "err": "哪个子空间更好地重建状态?", "output": "输出块", "next": "下一个 token",
        "leave": "简单:离开循环", "again": "困难:第二次循环",
        "r_n1": "两个秩为 16 的 PCA 子空间,由模型自己的回答拟合。",
        "r_n2": "没有梯度训练,也不需要经过输出块和输出头读取。",
        "r_n3": "每层一个缓存:每次循环覆盖本 token 的行,因此不留空洞。",
        "r_res": "真实提问:Vulkan 49.6 → 60.7 tok/s,M4 25.7 → 32.9",
        "f_title": "长提示词在显卡上占用很少",
        "f_sub": "一个全注意力层从提示词中保留什么。滑动层本来就只保留 1024 行。",
        "f_std": "两次循环,精确", "f_std2": "每行、每次循环都保存:内存和每个 token 的代价随文本增长",
        "f_tur": "Turbo", "f_near1": "最前几行", "f_far": "其余全部:每个头 32 个地标", "f_near2": "最后 1536 行",
        "f_keep": "选出的 512 行:精确", "f_q": "token 读取", "f_ex": "精确", "f_sk": "摘要", "f_res1": "32 000 行时 4.3 GB", "f_res2": "任意长度都是 0.5 GB",
        "x_title": "密码和代码保持精确",
        "x_sub": "远场会丢失的行,靠各层自身的行为找出,不使用词表。",
        "x_y": "行的得分", "x_x": "选择之前读取的提示词行", "x_fact": "“Remember this pass key: 653269.”", "x_rank": "它的 6 行:22 976 行中第 0 到 7 名", "x_line": "保留的行在这条线之上",
        "s1": "读取", "s1t": "提示词各行的键和值暂时保留。",
        "s2": "打分", "s2t": "每个 KV 头:键离地标张成空间有多远,以及值的范数。对 72 个排名做线性拟合。",
        "s3": "保留", "s3t": "在最后 512 行之前,得分最高的行及其相邻行离开远场,远场不含它们重新构建。",
        "s4": "回答", "s4t": "问题在精确事实上读取,并把它带到回答中。",
        "x_res": "100 000 字符中的事实:8 中 8,内存只有精确模型的十分之一",
        "c_title": "普通版与 Turbo:同一份权重",
        "c_sub": "q4tp,2.09 GB,NVIDIA L4((24 GB),Vulkan。精确模式是普通文件的两次循环。",
        "c_speed": "解码,tok/s", "c_mem": "设备 KV 内存,GB", "c_ctx": "上下文,token 数", "c_std": "普通", "c_tur": "Turbo", "c_dnf": "25 分钟未读完提示词",
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

    def line(self, x1, y1, x2, y2, stroke=LINE, sw=2, dash=None, marker=False):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        m = ' marker-end="url(#a%s)"' % stroke.strip("#") if marker else ""
        self.o.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{sw}" stroke-linecap="round"{d}{m}/>')

    def path(self, d, stroke, sw=3, fill="none", marker=False, dash=None):
        m = ' marker-end="url(#a%s)"' % stroke.strip("#") if marker else ""
        da = f' stroke-dasharray="{dash}"' if dash else ""
        self.o.append(f'<path d="{d}" stroke="{stroke}" stroke-width="{sw}" fill="{fill}" stroke-linecap="round" stroke-linejoin="round"{m}{da}/>')

    def dot(self, x, y, r, fill):
        self.o.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}" stroke="{SURF}" stroke-width="2"/>')

    def done(self, name):
        heads = "".join(
            f'<marker id="a{c.strip("#")}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="{c}"/></marker>'
            for c in (INK2, BLUE, ORANGE, AQUA, MUTE)
        )
        svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" viewBox="0 0 {self.w} {self.h}" font-family="{FONT}"><defs>{heads}</defs><rect width="{self.w}" height="{self.h}" fill="{SURF}"/>{"".join(self.o)}</svg>'
        p = os.path.join(out, name)
        open(p + ".svg", "w", encoding="utf-8").write(svg)
        subprocess.run(["rsvg-convert", "-w", str(self.w), p + ".svg", "-o", p + ".png"], check=True)
        os.remove(p + ".svg")


def head(s, t, title, sub):
    s.text(50, 66, title, 40, INK, "700")
    s.text(50, 108, sub, 23, INK2)


def box(s, x, y, w, h, t1, t2=None, fill=TINT, stroke=BLUE):
    s.rect(x, y, w, h, fill, 12, stroke)
    if t2:
        s.text(x + w / 2, y + h / 2 - 5, t1, 24 if len(t1) < 16 else 21, INK, "600", "middle")
        s.text(x + w / 2, y + h / 2 + 25, t2, 20, INK2, "400", "middle")
    else:
        s.text(x + w / 2, y + h / 2 + 8, t1, 24, INK, "600", "middle")


def router_fig(lang):
    t = T[lang]
    s = Svg(1600, 770)
    head(s, t, t["r_title"], t["r_sub"])
    y, h = 190, 110
    box(s, 50, y, 120, h, t["token"], None, "#f4f3ef", GREY)
    box(s, 225, y, 215, h, t["input"], t["sub16"])
    box(s, 495, y, 245, h, t["rec"], t["sub40"])
    # The router with its two subspaces.
    s.rect(800, 150, 390, 260, "#fff8f4", 14, ORANGE)
    s.text(995, 186, t["router"], 26, INK, "700", "middle")
    s.text(995, 216, t["state"], 19, INK2, "400", "middle")
    s.rect(825, 236, 165, 92, TINT, 10, BLUE)
    s.text(907, 276, t["easy"], 20 if lang != "ru" else 18, INK, "600", "middle")
    s.text(907, 306, "PCA, rank 16", 17, INK2, "400", "middle")
    s.rect(1000, 236, 165, 92, TINT2, 10, ORANGE)
    s.text(1082, 276, t["hard"], 20 if lang != "ru" else 18, INK, "600", "middle")
    s.text(1082, 306, "PCA, rank 16", 17, INK2, "400", "middle")
    s.text(995, 372, t["err"], 19, INK2, "400", "middle")
    box(s, 1270, y, 230, h, t["output"], t["sub16"])
    s.text(1385, 348, t["next"], 22, INK2, "400", "middle")
    s.line(1385, y + h + 4, 1385, 322, INK2, 3, marker=True)
    s.line(172, y + h / 2, 222, y + h / 2, INK2, 3, marker=True)
    s.line(442, y + h / 2, 492, y + h / 2, INK2, 3, marker=True)
    s.line(742, y + h / 2, 797, y + h / 2, INK2, 3, marker=True)
    s.line(1192, y + h / 2, 1267, y + h / 2, BLUE, 4, marker=True)
    s.text(1385, 172, t["leave"], 21, BLUE, "600", "middle")
    # The way back.
    s.path(f"M 995 412 L 995 470 L 617 470 L 617 {y + h + 6}", ORANGE, 4, marker=True)
    s.text(806, 505, t["again"], 22, ORANGE, "600", "middle")
    for i, n in enumerate(("r_n1", "r_n2", "r_n3")):
        s.dot(62, 560 + i * 40, 6, BLUE)
        s.text(80, 568 + i * 40, t[n], 23, INK)
    s.rect(50, 684, 1500, 58, TINT, 10)
    s.text(800, 722, t["r_res"], 26, INK, "700", "middle")
    s.done(f"router_{lang}")


def far_fig(lang):
    t = T[lang]
    s = Svg(1600, 690)
    head(s, t, t["f_title"], t["f_sub"])
    x0, x1 = 50, 1550
    # Exact: every row.
    s.text(x0, 180, t["f_std"], 28, INK, "700")
    s.text(x0 + (300 if lang == "ru" else 250 if lang == "en" else 200), 180, t["f_std2"], 21, INK2)
    n = 150
    w = (x1 - x0) / n
    for i in range(n):
        s.rect(x0 + i * w, 196, w - 2, 52, BLUE, 1)
    s.text(x1, 286, t["f_res1"], 26, INK, "700", "end")
    # Turbo.
    s.text(x0, 340, t["f_tur"], 28, INK, "700")
    y, h = 372, 76
    a, b, c = 80, 1040, 380
    gap = (x1 - x0 - a - b - c) / 2
    s.rect(x0, y, a, h, AQUA, 6)
    s.rect(x0 + a + gap, y, b, h, "#efeee9", 6, GREY)
    s.rect(x1 - c, y, c, h, AQUA, 6)
    s.text(x0 + a / 2, y + 112, t["f_near1"], 20, INK2, "400", "middle")
    s.text(x0 + a + gap + b / 2, y + h / 2 + 8, t["f_far"], 24, INK2, "500", "middle")
    s.text(x1 - c / 2, y + h / 2 + 8, t["f_near2"], 24, "#ffffff", "700", "middle")
    # Chosen rows inside the far stretch, pulled out above it.
    px = [0.03, 0.1, 0.17, 0.24, 0.77, 0.85, 0.94]
    for p in px:
        xx = x0 + a + gap + 20 + p * (b - 40)
        s.rect(xx, y - 8, 8, h + 16, ORANGE, 2)
    s.text(x0 + a + gap + b / 2, y + 112, t["f_keep"], 24, ORANGE, "700", "middle")
    # Legend of what a token reads.
    ly = 570
    s.text(x0, ly + 8, t["f_q"] + ":", 23, INK2)
    lx = x0 + (190 if lang != "zh" else 140)
    s.rect(lx, ly - 14, 34, 26, AQUA, 4)
    s.text(lx + 44, ly + 8, t["f_ex"], 23, INK)
    s.rect(lx + 150, ly - 14, 34, 26, ORANGE, 4)
    s.text(lx + 194, ly + 8, t["f_ex"], 23, INK)
    s.rect(lx + 300, ly - 14, 34, 26, "#efeee9", 4, GREY)
    s.text(lx + 344, ly + 8, t["f_sk"], 23, INK)
    s.rect(50, 610, 1500, 58, TINT, 10)
    s.text(800, 648, t["f_res2"], 28, INK, "700", "middle")
    s.done(f"far_field_{lang}")


def exact_fig(lang, sc, frows):
    t = T[lang]
    s = Svg(1600, 900)
    head(s, t, t["x_title"], t["x_sub"])
    px0, px1, py0, py1 = 110, 1550, 470, 150
    n = len(sc)
    lo, hi = float(np.percentile(sc, 50)), float(sc.max())
    bins = int(px1 - px0) // 2
    top = np.full(bins, -1e9)
    for i, v in enumerate(sc):
        b = min(bins - 1, i * bins // n)
        top[b] = max(top[b], v)
    Y = lambda v: py0 + (py1 - py0) * (v - lo) / (hi - lo)
    fb = {min(bins - 1, r * bins // n) for r in frows}
    for b in range(bins):
        x = px0 + b * 2
        c = ORANGE if b in fb else "#b9c9dc"
        if b not in fb and top[b] > lo:
            s.line(x, py0, x, Y(top[b]), c, 1.6)
    th = float(np.sort(sc)[-512])
    s.line(px0, Y(th), px1, Y(th), INK2, 2, "8 6")
    s.text(px0 + 10, Y(th) - 10, t["x_line"], 20, INK2)
    for b in fb:
        x = px0 + b * 2
        s.line(x, py0, x, Y(top[b]), ORANGE, 5)
    fx = px0 + min(fb) * 2
    anchor = "end" if fx > 900 else "start"
    dx = -16 if fx > 900 else 16
    s.text(fx + dx, py1 + 26, t["x_fact"], 24, INK, "700", anchor)
    s.text(fx + dx, py1 + 56, t["x_rank"], 21, ORANGE, "600", anchor)
    s.line(px0, py0, px1, py0, LINE, 2)
    s.text(px0, py0 + 34, "0", 20, INK2)
    s.text(px1, py0 + 34, f"{n:,}".replace(",", " "), 20, INK2, "400", "end")
    s.text((px0 + px1) / 2, py0 + 34, t["x_x"], 21, INK2, "400", "middle")
    s.o.append(f'<text transform="translate(66 {(py0 + py1) / 2}) rotate(-90)" font-size="21" fill="{INK2}" text-anchor="middle">{t["x_y"]}</text>')
    # Four steps.
    y = 545
    w = 362
    for i, k in enumerate(("s1", "s2", "s3", "s4")):
        x = 50 + i * (w + 17)
        s.rect(x, y, w, 240, TINT if i != 2 else TINT2, 12)
        s.dot(x + 32, y + 36, 19, BLUE if i != 2 else ORANGE)
        s.text(x + 32, y + 44, str(i + 1), 22, "#ffffff", "700", "middle")
        s.text(x + 62, y + 45, t[k], 27, INK, "700")
        for j, ln in enumerate(wrap(t[k + "t"], lang, 31 if lang != "zh" else 16)):
            s.text(x + 22, y + 90 + j * 30, ln, 21, INK)
    s.rect(50, 808, 1500, 62, TINT, 10)
    s.text(800, 848, t["x_res"], 24 if lang != "ru" else 23, INK, "700", "middle")
    s.done(f"exact_rows_{lang}")


def wrap(text, lang, n):
    if lang == "zh":
        return [text[i:i + n] for i in range(0, len(text), n)]
    out, cur = [], ""
    for word in text.split():
        if len(cur) + len(word) + 1 > n and cur:
            out.append(cur)
            cur = word
        else:
            cur = (cur + " " + word).strip()
    return out + [cur] if cur else out


def compare_fig(lang):
    t = T[lang]
    s = Svg(1600, 640)
    head(s, t, t["c_title"], t["c_sub"])
    ctx = [1000, 4000, 16000, 32000, 64000]
    labels = ["1 000", "4 000", "16 000", "32 000", "64 000"]
    std = [41.1, 35.1, 21.5, 14.1, None]
    tur = [60.6, 51.8, 52.2, 54.0, 53.0]
    mstd = [0.998, 0.998, 2.408, 4.287, None]
    mtur = [0.642, 0.642, 0.502, 0.502, 0.502]

    def panel(x0, x1, title, a, b, ymax, step, fmt):
        y0, y1 = 540, 200
        s.text(x0, 170, title, 26, INK, "700")
        Y = lambda v: y0 + (y1 - y0) * v / ymax
        lg = [math.log2(c) for c in ctx]
        X = lambda i: x0 + 50 + (x1 - x0 - 120) * (lg[i] - lg[0]) / (lg[-1] - lg[0])
        v = 0
        while v <= ymax + 1e-9:
            s.line(x0 + 40, Y(v), x1, Y(v), LINE, 1.5)
            s.text(x0 + 30, Y(v) + 7, fmt(v), 20, INK2, "400", "end")
            v += step
        for i, lb in enumerate(labels):
            s.text(X(i), y0 + 32, lb, 20, INK2, "400", "middle")
        s.text((x0 + x1) / 2, y0 + 66, t["c_ctx"], 20, INK2, "400", "middle")
        for ser, c in ((a, ORANGE), (b, BLUE)):
            pts = [(X(i), Y(v)) for i, v in enumerate(ser) if v is not None]
            s.path("M " + " L ".join(f"{p[0]:.1f} {p[1]:.1f}" for p in pts), c, 3)
            for p in pts:
                s.dot(p[0], p[1], 8, c)
        return X, Y

    n = (lambda v: f"{v:.1f}".replace(".", ",")) if lang == "ru" else (lambda v: f"{v:.1f}")
    X, Y = panel(50, 780, t["c_speed"], std, tur, 70, 10, lambda v: f"{v:.0f}")
    s.text(X(4) + 6, Y(53.0) - 22, f"{t['c_tur']} {n(53.0)}", 23, INK, "700", "end")
    s.text(X(0) + 6, Y(60.6) - 18, n(60.6), 21, INK, "400", "start")
    s.text(X(3) - 4, Y(14.1) - 22, f"{t['c_std']} {n(14.1)}", 23, INK, "700", "start")
    s.text(X(0) + 6, Y(41.1) + 34, n(41.1), 21, INK, "400", "start")
    s.text(X(4) + 30, Y(4), t["c_dnf"], 19, ORANGE, "600", "end")
    X, Y = panel(830, 1550, t["c_mem"], mstd, mtur, 5, 1, lambda v: f"{v:.0f}")
    s.text(X(3) - 12, Y(4.287) - 18, f"{t['c_std']} {n(4.3)}", 23, INK, "700", "end")
    s.text(X(4) + 6, Y(0.502) - 20, f"{t['c_tur']} {n(0.5)}", 23, INK, "700", "end")
    s.done(f"compare_{lang}")


if __name__ == "__main__":
    score_file, out = sys.argv[1:3]
    os.makedirs(out, exist_ok=True)
    sc = np.fromfile(score_file, dtype=np.float32)
    frows = list(range(17421, 17427))
    for lang in T:
        router_fig(lang)
        far_fig(lang)
        exact_fig(lang, sc, frows)
        compare_fig(lang)
    print("ok")
