#!/usr/bin/env python3
"""The figures of the CherryPit-1.2-2B Turbo page, from measured numbers.
cp_figures.py SCORES.bin FACT_ROW_FIRST FACT_ROW_LAST OUT_DIR FACTS_300K_OK FACTS_300K_TRIALS
(SCORES.bin: `CMF_DORY_EXACT_LOG` of a prompt with a fact; rsvg-convert on PATH)"""
import math, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dory"))
import card_figures as cf
from card_figures import Svg, SURF, INK, INK2, MUTE, LINE, BLUE, ORANGE, AQUA, GREY, TINT, TINT2, head

score_file, f0, f1, cf.out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
os.makedirs(cf.out, exist_ok=True)

# Measured on an RTX 5090, Vulkan: `cortiq bench --core --tokens 60 --ignore-eos --ctx N` (pod/turbobench.sh).
CTX = [1000, 4000, 16000, 32000, 64000]
LABELS = ["1 000", "4 000", "16 000", "32 000", "64 000"]
# Memory: nvidia-smi's peak in MiB / 1024, the unit a card's label uses.
STD = dict(dec=[108.0, 111.9, 90.5, 73.7, 42.9], first=[0.43, 2.31, 18.11, 61.67, 228.53], vram=[2.1, 2.4, 3.5, 4.8, 7.4])
TUR = dict(dec=[109.4, 108.2, 66.9, 67.4, 67.0], first=[0.45, 2.29, 48.80, 86.41, 171.90], vram=[1.9, 2.4, 2.9, 2.7, 3.2])
FACTS = dict(en="A planted fact: 8 of 8 in 100 000 characters, {f300} in 300 000 characters (about 67 000 tokens)",
             ru="Факт в тексте: 8 из 8 в 100 000 символов, {f300} в 300 000 символов (около 67 000 токенов)",
             zh="插入的事实：100 000 字符中 8/8，300 000 字符（约 67 000 token）中 {f300}")

T = {
    "en": {
        "t": "Longer prompts cost no more on the card",
        "sub": "CherryPit-1.2-2B q4tp, RTX 5090, Vulkan. Prompts below 8192 rows run exact in both files.",
        "dec": "Decode, tok/s", "first": "First token, s", "vram": "Peak device memory, GB", "ctx": "prompt, tokens",
        "std": "Standard", "tur": "Turbo", "card": "4 GB card",
        "c_t": "A long prompt on a small card",
        "c_sub": "Device memory stays where the first 8192 rows left it. Later rows leave the card as they are read.",
        "a": "first 8192 rows", "a2": "exact, on the card",
        "b": "the rest of the prompt", "b2": "each chunk's keys and values go to RAM or a fast SSD",
        "c": "last 512", "c2": "read against the rest",
        "s1": "Choose", "s1t": "The rows that hold passwords, codes and names are found among all the stored rows.",
        "s2": "Build", "s2t": "The far field is built anew from the stored rows, one layer at a time: 32 landmarks per head.",
        "s3": "Keep exact", "s3t": "248 chosen rows, the first 4 and the last 1800 stay exact on the card.",
        "s4": "Decode", "s4t": "A token reads the exact rows and the far field, so its cost stops growing with the text.",
        "res": "3.2 GB of device memory and 67 tok/s at 64 000 tokens; the standard file 7.4 GB and 43 tok/s",
        "x_t": "Passwords and codes stay exact",
        "x_sub": "Each row is scored by what the layers do with it, with no word lists.",
        "x_y": "row’s score", "x_x": "prompt rows read before the choice", "x_fact": "“Remember this pass key.”",
        "x_rank": "its rows: places {} of {}", "x_line": "the rows kept lie above this line",
        "x_1": "Per KV head of each layer, two numbers per row: how far its key lies from the span of the landmarks, and its value’s norm.",
        "x_2": "Each becomes a rank within the prompt, and a linear fit over 168 ranks gives the score, stored in the file.",
        "x_3": "The choice comes before the prompt’s last 512 rows: a question reads the exact fact and carries it on to the answer.",
    },
    "ru": {
        "t": "Длинный промпт не занимает больше памяти карты",
        "sub": "CherryPit-1.2-2B q4tp, RTX 5090, Vulkan. Промпты короче 8192 строк в обоих файлах идут точно.",
        "dec": "Декод, ток/с", "first": "Первый токен, с", "vram": "Пик памяти карты, ГБ", "ctx": "промпт, токенов",
        "std": "Обычный", "tur": "Турбо", "card": "карта 4 ГБ",
        "c_t": "Длинный промпт на маленькой карте",
        "c_sub": "Память карты остаётся там, где её оставили первые 8192 строки. Остальные строки уходят с карты по мере чтения.",
        "a": "первые 8192 строки", "a2": "точно, на карте",
        "b": "остальной промпт", "b2": "ключи и значения каждого куска уходят в RAM или на быстрый SSD",
        "c": "последние 512", "c2": "читают всё остальное",
        "s1": "Выбор", "s1t": "Среди всех сохранённых строк находятся те, где пароли, коды и имена.",
        "s2": "Сборка", "s2t": "Дальнее поле строится заново из сохранённых строк, слой за слоем: 32 опорные точки на голову.",
        "s3": "Точные", "s3t": "248 выбранных строк, первые 4 и последние 1800 остаются на карте точными.",
        "s4": "Декод", "s4t": "Токен читает точные строки и дальнее поле, и его цена перестаёт расти вместе с текстом.",
        "res": "3,2 ГБ памяти карты и 67 ток/с на 64 000 токенов; у обычного файла 7,4 ГБ и 43 ток/с",
        "x_t": "Пароли и коды остаются точными",
        "x_sub": "Строка получает оценку по тому, что с ней делают слои. Списков слов нет.",
        "x_y": "оценка строки", "x_x": "строки промпта, прочитанные до выбора", "x_fact": "«Запомни этот ключ.»",
        "x_rank": "его строки: места {} из {}", "x_line": "оставленные строки лежат выше этой линии",
        "x_1": "На каждую KV-голову каждого слоя два числа на строку: насколько ключ далёк от опорных точек и норма значения.",
        "x_2": "Каждое становится рангом внутри промпта, и линейная модель по 168 рангам даёт оценку. Веса лежат в файле.",
        "x_3": "Выбор идёт до последних 512 строк промпта: вопрос читает точный факт и несёт его к ответу.",
    },
    "zh": {
        "t": "提示再长，显存也不再增加",
        "sub": "CherryPit-1.2-2B q4tp，RTX 5090，Vulkan。短于 8192 行的提示在两个文件中都走精确模式。",
        "dec": "解码，tok/s", "first": "首个 token，秒", "vram": "显存峰值，GB", "ctx": "提示，token 数",
        "std": "普通", "tur": "Turbo", "card": "4 GB 显卡",
        "c_t": "小显卡上的长提示",
        "c_sub": "显存停留在前 8192 行占用的位置，其余的行一读完就离开显卡。",
        "a": "前 8192 行", "a2": "精确，在显卡上",
        "b": "提示的其余部分", "b2": "每块的键和值进入内存或高速 SSD",
        "c": "最后 512 行", "c2": "读取前面所有内容",
        "s1": "选择", "s1t": "在所有保存的行中找出含有密码、编码和名字的行。",
        "s2": "构建", "s2t": "用保存的行逐层重建远场：每个头 32 个地标。",
        "s3": "保持精确", "s3t": "选出的 248 行、前 4 行和后 1800 行以精确形式留在显卡上。",
        "s4": "解码", "s4t": "每个 token 读取精确行和远场，成本不再随文本增长。",
        "res": "64 000 token 时显存 3.2 GB、解码 67 tok/s；普通文件为 7.4 GB、43 tok/s",
        "x_t": "密码和编码保持精确",
        "x_sub": "根据各层对每一行的处理来打分，不用词表。",
        "x_y": "行的分数", "x_x": "选择之前读取的提示行", "x_fact": "“记住这个密钥。”",
        "x_rank": "它的行：第 {} 位，共 {} 行", "x_line": "保留的行在这条线之上",
        "x_1": "每层的每个 KV 头，每行两个数：键与地标张成空间的距离，以及值的范数。",
        "x_2": "两者都转为提示内的排名，再由 168 个排名的线性拟合给出分数，权重存放在文件中。",
        "x_3": "选择发生在提示最后 512 行之前：问题读到精确的事实，并把它带到答案中。",
    },
}


def wrap(text, lang, n):
    return cf.wrap(text, lang, n)


def compare(lang):
    t = T[lang]
    s = Svg(1600, 680)
    head(s, t, t["t"], t["sub"])
    # A legend, and each line named at its end.
    s.path("M 50 148 L 90 148", ORANGE, 3)
    s.dot(70, 148, 8, ORANGE)
    s.text(102, 156, t["std"], 22, INK)
    s.path("M 250 148 L 290 148", BLUE, 3)
    s.dot(270, 148, 8, BLUE)
    s.text(302, 156, t["tur"], 22, INK)
    n1 = (lambda v: f"{v:.1f}".replace(".", ",")) if lang == "ru" else (lambda v: f"{v:.1f}")

    def panel(x0, x1, title, a, b, ymax, step, ends):
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
            if i in (0, 2, 4):
                s.text(X(i), y0 + 30, lb, 19, INK2, "400", "middle")
        s.text((x0 + x1) / 2, y0 + 62, t["ctx"], 19, INK2, "400", "middle")
        for ser, c in ((a, ORANGE), (b, BLUE)):
            pts = [(X(i), Y(v)) for i, v in enumerate(ser)]
            s.path("M " + " L ".join(f"{p[0]:.1f} {p[1]:.1f}" for p in pts), c, 3)
            for p in pts:
                s.dot(p[0], p[1], 7, c)
        s.text(X(4), Y(a[4]) + ends[0], n1(a[4]) if ymax < 40 else f"{a[4]:.0f}", 22, INK, "700", "end")
        s.text(X(4), Y(b[4]) + ends[1], n1(b[4]) if ymax < 40 else f"{b[4]:.0f}", 22, INK, "700", "end")
        return X, Y

    panel(50, 540, t["dec"], STD["dec"], TUR["dec"], 120, 20, (34, -16))
    panel(570, 1060, t["first"], STD["first"], TUR["first"], 250, 50, (-16, 34))
    X, Y = panel(1090, 1550, t["vram"], STD["vram"], TUR["vram"], 8, 2, (-16, 34))
    s.line(1134, Y(4), 1550, Y(4), INK2, 2, "8 6")
    s.text(1140, Y(4) - 10, t["card"], 19, INK2)
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
        for j, ln in enumerate(wrap(t[k + "t"], lang, 29 if lang != "zh" else 15)):
            s.text(x + 22, yy + 92 + j * 30, ln, 21, INK)
    s.rect(50, 548, 1500, 62, TINT, 10)
    s.text(800, 588, t["res"], 24 if lang != "ru" else 23, INK, "700", "middle")
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
    span = f"{ranks[0]}–{ranks[-1]}"
    s.text(fx + dx, py1 + 26, t["x_fact"], 24, INK, "700", anchor)
    s.text(fx + dx, py1 + 56, t["x_rank"].format(span, f"{n:,}".replace(",", " ")), 21, ORANGE, "600", anchor)
    s.line(px0, py0, px1, py0, LINE, 2)
    s.text(px0, py0 + 34, "0", 20, INK2)
    s.text(px1, py0 + 34, f"{n:,}".replace(",", " "), 20, INK2, "400", "end")
    s.text((px0 + px1) / 2, py0 + 34, t["x_x"], 21, INK2, "400", "middle")
    s.o.append(f'<text transform="translate(66 {(py0 + py1) / 2}) rotate(-90)" font-size="21" fill="{INK2}" text-anchor="middle">{t["x_y"]}</text>')
    yy = 545
    ww = 494
    for i, k in enumerate(("x_1", "x_2", "x_3")):
        x = 50 + i * (ww + 9)
        s.rect(x, yy, ww, 140, TINT if i != 2 else TINT2, 12)
        for j, ln in enumerate(wrap(t[k], lang, 43 if lang != "zh" else 22)):
            s.text(x + 22, yy + 44 + j * 30, ln, 21, INK)
    s.rect(50, 708, 1500, 62, TINT, 10)
    s.text(800, 748, FACTS[lang].format(f300=F300[lang]), 24 if lang != "ru" else 23, INK, "700", "middle")
    s.done(f"exact_rows_{lang}")


F300 = dict(en=sys.argv[5] + " of " + sys.argv[6], ru=sys.argv[5] + " из " + sys.argv[6], zh=sys.argv[5] + "/" + sys.argv[6])
sc = np.fromfile(score_file, dtype=np.float32)
rows = list(range(f0, f1 + 1))
for lang in T:
    compare(lang)
    ceiling(lang)
    exact(lang, sc, rows)
print("ok", cf.out, "places of the fact:", sorted(int(np.sum(sc > sc[r])) for r in rows))
