#!/usr/bin/env python3
"""The figures of the Ripple-65B page, from measured numbers.
ripple_figures.py SCORES.bin FACT_ROW_FIRST FACT_ROW_LAST OUT_DIR
(SCORES.bin: `CMF_DORY_EXACT_LOG` of a prompt with a fact, scripts/ripple/pod/quality.sh; rsvg-convert on PATH)"""
import math, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dory"))
import card_figures as cf
from card_figures import Svg, INK, INK2, LINE, BLUE, ORANGE, GREY, TINT, TINT2, head

score_file, f0, f1, cf.out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
os.makedirs(cf.out, exist_ok=True)

# Measured on an NVIDIA A40 (46 GB), Vulkan: `cortiq bench --core --tokens 60 --ignore-eos`, the q4tp file.
# Memory: nvidia-smi's peak in MiB / 1000, the unit the card's label uses (46 GB), as the page's tables.
CTX = [4000, 16000, 32000, 64000]
LABELS = ["4 000", "16 000", "32 000", "64 000"]
STD = dict(dec=[12.1, 11.1, 10.2, 8.8], first=[48.4, 212.9, 473.2, 1138.2], vram=[35.3, 36.8, 39.1, 43.2])
TUR = dict(dec=[12.0, 11.7, 11.8, 11.8], first=[48.0, 251.2, 491.1, 987.8], vram=[35.3, 36.6, 36.6, 36.6])

# PPL over the first 2048 tokens of wikitext-2 test; BF16 6.940.
HOT = [("q2mix", 9.265, GREY), ("q2hot", 7.724, GREY), ("q2", 7.432, BLUE), ("q4", 7.468, GREY), ("q4kv", 7.089, BLUE)]
BF16 = 6.940

T = {
    "en": {
        "t": "Longer prompts cost no more on the card",
        "sub": "Ripple-65B q4tp, 34.6 GB, NVIDIA A40 46 GB, Vulkan. Prompts below 8192 rows run exact in both files.",
        "dec": "Decode, tok/s", "first": "First token, s", "vram": "Peak device memory, GB", "ctx": "prompt, tokens",
        "std": "Standard", "tur": "Turbo",
        "c_t": "A long prompt on a 46 GB card",
        "c_sub": "Device memory stays where the first part of the prompt left it. The rest of the rows leave the card as they are read.",
        "a": "first 8192 rows", "a2": "exact, on the card",
        "b": "the rest of the prompt", "b2": "each chunk's keys and values go to RAM or a fast SSD",
        "c": "last 512", "c2": "read against the rest",
        "s1": "Choose", "s1t": "The rows that hold passwords, codes and names are found among all the stored rows.",
        "s2": "Build", "s2t": "The far field is built anew from the stored rows, one layer at a time: 32 landmarks per head of the 16 full layers.",
        "s3": "Keep exact", "s3t": "248 chosen rows, the first 4 and the last 1800 stay exact on the card. The 64 sliding layers hold 1024 rows anyway.",
        "s4": "Decode", "s4t": "A token reads the exact rows and the far field. The drafter's tokens are verified together, so a token costs less.",
        "res": "64 000 tokens: 36.6 GB instead of 43.2, 11.8 tok/s instead of 8.8, the first token in 988 s instead of 1138.",
        "x_t": "Passwords and codes stay exact",
        "x_sub": "Each row is scored by what the layers do with it, with no word lists.",
        "x_y": "row’s score", "x_x": "prompt rows read before the choice", "x_fact": "“Remember this pass key: 653269.”",
        "x_rank": "the digits’ two rows rank {} of {}", "x_line": "the rows kept lie above this line",
        "x_1": "Per KV head of each full layer, two numbers per row: how far its key lies from the span of the landmarks, and its value’s norm.",
        "x_2": "Each becomes a rank within the prompt, and a linear fit over 256 ranks gives the score, stored in the file.",
        "x_3": "The choice comes before the prompt’s last 512 rows: a question reads the exact fact and carries it on to the answer.",
        "x_res": "Four facts in 60 000 characters: three answered; the fourth was still reasoning at its 300-token limit.",
        "h_t": "Two bits, with the busiest neurons at eight",
        "h_sub": "Perplexity on the first 2048 tokens of wikitext-2 test; BF16 6.94. The two files that go out are in blue.",
        "q2mix": "q2tp mix: layer 0 and the last two at eight bits", "q2hot": "+ the 5 % of each layer's FFN neurons that carry half the activation energy",
        "q2": "+ K and V projections at eight bits — the q2tp file, 29.2 GB", "q4": "q4tp", "q4kv": "q4tp with K and V at eight bits — the q4tp file, 34.6 GB",
        "h_bf": "BF16, 130 GB", "h_res": "The two-bit file reaches the four-bit one's perplexity at 85 % of its size; 1–2 % of hot neurons at four bits change nothing.",
    },
    "ru": {
        "t": "Длинный промпт не занимает больше памяти карты",
        "sub": "Ripple-65B q4tp, 34,6 ГБ, NVIDIA A40 46 ГБ, Vulkan. Промпты короче 8192 строк в обоих файлах идут точно.",
        "dec": "Декод, ток/с", "first": "Первый токен, с", "vram": "Пик памяти карты, ГБ", "ctx": "промпт, токенов",
        "std": "Обычный", "tur": "Турбо",
        "c_t": "Длинный промпт на карте 46 ГБ",
        "c_sub": "Память карты остаётся там, где её оставила первая часть промпта. Остальные строки уходят с карты по мере чтения.",
        "a": "первые 8192 строки", "a2": "точно, на карте",
        "b": "остальной промпт", "b2": "ключи и значения каждого куска уходят в RAM или на быстрый SSD",
        "c": "последние 512", "c2": "читают всё остальное",
        "s1": "Выбор", "s1t": "Среди всех сохранённых строк находятся те, где пароли, коды и имена.",
        "s2": "Сборка", "s2t": "Дальнее поле строится заново из сохранённых строк, слой за слоем: 32 опорные точки на голову в 16 полных слоях.",
        "s3": "Точные", "s3t": "248 выбранных строк, первые 4 и последние 1800 остаются на карте точными. 64 скользящих слоя и так держат 1024 строки.",
        "s4": "Декод", "s4t": "Токен читает точные строки и дальнее поле. Черновики проверяются вместе, и токен обходится дешевле.",
        "res": "64 000 токенов: 36,6 ГБ вместо 43,2, 11,8 ток/с вместо 8,8, первый токен за 988 с вместо 1138.",
        "x_t": "Пароли и коды остаются точными",
        "x_sub": "Строка получает оценку по тому, что с ней делают слои. Списков слов нет.",
        "x_y": "оценка строки", "x_x": "строки промпта, прочитанные до выбора", "x_fact": "«Запомни этот ключ: 653269.»",
        "x_rank": "две строки цифр: места {} из {}", "x_line": "оставленные строки лежат выше этой линии",
        "x_1": "На каждую KV-голову каждого полного слоя два числа на строку: насколько ключ далёк от опорных точек и норма значения.",
        "x_2": "Каждое становится рангом внутри промпта, и линейная модель по 256 рангам даёт оценку. Веса лежат в файле.",
        "x_3": "Выбор идёт до последних 512 строк промпта: вопрос читает точный факт и несёт его к ответу.",
        "x_res": "Четыре факта в 60 000 символов: три отвечены, четвёртый модель ещё обдумывала на пределе 300 токенов.",
        "h_t": "Два бита, а самые занятые нейроны — восемь",
        "h_sub": "Перплексия на первых 2048 токенах wikitext-2 test; BF16 6,94. Синим — два файла, которые выходят.",
        "q2mix": "смесь q2tp: слой 0 и два последних в восемь бит", "q2hot": "+ 5 % нейронов FFN каждого слоя, несущих половину энергии активаций",
        "q2": "+ проекции K и V в восемь бит — файл q2tp, 29,2 ГБ", "q4": "q4tp", "q4kv": "q4tp с K и V в восемь бит — файл q4tp, 34,6 ГБ",
        "h_bf": "BF16, 130 ГБ", "h_res": "Двухбитный файл выходит на перплексию четырёхбитного при 85 % его размера; 1–2 % горячих нейронов в четырёх битах ничего не меняют.",
    },
    "zh": {
        "t": "提示再长,显存也不再增加",
        "sub": "Ripple-65B q4tp,34.6 GB,NVIDIA A40 46 GB,Vulkan。短于 8192 行的提示在两个文件中都走精确模式。",
        "dec": "解码,tok/s", "first": "首个 token,秒", "vram": "显存峰值,GB", "ctx": "提示,token 数",
        "std": "普通", "tur": "Turbo",
        "c_t": "46 GB 显卡上的长提示",
        "c_sub": "显存停留在提示前段占用的位置,其余的行一读完就离开显卡。",
        "a": "前 8192 行", "a2": "精确,在显卡上",
        "b": "提示的其余部分", "b2": "每块的键和值进入内存或高速 SSD",
        "c": "最后 512 行", "c2": "读取前面所有内容",
        "s1": "选择", "s1t": "在所有保存的行中找出含有密码、编码和名字的行。",
        "s2": "构建", "s2t": "用保存的行逐层重建远场:16 个全注意力层每个头 32 个地标。",
        "s3": "保持精确", "s3t": "选出的 248 行、前 4 行和后 1800 行以精确形式留在显卡上。64 个滑动层本来只保留 1024 行。",
        "s4": "解码", "s4t": "每个 token 读取精确行和远场。草稿一起验证,单个 token 的成本更低。",
        "res": "64 000 token:显存 36.6 GB 而非 43.2,解码 11.8 tok/s 而非 8.8,首个 token 988 秒而非 1138 秒。",
        "x_t": "密码和编码保持精确",
        "x_sub": "根据各层对每一行的处理来打分,不用词表。",
        "x_y": "行的分数", "x_x": "选择之前读取的提示行", "x_fact": "“记住这个密钥:653269。”",
        "x_rank": "数字所在的两行名次:{},共 {} 行", "x_line": "保留的行在这条线之上",
        "x_1": "每个全注意力层的每个 KV 头,每行两个数:键与地标张成空间的距离,以及值的范数。",
        "x_2": "两者都转为提示内的排名,再由 256 个排名的线性拟合给出分数,权重存放在文件中。",
        "x_3": "选择发生在提示最后 512 行之前:问题读到精确的事实,并把它带到答案中。",
        "x_res": "60 000 个字符中的四个事实:答对三个,第四个在 300 个 token 用完时模型仍在推理。",
        "h_t": "两比特权重,最活跃的神经元用八比特",
        "h_sub": "wikitext-2 test 前 2048 个 token 的困惑度;BF16 为 6.94。蓝色是发布的两个文件。",
        "q2mix": "q2tp 混合:第 0 层和最后两层为八比特", "q2hot": "+ 每层 5 % 的 FFN 神经元(承载一半激活能量)",
        "q2": "+ K、V 投影为八比特 —— q2tp 文件,29.2 GB", "q4": "q4tp", "q4kv": "q4tp,K、V 为八比特 —— q4tp 文件,34.6 GB",
        "h_bf": "BF16,130 GB", "h_res": "两比特文件以四比特文件 85 % 的体积达到同样的困惑度;四比特下 1–2 % 的热神经元没有改变。",
    },
}


def wrap(text, lang, n):
    return cf.wrap(text, lang, n)


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
            s.text(X(i), y0 + 30, lb, 19, INK2, "400", "middle")
        s.text((x0 + x1) / 2, y0 + 62, t["ctx"], 19, INK2, "400", "middle")
        for ser, c in ((a, ORANGE), (b, BLUE)):
            pts = [(X(i), Y(v)) for i, v in enumerate(ser)]
            s.path("M " + " L ".join(f"{p[0]:.1f} {p[1]:.1f}" for p in pts), c, 3)
            for p in pts:
                s.dot(p[0], p[1], 7, c)
        last = len(a) - 1
        s.text(X(last), Y(a[last]) + ends[0], n1(a[last]) if ymax < 60 else f"{a[last]:.0f}", 22, INK, "700", "end")
        s.text(X(last), Y(b[last]) + ends[1], n1(b[last]) if ymax < 60 else f"{b[last]:.0f}", 22, INK, "700", "end")
        return X, Y

    panel(50, 540, t["dec"], STD["dec"], TUR["dec"], 15, 3, (30, -16))
    panel(570, 1060, t["first"], STD["first"], TUR["first"], 1200, 200, (-16, 34))
    panel(1090, 1550, t["vram"], STD["vram"], TUR["vram"], 50, 10, (-16, 34))
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
    s.text(800, 748, t["x_res"], 24 if lang != "ru" else 23, INK, "700", "middle")
    s.done(f"exact_rows_{lang}")


def hot(lang):
    """Perplexity of the two-bit mix as hot neurons and then K and V go to eight bits, beside q4tp."""
    t = T[lang]
    s = Svg(1600, 640)
    head(s, t, t["h_t"], t["h_sub"])
    x0, x1 = 760, 1500
    lo, hi = 6.5, 9.6
    X = lambda v: x0 + (x1 - x0) * (v - lo) / (hi - lo)
    y = 170
    for key, v, c in HOT:
        label = t[key]
        s.text(x0 - 20, y + 20, label if len(label) < 62 or lang == "zh" else label, 20 if len(label) < 56 else 18, INK, "400", "end")
        s.rect(x0, y + 4, X(v) - x0, 26, c, 4)
        s.text(X(v) + 10, y + 24, f"{v:.2f}".replace(".", "," if lang == "ru" else "."), 22, INK, "700")
        y += 66
    s.line(X(BF16), 160, X(BF16), y - 20, INK2, 2, "8 6")
    s.text(X(BF16) + 8, y - 2, f"{t['h_bf']}: {BF16:.2f}".replace(".", "," if lang == "ru" else "."), 20, INK2)
    s.line(x0, y + 10, x1, y + 10, LINE, 2)
    for v in (7, 8, 9):
        s.text(X(v), y + 40, str(v), 19, INK2, "400", "middle")
    s.rect(50, 558, 1500, 62, TINT, 10)
    s.text(800, 598, t["h_res"], 23 if lang != "ru" else 21, INK, "700", "middle")
    s.done(f"hot_{lang}")


sc = np.fromfile(score_file, dtype=np.float32)
rows = list(range(f0, f1 + 1))
for lang in T:
    compare(lang)
    ceiling(lang)
    exact(lang, sc, rows)
    hot(lang)
print("ok", cf.out, "places of the fact:", sorted(int(np.sum(sc > sc[r])) for r in rows))
