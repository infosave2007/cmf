#!/usr/bin/env python3
"""The figures of the Spark-X2.6-4B Turbo CUDA page, in English, Russian and Chinese, from measured numbers.
card_figures.py OUT_DIR      (rsvg-convert on PATH; the numbers are the medians of docs/CUDA.md)"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dory"))
import card_figures as cf
from card_figures import Svg, INK, INK2, LINE, BLUE, ORANGE, AQUA, TINT, TINT2, head

cf.out = sys.argv[1]
os.makedirs(cf.out, exist_ok=True)

CTX = ["1 000", "4 000", "16 000"]
# NVIDIA A40, `cortiq bench --core`, medians of three interleaved runs. Memory: nvidia-smi peak, MiB.
DATA = {
    "q4tp": {
        "prompt": {"cuda": [6132, 6096, 4808], "vk": [3177, 3789, 2949]},
        "decode": {"cuda": [116.9, 111.5, 92.7], "vk": [82.9, 80.2, 54.3]},
        "mem": {"cuda": [3587, 3993, 5009], "vk": [3697, 4569, 5622]},
    },
    "q8_2f": {
        "prompt": {"cuda": [5926, 5927, 4732], "vk": [513, 528, 507]},
        "decode": {"cuda": [82.7, 80.0, 69.4], "vk": [63.5, 60.1, 45.3]},
        "mem": {"cuda": [5361, 5767, 6783], "vk": [5326, 6250, 7319]},
    },
}

# Past the seal of 8 192 rows, q4tp: the far field on CUDA (the prompt read whole, one seal), the far field
# over Vulkan (sealed at 8 192 rows) and CUDA with exact attention — the medians `pod/table_turbo.py --json`
# wrote to figures_turbo.json. First token: medians of three runs. `decode` and `mem` (the card while it
# decodes, MiB): a run of 500 tokens.
CTX_T = ["16 000", "32 000", "64 000"]
_J = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures_turbo.json")))
_S = (("cuda", "cut"), ("vk", "vkt"), ("exact", "cux"))
DATA_T = {
    key: {name: [_J[f"q4_{c}_{tag}"][field] for c in (16000, 32000, 64000)] for name, tag in _S}
    for key, field in (("first", "first"), ("decode", "decode_long"), ("mem", "decode_mem"))
}

T = {
    "en": {
        "t": "CUDA and Vulkan on the same card",
        "sub": "Spark-X2.6-4B, NVIDIA A40, cortiq bench --core: 60 tokens after the prompt, medians of three runs, exact attention on both paths.",
        "q4tp": "q4tp · 4-bit · 2.15 GB", "q8_2f": "q8_2f · 8-bit · 4.13 GB",
        "prompt": "Prompt, tok/s", "decode": "Decode, tok/s", "mem": "Peak device memory, GB", "ctx": "prompt, tokens",
        "res": "A 4 000-token prompt on q4tp: first token in 0.66 s instead of 1.06, then 111.5 tok/s instead of 80.2",
        "tb_t": "Long prompts: the far field on CUDA",
        "tb_sub": "Spark-X2.6-4B q4tp, NVIDIA A40, cortiq bench --core. First token: medians of three runs. Decode and memory: 500 tokens.",
        "s_cuda": "CUDA, far field", "s_vk": "Vulkan, far field", "s_exact": "CUDA, exact attention",
        "first": "First token, s", "memd": "Device memory while decoding, GB",
        "tb_res": "At 64 000 tokens: first token in {a} s, then {b} tok/s on CUDA; {c} s and {d} tok/s over Vulkan",
        "far": "Past the seal of 8 192 rows: the far field",
        "f1": "The prompt, whole", "f1t": "Read with exact attention on the card, 1024 rows a chunk.",
        "f2": "One seal", "f2t": "248 rows are chosen to stay exact; 32 landmarks a head take the rest.",
        "f3": "State on the card", "f3t": "First rows, exact rows, the last 1800 rows and the far sums: 18 MB a layer.",
        "f4": "A token", "f4t": "Reads the near rows and the far part under one shift. Its cost stops growing.",
        "p_t": "A token and a prompt on the card",
        "p_sub": "CUDA kernels of cortiq’s own for decode, cuBLAS on tensor cores for the prompt. The file is the same on every backend.",
        "dec": "Decode: one token", "pro": "Prompt: 1024 rows a chunk",
        "d1": "Weights in their codec", "d1t": "q4tp and q8_2f stay packed on the card. Each matrix is read once a token.",
        "d2": "One kernel, several matrices", "d2t": "q, k, v and the head gate in one launch. A warp owns a matrix row.",
        "d3": "Attention", "d3t": "Three small kernels over the K/V mirror. A key row is read once for its four query heads.",
        "d4": "Head", "d4t": "Logits for 131 072 tokens, read back once a token.",
        "p1": "Weights to f16 planes", "p1t": "Each weight is expanded once a chunk.",
        "p2": "cuBLAS on tensor cores", "p2t": "f16 operands, f32 accumulation and output.",
        "p3": "Attention", "p3t": "Two batched GEMMs a layer around a causal softmax.",
        "p4": "K/V mirror", "p4t": "The prompt’s rows stay on the card for decode.",
        "p_res": "The Rust side is cudarc: the driver API, NVRTC and cuBLAS. The kernels are CUDA C, compiled at first use for the card.",
    },
    "ru": {
        "t": "CUDA и Vulkan на одной карте",
        "sub": "Spark-X2.6-4B, NVIDIA A40, cortiq bench --core: 60 токенов после промпта, медианы трёх прогонов, точное внимание на обоих путях.",
        "q4tp": "q4tp · 4 бита · 2,15 ГБ", "q8_2f": "q8_2f · 8 бит · 4,13 ГБ",
        "prompt": "Промпт, ток/с", "decode": "Декод, ток/с", "mem": "Пик памяти карты, ГБ", "ctx": "промпт, токенов",
        "res": "Промпт в 4 000 токенов на q4tp: первый токен через 0,66 с вместо 1,06, затем 111,5 ток/с вместо 80,2",
        "tb_t": "Длинные промпты: дальнее поле на CUDA",
        "tb_sub": "Spark-X2.6-4B q4tp, NVIDIA A40, cortiq bench --core. Первый токен: медианы трёх прогонов. Декод и память: 500 токенов.",
        "s_cuda": "CUDA, дальнее поле", "s_vk": "Vulkan, дальнее поле", "s_exact": "CUDA, точное внимание",
        "first": "Первый токен, с", "memd": "Память карты при декоде, ГБ",
        "tb_res": "На 64 000 токенов: первый токен через {a} с, затем {b} ток/с на CUDA; {c} с и {d} ток/с через Vulkan",
        "far": "После печати на 8 192 строках: дальнее поле",
        "f1": "Промпт целиком", "f1t": "Читается с точным вниманием на карте, кусками по 1024 строки.",
        "f2": "Одна печать", "f2t": "248 строк выбираются точными; остальное берут 32 опорные точки на голову.",
        "f3": "Состояние на карте", "f3t": "Первые строки, точные строки, последние 1800 и дальние суммы: 18 МБ на слой.",
        "f4": "Токен", "f4t": "Читает ближние строки и дальнюю часть под одним сдвигом. Его цена больше не растёт.",
        "p_t": "Токен и промпт на карте",
        "p_sub": "Собственные ядра CUDA для декода, cuBLAS на тензорных ядрах для промпта. Файл один и тот же на всех бэкендах.",
        "dec": "Декод: один токен", "pro": "Промпт: куски по 1024 строки",
        "d1": "Веса в своём кодеке", "d1t": "q4tp и q8_2f лежат на карте упакованными. Каждая матрица читается один раз на токен.",
        "d2": "Одно ядро, несколько матриц", "d2t": "q, k, v и гейт голов за один запуск. Варп владеет строкой матрицы.",
        "d3": "Внимание", "d3t": "Три небольших ядра над зеркалом K/V. Строка ключей читается один раз на четыре головы запроса.",
        "d4": "Голова", "d4t": "Логиты для 131 072 токенов, одно чтение на токен.",
        "p1": "Веса в плоскости f16", "p1t": "Каждый вес разворачивается один раз на кусок.",
        "p2": "cuBLAS на тензорных ядрах", "p2t": "Операнды f16, накопление и результат f32.",
        "p3": "Внимание", "p3t": "Два пакетных GEMM на слой вокруг причинного softmax.",
        "p4": "Зеркало K/V", "p4t": "Строки промпта остаются на карте для декода.",
        "p_res": "Со стороны Rust работает cudarc: driver API, NVRTC и cuBLAS. Ядра написаны на CUDA C и компилируются при первом запуске под карту.",
    },
    "zh": {
        "t": "同一块显卡上的 CUDA 与 Vulkan",
        "sub": "Spark-X2.6-4B，NVIDIA A40，cortiq bench --core：提示之后生成 60 个 token，三次运行的中位数，两条路径都是精确注意力。",
        "q4tp": "q4tp · 4 位 · 2.15 GB", "q8_2f": "q8_2f · 8 位 · 4.13 GB",
        "prompt": "提示，tok/s", "decode": "解码，tok/s", "mem": "显存峰值，GB", "ctx": "提示，token 数",
        "res": "q4tp 读 4 000 token 的提示：首个 token 0.66 秒（Vulkan 1.06 秒），随后 111.5 tok/s（Vulkan 80.2）",
        "tb_t": "长提示：CUDA 上的远场",
        "tb_sub": "Spark-X2.6-4B q4tp，NVIDIA A40，cortiq bench --core。首个 token：三次运行的中位数；解码和显存：500 个 token。",
        "s_cuda": "CUDA，远场", "s_vk": "Vulkan，远场", "s_exact": "CUDA，精确注意力",
        "first": "首个 token，秒", "memd": "解码时的显存，GB",
        "tb_res": "64 000 token 时：CUDA 首个 token {a} 秒，随后 {b} tok/s；Vulkan {c} 秒、{d} tok/s",
        "far": "超过 8 192 行的封口之后：远场",
        "f1": "整段提示", "f1t": "在显卡上以精确注意力读取，每块 1024 行。",
        "f2": "一次封口", "f2t": "选出 248 行保持精确；其余交给每个头的 32 个地标。",
        "f3": "显卡上的状态", "f3t": "最前几行、精确行、最后 1800 行和远场累加量：每层 18 MB。",
        "f4": "一个 token", "f4t": "在同一个偏移下读取近处的行和远场部分，代价不再增长。",
        "p_t": "显卡上的一个 token 和一段提示",
        "p_sub": "解码用 cortiq 自己的 CUDA 内核，提示用张量核上的 cuBLAS。所有后端用的是同一个文件。",
        "dec": "解码：一次一个 token", "pro": "提示：每块 1024 行",
        "d1": "权重保持原编码", "d1t": "q4tp 和 q8_2f 以压缩形式留在显卡上，每个 token 把每个矩阵读一次。",
        "d2": "一个内核，多个矩阵", "d2t": "q、k、v 和头门一次启动完成，一个 warp 负责矩阵的一行。",
        "d3": "注意力", "d3t": "K/V 镜像上的三个小内核，一行键为四个查询头只读一次。",
        "d4": "输出头", "d4t": "131 072 个 token 的 logits，每个 token 读回一次。",
        "p1": "权重展开为 f16 平面", "p1t": "每个权重每块展开一次。",
        "p2": "张量核上的 cuBLAS", "p2t": "操作数 f16，累加和输出 f32。",
        "p3": "注意力", "p3t": "每层两次批量 GEMM，中间是因果 softmax。",
        "p4": "K/V 镜像", "p4t": "提示的行留在显卡上，供解码读取。",
        "p_res": "Rust 一侧是 cudarc：driver API、NVRTC 和 cuBLAS。内核用 CUDA C 编写，首次使用时按显卡编译。",
    },
}


def num(lang, v, d):
    s = f"{v:.{d}f}"
    return s.replace(".", ",") if lang == "ru" else s


def wrap(text, lang, n):
    """Lines of at most `n` units. Chinese: a CJK character is one unit and a Latin one about half; a Latin
    word or a number is never split, and punctuation never opens a line."""
    if lang != "zh":
        return cf.wrap(text, lang, n)
    toks = re.findall(r"[A-Za-z0-9_./-]+(?: [0-9]+)*|\s+|.", text)
    width = lambda tok: sum(1.0 if ord(c) > 0x2E7F else 0.55 for c in tok)
    lines, cur, w = [], "", 0.0
    for tok in toks:
        if w + width(tok) > n and cur.strip() and tok not in "，。、：；）" and not tok.isspace():
            lines.append(cur.strip())
            cur, w = "", 0.0
        cur += tok
        w += width(tok)
    return lines + [cur.strip()] if cur.strip() else lines


def bar(s, x, base, w, h, fill, r=4):
    """A bar anchored to the baseline with a rounded data end."""
    h = max(h, 2.0)
    r = min(r, h, w / 2)
    y = base - h
    s.o.append(
        f'<path d="M {x:.1f} {base:.1f} L {x:.1f} {y + r:.1f} Q {x:.1f} {y:.1f} {x + r:.1f} {y:.1f} L {x + w - r:.1f} {y:.1f} '
        f'Q {x + w:.1f} {y:.1f} {x + w:.1f} {y + r:.1f} L {x + w:.1f} {base:.1f} Z" fill="{fill}"/>'
    )


def speed(lang):
    t = T[lang]
    s = Svg(1600, 952)
    head(s, t, t["t"], t["sub"])
    for x, c, name in ((50, BLUE, "CUDA"), (190, ORANGE, "Vulkan")):
        s.rect(x, 138, 22, 22, c, 5)
        s.text(x + 32, 156, name, 22, INK)
    cols = ((50, 540, "prompt", 8000, 2000, 0), (570, 1060, "decode", 125, 25, 1), (1090, 1550, "mem", 8, 2, 1))
    for row, q in enumerate(("q4tp", "q8_2f")):
        top = 200 + row * 325
        s.text(50, top + 20, t[q], 27, INK, "700")
        y1, y0 = top + 92, top + 262
        for x0, x1, key, ymax, step, d in cols:
            s.text(x0, top + 62, t[key], 21, INK2, "600")
            Y = lambda v: y0 + (y1 - y0) * v / ymax
            v = 0
            while v <= ymax + 1e-9:
                s.line(x0 + 56, Y(v), x1, Y(v), LINE, 2 if v == 0 else 1.2)
                s.text(x0 + 46, Y(v) + 7, f"{v:.0f}", 18, INK2, "400", "end")
                v += step
            gw = (x1 - x0 - 56) / 3
            bw = 58
            for i, lb in enumerate(CTX):
                cx = x0 + 56 + gw * (i + 0.5)
                for j, (ser, c) in enumerate((("cuda", BLUE), ("vk", ORANGE))):
                    val = DATA[q][key][ser][i]
                    if key == "mem":
                        val /= 1024
                    x = cx - bw - 2 + j * (bw + 4)
                    bar(s, x, y0, bw, y0 - Y(val), c)
                    dd = 0 if key == "prompt" else d
                    s.text(x + bw / 2, Y(val) - 8, num(lang, val, dd), 18, INK if j == 0 else INK2, "700" if j == 0 else "400", "middle")
                s.text(cx, y0 + 26, lb, 18, INK2, "400", "middle")
        if row == 0:
            s.line(50, top + 305, 1550, top + 305, LINE, 1.5)
    for x0, x1 in ((50, 540), (570, 1060), (1090, 1550)):
        s.text((x0 + 56 + x1) / 2, 200 + 325 + 262 + 54, t["ctx"], 18, INK2, "400", "middle")
    s.rect(50, 868, 1500, 58, TINT, 10)
    s.text(800, 905, t["res"], 23 if lang != "ru" else 22, INK, "700", "middle")
    s.done(f"speed_{lang}")


def turbo(lang):
    """Past the seal: first token, decode and the card's memory while decoding, against the prompt's length."""
    t = T[lang]
    s = Svg(1600, 722)
    head(s, t, t["tb_t"], t["tb_sub"])
    series = (("cuda", BLUE, t["s_cuda"]), ("vk", ORANGE, t["s_vk"]), ("exact", AQUA, t["s_exact"]))
    x = 50
    for _, c, name in series:
        s.path(f"M {x} 150 L {x + 40} 150", c, 3)
        s.dot(x + 20, 150, 8, c)
        s.text(x + 52, 158, name, 22, INK)
        x += 92 + len(name) * (22 if lang == "zh" else 12.5)
    last = len(CTX_T) - 1

    def panel(x0, x1, title, key, ymax, step, d, scale=1.0):
        y0, y1 = 560, 250
        s.text(x0, 218, title, 25, INK, "700")
        Y = lambda v: y0 + (y1 - y0) * v / ymax
        X = lambda i: x0 + 70 + (x1 - x0 - 150) * i / last
        v = 0
        while v <= ymax + 1e-9:
            s.line(x0 + 44, Y(v), x1, Y(v), LINE, 2 if v == 0 else 1.2)
            s.text(x0 + 34, Y(v) + 7, f"{v:.0f}", 19, INK2, "400", "end")
            v += step
        for i, lb in enumerate(CTX_T):
            s.text(X(i), y0 + 30, lb, 19, INK2, "400", "middle")
        s.text((x0 + 44 + x1) / 2, y0 + 62, t["ctx"], 19, INK2, "400", "middle")
        ends = []
        for ser, c, _ in series:
            vals = [a * scale for a in DATA_T[key][ser]]
            pts = [(X(i), Y(a)) for i, a in enumerate(vals)]
            s.path("M " + " L ".join(f"{p[0]:.1f} {p[1]:.1f}" for p in pts), c, 3)
            for p in pts:
                s.dot(p[0], p[1], 7, c)
            ends.append([pts[-1][1], vals[-1], ser == "cuda"])
        # end labels in ink beside the last points, pushed apart where two lines end together
        ends.sort(key=lambda e: e[0])
        for i in range(1, len(ends)):
            if ends[i][0] - ends[i - 1][0] < 26:
                ends[i][0] = ends[i - 1][0] + 26
        for y, val, bold in ends:
            s.text(X(last) + 16, y + 8, num(lang, val, d), 21, INK, "700" if bold else "400")

    top = lambda key, scale=1.0: max(max(v) for v in DATA_T[key].values()) * scale
    f_max = 20 * (int(top("first") / 20) + 1)
    panel(50, 540, t["first"], "first", f_max, 20, 1)
    panel(570, 1060, t["decode"], "decode", 120, 20, 1)
    panel(1090, 1550, t["memd"], "mem", 2 * (int(top("mem", 1 / 1024) / 2) + 1), 2, 1, 1 / 1024)
    s.rect(50, 642, 1500, 58, TINT, 10)
    res = t["tb_res"].format(
        a=num(lang, DATA_T["first"]["cuda"][last], 1), b=num(lang, DATA_T["decode"]["cuda"][last], 1),
        c=num(lang, DATA_T["first"]["vk"][last], 1), d=num(lang, DATA_T["decode"]["vk"][last], 1),
    )
    s.text(800, 679, res, 23 if lang != "ru" else 22, INK, "700", "middle")
    s.done(f"turbo_{lang}")


def paths(lang):
    t = T[lang]
    n = 30 if lang != "zh" else 15.5
    lanes = (("dec", "d", BLUE, TINT), ("pro", "p", ORANGE, TINT2), ("far", "f", AQUA, "#e6f6ef"))
    text = {pre: [wrap(t[f"{pre}{i + 1}t"], lang, n) for i in range(4)] for _, pre, _, _ in lanes}
    hs = {pre: 84 + 28 * max(len(x) for x in text[pre]) for pre in text}
    s = Svg(1600, 150 + sum(hs.values()) + len(lanes) * 86 + 58 + 24)
    head(s, t, t["p_t"], t["p_sub"])
    ww, gap = 350, 33
    y = 150
    for key, pre, c, tint in lanes:
        hh = hs[pre]
        s.dot(64, y + 14, 11, c)
        s.text(88, y + 23, t[key], 27, INK, "700")
        yb = y + 46
        for i in range(4):
            x = 50 + i * (ww + gap)
            s.rect(x, yb, ww, hh, tint, 12)
            title = t[f"{pre}{i + 1}"]
            s.text(x + 20, yb + 42, title, 24 if len(title) < 23 else 21, INK, "700")
            for j, ln in enumerate(text[pre][i]):
                s.text(x + 20, yb + 80 + j * 28, ln, 20, INK)
            if i < 3:
                s.line(x + ww + 5, yb + hh / 2, x + ww + gap - 7, yb + hh / 2, INK2, 2.5, None, True)
        y = yb + hh + 40
    s.rect(50, y - 4, 1500, 58, "#f4f3ef", 10)
    s.text(800, y + 33, t["p_res"], 21 if lang == "en" else 20, INK, "600", "middle")
    s.done(f"paths_{lang}")


for lang in T:
    turbo(lang)
    speed(lang)
    paths(lang)
print("figures:", sorted(os.listdir(cf.out)))
