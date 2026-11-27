#!/usr/bin/env python3
"""The figures of the GigaAM-v4 page, in English, Russian and Chinese, from the measured numbers.

    card_figures.py DATA.json OUT_DIR   (rsvg-convert on PATH)

DATA.json is what bench_rows.py writes: {stand: {file tag: [rows]}} with a row {"secs": seconds of audio,
"time": seconds to the transcript (warm), "x": seconds of audio a second, "vram_gb" | "rss_gb": peak memory}.
One figure a stand, speed_<stand>_<lang>.png: the time to the transcript, the speed against real time and the
peak memory against the recording's length. Each line is named at its end, so the figures have no legend.
"""
import json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "dory"))
import card_figures as cf  # scripts/dory: the SVG writer every card figure shares
from card_figures import Svg, INK, INK2, LINE, BLUE, ORANGE, AQUA, GREY, TINT, head

L4 = {"en": "NVIDIA L4 (24 GB), Google Colab, 12 vCPU", "ru": "NVIDIA L4 (24 ГБ), Google Colab, 12 vCPU", "zh": "NVIDIA L4（24 GB），Google Colab，12 vCPU"}
STAND = {
    "cuda": L4,
    "vulkan": L4,
    "cpu": {"en": "Intel Xeon 2.2 GHz, 12 vCPU (the L4 instance's host), no GPU", "ru": "Intel Xeon 2,2 ГГц, 12 vCPU (хост инстанса с L4), без GPU", "zh": "Intel Xeon 2.2 GHz，12 vCPU（L4 实例的主机），不使用 GPU"},
    "metal": {"en": "MacBook Air M4 (24 GB), fanless", "ru": "MacBook Air M4 (24 ГБ), без вентилятора", "zh": "MacBook Air M4（24 GB），无风扇"},
    "cpu_m4": {"en": "MacBook Air M4 (24 GB), fanless, CPU only", "ru": "MacBook Air M4 (24 ГБ), без вентилятора, только CPU", "zh": "MacBook Air M4（24 GB），无风扇，仅 CPU"},
}
BACKEND = {"cuda": "CUDA", "vulkan": "Vulkan", "metal": "Metal", "cpu": "CPU", "cpu_m4": "CPU"}
# file tag -> (label, colour, stroke); the headline file last so it is drawn on top
FILES = [
    ("ctc-q4tp", "CTC q4tp", GREY, 3),
    ("rnnt-q4tp", "RNN-T q4tp", AQUA, 3),
    ("ctc-q8_2f", "CTC q8_2f", ORANGE, 3),
    ("rnnt-q8_2f", "RNN-T q8_2f", BLUE, 5),
]
T = {
    "en": {
        "title": "Speed against the recording's length",
        "sub": "GigaAM-v4, cortiq transcribe, second pass in one process; recordings over 30 s are cut by voice activity.",
        "time": "Time to the transcript, s", "x": "Seconds of audio a second", "vram": "Peak card memory, GB", "rss": "Peak process memory, GB",
        "len": "recording", "stand": "Stand", "s": "s", "min": "min", "h": "h",
    },
    "ru": {
        "title": "Скорость в зависимости от длины записи",
        "sub": "GigaAM-v4, cortiq transcribe, второй проход в одном процессе; записи длиннее 30 с режутся по голосовой активности.",
        "time": "Время до транскрипта, с", "x": "Секунд аудио в секунду", "vram": "Пик памяти карты, ГБ", "rss": "Пик памяти процесса, ГБ",
        "len": "запись", "stand": "Стенд", "s": "с", "min": "мин", "h": "ч",
    },
    "zh": {
        "title": "速度随录音长度的变化",
        "sub": "GigaAM-v4，cortiq transcribe，同一进程内的第二遍；超过 30 秒的录音按语音活动切分。",
        "time": "出稿时间，秒", "x": "每秒处理的音频秒数", "vram": "显存峰值，GB", "rss": "进程内存峰值，GB",
        "len": "录音", "stand": "测试平台", "s": "秒", "min": "分", "h": "时",
    },
}


def num(lang, v):
    d = 0 if v >= 100 else 1 if v >= 10 else 2
    s = f"{v:.{d}f}"
    return s.replace(".", ",") if lang == "ru" else s


def length(lang, secs):
    t = T[lang]
    if secs < 60:
        return f"{secs} {t['s']}"
    if secs < 3600:
        return f"{secs // 60} {t['min']}"
    return f"{secs // 3600} {t['h']}"


def pill(s, x, y, text):
    w = 18 + 13 * len(text)
    s.rect(x, y - 28, w, 38, TINT, 19, BLUE)
    s.text(x + w / 2, y - 2, text, 21, BLUE, "700", "middle")
    return w


def fig(lang, stand, d):
    t = T[lang]
    series = [(d[tag], col, sw, label) for tag, label, col, sw in FILES if d.get(tag)]
    if not series:
        return
    mem_key = "vram_gb" if any(r.get("vram_gb") for rows, *_ in series for r in rows) else "rss_gb"
    panels = [("time", "time", True), ("x", "x", False), ("vram" if mem_key == "vram_gb" else "rss", mem_key, False)]
    s = Svg(1600, 700)
    head(s, t, t["title"], t["sub"])
    w = pill(s, 50, 158, BACKEND[stand])
    s.text(50 + w + 16, 156, f"{t['stand']}: {STAND[stand][lang]}", 22, INK, "700")
    lens = sorted({r["secs"] for rows, *_ in series for r in rows})
    gap = 40
    pw = (1500 - gap * (len(panels) - 1)) / len(panels)
    for pi, (tkey, key, log) in enumerate(panels):
        x0 = 50 + pi * (pw + gap)
        x1 = x0 + pw
        y0, y1 = 580, 250
        s.text(x0, 222, t[tkey], 25, INK, "700")
        vals = [r[key] for rows, *_ in series for r in rows if r.get(key)]
        xr = x1 - 150
        # the lengths are a decade apart at most: equal steps on the axis, named by length
        X = lambda c: x0 + 55 + (xr - x0 - 55) * lens.index(c) / max(len(lens) - 1, 1)
        if log:
            lo, hi = math.floor(math.log10(min(vals))), math.ceil(math.log10(max(vals) * 1.05))
            Y = lambda v: y0 + (y1 - y0) * (math.log10(v) - lo) / (hi - lo)
            for e in range(lo, hi + 1):
                s.line(x0 + 30, Y(10**e), xr + 12, Y(10**e), LINE, 1.5)
                s.text(x0 + 22, Y(10**e) + 7, f"{10**e:g}".replace(".", "," if lang == "ru" else "."), 18, INK2, "400", "end")
        else:
            top = max(vals) * 1.2
            base = 10 ** math.floor(math.log10(top / 4))
            step = next(base * m for m in (1, 2, 5, 10) if top / (base * m) <= 8)
            Y = lambda v: y0 + (y1 - y0) * v / top
            v = 0
            while v <= top:
                s.line(x0 + 30, Y(v), xr + 12, Y(v), LINE, 1.5)
                s.text(x0 + 22, Y(v) + 7, f"{v:g}".replace(".", "," if lang == "ru" else "."), 18, INK2, "400", "end")
                v += step
        for c in lens:
            s.text(X(c), y0 + 30, length(lang, c), 17, INK2, "400", "middle")
        s.text((x0 + 55 + xr) / 2, y0 + 60, t["len"], 18, INK2, "400", "middle")
        ends = []
        # a line that coincides with the headline one (the card's memory, say) is drawn over it, dashed with
        # small dots, so both show
        lead = {r["secs"]: r[key] for r in series[-1][0] if r.get(key)}

        def same(rows):
            mine = {r["secs"]: r[key] for r in rows if r.get(key)}
            shared = set(mine) & set(lead)
            return bool(shared) and all(abs(mine[c] - lead[c]) <= 0.04 * max(mine[c], lead[c]) for c in shared)

        order = [series[-1]] + [x for x in series[:-1] if same(x[0])]
        order = [x for x in series[:-1] if not same(x[0])] + order
        for rows, col, sw, label in order:
            dashed = rows is not series[-1][0] and same(rows)
            pts = [(X(r["secs"]), Y(r[key]), r[key], r["secs"]) for r in rows if r.get(key)]
            if not pts:
                continue
            s.path("M " + " L ".join(f"{p[0]:.1f} {p[1]:.1f}" for p in pts), INK2 if dashed else col, 3 if dashed else sw, dash="9 9" if dashed else None)
            for p in pts:
                s.dot(p[0], p[1], 4 if dashed else 7 if sw > 3 else 6, INK2 if dashed else col)
            ends.append([pts[-1], col, label])
        # a line is named to the right of its last point; names closer than a line of text move apart
        ends.sort(key=lambda e: e[0][1])
        ys = [e[0][1] + 6 for e in ends]
        for _ in range(8):
            for i in range(1, len(ys)):
                if ys[i] - ys[i - 1] < 24:
                    mid = (ys[i] + ys[i - 1]) / 2
                    ys[i - 1], ys[i] = mid - 12, mid + 12
        for (last, col, label), y in zip(ends, ys):
            s.text(last[0] + 14, y, f"{label} {num(lang, last[2])}", 18, INK if col == BLUE else INK2, "700", "start")
    s.done(f"speed_{stand}_{lang}")


if __name__ == "__main__":
    data = json.load(open(sys.argv[1]))
    cf.out = sys.argv[2]
    os.makedirs(cf.out, exist_ok=True)
    for lang in T:
        for stand, d in data.items():
            if stand in STAND:
                fig(lang, stand, d)
    print("ok")
