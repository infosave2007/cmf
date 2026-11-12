#!/usr/bin/env python3
"""Figures of the Qwen3.8 Fast Code page: SVG in three languages, PNG through rsvg-convert.

    python3 docs/hf/figures/fastcode/make.py DATA.json OUT_DIR

DATA.json holds the measured rows (see `figures_data.json` next to this file); writes
fc_speed_{en,ru,zh}, fc_quality_{en,ru,zh} and fc_size_{en,ru,zh} as .svg and .png into OUT_DIR.
"""
import html, json, os, subprocess, sys

FONT = "Helvetica Neue, Helvetica, Arial, PingFang SC, Hiragino Sans GB, Hiragino Sans, sans-serif"
INK, INK2, MUTED, GRID = "#111827", "#374151", "#6b7280", "#e5e7eb"
# one colour per system, the same on every chart
COL = {"fastcode": "#1d4ed8", "strata": "#f59e0b", "strata_cmf": "#fbbf24", "q4tp": "#9ca3af", "hot": "#6b7280", "q2tp": "#d1d5db"}

T = {
    "en": dict(
        speed="Speed on one RTX PRO 6000 (96 GB)", speed_sub="same pod, same client, greedy, thinking off",
        decode="decode, tok/s", prefill="prompt reading, tok/s", ctx="prompt",
        quality="Accuracy on code", quality_sub="HumanEval pass@1 (164 tasks, greedy, no thinking) and code perplexity",
        he="HumanEval pass@1, %", ppl="code perplexity, lower is better",
        size="Size and memory", size_sub="file on disk and peak VRAM while serving",
        file="file, GB", vram="peak VRAM, GB", better="better",
        cards="Cards from 12 GB (CUDA)", cards_sub="one RTX 5090 held to the memory shown; the experts that do not fit run on the host's cores",
        cards_y="the aquarium page written to the end, tok/s", cards_x="card memory",
        cards_note=["the answer is the complete working page in every bar", "within 12 GB: 1 357 of 12 288 experts on the card,", "38 of the first 40 HumanEval tasks pass"],
    ),
    "ru": dict(
        speed="Скорость на одной RTX PRO 6000 (96 ГБ)", speed_sub="один под, один клиент, жадный выбор, без размышлений",
        decode="генерация, ток/с", prefill="чтение промпта, ток/с", ctx="промпт",
        quality="Точность на коде", quality_sub="HumanEval pass@1 (164 задачи, жадно, без размышлений) и перплексия на коде",
        he="HumanEval pass@1, %", ppl="перплексия на коде, меньше — лучше",
        size="Размер и память", size_sub="файл на диске и пик VRAM при работе сервера",
        file="файл, ГБ", vram="пик VRAM, ГБ", better="лучше",
        cards="Карты от 12 ГБ (CUDA)", cards_sub="одна RTX 5090, ограниченная указанной памятью; не поместившихся экспертов считают ядра процессора",
        cards_y="страница аквариума, написанная до конца, ток/с", cards_x="память карты",
        cards_note=["ответ в каждом столбце — полная рабочая страница", "в 12 ГБ: на карте 1 357 экспертов из 12 288,", "проходят 38 из первых 40 задач HumanEval"],
    ),
    "zh": dict(
        speed="单张 RTX PRO 6000（96 GB）上的速度", speed_sub="同一台机器、同一客户端、贪心解码、关闭思考",
        decode="生成，tok/s", prefill="读取提示词，tok/s", ctx="提示词",
        quality="代码准确率", quality_sub="HumanEval pass@1（164 题，贪心，不思考）与代码困惑度",
        he="HumanEval pass@1，%", ppl="代码困惑度，越低越好",
        size="大小与显存", size_sub="磁盘文件与服务时的显存峰值",
        file="文件，GB", vram="显存峰值，GB", better="更好",
        cards="12 GB 起的显卡（CUDA）", cards_sub="一张 RTX 5090，限制为图中显存；放不下的专家由主机 CPU 核心计算",
        cards_y="完整写出水族箱页面，tok/s", cards_x="显存",
        cards_note=["每根柱子的回答都是完整可运行的页面", "12 GB 内：显卡保留 12 288 个专家中的 1 357 个，", "HumanEval 前 40 题通过 38 题"],
    ),
}


def txt(x, y, s, size=13, fill=INK, anchor="start", weight="normal"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" font-weight="{weight}">{html.escape(str(s))}</text>')


def fmt(v):
    if v >= 1000:
        return f"{v:,.0f}".replace(",", " ")
    if v >= 100:
        return f"{v:.0f}"
    return f"{v:.1f}" if v >= 10 else f"{v:.2f}"


def svg(w, h, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
            f'<rect width="{w}" height="{h}" fill="#ffffff"/>{body}</svg>')


def grouped(x0, y0, w, h, groups, series, title, vmax=None, lower_better=False):
    """Horizontal groups (e.g. prompt lengths) of vertical bars (one per system)."""
    out = [txt(x0, y0 - 10, title, 13, INK2, weight="bold")]
    vals = [v for g in groups for v in g["vals"].values() if v is not None]
    vmax = vmax or max(vals) * 1.15
    out.append(f'<line x1="{x0}" y1="{y0 + h}" x2="{x0 + w}" y2="{y0 + h}" stroke="{GRID}"/>')
    gw = w / len(groups)
    bw = min(34, (gw - 18) / max(1, len(series)))
    for gi, g in enumerate(groups):
        gx = x0 + gi * gw + (gw - bw * len(series)) / 2
        for si, (key, _label) in enumerate(series):
            v = g["vals"].get(key)
            if v is None:
                continue
            bh = h * v / vmax
            bx = gx + si * bw
            out.append(f'<rect x="{bx + 2:.1f}" y="{y0 + h - bh:.1f}" width="{bw - 4:.1f}" height="{bh:.1f}" rx="3" fill="{COL[key]}"/>')
            out.append(txt(bx + bw / 2, y0 + h - bh - 5, fmt(v), 11, INK, "middle", "bold" if key == "fastcode" else "normal"))
        out.append(txt(x0 + gi * gw + gw / 2, y0 + h + 18, g["label"], 12, MUTED, "middle"))
    return "".join(out)


def legend(x, y, series):
    out, cx = [], x
    for key, label in series:
        out.append(f'<rect x="{cx}" y="{y - 10}" width="12" height="12" rx="2" fill="{COL[key]}"/>')
        out.append(txt(cx + 18, y, label, 12, INK2))
        cx += 26 + 7.2 * len(label)
    return "".join(out)


def hbars(x0, y0, w, rows, title, lower_better=False, unit=""):
    """One horizontal bar per system."""
    out = [txt(x0, y0 - 10, title, 13, INK2, weight="bold")]
    vmax = max(r[2] for r in rows) * 1.12
    for i, (key, label, v) in enumerate(rows):
        y = y0 + i * 30
        bw = w * v / vmax
        out.append(txt(x0 - 10, y + 15, label, 12, INK2, "end", "bold" if key == "fastcode" else "normal"))
        out.append(f'<rect x="{x0}" y="{y + 3}" width="{bw:.1f}" height="18" rx="3" fill="{COL[key]}"/>')
        out.append(txt(x0 + bw + 6, y + 16, fmt(v) + unit, 12, INK, weight="bold" if key == "fastcode" else "normal"))
    return "".join(out)


def main():
    data = json.load(open(sys.argv[1]))
    out_dir = sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    for lang, t in T.items():
        names = {k: v[lang] for k, v in data["names"].items()}
        series = [(k, names[k]) for k in data["speed_series"]]
        # speed: decode and prefill by prompt length
        groups_d = [{"label": f'{c["label"]}', "vals": c["decode"]} for c in data["speed"]]
        groups_p = [{"label": f'{c["label"]}', "vals": c["prefill"]} for c in data["speed"]]
        W = 980
        body = txt(30, 36, t["speed"], 20, INK, weight="bold") + txt(30, 58, t["speed_sub"], 13, MUTED)
        body += legend(30, 86, series)
        body += grouped(30, 130, 440, 230, groups_d, series, t["decode"])
        body += grouped(520, 130, 440, 230, groups_p, series, t["prefill"])
        body += txt(30 + 220, 400, t["ctx"], 12, MUTED, "middle") + txt(520 + 220, 400, t["ctx"], 12, MUTED, "middle")
        H = 420
        if data.get("cards"):
            # second row: the same model on smaller cards (CUDA, host cores for the experts that do not fit)
            y2 = 462
            body += f'<line x1="30" y1="{y2 - 32}" x2="{W - 30}" y2="{y2 - 32}" stroke="{GRID}"/>'
            body += txt(30, y2, t["cards"], 20, INK, weight="bold") + txt(30, y2 + 22, t["cards_sub"], 13, MUTED)
            groups_c = [{"label": c["label"], "vals": {"fastcode": c["decode"]}} for c in data["cards"]]
            body += grouped(30, y2 + 70, 440, 200, groups_c, [("fastcode", names["fastcode"])], t["cards_y"])
            body += txt(30 + 220, y2 + 70 + 200 + 40, t["cards_x"], 12, MUTED, "middle")
            for i, line in enumerate(t["cards_note"]):
                body += txt(520, y2 + 110 + 24 * i, line, 14, INK2)
            H = y2 + 70 + 200 + 58
        write(out_dir, f"fc_speed_{lang}", svg(W, H, body))
        # quality: HumanEval and code ppl
        he = [(k, names[k], v) for k, v in data["humaneval"]]
        pp = [(k, names[k], v) for k, v in data["ppl"]]
        body = txt(30, 36, t["quality"], 20, INK, weight="bold") + txt(30, 58, t["quality_sub"], 13, MUTED)
        body += hbars(220, 100, 230, he, t["he"], unit=" %")
        body += hbars(700, 100, 230, pp, t["ppl"], lower_better=True)
        h = 120 + 30 * max(len(he), len(pp))
        write(out_dir, f"fc_quality_{lang}", svg(W, h, body))
        # size and memory
        fs = [(k, names[k], v) for k, v in data["file_gb"]]
        vr = [(k, names[k], v) for k, v in data["vram_gb"]]
        body = txt(30, 36, t["size"], 20, INK, weight="bold") + txt(30, 58, t["size_sub"], 13, MUTED)
        body += hbars(220, 100, 230, fs, t["file"])
        body += hbars(700, 100, 230, vr, t["vram"])
        h = 120 + 30 * max(len(fs), len(vr))
        write(out_dir, f"fc_size_{lang}", svg(W, h, body))


def write(out_dir, name, s):
    p = os.path.join(out_dir, name + ".svg")
    open(p, "w").write(s)
    subprocess.run(["rsvg-convert", "--zoom", "2", "-o", os.path.join(out_dir, name + ".png"), p], check=True)


if __name__ == "__main__":
    main()
