#!/usr/bin/env python3
"""Figures of the Qwen3.8 Fast Code page, SVG in three languages, PNG through rsvg-convert.

    python3 docs/hf/figures/fastcode/make.py DATA.json OUT_DIR

DATA.json holds the measured rows (see `figures_data.json` next to this file); writes
fc_cuda_{en,ru,zh}, fc_vulkan_{en,ru,zh} and fc_quality_{en,ru,zh} as .svg and .png into OUT_DIR.

One figure is one stand and one cortiq backend, named under its title: numbers of two stands never share
a figure. Fast Code is the one series in colour and the thicker bar; what it is compared with is gray, and
every bar carries its value.
"""
import html, json, os, subprocess, sys

W, PAD = 880, 36
FONT = "Helvetica Neue, Helvetica, Arial, PingFang SC, Hiragino Sans GB, Hiragino Sans, sans-serif"
INK, INK2, MUTED = "#0f172a", "#334155", "#64748b"
GRID, AXIS, TILE, BORDER = "#eef1f5", "#cbd5e1", "#f4f6f9", "#e2e8f0"
# one colour per system, the same on every chart
COL = {"fastcode": "#2563eb", "strata": "#6f7a8c", "q4tp": "#a9b2bf"}
# value labels sit on a white halo, so a grid line never runs through a number
HALO = ' paint-order="stroke" stroke="#ffffff" stroke-width="3" stroke-linejoin="round"'

T = {
    "en": dict(
        engine="cortiq", model="Qwen3.8-Flash-Next Fast Code", stand="Stand:",
        gen="Code generation, tok/s", read="Prompt reading, tok/s", tok="tok/s", gb="GB",
        cuda=dict(
            badge="CUDA", title="Code generation on NVIDIA cards with 12–24 GB",
            stand="one RTX 5090, card memory held to 12 / 16 / 24 GB · cortiq 0.8.78, CUDA backend",
            legend=dict(fastcode="Fast Code on cortiq — 4-bit experts", strata="Strata Coder on Strata — 2-bit experts"),
            tiles=["code generation within {tiers} GB", "of the speed of the 2-bit Strata Coder",
                   ("4-bit", "", "expert weights · Strata Coder: 2-bit")],
            cap="by card memory", group="{} GB", ratio="of Strata Coder's speed",
            note=["Task: a program from a {ptok}-token spec, answered to the end · greedy, thinking off · mean of two runs.",
                  "Strata Coder: a 2-bit cut of the same model on its own engine, on the same card; expert cache {cache}."],
        ),
        vulkan=dict(
            badge="VULKAN", title="Fast Code on one card with 48 GB or more",
            stand="one RTX PRO 6000 96 GB · cortiq, Vulkan backend",
            legend=dict(fastcode="Fast Code — 256 experts a layer", q4tp="Full file — 512 experts a layer"),
            tiles=["prompt reading · {x}× the full file", "code generation after a 4K–32K prompt",
                   "peak card memory · full file {full} GB"],
            cap="by prompt length, tokens", group="{}",
            note=["One stream, greedy, thinking off, draft tokens on; a new session tag before every prompt, so nothing comes from a cache.",
                  "The full file is the same model with all 512 experts of a layer, 4-bit, on the same engine and the same card."],
        ),
        quality=dict(
            badge="ACCURACY", title="Accuracy on code",
            sub="Fast Code keeps the experts at 4 bits and the attention at 8 bits; Strata Coder is a 2-bit cut of the same model",
            rows=dict(fastcode="Fast Code", strata="Strata Coder", q4tp="Full file · 512 experts"),
            cols=[("Expert weights", "bits per weight"), ("Code perplexity", "fresh code · lower is better"),
                  ("HumanEval pass@1", "164 tasks, thinking off")],
            bits="{}-bit",
            note=["¹ Strata Coder's weights imported into CMF and run by cortiq: the same text and tokenizer for all three.",
                  "Perplexity on 8 192 tokens of code written after 20.10.2026."],
        ),
    ),
    "ru": dict(
        engine="cortiq", model="Qwen3.8-Flash-Next Fast Code", stand="Стенд:",
        gen="Генерация кода, ток/с", read="Чтение промпта, ток/с", tok="ток/с", gb="ГБ",
        cuda=dict(
            badge="CUDA", title="Генерация кода на картах NVIDIA 12–24 ГБ",
            stand="одна RTX 5090, память карты ограничена до 12 / 16 / 24 ГБ · cortiq 0.8.78, бэкенд CUDA",
            legend=dict(fastcode="Fast Code на cortiq — эксперты 4 бита", strata="Strata Coder на Strata — эксперты 2 бита"),
            tiles=["генерация кода в {tiers} ГБ", "скорости 2-битной Strata Coder",
                   ("4 бита", "", "веса экспертов · у Strata Coder 2 бита")],
            cap="по памяти карты", group="{} ГБ", ratio="от скорости Strata Coder",
            note=["Задача: программа по ТЗ на {ptok} токен, ответ до конца · жадный выбор, без рассуждений · среднее двух запусков.",
                  "Strata Coder — 2-битная сборка той же модели на своём движке, на той же карте; кеш экспертов {cache}."],
        ),
        vulkan=dict(
            badge="VULKAN", title="Fast Code на одной карте от 48 ГБ",
            stand="одна RTX PRO 6000 96 ГБ · cortiq, бэкенд Vulkan",
            legend=dict(fastcode="Fast Code — 256 экспертов слоя", q4tp="Полный файл — 512 экспертов слоя"),
            tiles=["чтение промпта · {x}× полного файла", "генерация кода после промпта 4K–32K",
                   "пик памяти · полный файл {full} ГБ"],
            cap="по длине промпта, токены", group="{}",
            note=["Один поток, жадный выбор, без рассуждений, черновые токены включены; перед каждым промптом новая метка сессии — кеш не участвует.",
                  "Полный файл — та же модель со всеми 512 экспертами слоя, 4 бита; тот же движок и та же карта."],
        ),
        quality=dict(
            badge="ТОЧНОСТЬ", title="Точность на коде",
            sub="Fast Code сохраняет экспертов в 4 битах и внимание в 8 битах; Strata Coder — 2-битная сборка той же модели",
            rows=dict(fastcode="Fast Code", strata="Strata Coder", q4tp="Полный файл · 512 экспертов"),
            cols=[("Веса экспертов", "бит на вес"), ("Перплексия на коде", "свежий код · меньше — лучше"),
                  ("HumanEval pass@1", "164 задачи, без рассуждений")],
            bits="{} бита",
            note=["¹ Веса Strata Coder, импортированные в CMF и запущенные cortiq: один текст и один токенизатор для всех трёх.",
                  "Перплексия на 8 192 токенах кода, написанного после 20.10.2026."],
        ),
    ),
    "zh": dict(
        engine="cortiq", model="Qwen3.8-Flash-Next Fast Code", stand="测试平台：",
        gen="生成代码，token/s", read="读取提示词，token/s", tok="token/s", gb="GB",
        cuda=dict(
            badge="CUDA", title="在 12–24 GB 的 NVIDIA 显卡上生成代码",
            stand="一张 RTX 5090，显存限制为 12 / 16 / 24 GB · cortiq 0.8.78，CUDA 后端",
            legend=dict(fastcode="Fast Code（cortiq）— 专家 4 位", strata="Strata Coder（Strata）— 专家 2 位"),
            tiles=["在 {tiers} GB 显存内生成代码", "相当于 2 位 Strata Coder 的速度",
                   ("4 位", "", "专家权重 · Strata Coder 为 2 位")],
            cap="按显存", group="{} GB", ratio="相当于 Strata Coder 的速度",
            note=["任务：按 {ptok} token 的需求写出完整程序，直到回答结束 · 贪心解码，关闭思考 · 两次运行的平均值。",
                  "Strata Coder：同一模型的 2 位裁剪版，运行于其自有引擎，同一张显卡；专家缓存 {cache}。"],
        ),
        vulkan=dict(
            badge="VULKAN", title="Fast Code：一张 48 GB 以上的显卡",
            stand="一张 RTX PRO 6000 96 GB · cortiq，Vulkan 后端",
            legend=dict(fastcode="Fast Code — 每层 256 个专家", q4tp="完整文件 — 每层 512 个专家"),
            tiles=["读取提示词 · 完整文件的 {x} 倍", "4K–32K 提示词后生成代码", "显存峰值 · 完整文件：{full} GB"],
            cap="按提示词长度（token）", group="{}",
            note=["单路，贪心解码，关闭思考，开启草稿 token；每个提示词前加新的会话标记，不使用缓存。",
                  "完整文件是同一模型，保留每层全部 512 个专家，4 位；同一引擎，同一张显卡。"],
        ),
        quality=dict(
            badge="准确率", title="代码准确率",
            sub="Fast Code 的专家保持 4 位，注意力保持 8 位；Strata Coder 是同一模型的 2 位裁剪版",
            rows=dict(fastcode="Fast Code", strata="Strata Coder", q4tp="完整文件 · 512 专家"),
            cols=[("专家权重", "每个权重的位数"), ("代码困惑度", "新代码 · 越低越好"), ("HumanEval pass@1", "164 题，关闭思考")],
            bits="{} 位",
            note=["¹ Strata Coder 的权重导入 CMF 后由 cortiq 运行：三者使用相同的文本与分词器。",
                  "困惑度取自 2026-10-20 之后写的 8 192 token 代码。"],
        ),
    ),
}


def tw(s, size):
    """Rough text width: enough to lay out pills and legends."""
    w = 0.0
    for ch in str(s):
        if ord(ch) >= 0x2E80:  # CJK and full-width forms
            w += 1.0
        elif ch == " ":
            w += 0.28
        elif ch in "ilI.,:;·'|!":
            w += 0.26
        elif ch.isdigit():
            w += 0.556
        elif ch.isupper():
            w += 0.68
        elif ch in "mwшщжыюм":
            w += 0.78
        else:
            w += 0.54
    return w * size


def txt(x, y, s, size=13, fill=INK, anchor="start", weight="normal", extra=""):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" font-weight="{weight}"{extra}>{html.escape(str(s))}</text>')


def spans(x, y, parts):
    """One line of text in several styles: parts = [(string, size, fill, weight)]."""
    out = f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}">'
    for i, (s, size, fill, weight) in enumerate(parts):
        out += (f'<tspan{" dx=" + chr(34) + "6" + chr(34) if i else ""} font-size="{size}" fill="{fill}" '
                f'font-weight="{weight}">{html.escape(str(s))}</tspan>')
    return out + "</text>"


def fmt(v):
    if v >= 1000:
        return f"{v:,.0f}".replace(",", " ")
    if v >= 100:
        return f"{v:.0f}"
    return f"{v:.1f}" if v >= 10 else f"{v:.3f}"


def svg(h, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{h}" viewBox="0 0 {W} {h}">'
            f'<rect x="0.5" y="0.5" width="{W - 1}" height="{h - 1}" rx="16" fill="#ffffff" stroke="{BORDER}"/>{body}</svg>')


def header(t, f):
    """Backend pill with the engine and the model next to it, the title, then the stand the figure was measured on."""
    bw = tw(f["badge"], 11) + 1.2 * len(f["badge"]) + 22
    out = f'<rect x="{PAD}" y="26" width="{bw:.1f}" height="22" rx="11" fill="{INK}"/>'
    out += txt(PAD + bw / 2 + 0.6, 41, f["badge"], 11, "#ffffff", "middle", "bold", ' letter-spacing="1.2"')
    out += spans(PAD + bw + 10, 41.5, [(t["engine"], 13, INK, "bold"), ("·  " + t["model"], 13, MUTED, "normal")])
    out += txt(PAD, 82, f["title"], 23, INK, weight="bold")
    if "stand" in f:
        out += spans(PAD, 105, [(t["stand"], 13, INK, "bold"), (f["stand"], 13, INK2, "normal")])
    else:
        out += txt(PAD, 105, f["sub"], 13, MUTED)
    return out


def tiles(y, items):
    """A row of stat tiles: (number, unit, caption)."""
    gap, h = 12, 64
    w = (W - 2 * PAD - gap * (len(items) - 1)) / len(items)
    out = ""
    for i, (hero, unit, cap) in enumerate(items):
        x = PAD + i * (w + gap)
        out += f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="{h}" rx="10" fill="{TILE}"/>'
        out += f'<rect x="{x:.1f}" y="{y + 12}" width="3" height="{h - 24}" rx="1.5" fill="{COL["fastcode"]}"/>'
        if tw(cap, 12) > w - 30:
            print("tile caption may overflow:", cap, file=sys.stderr)
        out += spans(x + 18, y + 29, [(hero, 20, INK, "bold"), (unit, 12.5, MUTED, "normal")])
        out += txt(x + 18, y + 49, cap, 12, INK2)
    return out


def legend(x, y, items):
    """Swatch and name per system; the first one is the system the page is about."""
    out, cx = "", x
    for i, (key, label) in enumerate(items):
        out += f'<rect x="{cx:.1f}" y="{y - 10.5}" width="12" height="12" rx="3" fill="{COL[key]}"/>'
        out += txt(cx + 19, y, label, 13, INK if i == 0 else INK2, weight="bold" if i == 0 else "normal")
        cx += 19 + tw(label, 13) * (1.06 if i == 0 else 1.0) + 28
    return out


def hbar(x, y, w, h, color, r=4):
    """A horizontal bar: rounded at the value end, square on the baseline."""
    r = min(r, w, h / 2)
    return (f'<path d="M{x:.1f},{y:.1f} H{x + w - r:.1f} Q{x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f} V{y + h - r:.1f} '
            f'Q{x + w:.1f},{y + h:.1f} {x + w - r:.1f},{y + h:.1f} H{x:.1f} Z" fill="{color}"/>')


def ticks(vmax):
    """Round steps, about five of them, none above the largest value: the top label keeps a clear field."""
    raw = vmax / 5
    mag = 10 ** (len(str(int(raw))) - 1)
    step = next(s * mag for s in (1, 2, 2.5, 5, 10) if s * mag >= raw)
    return [i * step for i in range(int(vmax / step) + 1)]


def hpanel(x, y, w, title, cap, labels, rows, series, show=fmt, ratio=None, ratio_head=""):
    """Groups of horizontal bars, one group per label: Fast Code first and thicker, the comparison under it.
    `ratio` adds a column of percentages on the right. Returns the markup and the y under the axis labels."""
    lw = max(tw(s, 13) for s in labels) + 16
    rw = tw(ratio_head, 11.5) + 12 if ratio else 0
    x0, pw = x + lw, w - lw - rw - 52
    vmax = max(v for k in series for v in rows[k])
    bh = [20 if k == "fastcode" else 12 for k in series]
    gh = sum(bh) + 3 * (len(series) - 1)
    pitch, top = gh + 18, y + 34
    bottom = top + len(labels) * pitch - 18 + 8
    out = spans(x, y + 12, [(title, 14, INK, "bold"), (cap, 12, MUTED, "normal")])
    for t in ticks(vmax):
        tx = x0 + pw * t / vmax
        out += f'<line x1="{tx:.1f}" y1="{top - 8}" x2="{tx:.1f}" y2="{bottom}" stroke="{AXIS if t == 0 else GRID}"/>'
        out += txt(tx, bottom + 16, f"{t:,.0f}".replace(",", " "), 11, MUTED, "middle")
    for gi, label in enumerate(labels):
        by = gy = top + gi * pitch
        out += txt(x0 - 10, gy + gh / 2 + 4.5, label, 13, INK, "end")
        for si, key in enumerate(series):
            v, ours = rows[key][gi], key == "fastcode"
            out += hbar(x0, by, pw * v / vmax, bh[si], COL[key])
            out += txt(x0 + pw * v / vmax + 7, by + bh[si] / 2 + (4.8 if ours else 4.2), show(v), 13.5 if ours else 12,
                       INK if ours else INK2, weight="bold" if ours else "normal", extra=HALO)
            by += bh[si] + 3
        if ratio:
            out += txt(x + w, gy + 15.5, f"{ratio[gi]} %", 15, INK, "end", "bold")
    if ratio:
        out += txt(x + w, top - 14, ratio_head, 11.5, MUTED, "end")
    return out, bottom + 16


def notes(y, lines):
    out = ""
    for line in lines:
        if tw(line, 11.5) > W - 2 * PAD:
            print("note may overflow:", line, file=sys.stderr)
        out += txt(PAD, y, line, 11.5, MUTED)
        y += 17
    return out, y


def cuda_figure(t, cu):
    """NVIDIA cards over CUDA: code generation by card memory against Strata Coder on the same card."""
    f = t["cuda"]
    sp = lambda vals: " / ".join(f"{v:,}".replace(",", " ") for v in vals)
    ratio = cu["decode_vs_strata_pct"]
    body = header(t, f) + tiles(124, [
        (" · ".join(fmt(v) for v in cu["decode"]["fastcode"]), t["tok"], f["tiles"][0].format(tiers=" · ".join(cu["tiers"]))),
        (f"{min(ratio)}–{max(ratio)} %", "", f["tiles"][1]),
        f["tiles"][2],
    ])
    series = ["fastcode", "strata"]
    body += legend(PAD, 222, [(k, f["legend"][k]) for k in series])
    labels = [f["group"].format(x) for x in cu["tiers"]]
    p, y = hpanel(PAD, 246, W - 2 * PAD, t["gen"], f["cap"], labels, cu["decode"], series,
                  show=lambda v: f"{v:.1f}", ratio=ratio, ratio_head=f["ratio"])
    n, y = notes(y + 30, [s.format(ptok=f'{cu["prompt_tokens"]:,}'.replace(",", " "), cache=sp(cu["strata_expert_cache"]))
                          for s in f["note"]])
    return svg(y + 8, body + p + n)


def vulkan_figure(t, vk):
    """One large card over Vulkan: prompt reading and code generation against the full file on the same engine."""
    f = t["vulkan"]
    read = sum(vk["prefill"]["fastcode"]) / len(vk["prefill"]["fastcode"])
    full = sum(vk["prefill"]["q4tp"]) / len(vk["prefill"]["q4tp"])
    dec = vk["decode"]["fastcode"]
    body = header(t, f) + tiles(124, [
        (f"~{round(read, -1):.0f}", t["tok"], f["tiles"][0].format(x=f"{read / full:.0f}")),
        (f"{min(dec):.0f}–{max(dec):.0f}", t["tok"], f["tiles"][1]),
        (f'{vk["vram_gb"]["fastcode"]}', t["gb"], f["tiles"][2].format(full=vk["vram_gb"]["q4tp"])),
    ])
    series = ["fastcode", "q4tp"]
    body += legend(PAD, 222, [(k, f["legend"][k]) for k in series])
    pw = (W - 2 * PAD - 40) / 2
    a, y = hpanel(PAD, 246, pw, t["read"], f["cap"], vk["prompts"], vk["prefill"], series)
    b, y = hpanel(PAD + pw + 40, 246, pw, t["gen"], f["cap"], vk["prompts"], vk["decode"], series)
    n, y = notes(y + 30, f["note"])
    return svg(y + 8, body + a + b + n)


def quality_figure(t, q):
    """One row per system, one column per measure, a bar and its value in every cell."""
    f = t["quality"]
    body = header(t, f)
    rows = ["fastcode", "strata", "q4tp"]
    label_w, top, pitch = 226, 136, 42
    cw = (W - 2 * PAD - label_w) / 3
    cols = [
        (q["expert_bits"], max(q["expert_bits"].values()) * 1.0, lambda v, k: f["bits"].format(v)),
        (q["ppl"], max(q["ppl"].values()) * 1.0, lambda v, k: f"{v:.3f}" + ("¹" if k == "strata" else "")),
        (q["humaneval"], 100.0, lambda v, k: f"{v:.1f} %"),
    ]
    for ci, (head, sub) in enumerate(f["cols"]):
        cx = PAD + label_w + ci * cw
        body += txt(cx, top, head, 14, INK, weight="bold") + txt(cx, top + 17, sub, 11.5, MUTED)
    y0 = top + 34
    for ri, key in enumerate(rows):
        y = y0 + ri * pitch
        body += f'<line x1="{PAD}" y1="{y}" x2="{W - PAD}" y2="{y}" stroke="{GRID}"/>'
        ours = key == "fastcode"
        body += f'<rect x="{PAD}" y="{y + 15}" width="12" height="12" rx="3" fill="{COL[key]}"/>'
        body += txt(PAD + 19, y + 25.5, f["rows"][key], 13, INK if ours else INK2, weight="bold" if ours else "normal")
        for ci, (vals, vmax, show) in enumerate(cols):
            cx = PAD + label_w + ci * cw
            bw = (cw - 86) * vals[key] / vmax
            body += hbar(cx, y + 13, bw, 16, COL[key])
            body += txt(cx + bw + 8, y + 25.5, show(vals[key], key), 13, INK if ours else INK2, weight="bold" if ours else "normal")
    y = y0 + len(rows) * pitch
    body += f'<line x1="{PAD}" y1="{y}" x2="{W - PAD}" y2="{y}" stroke="{GRID}"/>'
    n, y = notes(y + 26, f["note"])
    return svg(y + 8, body + n)


def main():
    data = json.load(open(sys.argv[1]))
    out_dir = sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    for lang, t in T.items():
        write(out_dir, f"fc_cuda_{lang}", cuda_figure(t, data["cuda"]))
        write(out_dir, f"fc_vulkan_{lang}", vulkan_figure(t, data["vulkan"]))
        write(out_dir, f"fc_quality_{lang}", quality_figure(t, data["quality"]))


def write(out_dir, name, s):
    p = os.path.join(out_dir, name + ".svg")
    open(p, "w").write(s)
    subprocess.run(["rsvg-convert", "--zoom", "2", "-o", os.path.join(out_dir, name + ".png"), p], check=True)


if __name__ == "__main__":
    main()
