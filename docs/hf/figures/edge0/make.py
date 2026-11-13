#!/usr/bin/env python3
"""Figures of the Edge0-145B / Edge0-120B CMF pages, SVG in three languages, PNG through rsvg-convert.

    python3 docs/hf/figures/edge0/make.py DATA.json OUT_DIR

DATA.json holds the measured rows (`figures_data.json` next to this file). Writes e0_{145,120}_{cuda,vulkan}_{en,ru,zh}
as .svg and .png: one figure is one stand and one cortiq backend, named under its title; the page's own model is the
one series, and every bar carries its value.
"""
import html, json, os, subprocess, sys

W, PAD = 880, 36
FONT = "Helvetica Neue, Helvetica, Arial, PingFang SC, Hiragino Sans GB, Hiragino Sans, sans-serif"
INK, INK2, MUTED = "#0f172a", "#334155", "#64748b"
GRID, AXIS, TILE, BORDER = "#eef1f5", "#cbd5e1", "#f4f6f9", "#e2e8f0"
COL = {"m145": "#2563eb", "m120": "#a9b2bf"}
COL_OWN = "#2563eb"
HALO = ' paint-order="stroke" stroke="#ffffff" stroke-width="3" stroke-linejoin="round"'

T = {
    "en": dict(
        engine="cortiq", stand="Stand:", tok="tok/s",
        read="Prompt reading, tok/s", gen="Code generation, tok/s", cap="by prompt length, tokens",
        name={"m145": "Edge0-145B", "m120": "Edge0-120B"},
        title={"cuda": "Edge0 on CUDA", "vulkan": "Edge0 on Vulkan"},
        stand_txt={"cuda": "one RTX PRO 6000 96 GB · cortiq, CUDA backend", "vulkan": "one RTX PRO 6000 96 GB · cortiq, Vulkan backend"},
        tiles=["prompt reading, 4K–32K", "code generation, 4K–32K", "HumanEval pass@1, 164 tasks"],
        note=["One stream, greedy, thinking off, MTP draft tokens on; a new session tag before every prompt, so nothing comes from a cache.",
              "Mean of two runs. The whole model is on the card; the experts are Edge0's NVFP4 weights."],
        cards_title={"cuda": "Edge0 on 12–32 GB cards, CUDA", "vulkan": "Edge0 on 12–32 GB cards, Vulkan"},
        cards_stand={"cuda": "one RTX PRO 6000 limited to the card size named · cortiq, CUDA backend",
                     "vulkan": "one RTX PRO 6000 limited to the card size named · cortiq, Vulkan backend"},
        cards_cap="by card memory, 4K-token prompt", gb="GB",
        cards_tiles=["code generation", "prompt reading, 4K tokens", "experts on the card"],
        cards_note={"cuda": ["The card keeps the experts used most; a decode step computes the others on the CPU, a prompt carries them up in batches.",
                             "One run a size; the rest of the card's memory was held by another process. One stream, greedy, thinking off."],
                    "vulkan": ["The card keeps the expert banks of as many layers as fit; the others stay in system memory and are read over the bus.",
                               "One run a size; the rest of the card's memory was held by another process. One stream, greedy, thinking off."]},
    ),
    "ru": dict(
        engine="cortiq", stand="Стенд:", tok="ток/с",
        read="Чтение промпта, ток/с", gen="Генерация кода, ток/с", cap="по длине промпта, токены",
        name={"m145": "Edge0-145B", "m120": "Edge0-120B"},
        title={"cuda": "Edge0 на CUDA", "vulkan": "Edge0 на Vulkan"},
        stand_txt={"cuda": "одна RTX PRO 6000 96 ГБ · cortiq, бэкенд CUDA", "vulkan": "одна RTX PRO 6000 96 ГБ · cortiq, бэкенд Vulkan"},
        tiles=["чтение промпта, 4K–32K", "генерация кода, 4K–32K", "HumanEval pass@1, 164 задачи"],
        note=["Один поток, жадный выбор, без рассуждений, черновики MTP включены; перед каждым промптом новая метка сессии — кеш не участвует.",
              "Среднее двух запусков. Вся модель на карте; эксперты — NVFP4-веса Edge0."],
        cards_title={"cuda": "Edge0 на картах 12–32 ГБ, CUDA", "vulkan": "Edge0 на картах 12–32 ГБ, Vulkan"},
        cards_stand={"cuda": "одна RTX PRO 6000, ограниченная до названного объёма · cortiq, бэкенд CUDA",
                     "vulkan": "одна RTX PRO 6000, ограниченная до названного объёма · cortiq, бэкенд Vulkan"},
        cards_cap="по памяти карты, промпт 4K токенов", gb="ГБ",
        cards_tiles=["генерация кода", "чтение промпта, 4K токенов", "экспертов на карте"],
        cards_note={"cuda": ["На карте самые используемые эксперты; остальных шаг генерации считает на CPU, промпт подгружает пакетами.",
                             "Один запуск на объём; остальную память карты держал другой процесс. Один поток, жадный выбор, без рассуждений."],
                    "vulkan": ["На карте банки экспертов стольких слоёв, сколько помещается; остальные в системной памяти, карта читает их по шине.",
                               "Один запуск на объём; остальную память карты держал другой процесс. Один поток, жадный выбор, без рассуждений."]},
    ),
    "zh": dict(
        engine="cortiq", stand="测试平台：", tok="token/s",
        read="读取提示词，token/s", gen="生成代码，token/s", cap="按提示词长度（token）",
        name={"m145": "Edge0-145B", "m120": "Edge0-120B"},
        title={"cuda": "Edge0 · CUDA 后端", "vulkan": "Edge0 · Vulkan 后端"},
        stand_txt={"cuda": "一张 RTX PRO 6000 96 GB · cortiq，CUDA 后端", "vulkan": "一张 RTX PRO 6000 96 GB · cortiq，Vulkan 后端"},
        tiles=["读取提示词，4K–32K", "生成代码，4K–32K", "HumanEval pass@1，164 题"],
        note=["单路，贪心解码，关闭思考，开启 MTP 草稿；每个提示词前加新的会话标记，因此不使用缓存。",
              "两次运行的平均值。整个模型在显卡上；专家为 Edge0 的 NVFP4 权重。"],
        cards_title={"cuda": "Edge0 · 12–32 GB 显卡，CUDA", "vulkan": "Edge0 · 12–32 GB 显卡，Vulkan"},
        cards_stand={"cuda": "一张 RTX PRO 6000，限制为所示显存 · cortiq，CUDA 后端",
                     "vulkan": "一张 RTX PRO 6000，限制为所示显存 · cortiq，Vulkan 后端"},
        cards_cap="按显存大小，4K token 提示词", gb="GB",
        cards_tiles=["生成代码", "读取提示词，4K token", "显卡上的专家"],
        cards_note={"cuda": ["显卡保留最常用的专家；其余专家在生成时由 CPU 计算，读取提示词时分批送入显卡。",
                             "每种显存运行一次；显卡其余显存由另一进程占用。单路，贪心解码，关闭思考。"],
                    "vulkan": ["显卡保留能放下的若干层的专家库；其余留在系统内存，由显卡经总线读取。",
                               "每种显存运行一次；显卡其余显存由另一进程占用。单路，贪心解码，关闭思考。"]},
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


def tiles(y, items, accent):
    """A row of stat tiles: (number, unit, caption)."""
    gap, h = 12, 64
    w = (W - 2 * PAD - gap * (len(items) - 1)) / len(items)
    out = ""
    for i, (hero, unit, cap) in enumerate(items):
        x = PAD + i * (w + gap)
        out += f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="{h}" rx="10" fill="{TILE}"/>'
        out += f'<rect x="{x:.1f}" y="{y + 12}" width="3" height="{h - 24}" rx="1.5" fill="{accent}"/>'
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


def hpanel(x, y, w, title, cap, labels, rows, series, own, show=fmt, ratio=None, ratio_head=""):
    """Groups of horizontal bars, one group per label: Fast Code first and thicker, the comparison under it.
    `ratio` adds a column of percentages on the right. Returns the markup and the y under the axis labels."""
    lw = max(tw(s, 13) for s in labels) + 16
    rw = tw(ratio_head, 11.5) + 12 if ratio else 0
    x0, pw = x + lw, w - lw - rw - 52
    vmax = max(v for k in series for v in rows[k])
    bh = [20 if k == own else 12 for k in series]
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
            v, ours = rows[key][gi], key == own
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




def header(t, backend, model):
    """Backend pill, engine and model, the title, the stand."""
    badge = backend.upper()
    bw = tw(badge, 11) + 1.2 * len(badge) + 22
    out = f'<rect x="{PAD}" y="26" width="{bw:.1f}" height="22" rx="11" fill="{INK}"/>'
    out += txt(PAD + bw / 2 + 0.6, 41, badge, 11, "#ffffff", "middle", "bold", ' letter-spacing="1.2"')
    out += spans(PAD + bw + 10, 41.5, [(t["engine"], 13, INK, "bold"), ("·  " + t["name"][model] + " CMF", 13, MUTED, "normal")])
    out += txt(PAD, 82, t["title"][backend], 23, INK, weight="bold")
    out += spans(PAD, 105, [(t["stand"], 13, INK, "bold"), (t["stand_txt"][backend], 13, INK2, "normal")])
    return out


def figure(t, data, backend, model):
    global COL
    d = data[backend]
    other = "m120" if model == "m145" else "m145"
    col = dict(COL)
    col[model], col[other] = COL_OWN, "#a9b2bf"
    saved, COL = COL, col
    try:
        mean = lambda vals: sum(vals) / len(vals)
        rd = [v for p, v in zip(d["prompts"], d["prefill"][model]) if p != "1K"]
        gn = [v for p, v in zip(d["prompts"], d["decode"][model]) if p != "1K"]
        body = header(t, backend, model) + tiles(124, [
            (f"{min(rd):,.0f}–{max(rd):,.0f}".replace(",", " "), t["tok"], t["tiles"][0]),
            (f"{min(gn):.0f}–{max(gn):.0f}", t["tok"], t["tiles"][1]),
            (f'{data["humaneval"][model]:.1f} %', "", t["tiles"][2]),
        ], COL_OWN)
        series = [model]
        pw = (W - 2 * PAD - 40) / 2
        a, y = hpanel(PAD, 216, pw, t["read"], t["cap"], d["prompts"], d["prefill"], series, model)
        b, y = hpanel(PAD + pw + 40, 216, pw, t["gen"], t["cap"], d["prompts"], d["decode"], series, model)
        n, y = notes(y + 30, t["note"])
        return svg(y + 8, body + a + b + n)
    finally:
        COL = saved


def figure_cards(t, data, backend, model):
    """The smaller cards of one backend: generation and prompt reading by card memory (None without the rows)."""
    global COL
    rows = data.get(backend + "_tiers", {}).get(model)
    if not rows or len(rows) < 2:
        return None
    saved, COL = COL, {model: COL_OWN}
    try:
        at = lambda r, key: r[key][r["ctx"].index("4K")] if "4K" in r["ctx"] else r[key][-1]
        labels = [f'{r["gb"]} {t["gb"]}' for r in rows]
        gen = [at(r, "decode") for r in rows]
        rd = [at(r, "prefill") for r in rows]
        pct = [r.get("experts_pct", 0.0) for r in rows]
        f0 = lambda v: f"{v:,.0f}".replace(",", " ")
        badge = backend.upper()
        bw = tw(badge, 11) + 1.2 * len(badge) + 22
        body = f'<rect x="{PAD}" y="26" width="{bw:.1f}" height="22" rx="11" fill="{INK}"/>'
        body += txt(PAD + bw / 2 + 0.6, 41, badge, 11, "#ffffff", "middle", "bold", ' letter-spacing="1.2"')
        body += spans(PAD + bw + 10, 41.5, [(t["engine"], 13, INK, "bold"), ("·  " + t["name"][model] + " CMF", 13, MUTED, "normal")])
        body += txt(PAD, 82, t["cards_title"][backend], 23, INK, weight="bold")
        body += spans(PAD, 105, [(t["stand"], 13, INK, "bold"), (t["cards_stand"][backend], 13, INK2, "normal")])
        body += tiles(124, [
            (f"{f0(min(gen))}–{f0(max(gen))}", t["tok"], t["cards_tiles"][0]),
            (f"{f0(min(rd))}–{f0(max(rd))}", t["tok"], t["cards_tiles"][1]),
            (f"{min(pct):.0f}–{max(pct):.0f} %", "", t["cards_tiles"][2]),
        ], COL_OWN)
        pw = (W - 2 * PAD - 40) / 2
        a, y = hpanel(PAD, 216, pw, t["gen"], t["cards_cap"], labels, {model: gen}, [model], model)
        b, y = hpanel(PAD + pw + 40, 216, pw, t["read"], t["cards_cap"], labels, {model: rd}, [model], model)
        n, y = notes(y + 30, t["cards_note"][backend])
        return svg(y + 8, body + a + b + n)
    finally:
        COL = saved


def write(out_dir, name, s):
    p = os.path.join(out_dir, name + ".svg")
    open(p, "w").write(s)
    subprocess.run(["rsvg-convert", "--zoom", "2", "-o", os.path.join(out_dir, name + ".png"), p], check=True)


def main():
    data = json.load(open(sys.argv[1]))
    out_dir = sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    for lang, t in T.items():
        for backend in ("cuda", "vulkan"):
            for model, tag in (("m145", "145"), ("m120", "120")):
                write(out_dir, f"e0_{tag}_{backend}_{lang}", figure(t, data, backend, model))
                cards = figure_cards(t, data, backend, model)
                if cards:
                    write(out_dir, f"e0_{tag}_{backend}_cards_{lang}", cards)


if __name__ == "__main__":
    main()
