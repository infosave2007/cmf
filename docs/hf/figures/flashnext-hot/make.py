#!/usr/bin/env python3
"""Figures of the Qwen3.8-Flash-Next hot page: SVG in three languages, PNG through rsvg-convert.

    python3 docs/hf/figures/flashnext-hot/make.py OUT_DIR

writes hot_build_{en,ru,zh}.svg/.png and hot_quality_{en,ru,zh}.svg/.png into OUT_DIR.
"""
import os, subprocess, sys, html

FONT = "Helvetica Neue, Helvetica, Arial, PingFang SC, Hiragino Sans GB, Hiragino Sans, sans-serif"
INK, INK2, MUTED, LINE, GRID = "#1f2937", "#4b5563", "#6b7280", "#9ca3af", "#e5e7eb"
ACC, ACC_L, ACC_BG, COLD, COLD_BG = "#1d4ed8", "#60a5fa", "#dbeafe", "#9ca3af", "#f3f4f6"

T = {
 "en": dict(
  b_title="How the hot file is built",
  p1="1 · Probe", p1a="8 192 tokens of code and text, host path:", p1b="Σ(w·act)² per layer, expert and neuron",
  p1c="the 28 busiest of 512 experts", p1d="carry 43 % of the layer's energy",
  p1e="the shared expert runs on every token", p1f="same 6.5 GB, two grains:",
  p1g="5 % of every expert's neurons", p1h="28 whole experts",
  p2="2 · Bake", p2a="512 experts of a layer, 2-bit triples (gate / up / down)",
  p2b="28 chosen: gate and up rows set to zero", p2c="8-bit tubes from the bf16 checkpoint",
  p2d="28 experts + the shared expert", p2e="6.5 GB in all, resident on the card",
  p2f="the other 484 stay 2-bit and stream from RAM",
  p3="3 · Run", p3a="one token, one layer", p3b="router: 10 of 512",
  p3c="2-bit triple", p3d="from the expert cache", p3e="8-bit tube", p3f="already on the card",
  p3g="MoE output", p3h="two small kernels add a winner's tube;", p3i="the shared tube is added on every token, × σ",
  q_title="Code perplexity against file size", q_x="file size, GB", q_y="code ppl on 1 536 tokens of Rust, lower is better",
  q_std2="2-bit file, 77 GB", q_neu="5 % of every expert's neurons at 8 bits", q_rnd="28 random experts",
  q_top="28 busiest experts, no shared tube", q_sh="shared expert at 8 bits", q_this="this file: shared + 28 busiest",
  q_std4="4-bit file, 97 GB", q_gap="83 % of the way to the 4-bit file",
 ),
 "ru": dict(
  b_title="Как собран hot-файл",
  p1="1 · Зонд", p1a="8 192 токена кода и текста, путь на хосте:", p1b="Σ(w·act)² по слою, эксперту и нейрону",
  p1c="28 самых занятых из 512 экспертов", p1d="несут 43 % энергии слоя",
  p1e="общий эксперт работает на каждом токене", p1f="те же 6,5 ГБ, два зерна:",
  p1g="5 % нейронов каждого эксперта", p1h="28 экспертов целиком",
  p2="2 · Сборка", p2a="512 экспертов слоя, 2-битные тройки (gate / up / down)",
  p2b="28 выбранных: строки gate и up обнулены", p2c="8-битные трубки из bf16-чекпойнта",
  p2d="28 экспертов + общий эксперт", p2e="всего 6,5 ГБ, резидентно на карте",
  p2f="остальные 484 остаются 2-битными и читаются из ОЗУ",
  p3="3 · Запуск", p3a="один токен, один слой", p3b="маршрутизатор: 10 из 512",
  p3c="2-битная тройка", p3d="из кэша экспертов", p3e="8-битная трубка", p3f="уже на карте",
  p3g="выход MoE", p3h="два маленьких ядра добавляют трубку победителя;", p3i="трубка общего эксперта — на каждом токене, × σ",
  q_title="Перплексия на коде против размера файла", q_x="размер файла, ГБ", q_y="ppl на 1 536 токенах Rust, меньше — лучше",
  q_std2="2-битный файл, 77 ГБ", q_neu="5 % нейронов каждого эксперта в 8 битах", q_rnd="28 случайных экспертов",
  q_top="28 самых занятых, без общего", q_sh="общий эксперт в 8 битах", q_this="этот файл: общий + 28 самых занятых",
  q_std4="4-битный файл, 97 ГБ", q_gap="83 % пути к 4-битному файлу",
 ),
 "zh": dict(
  b_title="hot 文件是如何构建的",
  p1="1 · 探测", p1a="8 192 个代码与文本 token，主机路径：", p1b="按层、专家、神经元统计 Σ(w·act)²",
  p1c="512 个专家中最活跃的 28 个", p1d="承载该层 43 % 的能量",
  p1e="共享专家在每个 token 上运行", p1f="同样 6.5 GB，两种粒度：",
  p1g="每个专家 5 % 的神经元", p1h="28 个完整专家",
  p2="2 · 烘焙", p2a="一层的 512 个专家，2 位三元组（gate / up / down）",
  p2b="选中的 28 个：gate 与 up 行置零", p2c="来自 bf16 检查点的 8 位管",
  p2d="28 个专家 + 共享专家", p2e="共 6.5 GB，常驻显卡",
  p2f="其余 484 个保持 2 位，从内存流式读取",
  p3="3 · 运行", p3a="一个 token，一层", p3b="路由：512 选 10",
  p3c="2 位三元组", p3d="来自专家缓存", p3e="8 位管", p3f="已在显卡上",
  p3g="MoE 输出", p3h="两个小内核加上被选中专家的管；", p3i="共享管在每个 token 上加入，× σ",
  q_title="代码困惑度与文件大小", q_x="文件大小，GB", q_y="1 536 个 Rust token 的 ppl，越低越好",
  q_std2="2 位文件，77 GB", q_neu="每个专家 5 % 的神经元为 8 位", q_rnd="随机 28 个专家",
  q_top="最活跃的 28 个专家，无共享管", q_sh="共享专家为 8 位", q_this="本文件：共享 + 最活跃的 28 个",
  q_std4="4 位文件，97 GB", q_gap="走完到 4 位文件 83 % 的路",
 ),
}

def txt(x, y, s, size=14, fill=INK, anchor="start", weight="normal"):
    return (f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" font-weight="{weight}">{html.escape(s)}</text>')

def arrow(x1, y1, x2, y2, color=INK2):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="1.6" marker-end="url(#ar)"/>'

HEAD = '''<defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
<path d="M 0 0 L 10 5 L 0 10 z" fill="%s"/></marker>
<pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
<line x1="0" y1="0" x2="0" y2="6" stroke="%s" stroke-width="2"/></pattern></defs>''' % (INK2, COLD)

def build(t):
    W, H = 1240, 550
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">', HEAD,
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>', txt(24, 36, t["b_title"], 22, INK, weight="bold")]
    # ---- panel 1: probe
    px, py = 24, 62
    o.append(txt(px, py + 18, t["p1"], 17, ACC, weight="bold"))
    o.append(txt(px, py + 42, t["p1a"], 13, INK2)); o.append(txt(px, py + 60, t["p1b"], 13, INK2))
    # sorted energy bars: 64 bars, decaying
    bx, by, bw, bh = px, py + 80, 360, 150
    o.append(f'<line x1="{bx}" y1="{by+bh}" x2="{bx+bw}" y2="{by+bh}" stroke="{LINE}" stroke-width="1"/>')
    n = 64
    for i in range(n):
        h = bh * (0.97 * (0.955 ** i) + 0.03)
        x = bx + i * (bw / n)
        fill = ACC if i < 28 * n // 512 + 3 else COLD   # the first few bars stand for the 28 busiest
        fill = ACC if i < 8 else COLD
        o.append(f'<rect x="{x:.1f}" y="{by+bh-h:.1f}" width="{bw/n-1.2:.1f}" height="{h:.1f}" fill="{fill}"/>')
    o.append(txt(bx, by + bh + 18, t["p1c"], 13, ACC, weight="bold")); o.append(txt(bx, by + bh + 36, t["p1d"], 13, ACC))
    o.append(txt(bx + bw, by + bh + 18, "512 →", 12, MUTED, anchor="end"))
    # shared expert box
    sy = by + bh + 52
    o.append(f'<rect x="{bx}" y="{sy}" width="{bw}" height="30" rx="5" fill="{ACC_BG}" stroke="{ACC}" stroke-width="1.2"/>')
    o.append(txt(bx + bw / 2, sy + 20, t["p1e"], 13, ACC, anchor="middle"))
    # two grains
    gy = sy + 50
    o.append(txt(bx, gy + 14, t["p1f"], 13, INK2))
    for k, (label, pct, fill) in enumerate([(t["p1g"], 32, COLD), (t["p1h"], 43, ACC)]):
        yy = gy + 26 + k * 30
        o.append(txt(bx, yy + 14, label, 13, INK))
        o.append(f'<rect x="{bx+240}" y="{yy+2}" width="{pct*2.2:.0f}" height="16" rx="3" fill="{fill}"/>')
        o.append(txt(bx + 240 + pct * 2.2 + 6, yy + 15, f"{pct} %", 13, INK, weight="bold"))
    # ---- panel 2: bake
    qx = 430
    o.append(f'<line x1="{qx-18}" y1="60" x2="{qx-18}" y2="{H-20}" stroke="{GRID}" stroke-width="1"/>')
    o.append(txt(qx, py + 18, t["p2"], 17, ACC, weight="bold"))
    o.append(txt(qx, py + 42, t["p2a"], 13, INK2))
    # grid of 512 = 32 x 16 small cells, 28 chosen hatched (the first 28 in a scattered pattern)
    cx, cy, cs, gap = qx, py + 56, 10, 2
    chosen = set([3, 17, 29, 44, 58, 71, 90, 102, 121, 135, 150, 166, 177, 190, 205, 221, 236, 250, 263, 280, 297, 311, 330, 348, 366, 385, 410, 437])
    for i in range(512):
        r, c = divmod(i, 32)
        x, y = cx + c * (cs + gap), cy + r * (cs + gap)
        if i in chosen:
            o.append(f'<rect x="{x}" y="{y}" width="{cs}" height="{cs}" fill="url(#hatch)" stroke="{ACC}" stroke-width="1"/>')
        else:
            o.append(f'<rect x="{x}" y="{y}" width="{cs}" height="{cs}" fill="{COLD_BG}" stroke="{COLD}" stroke-width="0.6"/>')
    gy2 = cy + 16 * (cs + gap) + 14
    o.append(txt(qx, gy2, t["p2b"], 13, INK))
    # arrow down to tubes
    o.append(arrow(qx + 180, gy2 + 10, qx + 180, gy2 + 40, ACC))
    ty = gy2 + 46
    o.append(f'<rect x="{qx}" y="{ty}" width="384" height="114" rx="6" fill="{ACC_BG}" stroke="{ACC}" stroke-width="1.4"/>')
    # tubes drawn as seven small rounded bars + one wider (the shared expert)
    for k in range(7):
        o.append(f'<rect x="{qx+14+k*20}" y="{ty+12}" width="12" height="30" rx="3" fill="{ACC}"/>')
    o.append(f'<rect x="{qx+14+7*20+6}" y="{ty+12}" width="20" height="30" rx="3" fill="{ACC_L}"/>')
    o.append(txt(qx + 14, ty + 62, t["p2c"], 13, ACC, weight="bold"))
    o.append(txt(qx + 14, ty + 82, t["p2d"], 13, INK))
    o.append(txt(qx + 14, ty + 102, t["p2e"], 13, INK))
    o.append(txt(qx, ty + 136, t["p2f"], 13, INK2))
    # ---- panel 3: run
    rx = 850
    o.append(f'<line x1="{rx-18}" y1="60" x2="{rx-18}" y2="{H-20}" stroke="{GRID}" stroke-width="1"/>')
    o.append(txt(rx, py + 18, t["p3"], 17, ACC, weight="bold"))
    o.append(txt(rx, py + 42, t["p3a"], 13, INK2))
    # router box
    rb = py + 64
    o.append(f'<rect x="{rx}" y="{rb}" width="360" height="36" rx="6" fill="#ffffff" stroke="{INK2}" stroke-width="1.2"/>')
    o.append(txt(rx + 180, rb + 24, t["p3b"], 14, INK, anchor="middle"))
    # two branches
    by3 = rb + 90
    o.append(arrow(rx + 100, rb + 36, rx + 80, by3, INK2)); o.append(arrow(rx + 260, rb + 36, rx + 280, by3, ACC))
    o.append(f'<rect x="{rx}" y="{by3}" width="160" height="70" rx="6" fill="{COLD_BG}" stroke="{COLD}" stroke-width="1.2"/>')
    for k in range(3):
        o.append(f'<rect x="{rx+14+k*18}" y="{by3+14}" width="10" height="42" rx="2" fill="{COLD}"/>')
    o.append(txt(rx + 76, by3 + 30, t["p3c"], 13, INK, weight="bold")); o.append(txt(rx + 76, by3 + 50, t["p3d"], 12, INK2))
    o.append(f'<rect x="{rx+200}" y="{by3}" width="160" height="70" rx="6" fill="{ACC_BG}" stroke="{ACC}" stroke-width="1.2"/>')
    o.append(f'<rect x="{rx+214}" y="{by3+14}" width="16" height="42" rx="4" fill="{ACC}"/>')
    o.append(txt(rx + 244, by3 + 30, t["p3e"], 13, ACC, weight="bold")); o.append(txt(rx + 244, by3 + 50, t["p3f"], 12, INK2))
    # plus and output
    oy = by3 + 70
    o.append(arrow(rx + 80, oy, rx + 170, oy + 50, INK2)); o.append(arrow(rx + 280, oy, rx + 190, oy + 50, ACC))
    o.append(f'<circle cx="{rx+180}" cy="{oy+62}" r="14" fill="#ffffff" stroke="{INK}" stroke-width="1.4"/>')
    o.append(txt(rx + 180, oy + 68, "+", 18, INK, anchor="middle", weight="bold"))
    o.append(arrow(rx + 180, oy + 76, rx + 180, oy + 104, INK2))
    o.append(f'<rect x="{rx+100}" y="{oy+106}" width="160" height="32" rx="6" fill="#ffffff" stroke="{INK2}" stroke-width="1.2"/>')
    o.append(txt(rx + 180, oy + 128, t["p3g"], 14, INK, anchor="middle", weight="bold"))
    o.append(txt(rx, oy + 170, t["p3h"], 13, INK2)); o.append(txt(rx, oy + 188, t["p3i"], 13, INK2))
    o.append('</svg>')
    return "\n".join(o)

def quality(t):
    W, H = 1100, 560
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">', HEAD,
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>', txt(24, 36, t["q_title"], 22, INK, weight="bold")]
    L, R, Tp, B = 90, 1060, 70, 480
    x0, x1, y0, y1 = 74.0, 100.0, 4.05, 4.50
    X = lambda g: L + (g - x0) / (x1 - x0) * (R - L)
    Y = lambda p: B - (p - y0) / (y1 - y0) * (B - Tp)
    for p in [4.1, 4.2, 4.3, 4.4, 4.5]:
        o.append(f'<line x1="{L}" y1="{Y(p):.1f}" x2="{R}" y2="{Y(p):.1f}" stroke="{GRID}" stroke-width="1"/>')
        o.append(txt(L - 10, Y(p) + 5, f"{p:.1f}", 12, MUTED, anchor="end"))
    for g in [75, 80, 85, 90, 95, 100]:
        o.append(txt(X(g), B + 20, str(g), 12, MUTED, anchor="middle"))
        o.append(f'<line x1="{X(g):.1f}" y1="{B}" x2="{X(g):.1f}" y2="{B+4}" stroke="{LINE}"/>')
    o.append(f'<line x1="{L}" y1="{B}" x2="{R}" y2="{B}" stroke="{LINE}" stroke-width="1"/>')
    o.append(txt((L + R) / 2, B + 44, t["q_x"], 13, INK2, anchor="middle"))
    o.append(f'<text x="{24}" y="{(Tp+B)/2}" font-family="{FONT}" font-size="13" fill="{INK2}" text-anchor="middle" transform="rotate(-90 24 {(Tp+B)/2})">{html.escape(t["q_y"])}</text>')
    # the 4-bit floor
    o.append(f'<line x1="{L}" y1="{Y(4.111):.1f}" x2="{R}" y2="{Y(4.111):.1f}" stroke="{INK2}" stroke-width="1" stroke-dasharray="4 4"/>')
    pts = [  # (GB, ppl, label, fill, stroke, r, dx, dy, anchor, bold)
        (76.88, 4.438, t["q_std2"], COLD, COLD, 7, 12, 4, "start", False),
        (83.57, 4.425, t["q_neu"], "#ffffff", COLD, 7, 12, -6, "start", False),
        (83.57, 4.388, t["q_rnd"], "#ffffff", COLD, 7, 12, 4, "start", False),
        (83.57, 4.327, t["q_top"], COLD, COLD, 7, 12, 4, "start", False),
        (77.17, 4.241, t["q_sh"], ACC_L, ACC, 7, 12, 4, "start", False),
        (83.81, 4.167, t["q_this"], ACC, ACC, 10, 16, 4, "start", True),
        (97.0, 4.111, t["q_std4"], COLD, COLD, 7, -12, -8, "end", False),
    ]
    for g, p, label, fill, stroke, r, dx, dy, anchor, bold in pts:
        o.append(f'<circle cx="{X(g):.1f}" cy="{Y(p):.1f}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
        o.append(txt(X(g) + dx, Y(p) + dy, f"{p:.3f} · {label}", 13, ACC if bold else INK, anchor, "bold" if bold else "normal"))
    # bracket: the way to the 4-bit file
    gx = X(83.81) - 40
    o.append(f'<line x1="{gx}" y1="{Y(4.438):.1f}" x2="{gx}" y2="{Y(4.111):.1f}" stroke="{ACC}" stroke-width="1.2"/>')
    o.append(f'<line x1="{gx-4}" y1="{Y(4.438):.1f}" x2="{gx+4}" y2="{Y(4.438):.1f}" stroke="{ACC}" stroke-width="1.2"/>')
    o.append(f'<line x1="{gx-4}" y1="{Y(4.111):.1f}" x2="{gx+4}" y2="{Y(4.111):.1f}" stroke="{ACC}" stroke-width="1.2"/>')
    o.append(f'<line x1="{gx-4}" y1="{Y(4.167):.1f}" x2="{gx+4}" y2="{Y(4.167):.1f}" stroke="{ACC}" stroke-width="2.4"/>')
    o.append(txt(gx - 10, (Y(4.167) + Y(4.438)) / 2 + 4, t["q_gap"], 12, ACC, anchor="end"))
    o.append('</svg>')
    return "\n".join(o)

def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out, exist_ok=True)
    for lang, t in T.items():
        for name, fn in [("hot_build", build), ("hot_quality", quality)]:
            svg = os.path.join(out, f"{name}_{lang}.svg"); png = svg[:-4] + ".png"
            open(svg, "w").write(fn(t))
            subprocess.run(["rsvg-convert", "--zoom", "2", "-o", png, svg], check=True)
            print(png)

if __name__ == "__main__":
    main()
