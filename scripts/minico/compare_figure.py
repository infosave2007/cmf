"""The card's figure: decode speed after a prompt, standard and turbo, measured on a MacBook Air M4.
compare_figure.py OUT_DIR"""
import os
import sys

from PIL import Image, ImageDraw, ImageFont

out = sys.argv[1]
CTX = [1000, 4000, 16000, 32000, 64000]
STD = [155.3, 136.9, 92.2, 65.1, 38.6]
TUR = [152.8, 137.5, 121.5, 104.2, 106.2]
SURF, INK, INK2, MUTE, LINE = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8983", "#dedcd5"
BLUE, ORANGE = "#2a78d6", "#eb6834"
T = {
    "en": ("Standard and turbo, the same 8-bit weights",
           "Decode after a prompt of that length, tok/s. MacBook Air M4, Metal.",
           "prompt, tokens", "standard", "turbo"),
    "ru": ("Обычный и турбо, те же 8-битные веса",
           "Генерация после промпта такой длины, ток/с. MacBook Air M4, Metal.",
           "промпт, токенов", "обычный", "турбо"),
    "zh": ("标准与 turbo，相同的 8 位权重",
           "该长度提示词之后的生成速度，token/s。MacBook Air M4，Metal。",
           "提示词长度，token", "标准", "turbo"),
}
FONTS = {
    "en": "/System/Library/Fonts/Supplemental/Arial.ttf",
    "ru": "/System/Library/Fonts/Supplemental/Arial.ttf",
    "zh": "/System/Library/Fonts/Hiragino Sans GB.ttc",
}
S = 2  # supersampling
W, H = 1200 * S, 640 * S
for lang, (title, sub, xl, ls, lt) in T.items():
    f = lambda px: ImageFont.truetype(FONTS[lang], px * S)
    bold = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf" if lang != "zh" else FONTS[lang], 34 * S)
    im = Image.new("RGB", (W, H), SURF)
    d = ImageDraw.Draw(im)
    d.text((60 * S, 36 * S), title, font=bold, fill=INK)
    d.text((60 * S, 86 * S), sub, font=f(22), fill=INK2)
    x0, x1, y0, y1 = 110 * S, 1120 * S, 540 * S, 150 * S
    ymax = 180.0
    X = lambda i: x0 + (x1 - x0) * i / (len(CTX) - 1)
    Y = lambda v: y0 - (y0 - y1) * v / ymax
    for v in range(0, 181, 30):
        d.line([(x0, Y(v)), (x1, Y(v))], fill=LINE, width=S)
        d.text((x0 - 14 * S, Y(v)), str(v), font=f(18), fill=MUTE, anchor="rm")
    for i, c in enumerate(CTX):
        d.text((X(i), y0 + 16 * S), f"{c:,}".replace(",", " "), font=f(19), fill=INK2, anchor="mt")
    d.text(((x0 + x1) / 2, y0 + 52 * S), xl, font=f(19), fill=MUTE, anchor="mt")
    for vals, col, dy in ((STD, ORANGE, 26), (TUR, BLUE, -26)):
        pts = [(X(i), Y(v)) for i, v in enumerate(vals)]
        d.line(pts, fill=col, width=5 * S, joint="curve")
        for i, (px, py) in enumerate(pts):
            d.ellipse([px - 8 * S, py - 8 * S, px + 8 * S, py + 8 * S], fill=col)
            if i >= 2 or col == BLUE:
                lab = f"{vals[i]:.0f}" if vals[i] >= 100 else f"{vals[i]:.1f}".rstrip("0").rstrip(".")
                if lang == "ru":
                    lab = lab.replace(".", ",")
                d.text((px, py + dy * S), lab, font=f(21), fill=col, anchor="mm")
    lx, ly = x1 - 300 * S, 160 * S
    for k, (lab, col) in enumerate(((lt, BLUE), (ls, ORANGE))):
        yy = ly + k * 34 * S
        d.line([(lx, yy), (lx + 44 * S, yy)], fill=col, width=5 * S)
        d.text((lx + 58 * S, yy), lab, font=f(22), fill=INK, anchor="lm")
    im.resize((W // S, H // S), Image.LANCZOS).save(os.path.join(out, f"compare_{lang}.png"), optimize=True)
    print("wrote", lang)
