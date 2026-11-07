"""Two figures of the turbo mode for the Minico card, in English, Russian and Chinese.
turbo_figures.py SCORES.bin TOKENS.bin TOKENIZER.json OUT_DIR
SCORES.bin: `CMF_DORY_EXACT_LOG` of the turbo file on a prompt with a planted pass key;
TOKENS.bin: `CMF_O1_TOKENS_DUMP` of the same run."""
import os
import re
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tokenizers import Tokenizer

scores_p, toks_p, tokj, out = sys.argv[1:5]
SURF, INK, INK2, MUTE, LINE = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8983", "#dedcd5"
BLUE, ORANGE, AQUA, GREY, TINT, TINT2 = "#2a78d6", "#eb6834", "#1baf7a", "#c9c7bf", "#eaf2fc", "#fdeee7"
S = 2

T = {
    "en": dict(
        f_title="Turbo: a long prompt in a state of fixed size",
        f_sub="One attention layer of the six, after a prompt of 64 000 tokens. Prompts under 8192 tokens stay exact.",
        std="Standard", std_t="every row exact: memory and the cost of a token grow with the text",
        tur="Turbo", sink="first 4 rows", far="everything else: 32 landmarks per head", keep="248 chosen rows, exact",
        near="last 1800 rows, exact", reads="A new token reads 2052 rows exactly and the landmarks for the rest, at any length.",
        std_r="decode 38.6 tok/s", tur_r="decode 106 tok/s",
        x_title="Passwords and codes stay exact",
        x_sub="Each row of the prompt gets a score from what the layers do with it, with no word lists.",
        x_y="row score", x_x="rows of a 23 465-token prompt, read before the choice",
        x_fact="“Remember this pass key: 653269.”", x_rank="its digits: places 0 and 2 of 23 453",
        x_kept="kept exact, with 2 neighbours each side", x_rest="go to the far field", x_win="last 1800",
        steps=[("Read", "the prompt exactly, up to its last 512 rows"),
               ("Score", "every row by its key and value, in all 6 layers"),
               ("Keep", "the best 248 rows exact, the rest as landmarks"),
               ("Answer", "the question reads the exact fact")],
        x_ans="The model's answer: “The pass key given above is 653269.”"),
    "ru": dict(
        f_title="Турбо: длинный промпт в состоянии постоянного размера",
        f_sub="Один слой внимания из шести после промпта в 64 000 токенов. Промпты короче 8192 токенов остаются точными.",
        std="Обычный", std_t="каждая строка точная: память и цена токена растут вместе с текстом",
        tur="Турбо", sink="первые 4 строки", far="всё остальное: 32 опорные точки на голову", keep="248 выбранных строк, точно",
        near="последние 1800 строк, точно", reads="Новый токен читает 2052 строки точно и опорные точки вместо остальных, при любой длине.",
        std_r="генерация 38,6 ток/с", tur_r="генерация 106 ток/с",
        x_title="Пароли и коды остаются точными",
        x_sub="Каждая строка промпта получает оценку по тому, что с ней делают слои, без списков слов.",
        x_y="оценка строки", x_x="строки промпта в 23 465 токенов, прочитанные до выбора",
        x_fact="«Remember this pass key: 653269.»", x_rank="его цифры: места 0 и 2 из 23 453",
        x_kept="точные, с 2 соседями с каждой стороны", x_rest="уходят в дальнее поле", x_win="последние 1800",
        steps=[("Чтение", "промпт точно, до последних 512 строк"),
               ("Оценка", "каждой строки по ключу и значению, во всех 6 слоях"),
               ("Отбор", "248 лучших строк точно, остальное — опорные точки"),
               ("Ответ", "вопрос читает точный факт")],
        x_ans="Ответ модели: «The pass key given above is 653269.»"),
    "zh": dict(
        f_title="turbo：长提示词保存在固定大小的状态中",
        f_sub="六个注意力层之一，64 000 token 提示词之后。8192 token 以下的提示词保持精确。",
        std="标准", std_t="每一行都精确保存：内存和每个 token 的开销随文本增长",
        tur="turbo", sink="前 4 行", far="其余：每个头 32 个地标", keep="选出的 248 行，精确",
        near="最后 1800 行，精确", reads="新 token 精确读取 2052 行，其余通过地标读取，与长度无关。",
        std_r="生成 38.6 token/s", tur_r="生成 106 token/s",
        x_title="密码和编码保持精确",
        x_sub="提示词的每一行按各层对它的作用打分，不使用词表。",
        x_y="行得分", x_x="23 465 token 提示词中、选择之前读取的行",
        x_fact="“Remember this pass key: 653269.”", x_rank="其数字：在 23 453 行中排第 0 和第 2",
        x_kept="保持精确，两侧各带 2 个相邻行", x_rest="进入远场", x_win="最后 1800 行",
        steps=[("读取", "精确读取提示词，直到最后 512 行"),
               ("打分", "在全部 6 层中按键和值为每行打分"),
               ("保留", "得分最高的 248 行精确保存，其余为地标"),
               ("回答", "问题读取精确的事实")],
        x_ans="模型的回答：“The pass key given above is 653269.”"),
}
REG = {"en": ("/System/Library/Fonts/Supplemental/Arial.ttf", 0), "ru": ("/System/Library/Fonts/Supplemental/Arial.ttf", 0),
       "zh": ("/System/Library/Fonts/Hiragino Sans GB.ttc", 0)}
BOLD = {"en": ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 0), "ru": ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 0),
        "zh": ("/System/Library/Fonts/STHeiti Medium.ttc", 0)}


def fonts(lang):
    r = lambda px: ImageFont.truetype(REG[lang][0], px * S, index=REG[lang][1])
    b = lambda px: ImageFont.truetype(BOLD[lang][0], px * S, index=BOLD[lang][1])
    return r, b


def save(im, name):
    im.resize((im.width // S, im.height // S), Image.LANCZOS).save(os.path.join(out, name), optimize=True)


# ── Figure 1: what a long prompt keeps ─────────────────────────────────────────────
rng = np.random.default_rng(7)
keep_x = np.sort(rng.choice(np.arange(60, 860), 26, replace=False))
for lang, t in T.items():
    r, b = fonts(lang)
    W, H = 1200 * S, 600 * S
    im = Image.new("RGB", (W, H), SURF)
    d = ImageDraw.Draw(im)
    d.text((60 * S, 36 * S), t["f_title"], font=b(32), fill=INK)
    d.text((60 * S, 84 * S), t["f_sub"], font=r(20), fill=INK2)
    X0, X1 = 60 * S, 1140 * S
    # standard
    y = 150 * S
    d.text((X0, y), t["std"], font=b(24), fill=INK)
    d.text((X1, y + 4 * S), t["std_r"], font=b(20), fill=ORANGE, anchor="ra")
    bar = (X0, y + 40 * S, X1, y + 96 * S)
    d.rectangle(bar, fill=TINT2)
    for x in range(bar[0], bar[2], 6 * S):
        d.line([(x, bar[1]), (x, bar[3])], fill=ORANGE, width=2 * S)
    d.text((X0, bar[3] + 12 * S), t["std_t"], font=r(19), fill=INK2)
    # turbo
    y = 330 * S
    d.text((X0, y), t["tur"], font=b(24), fill=INK)
    d.text((X1, y + 4 * S), t["tur_r"], font=b(20), fill=BLUE, anchor="ra")
    by0, by1 = y + 40 * S, y + 96 * S
    sink_x1 = X0 + 18 * S
    near_x0 = X1 - 250 * S
    d.rectangle((X0, by0, sink_x1, by1), fill=INK)
    d.rectangle((sink_x1, by0, near_x0, by1), fill=TINT)
    for k in range(32):  # landmarks
        cx = sink_x1 + (near_x0 - sink_x1) * (k + 0.5) / 32
        d.ellipse((cx - 5 * S, (by0 + by1) / 2 - 5 * S, cx + 5 * S, (by0 + by1) / 2 + 5 * S), fill=GREY)
    for kx in keep_x:  # chosen rows
        cx = sink_x1 + (near_x0 - sink_x1) * kx / 900
        d.rectangle((cx - 2 * S, by0, cx + 2 * S, by1), fill=BLUE)
    d.rectangle((near_x0, by0, X1, by1), fill=BLUE)
    # labels under the turbo bar
    ly = by1 + 14 * S
    d.line([(X0 + 9 * S, by1), (X0 + 9 * S, ly + 40 * S)], fill=INK, width=2 * S)
    d.text((X0 + 16 * S, ly + 28 * S), t["sink"], font=r(18), fill=INK)
    d.text(((sink_x1 + near_x0) / 2, ly), t["far"], font=r(18), fill=MUTE, anchor="ma")
    kx = sink_x1 + (near_x0 - sink_x1) * keep_x[len(keep_x) // 2] / 900
    d.text(((sink_x1 + near_x0) / 2, ly + 28 * S), t["keep"], font=b(18), fill=BLUE, anchor="ma")
    d.text((X1, ly), t["near"], font=b(18), fill=BLUE, anchor="ra")
    d.text((X0, 540 * S), t["reads"], font=b(20), fill=INK)
    save(im, f"far_field_{lang}.png")

# ── Figure 2: how the rows are chosen ──────────────────────────────────────────────
sc = np.fromfile(scores_p, dtype=np.float32)
ids = np.fromfile(toks_p, dtype=np.uint32).tolist()
tok = Tokenizer.from_file(tokj)
rows = len(sc)
lo, hi = 4, rows - 1800
order = lo + np.argsort(-sc[lo:hi])
kept = np.zeros(rows, bool)
n = 0
for rr in order:
    for j in range(max(lo, rr - 2), min(hi, rr + 3)):
        if n >= 248:
            break
        if not kept[j]:
            kept[j] = True
            n += 1
    if n >= 248:
        break
fact = [i for i in range(rows - 1) if tok.decode([ids[i]]) in ("653", "269")]
for lang, t in T.items():
    r, b = fonts(lang)
    W, H = 1200 * S, 780 * S
    im = Image.new("RGB", (W, H), SURF)
    d = ImageDraw.Draw(im)
    d.text((60 * S, 36 * S), t["x_title"], font=b(32), fill=INK)
    d.text((60 * S, 84 * S), t["x_sub"], font=r(20), fill=INK2)
    x0, x1, y0, y1 = 110 * S, 1140 * S, 480 * S, 165 * S
    vmin, vmax = -20.0, 32.0
    X = lambda i: x0 + (x1 - x0) * i / rows
    Y = lambda v: y0 - (y0 - y1) * (v - vmin) / (vmax - vmin)
    for v in range(-20, 31, 10):
        d.line([(x0, Y(v)), (x1, Y(v))], fill=LINE, width=S)
        d.text((x0 - 12 * S, Y(v)), str(v), font=r(16), fill=MUTE, anchor="rm")
    d.text((x0 - 70 * S, y1 - 34 * S), t["x_y"], font=r(17), fill=MUTE)
    d.text(((x0 + x1) / 2, y0 + 14 * S), t["x_x"], font=r(17), fill=MUTE, anchor="ma")
    # window: rows that stay exact anyway
    d.rectangle((X(hi), y1, x1, y0), fill=TINT)
    d.text(((X(hi) + x1) / 2, y0 - 10 * S), t["x_win"], font=r(15), fill=BLUE, anchor="md")
    step = 2
    for i in range(lo, hi, step):
        if kept[i:i + step].any():
            continue
        v = float(np.clip(sc[i:i + step].max(), vmin, vmax))
        d.ellipse((X(i) - 1.6 * S, Y(v) - 1.6 * S, X(i) + 1.6 * S, Y(v) + 1.6 * S), fill="#cfcdc5")
    for i in np.nonzero(kept)[0]:
        v = float(np.clip(sc[i], vmin, vmax))
        d.ellipse((X(i) - 3 * S, Y(v) - 3 * S, X(i) + 3 * S, Y(v) + 3 * S), fill=BLUE)
    for i in fact:
        v = float(sc[i])
        d.ellipse((X(i) - 8 * S, Y(v) - 8 * S, X(i) + 8 * S, Y(v) + 8 * S), outline=ORANGE, width=3 * S)
    fx, fy = X(fact[0]), Y(float(sc[fact].max()))
    d.line([(fx + 10 * S, fy), (fx + 70 * S, fy - 6 * S)], fill=ORANGE, width=2 * S)
    d.text((fx + 76 * S, fy - 26 * S), t["x_fact"], font=b(18), fill=ORANGE)
    d.text((fx + 76 * S, fy - 2 * S), t["x_rank"], font=r(17), fill=ORANGE)
    lx, ly = 300 * S, 136 * S
    d.ellipse((lx, ly - 5 * S, lx + 10 * S, ly + 5 * S), fill=BLUE)
    d.text((lx + 20 * S, ly), t["x_kept"], font=r(17), fill=INK, anchor="lm")
    lx += 20 * S + d.textlength(t["x_kept"], font=r(17)) + 40 * S
    d.ellipse((lx, ly - 5 * S, lx + 10 * S, ly + 5 * S), fill="#cfcdc5")
    d.text((lx + 20 * S, ly), t["x_rest"], font=r(17), fill=INK, anchor="lm")
    # steps
    sy = 530 * S
    bw = (1140 - 60 - 3 * 20) / 4 * S
    for k, (h, s) in enumerate(t["steps"]):
        bx = 60 * S + k * (bw + 20 * S)
        d.rounded_rectangle((bx, sy, bx + bw, sy + 132 * S), radius=10 * S, fill=TINT if k != 2 else "#d8e8fb")
        d.text((bx + 16 * S, sy + 14 * S), f"{k + 1}. {h}", font=b(20), fill=BLUE if k == 2 else INK)
        words, line, yy = s.split(" "), "", sy + 50 * S
        for w in (words if lang != "zh" else re.findall(r"[A-Za-z0-9.,]+|.", s)):
            trial = (line + (" " if line and lang != "zh" else "") + w)
            if d.textlength(trial, font=r(17)) > bw - 32 * S and w not in "，。：；、）":
                d.text((bx + 16 * S, yy), line, font=r(17), fill=INK2)
                yy += 24 * S
                line = w
            else:
                line = trial
        d.text((bx + 16 * S, yy), line, font=r(17), fill=INK2)
    d.text((60 * S, 714 * S), t["x_ans"], font=b(20), fill=INK)
    save(im, f"exact_rows_{lang}.png")
print("figures written to", out)
