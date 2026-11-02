#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Schemes of the Embryo-56M card in EN/RU/ZH: SVG, then PNG through rsvg-convert.

usage: diagrams.py <out dir>
"""
import os
import subprocess
import sys
from xml.sax.saxutils import escape

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
LINE = "#8d8b84"
GRID = "#e4e2dc"
PANEL = "#f3f2ee"
BOX = "#ffffff"
ANCHOR = "#dfe9f6"
ACCENT = "#2a78d6"
ACCENT_BG = "#e3eefb"
CPU = "#eb6834"

FONT = {
    "en": "Helvetica Neue, Helvetica, Arial, sans-serif",
    "ru": "Helvetica Neue, Helvetica, Arial, sans-serif",
    "zh": "Hiragino Sans GB, Heiti SC, sans-serif",
}


class Svg:
    def __init__(self, w, h, lang):
        self.w, self.h = w, h
        self.o = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" font-family="{FONT[lang]}">',
            "<defs>",
            f'<marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="{LINE}"/></marker>',
            f'<marker id="b" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="{ACCENT}"/></marker>',
            "</defs>",
            f'<rect width="{w}" height="{h}" fill="{SURFACE}"/>',
        ]

    def text(self, x, y, s, size=20, weight=400, anchor="start", fill=INK):
        self.o.append(
            f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{fill}">{escape(s)}</text>'
        )

    def lines(self, x, y, rows, size=20, weight=400, anchor="middle", fill=INK, lead=1.22):
        """Rows centred around y."""
        n = len(rows)
        top = y - (n - 1) * size * lead / 2 + size * 0.35
        for i, r in enumerate(rows):
            self.text(x, round(top + i * size * lead, 1), r, size, weight, anchor, fill)

    def rect(self, x, y, w, h, fill=BOX, stroke=LINE, rx=8, sw=1.5, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.o.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>'
        )

    def box(self, x, y, w, h, rows, size=20, weight=400, fill=BOX, stroke=LINE, ink=INK, rx=8):
        self.rect(x, y, w, h, fill, stroke, rx)
        if isinstance(rows, str):
            rows = [rows]
        self.lines(x + w / 2, y + h / 2, rows, size, weight, "middle", ink)

    def path(self, pts, accent=False, dash=None, head=True, sw=2):
        c = ACCENT if accent else LINE
        d = "M" + " L".join(f"{x} {y}" for x, y in pts)
        m = f' marker-end="url(#{"b" if accent else "a"})"' if head else ""
        da = f' stroke-dasharray="{dash}"' if dash else ""
        self.o.append(f'<path d="{d}" fill="none" stroke="{c}" stroke-width="{sw}"{da}{m}/>')

    def line(self, x1, y1, x2, y2, stroke=GRID, sw=1.5, dash=None):
        da = f' stroke-dasharray="{dash}"' if dash else ""
        self.o.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{sw}"{da}/>')

    def bar(self, x, base, w, h, fill):
        # flat at the baseline, 4 px round at the data end
        r = min(4, h)
        self.o.append(
            f'<path d="M{x} {base} V{base - h + r} Q{x} {base - h} {x + r} {base - h} H{x + w - r} Q{x + w} {base - h} {x + w} {base - h + r} V{base} Z" fill="{fill}"/>'
        )

    def dot(self, x, y, r, fill):
        self.o.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{SURFACE}" stroke-width="2"/>')

    def out(self):
        return "\n".join(self.o) + "\n</svg>\n"


# ---------------------------------------------------------------- strings

A = {
    "en": dict(
        title="One token through the network: all memory of the text sits in blocks of fixed size",
        token="token id", embed="embedding · 384", mixer="mixer", anchor="anchor",
        fnorm="final norm", head1="head: 128 clusters × 256 rows", head2="reads only clusters above the gate",
        then="then", stay="1.7 MB together",
        pa="Mixer: gated delta rule (6 layers of 8)",
        norm="norm", proj=["matrices", "q k v z a b"], delta=["delta", "rule"], gnorm=["gated", "norm"], outm=["matrix", "out"], res=["+ residual"],
        conv=["convolution ring", "kernel 4"], state=["state S", "4 heads × 128 × 128"], rw="reads and rewrites", shift="shifts",
        pb="Anchor: bounded attention (2 layers of 8)",
        qk=["matrices", "q k v"], score=["key scores,", "softmax"], sumv=["sum of", "values"],
        ring=["ring of keys and values, 2048 rows (4 sink keys live in the weights)", "one group of heads reads the last 128 keys, the other all 2048"],
        wr="writes over the oldest", rd="reads keys", rv="reads values",
        pc="Feed-forward: the router picks one expert (every layer)",
        err=["reconstruction error", "in 4 subspaces, rank 32"], small="smallest error",
        exp=["chosen expert", "gate · up · down"], shared=["shared part", "gate · up · down"], sumf=["sum,", "+ residual"],
        total="State of a sequence: 1.7 + 8.4 = 10.1 MB at any context length",
    ),
    "ru": dict(
        title="Один токен через сеть: вся память о тексте лежит в блоках постоянного размера",
        token="номер токена", embed="эмбеддинг · 384", mixer="миксер", anchor="якорь",
        fnorm="финальная норма", head1="голова: 128 кластеров × 256 строк", head2="читает кластеры выше порога",
        then="затем", stay="вместе 1,7 МБ",
        pa="Миксер: дельта-правило с затвором (6 слоёв из 8)",
        norm="норма", proj=["матрицы", "q k v z a b"], delta=["дельта-", "правило"], gnorm=["норма", "с затвором"], outm=["матрица", "выхода"], res=["+ остаток"],
        conv=["кольцо свёртки", "ядро 4"], state=["состояние S", "4 головы × 128 × 128"], rw="читает и переписывает", shift="сдвигает",
        pb="Якорь: ограниченное внимание (2 слоя из 8)",
        qk=["матрицы", "q k v"], score=["оценка", "ключей,", "softmax"], sumv=["сумма", "значений"],
        ring=["кольцо ключей и значений, 2048 строк (4 ключа-стока лежат в весах)", "одна группа голов читает последние 128 ключей, другая все 2048"],
        wr="пишет поверх старых", rd="читает ключи", rv="читает значения",
        pc="Прямой проход: роутер выбирает одного эксперта (каждый слой)",
        err=["ошибка реконструкции", "в 4 подпространствах, ранг 32"], small="наименьшая ошибка",
        exp=["выбранный эксперт", "gate · up · down"], shared=["общая часть", "gate · up · down"], sumf=["сумма,", "+ остаток"],
        total="Состояние последовательности: 1,7 + 8,4 = 10,1 МБ при любой длине контекста",
    ),
    "zh": dict(
        title="一个 token 穿过网络：关于文本的全部记忆都放在固定大小的块中",
        token="token 编号", embed="嵌入 · 384", mixer="混合层", anchor="锚点层",
        fnorm="最终归一化", head1="输出头：128 个簇 × 256 行", head2="只读取高于门限的簇",
        then="然后", stay="合计 1.7 MB",
        pa="混合层：门控 delta 规则（8 层中的 6 层）",
        norm="归一化", proj=["矩阵", "q k v z a b"], delta=["delta", "规则"], gnorm=["门控", "归一化"], outm=["输出", "矩阵"], res=["+ 残差"],
        conv=["卷积环", "核大小 4"], state=["状态 S", "4 头 × 128 × 128"], rw="读取并改写", shift="移位",
        pb="锚点层：有界注意力（8 层中的 2 层）",
        qk=["矩阵", "q k v"], score=["键打分，", "softmax"], sumv=["值", "求和"],
        ring=["键值环，2048 行（4 个汇聚键在权重中）", "一组头读取最近 128 个键，另一组读取全部 2048 个"],
        wr="覆盖最旧的行", rd="读取键", rv="读取值",
        pc="前馈：路由器选择一个专家（每层）",
        err=["重建误差", "4 个子空间，秩 32"], small="误差最小",
        exp=["被选中的专家", "gate · up · down"], shared=["共享部分", "gate · up · down"], sumf=["求和，", "+ 残差"],
        total="一个序列的状态：1.7 + 8.4 = 10.1 MB，与上下文长度无关",
    ),
}

R = {
    "en": dict(
        title="How a message finds its answer: the smallest reconstruction error",
        sub="Resonance routing: every record has an isolated subspace, no classifier",
        msg=["message"],
        d1=["Does it name a key", "from a table?"], d2=["Did an earlier turn", "name a key?"],
        yes="yes", no="no",
        p1=["φ: key tokens", "+ rest of the message"], e1=["error: subspaces of records", "with this key against", "the network's subspace"],
        p2=["φ: key of the earlier", "turn + this message"], e2=["error: follow-up subspaces,", "about the record against", "unrelated messages"],
        p3=["φ: the whole", "message"], e3=["error: two subspaces,", "question for a fact", "against ordinary chat"],
        win="record wins", about="follow-up", unrel="unrelated", fact="fact", chat="chat", net="network wins",
        o1="Record, word for word", o1x="What is valerian used for?",
        o2="The same record", o2x="Is it safe?",
        o3="Honest refusal", o3x="What is 2+2?",
        o4="The network writes", o4x="Write a poem about the sea.",
        file="One file, 91 MB", gen=["sealed network", "59 MB"],
        recs=["records appended next to it, 32 MB; the network's bytes do not change:", "plants 4265 · countries 194 · science 149 · elements 118 · people 56 · trivia 42", "how-to 39 · places 33 · books 32 · about itself 13 · refusal 1"],
    ),
    "ru": dict(
        title="Как сообщение находит ответ: наименьшая ошибка реконструкции",
        sub="Резонансный роутинг: у каждой записи своё изолированное подпространство, без классификатора",
        msg=["сообщение"],
        d1=["Называет ключ", "из таблицы?"], d2=["Ключ был назван", "в прошлой реплике?"],
        yes="да", no="нет",
        p1=["φ: токены ключа", "+ остальное сообщение"], e1=["ошибка: подпространства", "записей с этим ключом", "против подпространства сети"],
        p2=["φ: ключ прошлой", "реплики + сообщение"], e2=["ошибка: подпространства", "уточнений, о записи", "против посторонних реплик"],
        p3=["φ: всё", "сообщение"], e3=["ошибка: два подпространства,", "вопрос о факте", "против обычного чата"],
        win="запись ближе", about="о записи", unrel="посторонняя", fact="факт", chat="чат", net="сеть ближе",
        o1="Запись, слово в слово", o1x="Для чего применяют валериану?",
        o2="Та же запись", o2x="Какие есть противопоказания?",
        o3="Честный отказ", o3x="Сколько будет 2+2?",
        o4="Отвечает сеть", o4x="Напиши стихотворение о море.",
        file="Один файл, 91 МБ", gen=["запечатанная сеть", "59 МБ"],
        recs=["записи дописаны рядом, 32 МБ; байты сети не меняются:", "растения 4265 · страны 194 · наука 149 · элементы 118 · люди 56 · факты 42", "советы 39 · места 33 · книги 32 · о себе 13 · отказ 1"],
    ),
    "zh": dict(
        title="消息如何找到答案：重建误差最小者胜出",
        sub="共振路由：每条记录有独立的子空间，没有分类器",
        msg=["消息"],
        d1=["是否提到", "表中的键？"], d2=["之前的轮次", "提到过键吗？"],
        yes="是", no="否",
        p1=["φ：键的 token", "+ 消息其余部分"], e1=["在含此键的记录子空间", "和网络自身子空间中", "计算误差"],
        p2=["φ：之前轮次的键", "+ 本条消息"], e2=["在追问子空间中计算误差：", "关于该记录", "还是无关消息"],
        p3=["φ：整条", "消息"], e3=["在两个子空间中计算误差：", "询问事实", "还是普通聊天"],
        win="记录胜出", about="关于该记录", unrel="无关", fact="事实", chat="聊天", net="网络胜出",
        o1="记录，逐字回答", o1x="What is valerian used for?",
        o2="同一条记录", o2x="Is it safe?",
        o3="如实拒答", o3x="What is 2+2?",
        o4="由网络生成", o4x="Write a poem about the sea.",
        file="一个文件，91 MB", gen=["封存的网络", "59 MB"],
        recs=["记录追加在旁边，32 MB；网络的字节不变：", "植物 4265 · 国家 194 · 科学 149 · 元素 118 · 人物 56 · 常识 42", "方法 39 · 地点 33 · 书籍 32 · 自我介绍 13 · 拒答 1"],
    ),
}

V = {
    "en": dict(
        title="What this gives: measured numbers",
        t=[("91 MB", ["file with all records"]), ("10.1 MB", ["state at any", "context length"]), ("218 / 223", ["plant facts,", "held-out questions"]), ("38 / 44", ["facts outside the", "records: honest refusal"]), ("0", ["foreign weights: born and", "trained on one laptop GPU"])],
        sp="Decode speed, tokens per second", gpu="GPU", cpu="CPU", short="short prompt", deep="after 2000 tokens",
        m4="Apple M4", rtx="RTX 2000 Ada, EPYC 7443P",
        mem="State of one sequence, MB", ctx="context, tokens",
        kv=["ordinary KV cache, 8 layers", "of the same K/V: calculated"], em="Embryo-56M: measured", mb="MB",
    ),
    "ru": dict(
        title="Что это даёт: измеренные числа",
        t=[("91 МБ", ["файл со всеми записями"]), ("10,1 МБ", ["состояние при любой", "длине контекста"]), ("218 / 223", ["факты о растениях,", "отложенные вопросы"]), ("38 / 44", ["факты вне записей:", "честный отказ"]), ("0", ["чужих весов: рождена и", "обучена на GPU ноутбука"])],
        sp="Скорость декодирования, токенов в секунду", gpu="GPU", cpu="CPU", short="короткий промпт", deep="после 2000 токенов",
        m4="Apple M4", rtx="RTX 2000 Ada, EPYC 7443P",
        mem="Состояние последовательности, МБ", ctx="контекст, токенов",
        kv=["обычный KV-кеш, 8 слоёв", "с теми же K/V: расчёт"], em="Embryo-56M: измерено", mb="МБ",
    ),
    "zh": dict(
        title="带来什么：实测数字",
        t=[("91 MB", ["文件含全部记录"]), ("10.1 MB", ["状态与上下文", "长度无关"]), ("218 / 223", ["植物事实，", "留出的问题"]), ("38 / 44", ["记录之外的事实：", "如实拒答"]), ("0", ["他人权重：在一台笔记本", "GPU 上诞生并训练"])],
        sp="解码速度，token/秒", gpu="GPU", cpu="CPU", short="短提示", deep="2000 token 之后",
        m4="Apple M4", rtx="RTX 2000 Ada，EPYC 7443P",
        mem="一个序列的状态，MB", ctx="上下文，token",
        kv=["普通 KV 缓存，8 层", "相同 K/V：计算值"], em="Embryo-56M：实测", mb="MB",
    ),
}

# ---------------------------------------------------------------- figure 1


def architecture(lang):
    s = A[lang]
    g = Svg(1500, 1090, lang)
    g.text(40, 52, s["title"], 32, 700)
    # the path of a token
    x, w = 40, 330
    y = 90
    g.box(x, y, w, 50, s["token"])
    g.path([(x + w / 2, y + 50), (x + w / 2, y + 72)])
    y = 162
    g.box(x, y, w, 50, s["embed"])
    g.path([(x + w / 2, y + 50), (x + w / 2, y + 72)])
    y = 234
    rows = {}
    for i in range(8):
        anchor = i in (3, 7)
        ry = y + i * 62
        rows[i] = ry
        g.rect(x, ry, w, 54, ANCHOR if anchor else BOX, ACCENT if anchor else LINE)
        g.text(x + 24, ry + 35, str(i), 22, 700, "start", INK2)
        g.text(x + 70, ry + 35, s["anchor"] if anchor else s["mixer"], 22, 700 if anchor else 400)
        if i < 7:
            g.path([(x + w / 2, ry + 54), (x + w / 2, ry + 62)], head=False)
    y = rows[7] + 54
    g.path([(x + w / 2, y), (x + w / 2, y + 22)])
    y += 22
    g.box(x, y, w, 50, s["fnorm"])
    g.path([(x + w / 2, y + 50), (x + w / 2, y + 72)])
    y += 72
    g.rect(x, y, w, 86)
    g.lines(x + w / 2, y + 43, [s["head1"], s["head2"]], 18)

    px, pw = 430, 1030

    def stage(sx, sy, sw_, sh, rows_, gap=None, **kw):
        g.box(sx, sy, sw_, sh, rows_, kw.pop("size", 20), **kw)
        if gap:
            g.path([(sx + sw_, sy + sh / 2), (sx + sw_ + gap, sy + sh / 2)])

    # panel A: mixer
    py = 90
    g.rect(px, py, pw, 310, PANEL, GRID, 12)
    g.text(px + 24, py + 40, s["pa"], 24, 700)
    w1, gap, h1 = 138, 24, 74
    fx = px + 24
    fy = py + 66
    names = [[s["norm"]], s["proj"], s["delta"], s["gnorm"], s["outm"], s["res"]]
    for i, n in enumerate(names):
        stage(fx + i * (w1 + gap), fy, w1, h1, n, gap if i < 5 else None)
    # the state of the mixer
    sy = py + 196
    cx = fx + w1 + gap
    g.path([(cx + w1 / 2, fy + h1), (cx + w1 / 2, sy)], accent=True)
    g.text(cx + w1 / 2 - 10, fy + h1 + 34, s["shift"], 18, 400, "end", INK2)
    g.box(cx - 100, sy, w1 + 110, 84, s["conv"], 20, 400, ACCENT_BG, ACCENT)
    dx = fx + 2 * (w1 + gap)
    g.path([(dx + w1 / 2, fy + h1), (dx + w1 / 2, sy)], accent=True)
    g.text(dx + w1 / 2 + 10, fy + h1 + 34, s["rw"], 18, 400, "start", INK2)
    g.box(dx + 30, sy, 330, 84, s["state"], 20, 400, ACCENT_BG, ACCENT)
    g.text(dx + 376, sy + 52, s["stay"], 24, 700, "start", ACCENT)

    # panel B: anchor
    py = 416
    g.rect(px, py, pw, 310, PANEL, GRID, 12)
    g.text(px + 24, py + 40, s["pb"], 24, 700)
    fy = py + 66
    names = [[s["norm"]], s["qk"], s["score"], s["sumv"], s["outm"], s["res"]]
    for i, n in enumerate(names):
        stage(fx + i * (w1 + gap), fy, w1, h1, n, gap if i < 5 else None, size=18 if len(n) > 2 else 20)
    sy = py + 206
    qx = fx + w1 + gap
    g.rect(qx, sy, 680, 84, ACCENT_BG, ACCENT)
    g.lines(qx + 340, sy + 42, s["ring"], 18)
    g.text(qx + 696, sy + 52, "8.4 MB" if lang != "ru" else "8,4 МБ", 26, 700, "start", ACCENT)
    g.path([(qx + 20, fy + h1), (qx + 20, sy)], accent=True)
    g.text(qx + 30, fy + h1 + 42, s["wr"], 18, 400, "start", INK2)
    k2 = fx + 2 * (w1 + gap) + w1 - 20
    g.path([(k2, sy), (k2, fy + h1)], accent=True)
    g.text(k2 + 10, fy + h1 + 42, s["rd"], 18, 400, "start", INK2)
    k3 = fx + 3 * (w1 + gap) + w1 - 20
    g.path([(k3, sy), (k3, fy + h1)], accent=True)
    g.text(k3 + 10, fy + h1 + 42, s["rv"], 18, 400, "start", INK2)

    # panel C: router
    py = 742
    g.rect(px, py, pw, 250, PANEL, GRID, 12)
    g.text(px + 24, py + 40, s["pc"], 24, 700)
    fy = py + 66
    g.box(fx, fy, w1, 74, s["norm"])
    g.path([(fx + w1, fy + 37), (fx + w1 + 24, fy + 37)])
    ex = fx + w1 + 24
    g.box(ex, fy, 300, 74, s["err"], 19, 400, ACCENT_BG, ACCENT)
    px2 = ex + 300
    g.path([(px2, fy + 37), (px2 + 190, fy + 37)], accent=True)
    g.text(px2 + 95, fy + 25, s["small"], 18, 400, "middle", INK2)
    mx = px2 + 190
    g.box(mx, fy, 210, 74, s["exp"], 18)
    g.box(mx, fy + 92, 210, 74, s["shared"], 18)
    g.path([(fx + w1 / 2, fy + 74), (fx + w1 / 2, fy + 129), (mx, fy + 129)])
    sx = mx + 210 + 24
    g.path([(mx + 210, fy + 37), (sx, fy + 70)])
    g.path([(mx + 210, fy + 129), (sx, fy + 96)])
    g.box(sx, fy + 44, px + pw - 24 - sx, 78, s["sumf"], 18)
    g.text(px + 24, py + 226, "US App. 19/452,440", 18, 400, "start", INK2)

    # from the rows to the panels
    g.path([(x + w, rows[2] + 27), (405, rows[2] + 27), (405, 245), (px, 245)])
    g.path([(x + w, rows[3] + 27), (px, rows[3] + 27)])
    # after the mixer or the anchor comes the feed-forward part
    rail = fx + 6 * (w1 + gap) - gap + 26
    g.path([(rail - 26, 90 + 66 + 37), (rail, 90 + 66 + 37), (rail, 742)])
    g.path([(rail - 26, 416 + 66 + 37), (rail, 416 + 66 + 37)], head=False)
    g.text(rail - 8, 734, s["then"], 18, 400, "end", INK2)

    # what stays
    g.rect(40, 1008, 1420, 62, ACCENT, ACCENT, 8)
    g.text(64, 1048, s["total"], 24, 700, "start", "#ffffff")
    g.text(1436, 1048, "US App. 19/738,763", 20, 400, "end", "#ffffff")
    return g.out()


# ---------------------------------------------------------------- figure 2


def routing(lang):
    s = R[lang]
    g = Svg(1500, 1010, lang)
    g.text(40, 52, s["title"], 32, 700)
    g.text(40, 88, s["sub"], 20, 400, "start", INK2)
    g.text(1460, 88, "US App. 19/452,440", 20, 400, "end", INK2)
    c1, dw = 200, 240
    c2, pw = 490, 230
    c3, ew = 744, 300
    c4, ow, oh = 1184, 276, 92
    rh = 120
    r1, r2, r3 = 120, 320, 520
    end = c3 + ew
    rail = c4 - 22
    # the message
    g.box(40, r1 + 20, 130, 80, s["msg"], 21, 700)
    g.path([(170, r1 + 60), (c1, r1 + 60)])
    # decisions
    g.box(c1, r1, dw, rh, s["d1"], 20, 400, PANEL)
    g.box(c1, r2, dw, rh, s["d2"], 20, 400, PANEL)
    g.path([(c1 + dw / 2, r1 + rh), (c1 + dw / 2, r2)])
    g.text(c1 + dw / 2 + 12, r1 + rh + 48, s["no"], 18, 700, "start", INK2)
    g.path([(c1 + dw / 2, r2 + rh), (c1 + dw / 2, r3 + 60), (c2, r3 + 60)])
    g.text(c1 + dw / 2 + 12, r2 + rh + 40, s["no"], 18, 700, "start", INK2)
    # the three forms of φ and the errors
    for ry, p, e in ((r1, s["p1"], s["e1"]), (r2, s["p2"], s["e2"])):
        g.path([(c1 + dw, ry + 60), (c2, ry + 60)])
        g.text((c1 + dw + c2) / 2, ry + 50, s["yes"], 18, 700, "middle", INK2)
        g.box(c2, ry + 20, pw, 80, p, 19)
        g.path([(c2 + pw, ry + 60), (c3, ry + 60)])
        g.box(c3, ry + 8, ew, 104, e, 18, 400, ACCENT_BG, ACCENT)
    g.box(c2, r3 + 20, pw, 80, s["p3"], 19)
    g.path([(c2 + pw, r3 + 60), (c3, r3 + 60)])
    g.box(c3, r3 + 8, ew, 104, s["e3"], 18, 400, ACCENT_BG, ACCENT)
    # outcomes
    y1, y2, y3, y4 = r1 + 14, r2 + 14, r3 + 14, r3 + 150
    for yy, a, b, acc in ((y1, s["o1"], s["o1x"], True), (y2, s["o2"], s["o2x"], True), (y3, s["o3"], s["o3x"], False), (y4, s["o4"], s["o4x"], False)):
        g.rect(c4, yy, ow, oh, ACCENT if acc else BOX, ACCENT if acc else LINE)
        ink = "#ffffff" if acc else INK
        g.text(c4 + ow / 2, yy + 38, a, 21, 700, "middle", ink)
        g.text(c4 + ow / 2, yy + 68, "«" + b + "»" if lang == "ru" else b, 16, 400, "middle", ink if acc else INK2)
    mid = (end + rail) / 2
    g.path([(end, r1 + 60), (c4, r1 + 60)], accent=True)
    g.text(mid, r1 + 50, s["win"], 17, 400, "middle", INK2)
    g.path([(end, r2 + 60), (c4, r2 + 60)], accent=True)
    g.text(mid, r2 + 50, s["about"], 17, 400, "middle", INK2)
    g.path([(end, r3 + 60), (c4, r3 + 60)])
    g.text(mid, r3 + 50, s["fact"], 17, 400, "middle", INK2)
    # to the network
    cx = end - 70
    g.path([(cx, r3 + 112), (cx, y4 + 60), (c4, y4 + 60)])
    g.text(cx + 10, y4 + 50, s["chat"], 17, 400, "start", INK2)
    # an unrelated message goes on to the third row
    g.path([(c3 + 70, r2 + 112), (c3 + 70, r3 + 8)])
    g.text(c3 + 80, r2 + 150, s["unrel"], 17, 400, "start", INK2)
    # the network's subspace wins, along the gap before the outcomes
    g.path([(end, r1 + 96), (rail, r1 + 96), (rail, y4 + 28), (c4, y4 + 28)], dash="6 5")
    g.text(rail - 10, r1 + 120, s["net"], 17, 400, "end", INK2)
    # the file
    fy = 790
    g.rect(40, fy, 1420, 200, PANEL, GRID, 12)
    g.text(64, fy + 40, s["file"], 24, 700)
    g.text(1436, fy + 40, "US App. 19/731,402", 20, 400, "end", INK2)
    g.box(64, fy + 60, 280, 118, s["gen"], 22, 700, BOX, LINE)
    g.path([(344, fy + 119), (392, fy + 119)])
    g.rect(392, fy + 60, 1044, 118, ACCENT_BG, ACCENT)
    for yy, r in zip((fy + 90, fy + 124, fy + 158), s["recs"]):
        g.text(412, yy, r, 19)
    return g.out()


# ---------------------------------------------------------------- figure 3


def advantages(lang):
    s = V[lang]
    g = Svg(1500, 930, lang)
    g.text(40, 52, s["title"], 32, 700)
    # tiles
    tw, ty, th = 268, 84, 170
    for i, (big, small) in enumerate(s["t"]):
        tx = 40 + i * (tw + 20)
        g.rect(tx, ty, tw, th, PANEL, GRID, 12)
        g.text(tx + 20, ty + 66, big, 42, 700)
        for j, r in enumerate(small):
            g.text(tx + 20, ty + 108 + j * 26, r, 19, 400, "start", INK2)

    # speed: two small multiples on one scale
    top = 290
    g.text(40, top + 28, s["sp"], 24, 700)
    # legend
    lx = 720
    g.rect(lx, top + 12, 18, 18, ACCENT, ACCENT, 3, 0)
    g.text(lx + 26, top + 28, s["gpu"], 20)
    g.rect(lx + 90, top + 12, 18, 18, CPU, CPU, 3, 0)
    g.text(lx + 116, top + 28, s["cpu"], 20)
    base, height, vmax = 830, 460, 1000
    px0, pw0 = 90, 800
    for v in (0, 250, 500, 750, 1000):
        yy = base - height * v / vmax
        g.line(px0, yy, px0 + pw0, yy, GRID if v else LINE)
        g.text(px0 - 10, yy + 6, str(v), 17, 400, "end", INK2)
    data = [(s["m4"], [(650, 470), (535, 379)]), (s["rtx"], [(925, 612), (796, 405)])]
    bw, bgap, ggap = 70, 2, 44
    x = px0 + 28
    for name, groups in data:
        x0 = x
        for gi, (a, b) in enumerate(groups):
            for val, col, off in ((a, ACCENT, 0), (b, CPU, bw + bgap)):
                h = height * val / vmax
                g.bar(x + off, base, bw, h, col)
                g.text(x + off + bw / 2, base - h - 8, str(val), 19, 700, "middle", INK)
            g.text(x + bw + bgap / 2, base + 26, s["short"] if gi == 0 else s["deep"], 16, 400, "middle", INK2)
            x += 2 * bw + bgap + ggap
        g.text((x0 + x - ggap) / 2, base + 58, name, 20, 700, "middle")
        x += 24

    # memory: flat against growing
    mx0, mw, mbase, mheight = 1000, 440, 830, 460
    g.text(960, top + 28, s["mem"], 24, 700)
    ymax, xmax = 600, 32000
    for v in (0, 200, 400, 600):
        yy = mbase - mheight * v / ymax
        g.line(mx0, yy, mx0 + mw, yy, GRID if v else LINE)
        g.text(mx0 - 10, yy + 6, str(v), 17, 400, "end", INK2)
    for v in (0, 8000, 16000, 24000, 32000):
        xx = mx0 + mw * v / xmax
        g.text(xx, mbase + 26, f"{v // 1000}k" if v else "0", 17, 400, "middle", INK2)
    g.text(mx0 + mw / 2, mbase + 58, s["ctx"], 20, 400, "middle", INK2)
    kv_mb = 8 * 2 * 2 * 128 * 4 * xmax / 1e6
    yk = mbase - mheight * kv_mb / ymax
    g.path([(mx0, mbase), (mx0 + mw, yk)], dash="8 6", head=False, sw=2)
    g.dot(mx0 + mw, yk, 6, LINE)
    g.text(mx0 + mw - 6, yk - 16, f"{kv_mb:.0f} {s['mb']}", 22, 700, "end")
    g.text(mx0 + 10, yk + 16, s["kv"][0], 18, 400, "start", INK2)
    g.text(mx0 + 10, yk + 40, s["kv"][1], 18, 400, "start", INK2)
    yf = mbase - mheight * 10.1 / ymax
    g.path([(mx0, yf), (mx0 + mw, yf)], accent=True, head=False, sw=3)
    g.dot(mx0 + mw, yf, 7, ACCENT)
    tenth = "10,1" if lang == "ru" else "10.1"
    g.text(mx0 + mw - 6, yf - 44, f"{tenth} {s['mb']}", 22, 700, "end")
    g.text(mx0 + mw - 6, yf - 18, s["em"], 18, 400, "end", INK2)
    return g.out()


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    for name, fn in (("architecture", architecture), ("routing", routing), ("advantages", advantages)):
        for lang in ("en", "ru", "zh"):
            base = os.path.join(out, f"{name}_{lang}")
            with open(base + ".svg", "w", encoding="utf-8") as f:
                f.write(fn(lang))
            subprocess.run(["rsvg-convert", "-z", "2", "-o", base + ".png", base + ".svg"], check=True)


if __name__ == "__main__":
    main()
