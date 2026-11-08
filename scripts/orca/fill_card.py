#!/usr/bin/env python3
# Fill README.tmpl.md with the measured numbers -> README.md. fill_card.py TMPL OUT [FACTS_JSON] [ANSWER_MD]
import json, sys
tmpl, out = sys.argv[1], sys.argv[2]
facts = json.load(open(sys.argv[3])) if len(sys.argv) > 3 and sys.argv[3] != "-" else None
answer = open(sys.argv[4], encoding="utf-8").read().strip() if len(sys.argv) > 4 else "{{ANSWER}}"
s = open(tmpl, encoding="utf-8").read()
SHA = "6ff230576c09672102f1f0dbc6020721bc1bad548aa4df9bb31e3c7d1251d5f7"
PR = "https://github.com/infosave2007/cmf/pull/99"
rows = [("1 000", "9.1", "8.9", "27.4", "29.9", "14.1", "14.1"),
        ("4 000", "37.0", "36.7", "25.8", "26.7", "14.5", "14.5"),
        ("16 000", None, "159", None, "26.7", None, "14.8"),
        ("32 000", None, "324", None, "26.3", None, "14.8"),
        ("64 000", None, "652", None, "26.0", None, "14.8")]
def bench(lang):
    H = {"en": ("prompt, tokens", "first token, s (standard)", "first token, s (**turbo**)", "decode, tok/s (standard)",
                "decode, tok/s (**turbo**)", "peak VRAM, GB (standard)", "peak VRAM, GB (**turbo**)", "does not fit"),
         "ru": ("промпт, токенов", "первый токен, с (обычный)", "первый токен, с (**турбо**)", "декод, ток/с (обычный)",
                "декод, ток/с (**турбо**)", "пик памяти карты, ГБ (обычный)", "пик памяти карты, ГБ (**турбо**)", "не помещается"),
         "zh": ("提示 token 数", "首个 token，秒（普通）", "首个 token，秒（turbo）", "解码 tok/s（普通）",
                "解码 tok/s（turbo）", "显存峰值 GB（普通）", "显存峰值 GB（turbo）", "放不下")}[lang]
    dec = (lambda v: v.replace(".", ",")) if lang == "ru" else (lambda v: v)
    b = (lambda v: f"**{dec(v)}**") if lang != "zh" else dec
    lines = ["| " + " | ".join(H[:7]) + " |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        std = lambda v: dec(v) if v else ("—" if r[1] else H[7])
        lines.append(f"| {r[0]} | {dec(r[1]) if r[1] else H[7]} | {b(r[2])} | {dec(r[3]) if r[3] else '—'} | {b(r[4])} | "
                     f"{dec(r[5]) if r[5] else '—'} | {b(r[6])} |")
    note = {"en": "The first token includes moving the weights to the card (about 5 s); the prompt itself is read at "
                  "98–112 tok/s. `bench` decodes with the MTP drafts on. The standard file runs out of device memory "
                  "from 16 000 tokens on this 16 GB card. Host RAM of the turbo file while a long prompt is read: 4.1 GB "
                  "at 16 000, 7.1 GB at 32 000, 13.3 GB at 64 000 (`--o1-spill` moves it to a disk).",
            "ru": "Первый токен включает перенос весов на карту (около 5 с); сам промпт читается со скоростью "
                  "98–112 ток/с. `bench` декодирует с черновиками MTP. Обычный файл с 16 000 токенов не помещается "
                  "на эту карту 16 ГБ. RAM хоста у турбо-файла при чтении длинного промпта: 4,1 ГБ на 16 000, 7,1 ГБ "
                  "на 32 000, 13,3 ГБ на 64 000 (`--o1-spill` переносит их на диск).",
            "zh": "首个 token 包含把权重搬到显卡上的时间（约 5 秒）；提示本身的读取速度为 98–112 tok/s。`bench` 解码时开启 MTP 草稿。"
                  "在这块 16 GB 显卡上，普通文件从 16 000 token 起显存不足。读取长提示时 turbo 文件占用的主机内存：16 000 时 4.1 GB，"
                  "32 000 时 7.1 GB，64 000 时 13.3 GB（`--o1-spill` 可移到磁盘）。"}[lang]
    return "\n".join(lines) + "\n\n" + note
P = [("code", "a Rust function with tests", "функция на Rust с тестами", "带测试的 Rust 函数", "13.5", "28.2", "76"),
     ("review", "a security review of a Flask handler", "ревью безопасности обработчика Flask", "Flask 处理函数的安全审查", "13.4", "26.5", "70"),
     ("logs", "triage of auth log lines", "разбор строк auth-лога", "认证日志分诊", "13.3", "23.6", "64"),
     ("prose", "two paragraphs of prose", "два абзаца прозы", "两段散文", "13.5", "19.6", "48"),
     ("ru", "a Russian answer", "ответ на русском", "俄语回答", "13.5", "19.5", "48")]
def prompts(lang):
    H = {"en": ("prompt", "no drafts, tok/s", "MTP drafts, tok/s", "drafts kept"),
         "ru": ("промпт", "без черновиков, ток/с", "черновики MTP, ток/с", "принято черновиков"),
         "zh": ("提示", "无草稿 tok/s", "MTP 草稿 tok/s", "草稿接受率")}[lang]
    i = {"en": 1, "ru": 2, "zh": 3}[lang]
    dec = (lambda v: v.replace(".", ",")) if lang == "ru" else (lambda v: v)
    lines = ["| " + " | ".join(H) + " |", "|---|---:|---:|---:|"]
    for p in P:
        lines.append(f"| {p[i]} | {dec(p[4])} | {dec(p[5])} | {p[6]} % |" if lang != "zh" else f"| {p[i]} | {p[4]} | {p[5]} | {p[6]}% |")
    return "\n".join(lines)
REAL = {"en": "A real answer: a one-paragraph summary of 100 000 characters of Wikipedia (about 24 600 tokens), correct "
              "(the Second Battle of Naktong Bulge), first token 252 s, decode 20.8 tok/s.",
        "ru": "Реальный ответ: краткое содержание 100 000 символов Википедии одним абзацем (около 24 600 токенов), "
              "верное (Вторая битва за выступ Нактонг), первый токен через 252 с, декод 20,8 ток/с.",
        "zh": "真实回答：用一段话概括 100 000 字符的维基百科（约 24 600 token），回答正确（第二次洛东江突出部战役），首个 token 252 秒，解码 20.8 tok/s。"}
VER = {"en": f"Measured with cortiq built from [infosave2007/cmf#99]({PR}), which comes in the next release; cortiq "
             "0.8.63 runs the same file and reads prompts at about 75 tok/s instead of 98–112.",
       "ru": f"Замерено на cortiq, собранном из [infosave2007/cmf#99]({PR}), он войдёт в следующий релиз; cortiq 0.8.63 "
             "запускает тот же файл и читает промпт примерно со скоростью 75 ток/с вместо 98–112.",
       "zh": f"测量所用的 cortiq 由 [infosave2007/cmf#99]({PR}) 构建，将包含在下一个版本中；cortiq 0.8.63 也能运行同一文件，"
             "但读取提示约为 75 tok/s，而不是 98–112。"}
if facts:
    F = {"en": f"Facts planted at random places of 100 000 characters (a pass key, an access code, a password, a room number): "
               f"{facts['turbo']} with the exact rows; a far field without them finds {facts['noexact']}.",
         "ru": f"Факты в случайных местах 100 000 символов (ключ, код доступа, пароль, номер комнаты): {facts['turbo_ru']} с "
               f"точными строками; дальнее поле без них находит {facts['noexact_ru']}.",
         "zh": f"在 100 000 字符的随机位置插入事实（密钥、访问码、密码、房间号）：有精确行时 {facts['turbo_zh']}；没有精确行的远场为 {facts['noexact_zh']}。"}
else:
    F = {"en": "{{FACTS_EN}}", "ru": "{{FACTS_RU}}", "zh": "{{FACTS_ZH}}"}
rep = {
    "{{SHA}}": SHA, "{{T64_VRAM}}": "14.8", "{{T64_VRAM_RU}}": "14,8", "{{T64_DEC}}": "26.0", "{{T64_DEC_RU}}": "26,0",
    "{{STD64}}": "runs out of device memory from 16 000 tokens on", "{{STD64_RU}}": "не помещается на карту уже с 16 000 токенов",
    "{{STD64_ZH}}": "从 16 000 token 起就放不进显卡",
    "{{VERSION_EN}}": VER["en"], "{{VERSION_RU}}": VER["ru"], "{{VERSION_ZH}}": VER["zh"],
    "{{BENCH_TABLE_EN}}": bench("en"), "{{BENCH_TABLE_RU}}": bench("ru"), "{{BENCH_TABLE_ZH}}": bench("zh"),
    "{{PROMPT_TABLE_EN}}": prompts("en"), "{{PROMPT_TABLE_RU}}": prompts("ru"), "{{PROMPT_TABLE_ZH}}": prompts("zh"),
    "{{REAL_EN}}": REAL["en"], "{{REAL_RU}}": REAL["ru"], "{{REAL_ZH}}": REAL["zh"],
    "{{PPL_W}}": "8.89", "{{PPL_BASE}}": "8.79", "{{PPL_C}}": "3.79", "{{PPL_W_RU}}": "8,89", "{{PPL_BASE_RU}}": "8,79",
    "{{PPL_C_RU}}": "3,79", "{{FACTS_EN}}": F["en"], "{{FACTS_RU}}": F["ru"], "{{FACTS_ZH}}": F["zh"], "{{ANSWER}}": answer,
}
for k, v in rep.items():
    s = s.replace(k, v)
open(out, "w", encoding="utf-8").write(s)
left = sorted(set(__import__("re").findall(r"\{\{[A-Z0-9_]+\}\}", s)))
print("left:", left)
