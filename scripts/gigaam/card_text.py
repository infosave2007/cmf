#!/usr/bin/env python3
"""The GigaAM-v4 page (docs/hf/GigaAM-v4-cmf.md) in English, Russian and Chinese, from the measured numbers.

    card_text.py FIGURES.json WER_DIR FILES.json [--ref REF_SPEED.txt] [--install "cargo install …"] > card.md

FIGURES.json: what bench_rows.py writes ({stand: {file tag: [rows]}}).
WER_DIR: the wer/<rev>_<set>_<field>.json files wer.py writes (sets far, ruls, crowd; fields e2e, raw).
FILES.json: {"<tag>": {"name": file name on the Hub, "size": bytes, "sha256": hex}} for the published files.
REF_SPEED.txt: lines of ref_bench.py (`<rev> ref <device> <f32|f16> s<SECONDS>: warm <s> | <x> real time`).

Every number on the page comes from these files; the prose around them is here.
"""
import argparse, json, os, re, sys

REPO = "infosave/GigaAM-v4-cmf"
ASSETS = f"https://huggingface.co/{REPO}/resolve/main/assets"
SETS = [("far", "Golos Farfield", 1916), ("crowd", "Golos Crowd", 9994), ("ruls", "Russian LibriSpeech", 1352)]
STANDS = ["cuda", "vulkan", "metal", "cpu_m4", "cpu"]

ap = argparse.ArgumentParser()
ap.add_argument("figures")
ap.add_argument("wer_dir")
ap.add_argument("files")
ap.add_argument("--ref")
ap.add_argument("--install", default="cargo install cortiq-cli")
ap.add_argument("--version-note", default="")
a = ap.parse_args()

fig = json.load(open(a.figures))
files = json.load(open(a.files))
wer = {}
for rev in ("rnnt", "ctc"):
    for s, _, _ in SETS:
        for field in ("e2e", "raw"):
            p = os.path.join(a.wer_dir, f"{rev}_{s}_{field}.json")
            if os.path.exists(p):
                wer[(rev, s, field)] = json.load(open(p))
ref_speed = {}
if a.ref:
    for line in open(a.ref, errors="replace"):
        m = re.match(r"\s*(\w+) ref (\w+) (f32|f16) s(\d+): warm ([\d.]+)", line)
        if m:
            ref_speed[(m.group(1), m.group(2), m.group(3), int(m.group(4)))] = float(m.group(5))

TAGS = [t for t in ("rnnt-q8_2f", "ctc-q8_2f", "rnnt-q4tp", "ctc-q4tp") if t in files]
MAIN = "rnnt-q8_2f"
main_name = files[MAIN]["name"]


def n(lang, v, d=None):
    """A number in the language's notation (decimal comma in Russian)."""
    if d is None:
        d = 0 if abs(v) >= 100 else 1 if abs(v) >= 10 else 2 if abs(v) >= 1 else 3
    s = f"{v:.{d}f}"
    return s.replace(".", ",") if lang == "ru" else s


def grp(v):
    """An integer with thin groups of three: 13 261."""
    return f"{v:,}".replace(",", " ")


def dur(lang, secs):
    names = {"en": ("s", "min", "h"), "ru": ("с", "мин", "ч"), "zh": ("秒", "分钟", "小时")}[lang]
    if secs < 60:
        return f"{secs} {names[0]}"
    if secs < 3600:
        return f"{secs // 60} {names[1]}"
    return f"{secs // 3600} {names[2]}"


def row(stand, tag, secs):
    return next((r for r in fig.get(stand, {}).get(tag, []) if r["secs"] == secs), None)


def best(stand, tag):
    rows = fig.get(stand, {}).get(tag, [])
    return max(rows, key=lambda r: r["x"]) if rows else None


def w(rev, s, field, name):
    e = wer.get((rev, s, field), {}).get("files", {}).get(name)
    return e


def gb(b):
    return b / 1e9 if b >= 1e9 else None


def size(lang, b):
    return f"{n(lang, b / 1e6, 0)} MB" if lang == "en" else (f"{n(lang, b / 1e6, 0)} МБ" if lang == "ru" else f"{n(lang, b / 1e6, 0)} MB")


def speed_table(lang, stand, T):
    tags = [t for t in TAGS if fig.get(stand, {}).get(t)]
    if not tags:
        return ""
    mem_key = "vram_gb" if any(r.get("vram_gb") for t in tags for r in fig[stand][t]) else "rss_gb"
    lens = sorted({r["secs"] for t in tags for r in fig[stand][t]})
    label = {"rnnt-q8_2f": "RNN-T `q8_2f`", "ctc-q8_2f": "CTC `q8_2f`", "rnnt-q4tp": "RNN-T `q4tp`", "ctc-q4tp": "CTC `q4tp`"}
    shown = [t for t in tags if t.endswith("q8_2f")] or tags
    head = [T["rec"]] + [f"{T['time']} ({label[t]})" for t in shown] + [f"{T['x']} ({label[t]})" for t in shown] + [T[mem_key]]
    out = ["| " + " | ".join(head) + " |", "|---:" * len(head) + "|"]
    for s in lens:
        cells = [dur(lang, s)]
        rows = [row(stand, t, s) for t in shown]
        cells += [n(lang, r["time"]) if r else "—" for r in rows]
        cells += [n(lang, r["x"], 0) if r else "—" for r in rows]
        mem = next((r[mem_key] for r in rows if r and r.get(mem_key)), None)
        cells.append(n(lang, mem, 1) if mem else "—")
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def wer_table(lang, T, field):
    names = [("reference", T["ref_row"]), ("f16_exact", T["f16_row"])] + [(t.split("-")[1], None) for t in TAGS]
    out = []
    for rev in ("rnnt", "ctc"):
        rev_tags = [t for t in TAGS if t.startswith(rev)]
        if not rev_tags or not any((rev, s, field) in wer for s, _, _ in SETS):
            continue
        head = [T["file"]] + [f"{name}, {grp(clips)}" for _, name, clips in SETS] + [T["same"]]
        out += [f"**{'RNN-T' if rev == 'rnnt' else 'CTC'}**", "", "| " + " | ".join(head) + " |", "|---|" + "---:|" * (len(head) - 1)]
        rows = [("reference", T["ref_row"]), ("f16_exact", T["f16_row"])] + [(t.split("-", 1)[1], f"`{files[t]['name']}`") for t in rev_tags]
        for key, title in rows:
            cells, same, total = [title], 0, 0
            for s, _, clips in SETS:
                e = w(rev, s, field, key)
                cells.append(n(lang, e["wer"], 2) if e else "—")
                if e and "identical" in e:
                    same += e["identical"]
                    total += wer[(rev, s, field)]["clips"]
            cells.append(f"{grp(same)} / {grp(total)}" if total else "—")
            out.append("| " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out).rstrip()


TXT = {
    "en": {
        "rec": "recording", "time": "time, s", "x": "× real time", "vram_gb": "peak card memory, GB", "rss_gb": "peak process memory, GB",
        "file": "file", "same": "transcripts identical to the reference", "ref_row": "reference: the authors' PyTorch code, float32",
        "f16_row": "CMF, the released half-precision weights (not uploaded)",
    },
    "ru": {
        "rec": "запись", "time": "время, с", "x": "× реального времени", "vram_gb": "пик памяти карты, ГБ", "rss_gb": "пик памяти процесса, ГБ",
        "file": "файл", "same": "транскрипты, совпавшие с эталоном", "ref_row": "эталон: код авторов на PyTorch, float32",
        "f16_row": "CMF, исходные веса половинной точности (не выложен)",
    },
    "zh": {
        "rec": "录音", "time": "耗时，秒", "x": "倍实时", "vram_gb": "显存峰值，GB", "rss_gb": "进程内存峰值，GB",
        "file": "文件", "same": "与参考实现完全一致的转写", "ref_row": "参考：作者的 PyTorch 代码，float32",
        "f16_row": "CMF，原始半精度权重（未上传）",
    },
}


def facts(lang):
    """The measured facts the opening bullets and the sections quote."""
    f = {}
    for stand in STANDS:
        for tag in TAGS:
            for secs in (30, 1800, 7200):
                r = row(stand, tag, secs)
                if r:
                    f[(stand, tag, secs)] = r
    return f


F = facts("en")


def pick(stand, tag, *lens):
    for s in lens:
        if (stand, tag, s) in F:
            return s, F[(stand, tag, s)]
    return None, None


def emit(lang):
    T = TXT[lang]
    o = []
    P = o.append
    c30 = F.get(("cuda", MAIN, 30))
    cl_s, cl = pick("cuda", MAIN, 1800, 7200, 300)
    v30 = F.get(("vulkan", MAIN, 30))
    vl_s, vl = pick("vulkan", MAIN, 1800, 7200, 300)
    m30 = F.get(("metal", MAIN, 30))
    ml_s, ml = pick("metal", MAIN, 1800, 300)
    p30 = F.get(("cpu_m4", MAIN, 30))
    pl_s, pl = pick("cpu_m4", MAIN, 1800, 300)
    x30 = F.get(("cpu", MAIN, 30))
    we = {s: (w("rnnt", s, "e2e", "q8_2f"), w("rnnt", s, "e2e", "reference")) for s, _, _ in SETS}
    exact = [(w("rnnt", s, "e2e", "f16_exact"), wer.get(("rnnt", s, "e2e"), {}).get("clips", 0)) for s, _, _ in SETS]
    exact_same = sum(e["identical"] for e, _ in exact if e)
    exact_total = sum(c for e, c in exact if e)
    ref30 = ref_speed.get(("rnnt", "cuda", "f32", 30))
    refl = ref_speed.get(("rnnt", "cuda", "f32", cl_s)) if cl_s else None
    refh = ref_speed.get(("rnnt", "cuda", "f16", cl_s)) if cl_s else None
    ms = lambda v: n(lang, v * 1000, 0)
    if lang == "en":
        P("# GigaAM-v4 — CMF\n")
        P("[GigaAM-v4](https://huggingface.co/ai-sage/GigaAM-v4) is a speech recognition model for Russian and English: a")
        P("238 M-parameter Zapformer encoder with two recognition heads on top — formatted text with case, punctuation and")
        P("digits, and plain lower-case words — as RNN-T or CTC. Here it is packaged as single")
        P("[CMF](https://github.com/infosave2007/cmf) files for `cortiq`, a Rust inference engine with no Python and no ML")
        P("framework: NVIDIA GPUs through CUDA or Vulkan, AMD and Intel GPUs through Vulkan/DX12, Apple silicon through Metal,")
        P("and the CPU. A recording of any length goes in: each file carries the model's own voice-activity network, which")
        P("cuts long audio into segments of up to 30 s. Measured:\n")
        if cl and c30:
            mem = f", in {n(lang, cl['vram_gb'], 1)} GB of the card" if cl.get("vram_gb") else ""
            P(f"- **NVIDIA L4 (24 GB), CUDA:** {dur(lang, cl_s)} of speech in {n(lang, cl['time'])} s — {n(lang, cl['x'], 0)} times faster than real")
            P(f"  time; a 30 s clip in {ms(c30['time'])} ms{mem}.")
        if vl and v30:
            P(f"- **The same card over Vulkan:** {dur(lang, vl_s)} in {n(lang, vl['time'])} s ({n(lang, vl['x'], 0)}× real time), a 30 s clip in {ms(v30['time'])} ms.")
        if m30 and p30:
            tail = f"; {dur(lang, ml_s)} in {n(lang, ml['time'])} s and {n(lang, pl['time'])} s" if ml and pl else ""
            P(f"- **MacBook Air M4 (24 GB):** a 30 s clip in {ms(m30['time'])} ms through Metal and {ms(p30['time'])} ms on the CPU alone{tail}.")
        if all(q and r for q, r in we.values()):
            pairs = ", ".join(f"{n(lang, we[s][0]['wer'], 2)} % against {n(lang, we[s][1]['wer'], 2)} % on {name}" for s, name, _ in SETS)
            P(f"- **Accuracy:** the word error rate of the 8-bit file is the reference implementation's: {pairs}.")
            if exact_total:
                P(f"  With the released half-precision weights {grp(exact_same)} of {grp(exact_total)} transcripts are the reference's, character for character.")
        if refl and refh and cl:
            P(f"- **Against the authors' PyTorch code on the same L4:** the same {dur(lang, cl_s)} take it {n(lang, refl)} s in float32 and {n(lang, refh)} s with")
            P(f"  its half-precision encoder; `cortiq` takes {n(lang, cl['time'])} s.")
        P("\n[English](#quick-start) · [Русский](#документация-на-русском) · [中文](#中文文档)\n")
        P("## Quick start\n")
        P("```bash")
        P(a.install)
        P(f"hf download {REPO} {main_name} --local-dir .")
        P(f"cortiq transcribe {main_name} speech.wav")
        P("```\n")
        if a.version_note:
            P(a.version_note + "\n")
        P("Prebuilt binaries for Linux, macOS and Windows are on the [releases page](https://github.com/infosave2007/cmf/releases).")
        P("Add `--features cuda` to the install for the CUDA backend (`CMF_GPU=cuda`). WAV is read directly; mp3, flac, ogg,")
        P("m4a and the rest go through `ffmpeg` when it is on the PATH. Any sample rate and channel count: the audio is")
        P("mixed down to mono at 16 kHz.\n")
        P("## Files\n")
        P("| file | heads | weights | size | WER, Golos Farfield | WER, Golos Crowd | WER, Russian LibriSpeech |")
        P("|---|---|---|---:|---:|---:|---:|")
        for t in TAGS:
            rev, q = t.split("-", 1)
            cells = [f"`{files[t]['name']}`", "RNN-T" if rev == "rnnt" else "CTC", ("8-bit, **default**" if t == MAIN else "8-bit") if q == "q8_2f" else "4-bit", size(lang, files[t]["size"])]
            for s, _, _ in SETS:
                e = w(rev, s, "e2e", q)
                cells.append(n(lang, e["wer"], 2) if e else "—")
            P("| " + " | ".join(cells) + " |")
        for rev in ("rnnt", "ctc"):
            if any(t.startswith(rev) for t in TAGS):
                cells = [f"reference: the authors' PyTorch code, {'RNN-T' if rev == 'rnnt' else 'CTC'}", "", "float32", "—"]
                for s, _, _ in SETS:
                    e = w(rev, s, "e2e", "reference")
                    cells.append(n(lang, e["wer"], 2) if e else "—")
                P("| " + " | ".join(cells) + " |")
        P("")
        P("Word error rate of the formatted transcript (`e2e` head) on the test sets' own clips, the reference run on the")
        P("same clips, both sides through the reference package's text normalization (lower case, ё → е, punctuation")
        P("removed; numbers the model writes as digits count against a reference that spells them out, for every row")
        P("alike). RNN-T is the more accurate of the two and the upstream default; CTC decodes in one pass and is the")
        P("faster one. Every file holds both heads, the mel front end's filter bank and the voice-activity network.")
        P("Not here: the `ssl` revision (an encoder without a recognition head) and the streaming modes (160 / 320 / 640 ms")
        P("of look-ahead) — these files run the full-context mode.\n")
        P("## Performance\n")
        P("One recording a run, `cortiq transcribe --stats`, the second pass in one process (weights already on the device):")
        P("the time from samples to transcript — voice-activity cut, features, encoder, decoding — and the recording's")
        P("length divided by it. The recordings are test-set clips laid end to end. Card memory is the peak `nvidia-smi`")
        P("shows above the idle card. Read each backend's section on its own.\n")
        sec = {
            "cuda": ("### CUDA · NVIDIA L4", "Stand: NVIDIA L4 (24 GB), Google Colab (12 vCPU), the CUDA backend (`CMF_GPU=cuda`). The voice-activity network, the convolutional front end and the encoder run on the card (TF32 products); the host computes the mel features and the greedy decoding.", "The files on CUDA"),
            "vulkan": ("### Vulkan · NVIDIA L4", "Stand: the same NVIDIA L4 (24 GB), over Vulkan (the default backend off macOS).", "The files on Vulkan"),
            "metal": ("### Metal · MacBook Air M4", "Stand: MacBook Air M4 (24 GB), fanless, Metal (the default backend on macOS). The encoder runs on the GPU; memory is the process's (the Mac's memory is shared with the GPU).", "The files on Metal"),
            "cpu_m4": ("### CPU · MacBook Air M4", "Stand: MacBook Air M4 (24 GB), fanless, CPU only (`CMF_GPU=0`): Accelerate's matrix units do the products.", "The files on the M4's CPU"),
            "cpu": ("### CPU · Xeon, 12 vCPU", "Stand: the L4 instance's host, Intel Xeon 2.2 GHz, 12 vCPU, no GPU (`CMF_GPU=0`).", "The files on a server CPU"),
        }
        for stand in STANDS:
            tbl = speed_table(lang, stand, T)
            if not tbl:
                continue
            P(sec[stand][0] + "\n")
            P(sec[stand][1] + "\n")
            P(f"![{sec[stand][2]}]({ASSETS}/speed_{stand}_en.png)\n")
            P(tbl + "\n")
        P("## Usage\n")
        P("```bash")
        P(f"cortiq transcribe {main_name} speech.wav                 # the formatted transcript")
        P(f"cortiq transcribe {main_name} speech.wav --head raw      # lower-case words, no punctuation")
        P(f"cortiq transcribe {main_name} speech.wav --timestamps    # every word with its start and end")
        P(f"cortiq transcribe {main_name} a.wav b.mp3 c.ogg          # several files, one load")
        P(f"cortiq transcribe {main_name} talk.wav --json --head both  # text, segments, word times, timings")
        P("```\n")
        P("A recording of up to 30 s is one segment. A longer one is cut where the voice-activity network finds pauses")
        P("(segments of at most 30 s, as the reference package's `transcribe_longform` cuts them) and the segments'")
        P("transcripts are joined; `--vad` applies the cut to a short clip too. `--stats` prints the time of each stage.\n")
        P("### OpenAI-compatible server\n")
        P("```bash")
        P(f"cortiq serve {main_name} --port 8080\n")
        P("curl http://localhost:8080/v1/audio/transcriptions -F file=@speech.wav")
        P('curl http://localhost:8080/v1/audio/transcriptions -F file=@talk.mp3 \\')
        P('  -F response_format=verbose_json -F "timestamp_granularities[]=word"')
        P("curl http://localhost:8080/v1/audio/transcriptions -F file=@talk.mp3 -F response_format=srt")
        P("```\n")
        P("`POST /v1/audio/transcriptions` takes the multipart form the OpenAI clients send (`file`, `response_format`:")
        P("`json`, `text`, `verbose_json`, `srt`, `vtt`; `timestamp_granularities[]`: `segment`, `word`), or JSON with")
        P('`file` as a `data:` / `http(s)` URL or `{"data": <base64>}`. `head=raw` asks for the plain-words transcript.')
        P("`/v1/models` and `/healthz` describe the loaded file.\n")
        P("### Settings\n")
        P("| variable | effect |")
        P("|---|---|")
        P("| `CMF_GPU=cuda` / `1` / `0` | the CUDA backend / the default GPU backend (Vulkan, Metal) / the CPU |")
        P("| `CMF_CUDA_PREC=exact` | plain float32 products on CUDA instead of TF32: the CPU's transcript, character for character, at about half the encoder speed |")
        P("| `CMF_GIGAAM_GEMM=f32` | the same on Vulkan |")
        P("| `CMF_GIGAAM_DENSE=1` | CPU: keep float copies of the weights (about 0.95 GB) — a short clip then pays no unpacking |")
        P("| `CMF_THREADS=N` | CPU threads |")
        P("")
        P("## Accuracy\n")
        P("Word error rate, %, of each file and of the reference implementation on the same clips with the same text")
        P("normalization; the last column counts the clips whose normalized transcript is the reference's exactly.")
        P("CMF rows ran on CUDA at its default precision.\n")
        P("Formatted transcript (`e2e` head):\n")
        P(wer_table(lang, T, "e2e") + "\n")
        P("Plain words (`raw` head):\n")
        P(wer_table(lang, T, "raw") + "\n")
        P("## Verify the download\n")
        P("```bash")
        P(f"cortiq verify {main_name}")
        P(f"cortiq info {main_name}")
        P("```\n")
        P("| file | sha256 |")
        P("|---|---|")
        for t in TAGS:
            P(f"| `{files[t]['name']}` | `{files[t]['sha256']}` |")
        P("")
        P("Weights: [MIT](https://huggingface.co/ai-sage/GigaAM-v4/blob/main/LICENSE), from")
        P("[ai-sage/GigaAM-v4](https://huggingface.co/ai-sage/GigaAM-v4). The model card's notes on intended use and")
        P("limitations apply to these files as well.\n")
        P("---\n")
    elif lang == "ru":
        P("## Документация на русском\n")
        P("[GigaAM-v4](https://huggingface.co/ai-sage/GigaAM-v4) — модель распознавания речи для русского и английского:")
        P("Zapformer-энкодер на 238 млн параметров и две головы распознавания над ним — оформленный текст с регистром,")
        P("пунктуацией и цифрами и простой текст строчными словами — в вариантах RNN-T и CTC. Здесь она упакована в")
        P("отдельные файлы [CMF](https://github.com/infosave2007/cmf) для `cortiq` — движка на Rust без Python и")
        P("ML-фреймворков: GPU NVIDIA через CUDA или Vulkan, AMD и Intel через Vulkan/DX12, Apple silicon через Metal, а")
        P("также CPU. На вход идёт запись любой длины: в каждом файле лежит собственная сеть голосовой активности модели,")
        P("которая режет длинное аудио на сегменты до 30 с. Измерено:\n")
        if cl and c30:
            mem = f", в {n(lang, cl['vram_gb'], 1)} ГБ карты" if cl.get("vram_gb") else ""
            P(f"- **NVIDIA L4 (24 ГБ), CUDA:** {dur(lang, cl_s)} речи за {n(lang, cl['time'])} с — в {n(lang, cl['x'], 0)} раз быстрее реального")
            P(f"  времени; клип 30 с — за {ms(c30['time'])} мс{mem}.")
        if vl and v30:
            P(f"- **Та же карта через Vulkan:** {dur(lang, vl_s)} за {n(lang, vl['time'])} с ({n(lang, vl['x'], 0)}× реального времени), клип 30 с — за {ms(v30['time'])} мс.")
        if m30 and p30:
            tail = f"; {dur(lang, ml_s)} — за {n(lang, ml['time'])} с и {n(lang, pl['time'])} с" if ml and pl else ""
            P(f"- **MacBook Air M4 (24 ГБ):** клип 30 с за {ms(m30['time'])} мс через Metal и за {ms(p30['time'])} мс на одном CPU{tail}.")
        if all(q and r for q, r in we.values()):
            pairs = ", ".join(f"{n(lang, we[s][0]['wer'], 2)} % против {n(lang, we[s][1]['wer'], 2)} % на {name}" for s, name, _ in SETS)
            P(f"- **Точность:** доля ошибок по словам у 8-битного файла та же, что у эталонной реализации: {pairs}.")
            if exact_total:
                P(f"  С исходными весами половинной точности {grp(exact_same)} из {grp(exact_total)} транскриптов совпадают с эталоном символ в символ.")
        if refl and refh and cl:
            P(f"- **Против кода авторов на PyTorch на той же L4:** те же {dur(lang, cl_s)} занимают у него {n(lang, refl)} с во float32 и {n(lang, refh)} с с")
            P(f"  энкодером половинной точности; у `cortiq` — {n(lang, cl['time'])} с.")
        P("\nБыстрый старт:\n")
        P("```bash")
        P(a.install)
        P(f"hf download {REPO} {main_name} --local-dir .")
        P(f"cortiq transcribe {main_name} speech.wav")
        P("```\n")
        P("WAV читается напрямую; mp3, flac, ogg, m4a и прочее — через `ffmpeg`, если он есть в PATH. Частота и число")
        P("каналов любые: звук сводится в моно 16 кГц. Таблица файлов с WER — в разделе [Files](#files): RNN-T точнее и")
        P("стоит у авторов по умолчанию, CTC декодирует за один проход и быстрее. В каждом файле обе головы и сеть")
        P("голосовой активности. Не выложены ревизия `ssl` (энкодер без головы распознавания) и потоковые режимы")
        P("(160 / 320 / 640 мс заглядывания вперёд): файлы работают в режиме полного контекста.\n")
        P("### Скорость\n")
        P("Одна запись за прогон, `cortiq transcribe --stats`, второй проход в одном процессе (веса уже на устройстве):")
        P("время от отсчётов до транскрипта — нарезка по голосовой активности, признаки, энкодер, декодирование — и длина")
        P("записи, делённая на него. Записи — клипы тестовых наборов, уложенные подряд. Память карты — пик по `nvidia-smi`")
        P("сверх простаивающей карты. Каждый раздел читается сам по себе.\n")
        sec = {
            "cuda": ("**CUDA · NVIDIA L4**", "Стенд: NVIDIA L4 (24 ГБ), Google Colab (12 vCPU), бэкенд CUDA (`CMF_GPU=cuda`). Сеть голосовой активности, свёрточный вход и энкодер считаются на карте (произведения TF32); на хосте — mel-признаки и жадное декодирование."),
            "vulkan": ("**Vulkan · NVIDIA L4**", "Стенд: та же NVIDIA L4 (24 ГБ), Vulkan (бэкенд по умолчанию вне macOS)."),
            "metal": ("**Metal · MacBook Air M4**", "Стенд: MacBook Air M4 (24 ГБ), без вентилятора, Metal (бэкенд по умолчанию на macOS). Энкодер считается на GPU; память — всего процесса (память Mac общая с GPU)."),
            "cpu_m4": ("**CPU · MacBook Air M4**", "Стенд: MacBook Air M4 (24 ГБ), без вентилятора, только CPU (`CMF_GPU=0`): произведения считают матричные блоки Accelerate."),
            "cpu": ("**CPU · Xeon, 12 vCPU**", "Стенд: хост инстанса с L4, Intel Xeon 2,2 ГГц, 12 vCPU, без GPU (`CMF_GPU=0`)."),
        }
        for stand in STANDS:
            tbl = speed_table(lang, stand, T)
            if not tbl:
                continue
            P(sec[stand][0] + "\n")
            P(sec[stand][1] + "\n")
            P(f"![{sec[stand][0].strip('*')}]({ASSETS}/speed_{stand}_ru.png)\n")
            P(tbl + "\n")
        P("### Использование\n")
        P("```bash")
        P(f"cortiq transcribe {main_name} speech.wav                 # оформленный транскрипт")
        P(f"cortiq transcribe {main_name} speech.wav --head raw      # строчные слова без пунктуации")
        P(f"cortiq transcribe {main_name} speech.wav --timestamps    # каждое слово с началом и концом")
        P(f"cortiq transcribe {main_name} a.wav b.mp3 c.ogg          # несколько файлов за одну загрузку")
        P(f"cortiq transcribe {main_name} talk.wav --json --head both  # текст, сегменты, времена слов, тайминги")
        P("```\n")
        P("Запись до 30 с — один сегмент. Более длинная режется там, где сеть голосовой активности находит паузы")
        P("(сегменты не длиннее 30 с, как режет `transcribe_longform` из пакета авторов), транскрипты сегментов")
        P("соединяются; `--vad` включает нарезку и для короткого клипа. `--stats` печатает время каждой стадии.\n")
        P("Сервер с API в формате OpenAI:\n")
        P("```bash")
        P(f"cortiq serve {main_name} --port 8080\n")
        P("curl http://localhost:8080/v1/audio/transcriptions -F file=@speech.wav")
        P('curl http://localhost:8080/v1/audio/transcriptions -F file=@talk.mp3 \\')
        P('  -F response_format=verbose_json -F "timestamp_granularities[]=word"')
        P("```\n")
        P("`POST /v1/audio/transcriptions` принимает multipart-форму клиентов OpenAI (`file`, `response_format`: `json`,")
        P("`text`, `verbose_json`, `srt`, `vtt`; `timestamp_granularities[]`: `segment`, `word`) или JSON, где `file` —")
        P('URL `data:` / `http(s)` либо `{"data": <base64>}`. `head=raw` запрашивает транскрипт простыми словами.\n')
        P("Настройки: `CMF_GPU=cuda` / `1` / `0` — бэкенд CUDA / GPU по умолчанию (Vulkan, Metal) / CPU;")
        P("`CMF_CUDA_PREC=exact` (на Vulkan `CMF_GIGAAM_GEMM=f32`) — обычные float32-произведения вместо TF32: транскрипт")
        P("CPU символ в символ при примерно вдвое меньшей скорости энкодера; `CMF_GIGAAM_DENSE=1` — на CPU держать")
        P("веса во float (около 0,95 ГБ), короткий клип тогда не платит за распаковку; `CMF_THREADS=N` — потоки CPU.\n")
        P("### Точность\n")
        P("Доля ошибок по словам, %, у каждого файла и у эталонной реализации на тех же клипах с той же нормализацией")
        P("текста (нижний регистр, ё → е, без пунктуации; числа, которые модель пишет цифрами, считаются ошибкой против")
        P("разметки прописью — одинаково для всех строк). Последний столбец — клипы, чей нормализованный транскрипт")
        P("совпал с эталоном полностью. Строки CMF сняты на CUDA с точностью по умолчанию.\n")
        P("Оформленный транскрипт (голова `e2e`):\n")
        P(wer_table(lang, T, "e2e") + "\n")
        P("Простые слова (голова `raw`):\n")
        P(wer_table(lang, T, "raw") + "\n")
        P("### Проверка загрузки\n")
        P(f"`cortiq verify {main_name}` сверяет хеши секций файла; sha256 файлов — в таблице")
        P("[Verify the download](#verify-the-download). Веса: [MIT](https://huggingface.co/ai-sage/GigaAM-v4/blob/main/LICENSE),")
        P("из [ai-sage/GigaAM-v4](https://huggingface.co/ai-sage/GigaAM-v4); замечания карточки модели о назначении и")
        P("ограничениях относятся и к этим файлам.\n")
        P("---\n")
    else:
        P("## 中文文档\n")
        P("[GigaAM-v4](https://huggingface.co/ai-sage/GigaAM-v4) 是面向俄语和英语的语音识别模型：2.38 亿参数的 Zapformer")
        P("编码器，上面有两个识别头——带大小写、标点和数字的规范文本，以及纯小写词——分为 RNN-T 和 CTC 两种版本。这里把它")
        P("打包成单个 [CMF](https://github.com/infosave2007/cmf) 文件，由 `cortiq` 运行：这是一个用 Rust 写的推理引擎，不依赖")
        P("Python 和任何机器学习框架，NVIDIA 显卡走 CUDA 或 Vulkan，AMD 和 Intel 显卡走 Vulkan/DX12，Apple 芯片走 Metal，也可以")
        P("只用 CPU。任意长度的录音都可以直接输入：每个文件都带有模型自己的语音活动检测网络，把长音频切成不超过 30 秒的片段。")
        P("实测：\n")
        if cl and c30:
            mem = f"，占用显存 {n(lang, cl['vram_gb'], 1)} GB" if cl.get("vram_gb") else ""
            P(f"- **NVIDIA L4（24 GB），CUDA：** {dur(lang, cl_s)}语音用时 {n(lang, cl['time'])} 秒，是实时的 {n(lang, cl['x'], 0)} 倍；30 秒的片段用时 {ms(c30['time'])} 毫秒{mem}。")
        if vl and v30:
            P(f"- **同一张卡走 Vulkan：** {dur(lang, vl_s)}用时 {n(lang, vl['time'])} 秒（{n(lang, vl['x'], 0)} 倍实时），30 秒的片段用时 {ms(v30['time'])} 毫秒。")
        if m30 and p30:
            tail = f"；{dur(lang, ml_s)}分别用时 {n(lang, ml['time'])} 秒和 {n(lang, pl['time'])} 秒" if ml and pl else ""
            P(f"- **MacBook Air M4（24 GB）：** 30 秒的片段，Metal 用时 {ms(m30['time'])} 毫秒，只用 CPU 用时 {ms(p30['time'])} 毫秒{tail}。")
        if all(q and r for q, r in we.values()):
            pairs = "，".join(f"{name} 上 {n(lang, we[s][0]['wer'], 2)} % 对 {n(lang, we[s][1]['wer'], 2)} %" for s, name, _ in SETS)
            P(f"- **准确率：** 8 位文件的词错误率与参考实现相同：{pairs}。")
            if exact_total:
                P(f"  使用原始半精度权重时，{grp(exact_total)} 条转写中有 {grp(exact_same)} 条与参考实现逐字符一致。")
        if refl and refh and cl:
            P(f"- **与同一张 L4 上作者的 PyTorch 代码相比：** 同样的{dur(lang, cl_s)}，它用 float32 需要 {n(lang, refl)} 秒，用半精度编码器需要 {n(lang, refh)} 秒；`cortiq` 用时 {n(lang, cl['time'])} 秒。")
        P("\n快速开始：\n")
        P("```bash")
        P(a.install)
        P(f"hf download {REPO} {main_name} --local-dir .")
        P(f"cortiq transcribe {main_name} speech.wav")
        P("```\n")
        P("WAV 直接读取；mp3、flac、ogg、m4a 等在 PATH 中有 `ffmpeg` 时通过它解码。采样率和声道数不限：音频会混成 16 kHz 单声道。")
        P("带 WER 的文件表见 [Files](#files)：RNN-T 更准确，也是上游的默认版本；CTC 一遍解码，速度更快。每个文件都包含两个识别头")
        P("和语音活动检测网络。未提供的有：`ssl` 版本（没有识别头的编码器）和流式模式（160 / 320 / 640 毫秒前瞻）——这些文件以")
        P("全上下文模式运行。\n")
        P("### 性能\n")
        P("每次运行一段录音，`cortiq transcribe --stats`，同一进程内的第二遍（权重已在设备上）：从采样到转写的时间——语音活动")
        P("切分、特征、编码器、解码——以及录音长度与它之比。录音由测试集的片段首尾相接而成。显存为 `nvidia-smi` 显示的、高于")
        P("空闲状态的峰值。各后端的小节请分别阅读。\n")
        sec = {
            "cuda": ("**CUDA · NVIDIA L4**", "测试平台：NVIDIA L4（24 GB），Google Colab（12 vCPU），CUDA 后端（`CMF_GPU=cuda`）。语音活动检测网络、卷积前端和编码器在显卡上运行（TF32 乘法）；主机计算 mel 特征和贪心解码。"),
            "vulkan": ("**Vulkan · NVIDIA L4**", "测试平台：同一张 NVIDIA L4（24 GB），走 Vulkan（macOS 之外的默认后端）。"),
            "metal": ("**Metal · MacBook Air M4**", "测试平台：MacBook Air M4（24 GB），无风扇，Metal（macOS 上的默认后端）。编码器在 GPU 上运行；内存为整个进程的占用（Mac 的内存与 GPU 共享）。"),
            "cpu_m4": ("**CPU · MacBook Air M4**", "测试平台：MacBook Air M4（24 GB），无风扇，仅 CPU（`CMF_GPU=0`）：乘法由 Accelerate 的矩阵单元完成。"),
            "cpu": ("**CPU · Xeon，12 vCPU**", "测试平台：L4 实例的主机，Intel Xeon 2.2 GHz，12 vCPU，不使用 GPU（`CMF_GPU=0`）。"),
        }
        for stand in STANDS:
            tbl = speed_table(lang, stand, T)
            if not tbl:
                continue
            P(sec[stand][0] + "\n")
            P(sec[stand][1] + "\n")
            P(f"![{sec[stand][0].strip('*')}]({ASSETS}/speed_{stand}_zh.png)\n")
            P(tbl + "\n")
        P("### 使用\n")
        P("```bash")
        P(f"cortiq transcribe {main_name} speech.wav                 # 规范文本")
        P(f"cortiq transcribe {main_name} speech.wav --head raw      # 小写词，无标点")
        P(f"cortiq transcribe {main_name} speech.wav --timestamps    # 每个词的起止时间")
        P(f"cortiq transcribe {main_name} a.wav b.mp3 c.ogg          # 多个文件，只加载一次")
        P(f"cortiq transcribe {main_name} talk.wav --json --head both  # 文本、片段、词时间、各阶段耗时")
        P("```\n")
        P("不超过 30 秒的录音是一个片段。更长的录音在语音活动检测网络找到停顿的位置切开（片段不超过 30 秒，与作者软件包的")
        P("`transcribe_longform` 的切法相同），各片段的转写再拼接起来；`--vad` 让短片段也走切分。`--stats` 打印各阶段的耗时。\n")
        P("OpenAI 兼容服务：\n")
        P("```bash")
        P(f"cortiq serve {main_name} --port 8080\n")
        P("curl http://localhost:8080/v1/audio/transcriptions -F file=@speech.wav")
        P('curl http://localhost:8080/v1/audio/transcriptions -F file=@talk.mp3 \\')
        P('  -F response_format=verbose_json -F "timestamp_granularities[]=word"')
        P("```\n")
        P("`POST /v1/audio/transcriptions` 接受 OpenAI 客户端发送的 multipart 表单（`file`；`response_format`：`json`、`text`、")
        P("`verbose_json`、`srt`、`vtt`；`timestamp_granularities[]`：`segment`、`word`），也接受 JSON，其中 `file` 为 `data:` /")
        P('`http(s)` URL 或 `{"data": <base64>}`。`head=raw` 请求纯词转写。\n')
        P("设置：`CMF_GPU=cuda` / `1` / `0` —— CUDA 后端 / 默认 GPU 后端（Vulkan、Metal）/ CPU；`CMF_CUDA_PREC=exact`（Vulkan 上为")
        P("`CMF_GIGAAM_GEMM=f32`）—— 用普通 float32 乘法代替 TF32：得到与 CPU 逐字符相同的转写，编码器速度约减半；")
        P("`CMF_GIGAAM_DENSE=1` —— 在 CPU 上保留浮点权重副本（约 0.95 GB），短片段不再为解包付出时间；`CMF_THREADS=N` —— CPU 线程数。\n")
        P("### 准确率\n")
        P("各文件与参考实现在相同片段上、使用相同文本规范化（小写、ё → е、去掉标点；模型写成数字的数，相对于用文字标注的")
        P("参考文本计为错误，对每一行都一样）得到的词错误率（%）。最后一列是规范化后的转写与参考实现完全一致的片段数。")
        P("CMF 各行在 CUDA 上以默认精度测得。\n")
        P("规范文本（`e2e` 头）：\n")
        P(wer_table(lang, T, "e2e") + "\n")
        P("纯词（`raw` 头）：\n")
        P(wer_table(lang, T, "raw") + "\n")
        P("### 校验下载\n")
        P(f"`cortiq verify {main_name}` 校验文件各部分的哈希；各文件的 sha256 见")
        P("[Verify the download](#verify-the-download) 表。权重：[MIT](https://huggingface.co/ai-sage/GigaAM-v4/blob/main/LICENSE)，")
        P("来自 [ai-sage/GigaAM-v4](https://huggingface.co/ai-sage/GigaAM-v4)；模型卡片中关于用途和局限的说明同样适用于这些文件。")
    return "\n".join(o)


front = f"""---
license: mit
library_name: cortiq
base_model: ai-sage/GigaAM-v4
base_model_relation: quantized
pipeline_tag: automatic-speech-recognition
tags:
  - cmf
  - cortiq
  - quantized
  - speech-recognition
  - asr
  - rnnt
  - ctc
  - cuda
  - vulkan
  - metal
language:
  - ru
  - en
---

"""
sys.stdout.write(front + emit("en") + "\n" + emit("ru") + "\n" + emit("zh") + "\n")
