#!/usr/bin/env python3
"""The GigaAM-v4 sections of README.md / README.ru.md / README.zh.md and the CHANGELOG entry, with the measured
numbers filled in from the same data the card uses.

    readme_sections.py FIGURES.json WER_DIR > sections.txt

Prints four blocks separated by lines `=== <name> ===` (en, ru, zh, changelog); they are pasted into the files by
hand (the READMEs are not generated).
"""
import json, os, sys

fig = json.load(open(sys.argv[1]))
wer_dir = sys.argv[2]
MAIN = "rnnt-q8_2f"


def row(stand, secs):
    return next((r for r in fig.get(stand, {}).get(MAIN, []) if r["secs"] == secs), None)


def t(stand, secs=1800):
    r = row(stand, secs)
    return r["time"] if r else None


def x(stand, secs=1800):
    r = row(stand, secs)
    return r["x"] if r else None


def f(v, lang="en"):
    d = 0 if v >= 100 else 1 if v >= 10 else 2
    s = f"{v:.{d}f}"
    return s.replace(".", ",") if lang == "ru" else s


clips = same = 0
for s in ("far", "crowd", "ruls"):
    p = os.path.join(wer_dir, f"rnnt_{s}_e2e.json")
    if os.path.exists(p):
        w = json.load(open(p))
        clips += w["clips"]
        same += w["files"].get("f16_exact", {}).get("identical", 0)

clips, same = f"{clips:,}".replace(",", " "), f"{same:,}".replace(",", " ")
FILE = "GigaAM-v4-rnnt-q8_2f.cmf"
CMD = f"""```sh
cortiq transcribe {FILE} meeting.wav
cortiq transcribe {FILE} meeting.mp3 --timestamps
cortiq transcribe {FILE} a.wav b.wav --json --head both
cortiq serve {FILE} --port 8080   # POST /v1/audio/transcriptions
```"""

print("=== en ===")
print(f"""### GigaAM-v4

`cortiq` runs [GigaAM-v4](https://huggingface.co/ai-sage/GigaAM-v4), a speech recognition model for Russian and
English (a 238 M-parameter Zapformer encoder with RNN-T or CTC heads: a formatted transcript with case,
punctuation and digits, and a plain one), end to end in Rust: the mel front end, the encoder, greedy RNN-T and
CTC decoding with word times, and the model's own voice-activity network, which cuts a recording of any length
into segments. Ready files are on [Hugging Face](https://huggingface.co/infosave/GigaAM-v4-cmf):

{CMD}

On a GPU the encoder stays on the card from the first layer to the last — on CUDA and Vulkan the voice-activity
network and the convolutional front end too. Half an hour of speech takes {f(t('cuda'))} s on an NVIDIA L4 through CUDA
({f(x('cuda'))} times faster than real time), {f(t('vulkan'))} s through Vulkan, {f(t('metal'))} s on a MacBook Air M4 through Metal and
{f(t('cpu_m4'))} s on its CPU. The 8-bit file's word error rate is the reference implementation's on Golos and Russian
LibriSpeech ({clips} clips), and with the released half-precision weights {same} of the transcripts are the
reference's character for character. The server answers `POST /v1/audio/transcriptions` in the OpenAI shape
(`json`, `text`, `verbose_json` with word times, `srt`, `vtt`). `cortiq convert --model <checkpoint dir>
--quant q8_2f | q4tp | f16` packs the `rnnt` or the `ctc` revision of the upstream repository.""")

print("=== ru ===")
print(f"""### GigaAM-v4

`cortiq` целиком на Rust исполняет [GigaAM-v4](https://huggingface.co/ai-sage/GigaAM-v4) — модель распознавания
речи для русского и английского (Zapformer-энкодер на 238 млн параметров с головами RNN-T или CTC: оформленный
транскрипт с регистром, пунктуацией и цифрами и простой): мел-фронтенд, энкодер, жадное декодирование RNN-T и
CTC с временами слов и собственную сеть голосовой активности модели, которая режет запись любой длины на
сегменты. Готовые файлы — на [Hugging Face](https://huggingface.co/infosave/GigaAM-v4-cmf):

{CMD.replace('meeting', 'встреча')}

На GPU энкодер остаётся на карте от первого слоя до последнего, а на CUDA и Vulkan — ещё и сеть голосовой
активности со свёрточным входом. Полчаса речи занимают {f(t('cuda'), 'ru')} с на NVIDIA L4 через CUDA (в {f(x('cuda'), 'ru')} раз быстрее
реального времени), {f(t('vulkan'), 'ru')} с через Vulkan, {f(t('metal'), 'ru')} с на MacBook Air M4 через Metal и {f(t('cpu_m4'), 'ru')} с на его CPU. Доля
ошибок по словам у 8-битного файла та же, что у эталонной реализации, на Golos и Russian LibriSpeech
({clips} клипов), а с исходными весами половинной точности {same} транскриптов совпадают с эталоном символ в
символ. Сервер отвечает на `POST /v1/audio/transcriptions` в формате OpenAI (`json`, `text`, `verbose_json` с
временами слов, `srt`, `vtt`). `cortiq convert --model <каталог чекпоинта> --quant q8_2f | q4tp | f16`
упаковывает ревизию `rnnt` или `ctc` исходного репозитория.""")

print("=== zh ===")
print(f"""## GigaAM-v4：俄语和英语语音识别

`cortiq` 完全用 Rust 运行 [GigaAM-v4](https://huggingface.co/ai-sage/GigaAM-v4)——面向俄语和英语的语音识别模型
（2.38 亿参数的 Zapformer 编码器，带 RNN-T 或 CTC 识别头：一份带大小写、标点和数字的规范转写，一份纯文本转写）：
mel 前端、编码器、带词时间的 RNN-T 与 CTC 贪心解码，以及模型自带的语音活动检测网络，它把任意长度的录音切成片段。
现成文件在 [Hugging Face](https://huggingface.co/infosave/GigaAM-v4-cmf)：

{CMD}

在 GPU 上，编码器从第一层到最后一层都留在显卡上；在 CUDA 和 Vulkan 上，语音活动检测网络和卷积前端也在显卡上。
半小时语音在 NVIDIA L4 上走 CUDA 用时 {f(t('cuda'))} 秒（实时的 {f(x('cuda'))} 倍），走 Vulkan 用时 {f(t('vulkan'))} 秒，在 MacBook Air M4 上走
Metal 用时 {f(t('metal'))} 秒，只用其 CPU 用时 {f(t('cpu_m4'))} 秒。8 位文件在 Golos 和 Russian LibriSpeech（{clips} 个片段）上的词错误率
与参考实现相同；使用原始半精度权重时，{same} 条转写与参考实现逐字符一致。服务端以 OpenAI 的格式响应
`POST /v1/audio/transcriptions`（`json`、`text`、带词时间的 `verbose_json`、`srt`、`vtt`）。
`cortiq convert --model <检查点目录> --quant q8_2f | q4tp | f16` 打包上游仓库的 `rnnt` 或 `ctc` 版本。""")

print("=== changelog ===")
print(f"""- GigaAM-v4 (ai-sage/GigaAM-v4), speech recognition for Russian and English: `cortiq convert` packs the `rnnt`
  and `ctc` revisions (`--quant q8_2f | q4tp | f16`; both recognition heads, the mel filter bank and the
  voice-activity network in one file), `cortiq transcribe` gives the formatted or the plain transcript
  (`--head e2e | raw | both`), word times (`--timestamps`), JSON (`--json`), and cuts a recording over 30 s by voice
  activity as the reference package does; `cortiq serve` on such a file answers `POST /v1/audio/transcriptions`
  in the OpenAI shape (multipart or JSON in; `json`, `text`, `verbose_json`, `srt`, `vtt` out). The encoder runs
  resident on CUDA, Vulkan and Metal (`gpu::gigaam_encode`), with the convolutional front end and the
  voice-activity network on the card too on CUDA and Vulkan; the host path multiplies the int8 weights a
  cache-sized tile at a time through a float GEMM (Accelerate on macOS), and a short batch straight from the
  int8 rows on AVX2. NVIDIA L4: half an hour of speech in {f(t('cuda'))} s through CUDA ({f(x('cuda'))}x real time) and {f(t('vulkan'))} s
  through Vulkan; MacBook Air M4: {f(t('metal'))} s through Metal, {f(t('cpu_m4'))} s on the CPU. Against the authors' PyTorch code on
  {clips} clips of Golos and Russian LibriSpeech the 8-bit file has the same word error rate, and the
  half-precision weights give {same} identical transcripts.""")
