---
license: other
license_name: lfm1.0
license_link: LICENSE
library_name: cortiq
base_model: LiquidAI/d1-Omni-2B
base_model_relation: quantized
pipeline_tag: any-to-any
tags:
  - cmf
  - cortiq
  - quantized
  - q4tp
  - liquid
  - lfm3
  - decision
  - classification
  - extraction
  - system-one
  - audio
  - omni
  - vulkan
  - metal
language:
  - en
  - ar
  - zh
  - fr
  - de
  - it
  - ja
  - ko
  - pt
  - es
  - vi
  - th
  - id
  - hi
  - ru
  - pl
---

# d1-Omni-2B — CMF

[Liquid AI d1-Omni-2B](https://huggingface.co/LiquidAI/d1-Omni-2B) — the
omni-modal 2.2B decision model (System One) on LFM3-Omni-2B: named questions
over a state — text, JSON, pictures or audio clips — answered from one forward
pass as typed answers with calibrated probabilities, with no generated tokens —
packaged as single [CMF](https://github.com/infosave2007/cmf) files. They run
with `cortiq`, a Rust inference engine with no Python and no ML framework:
NVIDIA, AMD and Intel GPUs through Vulkan/DX12, Apple silicon through Metal,
and the CPU.

[English](#quick-start) · [Русский](#документация-на-русском) · [中文](#中文说明)

## Quick start

```bash
cargo install cortiq-cli
hf download infosave/d1-Omni-2B-cmf d1-omni-2b-q8_2f.cmf --local-dir .
cortiq system-one d1-omni-2b-q8_2f.cmf --audio call.wav \
  --questions '{"booking": {"type": "noul", "instructions": "Is the speaker making a reservation?"},
                "people": {"type": "number", "instructions": "For how many people?", "range": [0, 10], "step": 1},
                "kind": {"type": "choice", "instructions": "What is the request about?",
                         "criteria": {"restaurant": "A table or a meal", "travel": "Flights or hotels",
                                      "support": "A product problem"}}}'
```

For a 3-second clip saying *"Please book a table for four people at 7.30
tonight"* the answers are `booking` 0.95, `people` 4, `kind` `restaurant`.

Use cortiq 0.8.50 or later; prebuilt binaries for Linux, macOS and Windows are
on the [releases page](https://github.com/infosave2007/cmf/releases). `--state`
and `--questions` take text or `@file` (a `.json` state file is read as JSON);
`--image` adds a picture and `--audio` a clip (WAV, other formats through
`ffmpeg`; path, data URL or HTTP(S) URL; both repeatable, pictures first).

## Files

| file | weights | vision and audio encoders | size | answers = original | recommended for |
|---|---|---|---:|---:|---|
| `d1-omni-2b-q8_2f.cmf` | 8-bit | 16-bit | 2,633,401,242 B (2.63 GB) | 22 of 22 | **default** |
| `d1-omni-2b-q4tp.cmf` | 4-bit | 16-bit | 1,803,686,810 B (1.80 GB) | 21 of 22 | the smallest memory footprint |

*Answers = original*: the choice, level, span or value of every question of
five test requests (a support ticket, an invoice, a JSON record, a picture
with counts and boxes, and a spoken request) against the bf16 checkpoint run
in float32 by Transformers; the largest probability difference is 0.005
(`q8_2f`) and 0.13 (`q4tp`, whose one different answer — `billing` against
`shipping` — is a near tie in the original too: 0.52 against 0.46). The
calibration the release ships (`calibration.json`) is in the file and applied
to every answer.

## Performance

One request at a time through `cortiq serve`, after the first call:

| hardware | file | 1 question (43 tokens) | 3 questions (123 tokens) | 3 questions about a 3.3 s clip |
|---|---|---:|---:|---:|
| NVIDIA A40 (Vulkan) | `q4tp` | 40–50 ms | 64 ms | 0.47 s |
| NVIDIA A40 (Vulkan) | `q8_2f` | 55–66 ms | 75 ms | 0.67 s |
| Apple M4 (Metal) | `q4tp` | 85 ms | 157 ms | 0.30 s |

The whole forward pass — every layer over the packed state and questions, then
the vocabulary head over the answer slots — runs in one GPU submission. The
vision and audio encoders run in float32 on the CPU before it (a 640×480
picture: 0.5 s on the M4, ~3 s on the A40 host's CPU).

## Questions

| `type` | question | answer |
|---|---|---|
| `noul` | yes or no | `noul`: P(yes) |
| `choice` | one of the options `criteria` names (`{name: description}`) | `choice`, `confidence`, `probabilities` |
| `choice` with `"multiple": true` | any of the options | `choices`, `probabilities` (each option's own P(yes)) |
| `score` | one of 2 to 10 ordered levels (`criteria` lowest first) | `score` (the median level), `confidence`, `probabilities`, `legend` |
| `text`, `number`, `date` | a value read from the state | the value, `span`, `found` |
| `number` with `range` (and `step`) | an estimate | `number` (the median), `interval` (p10, p90) |
| `box`, `boxes` (`max`) | one object, or every instance, in the picture | `box` (top, bottom, left, right as fractions), `found`, `confidence`, or `boxes` and `count` |

A value or a box whose `found` is under 0.5 is `null`. Up to 16 questions are
read together after one state (the release's set size); more run in further
passes over the same state.

### Server

```bash
cortiq serve d1-omni-2b-q8_2f.cmf --port 8080

curl http://localhost:8080/v1/system_one \
  -H "Content-Type: application/json" \
  -d '{"state": null,
       "questions": {"people": {"type": "number", "instructions": "For how many people?", "range": [0, 10], "step": 1}},
       "media": [{"type": "audio", "path": "call.wav"}]}'
```

`media` takes pictures (a path, an HTTP(S) or `data:` URL, or
`{"type": "image_url", "image_url": {"url": ...}}`) and audio clips
(`{"type": "audio", "path": ...}` or `{"type": "input_audio", "input_audio":
{"data": <base64>, "format": "wav"}}`). Video, spoken questions and batched
states (`system_one_batch`) are not supported yet.

## Hardware

**NVIDIA / AMD / Intel (Vulkan, DX12).** Both files fit any card with 3 GB of
free memory; on headless Linux install `libvulkan1 libglvnd0 libegl1 libgl1
libglx0` and set `XDG_RUNTIME_DIR=/tmp`. **Apple silicon:** the Metal backend
is the default. **CPU only:** `CMF_GPU=0`.

## Verify the download

```bash
sha256sum -c d1-omni-2b-q8_2f.cmf.sha256
cortiq verify d1-omni-2b-q8_2f.cmf
```

| file | sha256 |
|---|---|
| `d1-omni-2b-q8_2f.cmf` | `c8906f4f5914f1dfcd80aad0c84aa87ce13eb8fe396c200885289aedaed7bca0` |
| `d1-omni-2b-q4tp.cmf` | `80e8ec2ea5a95233e031e6aa5379df029bfe4df274def9064cf8d5278bd19fa9` |

## Reproduce

```bash
cortiq convert --model LiquidAI/d1-Omni-2B --quant q8_2f \
  --output d1-omni-2b-q8_2f.cmf --also-output q4tp=d1-omni-2b-q4tp.cmf
```

Scripts (Transformers reference, answer parity, latency): [scripts/lfm3](https://github.com/infosave2007/cmf/tree/master/scripts/lfm3).

## License

These files are a modified form of Liquid AI's d1-Omni-2B: the weights were
converted to CMF and the language model quantized (8-bit `q8_2f`, 4-bit `q4tp`)
by cortiq. They remain under the LFM Open License v1.0, a copy of which is in
this repository ([LICENSE](LICENSE)); among its terms, commercial use by an
entity with annual revenue of US$10 million or more is not licensed. See the
[upstream card](https://huggingface.co/LiquidAI/d1-Omni-2B).

---

## Документация на русском

[d1-Omni-2B](https://huggingface.co/LiquidAI/d1-Omni-2B) от Liquid AI —
омни-модальная модель решений (System One) на 2.2 млрд параметров поверх
LFM3-Omni-2B: именованные вопросы к состоянию — тексту, JSON, изображениям или
аудиоклипам — получают типизированные ответы с откалиброванными вероятностями
за один прямой проход, без генерации токенов, — в виде файлов
[CMF](https://github.com/infosave2007/cmf) для движка `cortiq` на Rust без Python.

```bash
cargo install cortiq-cli
hf download infosave/d1-Omni-2B-cmf d1-omni-2b-q8_2f.cmf --local-dir .
cortiq system-one d1-omni-2b-q8_2f.cmf --audio call.wav \
  --questions '{"people": {"type": "number", "instructions": "Сколько человек?", "range": [0, 10], "step": 1}}'
```

| файл | веса | размер | ответы = оригинал | назначение |
|---|---|---:|---:|---|
| `d1-omni-2b-q8_2f.cmf` | 8 бит | 2.63 ГБ | 22 из 22 | **по умолчанию** |
| `d1-omni-2b-q4tp.cmf` | 4 бита | 1.80 ГБ | 21 из 22 | минимум памяти |

Нужен cortiq 0.8.50 или новее. Типы вопросов те же, что у d1-2B (`noul`,
`choice`, `score`, `text`/`number`/`date`, оценки, `box`/`boxes`); `--image` и
`--audio` добавляют изображения и аудиоклипы. Весь проход идёт одной отправкой
на GPU: на NVIDIA A40 один вопрос — 40–50 мс (`q4tp`), на Apple M4 (Metal) —
85 мс; три вопроса о 3-секундном клипе — 0.30 с на M4. `cortiq serve` отдаёт
`POST /v1/system_one` (`state`, `questions`, `media` с изображениями и аудио).
Видео, вопросы голосом и пакетные запросы пока не поддерживаются. Файлы —
изменённая форма d1-Omni-2B от Liquid AI по LFM Open License v1.0 (копия —
[LICENSE](LICENSE)); коммерческое использование организацией с годовой
выручкой от 10 млн долларов США лицензией не покрывается.

## 中文说明

[d1-Omni-2B](https://huggingface.co/LiquidAI/d1-Omni-2B) 是 Liquid AI 基于
LFM3-Omni-2B 的 2.2B 全模态决策模型（System One）：对一个状态（文本、JSON、图片或
音频片段）提出若干具名问题，一次前向即可得到带校准概率的类型化答案，不生成任何
token。本仓库提供可由 Rust 推理引擎 `cortiq`（0.8.50 及以上）加载的 CMF 文件。

```bash
cargo install cortiq-cli
hf download infosave/d1-Omni-2B-cmf d1-omni-2b-q8_2f.cmf --local-dir .
cortiq system-one d1-omni-2b-q8_2f.cmf --audio call.wav \
  --questions '{"people": {"type": "number", "instructions": "一共几个人？", "range": [0, 10], "step": 1}}'
```

| 文件 | 权重 | 大小 | 答案与原模型一致 | 适用 |
|---|---|---:|---:|---|
| `d1-omni-2b-q8_2f.cmf` | 8 位 | 2.63 GB | 22/22 | **默认** |
| `d1-omni-2b-q4tp.cmf` | 4 位 | 1.80 GB | 21/22 | 内存最小 |

整个前向在一次 GPU 提交中完成：NVIDIA A40 上单个问题 40–50 ms（`q4tp`），Apple M4
（Metal）上 85 ms；关于 3 秒音频的三个问题在 M4 上 0.30 s。`cortiq serve` 提供
`POST /v1/system_one`（`media` 可含图片与音频）。暂不支持视频、语音提问与批量请求。
这些文件是 Liquid AI d1-Omni-2B 的修改形式，仍受 LFM Open License v1.0 约束（副本见
[LICENSE](LICENSE)）；年收入 1000 万美元及以上的实体进行商业使用不在许可范围内。
