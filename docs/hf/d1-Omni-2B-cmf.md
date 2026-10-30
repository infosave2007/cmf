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
over a state (text, JSON, pictures or audio clips) answered from one forward
pass as typed answers with calibrated
probabilities, with no generated tokens — packaged as single
[CMF](https://github.com/infosave2007/cmf) files. They run with `cortiq`, a
Rust inference engine with no Python and no ML framework: NVIDIA, AMD and Intel
GPUs through Vulkan/DX12, Apple silicon through Metal, and the CPU.

[English](#quick-start) · [Русский](#документация-на-русском) · [中文](#中文说明)

## Quick start

```bash
cargo install cortiq-cli
hf download infosave/d1-Omni-2B-cmf d1-omni-2b-q8_2f.cmf --local-dir .
cortiq system-one d1-omni-2b-q8_2f.cmf \
  --state "I was charged twice and my parcel never arrived. Please refund one charge." \
  --questions '{"refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
                "team": {"type": "choice", "instructions": "Which team should handle this?",
                         "criteria": {"billing": "Charges, refunds, invoices",
                                      "shipping": "Delivery and parcels",
                                      "technical": "App or site faults"}}}'
```

```json
{"answers": {"refund": {"type": "noul", "noul": 0.9999},
             "team": {"type": "choice", "choice": "billing", "confidence": 0.87, "probabilities": {...}}},
 "usage": {"input_tokens": 123, "output_tokens": 0}}
```

Use cortiq 0.8.50 or later; prebuilt binaries for Linux, macOS and Windows are
on the [releases page](https://github.com/infosave2007/cmf/releases). `--state`
and `--questions` take text or `@file` (a `.json` state file is read as JSON);
`--image` adds a picture (path, data URL or HTTP(S) URL, repeatable).

## Files

| file | weights | vision encoder | size | answers = original | recommended for |
|---|---|---|---:|---:|---|
| `d1-omni-2b-q8_2f.cmf` | 8-bit | 16-bit | 2,135,818,786 B (2.14 GB) | 19 of 19 | **default** |
| `d1-omni-2b-q4tp.cmf` | 4-bit | 16-bit | 1,306,104,354 B (1.31 GB) | 19 of 19 (one estimate 2 steps off) | the smallest memory footprint |

*Answers = original*: the choice, level, span or value of every question of
four test requests (a support ticket, an invoice, a JSON record and a picture
with counts and boxes) against the bf16 checkpoint run in float32 by
Transformers; the largest probability difference is 0.014 (`q8_2f`) and 0.07
(`q4tp`). An f16 conversion matches Transformers float32 to about 1e-6 on every
probability. The calibration the release ships (`calibration.json`) is in the
file and applied to every answer.

## Performance

One request at a time through `cortiq serve`, after the first call:

| hardware | file | 1 question (43 tokens) | 3 questions (123 tokens) | 5 questions over a ~200-token state |
|---|---|---:|---:|---:|
| NVIDIA A40 (Vulkan) | `q4tp` | 40 ms | 66 ms | 100–115 ms |
| NVIDIA A40 (Vulkan) | `q8_2f` | 68 ms | 74 ms | 115–127 ms |
| Apple M4 (Metal) | `q4tp` | 85 ms | 160 ms | 255–310 ms |

The whole forward pass — every layer over the packed state and questions,
then the vocabulary head over the answer slots — runs in one GPU submission.
A picture adds its vision encoder (0.5 s for 640×480 on the M4).

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

```bash
cortiq system-one d1-omni-2b-q8_2f.cmf --image cats.jpg \
  --questions '{"cats": {"type": "number", "instructions": "How many cats are there?", "range": [0, 10], "step": 1},
                "remote": {"type": "box", "instructions": "Where is the remote control on the left?"}}'
```

### Server

```bash
cortiq serve d1-omni-2b-q8_2f.cmf --port 8080

curl http://localhost:8080/v1/system_one \
  -H "Content-Type: application/json" \
  -d '{"state": "ACME SA\nInvoice 2026-117\nTotal due: 1 234,50 EUR",
       "questions": {"total": {"type": "number", "instructions": "How much is due?"}},
       "media": []}'
```

`media` takes pictures (a path, an HTTP(S) or `data:` URL, or
`{"type": "image_url", "image_url": {"url": ...}}`). Video and batched states
(`system_one_batch`) are not supported yet.

## Hardware

**NVIDIA / AMD / Intel (Vulkan, DX12).** Both files fit any card with 2.5 GB of
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
| `d1-omni-2b-q8_2f.cmf` | `afee01128b798277a49ddd97ebfa37bb8f94cca16e245f4f40065cab1a3c2272` |
| `d1-omni-2b-q4tp.cmf` | `65ce90e5df70f591bc1594c40c9425a39770d257433a51ee30b01892eb638e04` |

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

[d1-2B](https://huggingface.co/LiquidAI/d1-Omni-2B) от Liquid AI — модель решений
(System One) на 2 млрд параметров поверх LFM3-VL-2B: именованные вопросы к
состоянию (текст, JSON или изображения) получают типизированные ответы с
откалиброванными вероятностями за один прямой проход, без генерации токенов, —
в виде файлов [CMF](https://github.com/infosave2007/cmf) для движка `cortiq`
на Rust без Python.

```bash
cargo install cortiq-cli
hf download infosave/d1-Omni-2B-cmf d1-omni-2b-q8_2f.cmf --local-dir .
cortiq system-one d1-omni-2b-q8_2f.cmf --state "Мне дважды списали деньги, посылка не пришла." \
  --questions '{"refund": {"type": "noul", "instructions": "Клиент просит возврат?"}}'
```

| файл | веса | размер | ответы = оригинал | назначение |
|---|---|---:|---:|---|
| `d1-omni-2b-q8_2f.cmf` | 8 бит | 2.14 ГБ | 19 из 19 | **по умолчанию** |
| `d1-omni-2b-q4tp.cmf` | 4 бита | 1.31 ГБ | 19 из 19 (одна оценка на 2 шага) | минимум памяти |

Нужен cortiq 0.8.50 или новее. Типы вопросов: `noul` (да/нет), `choice` (один
или несколько вариантов), `score` (уровни), `text`/`number`/`date` (значения из
состояния), `number` с `range` (оценка с интервалом), `box`/`boxes` (рамки на
изображении). Весь проход — слои над упакованным состоянием и вопросами и
голова словаря над слотами ответов — идёт одной отправкой на GPU: на NVIDIA A40
один вопрос отвечается за 40 мс (`q4tp`), на Apple M4 (Metal) — за 85 мс.
`cortiq serve` отдаёт `POST /v1/system_one` (`state`, `questions`, `media`).
Видео и пакетные запросы пока не поддерживаются. Файлы — изменённая форма d1-2B
от Liquid AI по LFM Open License v1.0 (копия — [LICENSE](LICENSE));
коммерческое использование организацией с годовой выручкой от 10 млн долларов
США лицензией не покрывается.

## 中文说明

[d1-2B](https://huggingface.co/LiquidAI/d1-Omni-2B) 是 Liquid AI 基于 LFM3-VL-2B 的
2B 决策模型（System One）：对一个状态（文本、JSON 或图片）提出若干具名问题，一次
前向即可得到带校准概率的类型化答案，不生成任何 token。本仓库提供可由 Rust 推理
引擎 `cortiq`（0.8.50 及以上）加载的 CMF 文件。

```bash
cargo install cortiq-cli
hf download infosave/d1-Omni-2B-cmf d1-omni-2b-q8_2f.cmf --local-dir .
cortiq system-one d1-omni-2b-q8_2f.cmf --state "我被重复扣款，包裹也没到。" \
  --questions '{"refund": {"type": "noul", "instructions": "客户是否要求退款？"}}'
```

| 文件 | 权重 | 大小 | 答案与原模型一致 | 适用 |
|---|---|---:|---:|---|
| `d1-omni-2b-q8_2f.cmf` | 8 位 | 2.14 GB | 19/19 | **默认** |
| `d1-omni-2b-q4tp.cmf` | 4 位 | 1.31 GB | 19/19（一个估计值差 2 档） | 内存最小 |

整个前向（打包后的状态与问题经过所有层，再在答案位置上计算词表头）在一次 GPU
提交中完成：NVIDIA A40 上单个问题 40 ms（`q4tp`），Apple M4（Metal）上 85 ms。
`cortiq serve` 提供 `POST /v1/system_one`。暂不支持视频与批量请求。这些文件是
Liquid AI d1-Omni-2B 的修改形式，仍受 LFM Open License v1.0 约束（副本见
[LICENSE](LICENSE)）；年收入 1000 万美元及以上的实体进行商业使用不在许可范围内。
