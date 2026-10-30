---
license: other
license_name: lfm1.0
license_link: LICENSE
library_name: cortiq
base_model: LiquidAI/LFM3-1.7B
base_model_relation: quantized
pipeline_tag: text-generation
tags:
  - cmf
  - cortiq
  - quantized
  - q4tp
  - liquid
  - lfm3
  - edge
  - reasoning
  - tool-calling
  - vulkan
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

# LFM3-1.7B — CMF

[Liquid AI LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B) — a 1.7B text
model for local agents, tool calling and structured output on Liquid's
tri-hybrid architecture (24 layers: 13 gated short-convolution, 6 full-attention
and 5 sliding-window attention layers; 262,144-token context; thinking on by
default) — packaged as single [CMF](https://github.com/infosave2007/cmf) files.
They run with `cortiq`, a Rust inference engine with no Python and no ML
framework: NVIDIA, AMD and Intel GPUs through Vulkan/DX12, and the CPU.

[English](#quick-start) · [Русский](#документация-на-русском) · [中文](#中文说明)

## Quick start

```bash
cargo install cortiq-cli
hf download infosave/LFM3-1.7B-cmf lfm3-1.7b-q8_2f.cmf --local-dir .
cortiq run lfm3-1.7b-q8_2f.cmf --prompt "Write a Python function that merges two sorted lists." --max-tokens 2048
```

Use cortiq 0.8.49 or later; prebuilt binaries for Linux, macOS and Windows are
on the [releases page](https://github.com/infosave2007/cmf/releases). The
engine picks the GPU and its settings itself. The model thinks before it
answers, and the thinking counts toward `--max-tokens` (default 256), so give
it room; `--no-think` asks for a direct answer.

## Files

| file | weights | size | wiki ppl | top-1 vs original | recommended for |
|---|---|---:|---:|---:|---|
| `lfm3-1.7b-q8_2f.cmf` | 8-bit | 1,756,991,387 B (1.76 GB) | 13.14 | 97.7 % | **default**: near-lossless |
| `lfm3-1.7b-q4tp.cmf` | 4-bit | 927,275,931 B (0.93 GB) | 15.15 | 80.4 % | the smallest memory footprint |
| bf16 original (Transformers) | — | 3.46 GB | 13.10 | — | — |

Perplexity over the first 4096 tokens of wikitext-2 test (with BOS); *top-1 vs
original*: positions where the file's top prediction matches the bf16
checkpoint run in float32 by Transformers over the same tokens. The 1.7B
instruct model is sensitive to 4-bit weights: prefer `q8_2f` unless memory is
the limit. Norms, the short-convolution taps and the per-head attention gates
stay 16-bit in both files; the tied embedding / output head follows the file's
codec.

The conversion is exact: an f16 conversion of the checkpoint matches
Transformers float32 position for position on the CPU (perplexity 13.186 against 13.186, top-1
100 %), and greedy chat continuations are text-identical.

## Performance

`cortiq bench --core --ignore-eos`, one stream, cortiq 0.8.49, NVIDIA A40
(48 GB, Vulkan):

| file | decode, tok/s | decode at 1k / 4k | 1k / 4k-token prompt, tok/s | first token, 4k prompt |
|---|---:|---:|---:|---:|
| `q8_2f` | 166 | 135 / 130 | 749 / 623 | 7.6 s |
| `q4tp` | 231 | 184 / 165 | 695 / 574 | 8.0 s |

Resident weights: 1.73 GB (`q8_2f`), 0.90 GB (`q4tp`). Every layer of a
token runs in one GPU submission.

## Usage

```bash
cortiq run lfm3-1.7b-q8_2f.cmf                                  # interactive chat
cortiq run lfm3-1.7b-q8_2f.cmf --prompt "..." --max-tokens 4096 # one answer, thinking first
cortiq run lfm3-1.7b-q8_2f.cmf --prompt "..." --no-think        # direct answer
cortiq run lfm3-1.7b-q8_2f.cmf --prompt "..." --greedy          # deterministic output
```

The files carry Liquid's recommended general sampling — temperature 0.1,
top-k 50, repetition penalty 1.1 — as the `run` and `serve` defaults; for math
and science Liquid recommends `--temperature 1.0 --top-p 0.95 --top-k 20`.
The embedded chat template is the checkpoint's own (the generation prompt ends
with `<think>`; `--no-think` writes `<think></think>`).

### OpenAI-compatible server

```bash
cortiq serve lfm3-1.7b-q8_2f.cmf --port 8080

curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "lfm3", "max_tokens": 2048,
       "messages": [{"role": "user", "content": "Explain how a B-tree splits a full node."}]}'
```

Endpoints: `/v1/chat/completions` (streamed or not), `/v1/completions`,
`/v1/models`, plus a web dashboard on the same port. `--host 127.0.0.1` keeps
the server local. A thinking reply reads `<think>…</think>answer`;
`"enable_thinking": false` gives a direct answer.

### Tool calling

Send `tools` in the OpenAI format; the template lists them the way the model
was trained, and the model answers with
`<tool_call><function=NAME><parameter=KEY>value</parameter></function></tool_call>`.
The server returns that as `message.tool_calls` with
`finish_reason: "tool_calls"`. In the test conversation (a weather tool, then
the answer from its result) both files made the call with the right arguments
and answered from the result, with and without thinking.

## Hardware

**NVIDIA / AMD / Intel (Vulkan, DX12).** Both files fit any card with 2 GB of
free memory. `cortiq gpu` lists the adapters; `CMF_GPU_ADAPTER=1` (index or
name substring) picks one. On headless Linux install
`libvulkan1 libglvnd0 libegl1 libgl1 libglx0` and set `XDG_RUNTIME_DIR=/tmp`.

**CPU only.** `CMF_GPU=0`.

The numbers above are measured on NVIDIA through Vulkan; Apple silicon has not
been measured with this model yet.

## Long context

The model supports 262,144 tokens; the KV cache holds 32,768 by default.
`CMF_MAX_SEQ=262144` raises it. The 13 short-convolution layers keep a fixed
0.6 MB state and the 5 sliding-window layers only their last 256 positions, so
KV grows by about 25 KB per token (812 MB at 32k, ~6.5 GB at 262k). At a
32,768-token prompt (A40) the prompt reads at ~540 tok/s and decoding runs at
92 (`q8_2f`) / 117 (`q4tp`) tok/s.

## Verify the download

```bash
sha256sum -c lfm3-1.7b-q8_2f.cmf.sha256
cortiq verify lfm3-1.7b-q8_2f.cmf
cortiq info lfm3-1.7b-q8_2f.cmf
```

| file | sha256 |
|---|---|
| `lfm3-1.7b-q8_2f.cmf` | `a388bee1a90fc5f0602c03f7b393f762e908c78a151f4761310d355216272e55` |
| `lfm3-1.7b-q4tp.cmf` | `8bada149217fa5614d01eaf346d208089952e499912ba5a8e01273ce8fbe2a98` |

## Reproduce

```bash
cortiq convert --model LiquidAI/LFM3-1.7B --quant q8_2f \
  --output lfm3-1.7b-q8_2f.cmf --also-output q4tp=lfm3-1.7b-q4tp.cmf \
  --sampling temperature=0.1,top_k=50,repetition_penalty=1.1
```

The scripts used for these files (Transformers reference, per-position
comparison, chat parity, tool-call test, benchmarks) are in
[scripts/lfm3](https://github.com/infosave2007/cmf/tree/master/scripts/lfm3).

## License

These files are a modified form of Liquid AI's LFM3-1.7B: the weights were
converted to CMF and quantized (8-bit `q8_2f`, 4-bit `q4tp`) by cortiq. They
remain under the LFM Open License v1.0, a copy of which is in this repository
([LICENSE](LICENSE)); among its terms, commercial use by an entity with annual
revenue of US$10 million or more is not licensed. See the
[upstream card](https://huggingface.co/LiquidAI/LFM3-1.7B).

---

## Документация на русском

[LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B) от Liquid AI — текстовая
модель на 1.7 млрд параметров для локальных агентов, вызова инструментов и
структурированного вывода на трёхгибридной архитектуре Liquid (24 слоя: 13
слоёв короткой свёртки с гейтами, 6 слоёв полного внимания и 5 слоёв внимания
со скользящим окном; контекст 262 144 токена; рассуждения включены по
умолчанию), в виде отдельных файлов [CMF](https://github.com/infosave2007/cmf)
для движка `cortiq` на Rust без Python и ML-фреймворков.

### Быстрый старт

```bash
cargo install cortiq-cli
hf download infosave/LFM3-1.7B-cmf lfm3-1.7b-q8_2f.cmf --local-dir .
cortiq run lfm3-1.7b-q8_2f.cmf --prompt "Напиши функцию на Python, которая сливает два отсортированных списка." --max-tokens 2048
```

Нужен cortiq 0.8.49 или новее. Модель рассуждает перед ответом; рассуждение
входит в `--max-tokens` (по умолчанию 256), поэтому дайте запас; `--no-think`
просит ответ сразу.

### Файлы

| файл | веса | размер | ppl wiki | top-1 к оригиналу | назначение |
|---|---|---:|---:|---:|---|
| `lfm3-1.7b-q8_2f.cmf` | 8 бит | 1.76 ГБ | 13.14 | 97.7 % | **по умолчанию**: почти без потерь |
| `lfm3-1.7b-q4tp.cmf` | 4 бита | 0.93 ГБ | 15.15 | 80.4 % | минимум памяти |
| bf16-оригинал (Transformers) | — | 3.46 ГБ | 13.10 | — | — |

Инструктивная 1.7B чувствительна к 4-битным весам: берите `q8_2f`, если
память позволяет. Конвертация точная: f16-конвертация чекпойнта на CPU совпадает с
Transformers float32 по каждой позиции (ppl 13.186 против 13.186, top-1 100 %).

### Скорость

NVIDIA A40 (48 ГБ, Vulkan), cortiq 0.8.49:

| файл | генерация, ток/с | на глубине 1k / 4k | промпт 1k / 4k, ток/с | первый токен, промпт 4k |
|---|---:|---:|---:|---:|
| `q8_2f` | 166 | 135 / 130 | 749 / 623 | 7.6 с |
| `q4tp` | 231 | 184 / 165 | 695 / 574 | 8.0 с |

Сэмплирование по умолчанию — рекомендованное Liquid (temperature 0.1, top-k 50,
repetition penalty 1.1); для математики Liquid советует
`--temperature 1.0 --top-p 0.95 --top-k 20`. `cortiq serve` даёт
OpenAI-совместимый API с вызовом инструментов (`message.tool_calls`; проверено
на обоих файлах, с рассуждением и без). Контекст до 262 144 токенов:
`CMF_MAX_SEQ=262144` (по умолчанию 32 768); KV растёт примерно на 25 КБ на
токен (812 МБ при 32k), скользящие слои хранят только последние 256 позиций.
Apple silicon с этой моделью пока не измерялся.

Файлы — изменённая форма LFM3-1.7B от Liquid AI (конвертация в CMF и
квантование cortiq) и распространяются по LFM Open License v1.0 (копия —
[LICENSE](LICENSE)); в частности, коммерческое использование организацией с
годовой выручкой от 10 млн долларов США лицензией не покрывается.

## 中文说明

[LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B) 是 Liquid AI 的 1.7B
文本模型，面向本地智能体、工具调用与结构化输出，采用 Liquid 三混合架构
（24 层：13 层门控短卷积、6 层全注意力、5 层滑动窗口注意力；262,144 token
上下文；默认开启思考）。本仓库提供可由 Rust 推理引擎 `cortiq`（0.8.49 及以上）
加载的 CMF 文件。

```bash
cargo install cortiq-cli
hf download infosave/LFM3-1.7B-cmf lfm3-1.7b-q8_2f.cmf --local-dir .
cortiq run lfm3-1.7b-q8_2f.cmf --prompt "你好" --max-tokens 2048
```

| 文件 | 权重 | 大小 | wiki ppl | 与原模型 top-1 一致率 | 适用 |
|---|---|---:|---:|---:|---|
| `lfm3-1.7b-q8_2f.cmf` | 8 位 | 1.76 GB | 13.14 | 97.7 % | **默认**：近无损 |
| `lfm3-1.7b-q4tp.cmf` | 4 位 | 0.93 GB | 15.15 | 80.4 % | 内存最小 |
| bf16 原模型（Transformers） | — | 3.46 GB | 13.10 | — | — |

该指令模型对 4 位权重较敏感，内存允许时请使用 `q8_2f`。转换是精确的：检查点的
f16 转换在 CPU 上与 Transformers float32 逐位置一致（ppl 13.186 对 13.186，top-1
100 %）。NVIDIA A40（48 GB，Vulkan）上 `q8_2f` 生成 166 token/s、4k 提示词
623 token/s，`q4tp` 生成 231 token/s、4k 提示词 574 token/s。默认采样为 Liquid
推荐的 temperature 0.1、top-k 50、repetition penalty 1.1。`cortiq serve` 提供
OpenAI 兼容接口与工具调用。`CMF_MAX_SEQ=262144` 可开启完整 262,144 token 上下文
（默认 32,768）；KV 每 token 约 25 KB（32k 时 812 MB）。Apple silicon 尚未针对
该模型测量。

这些文件是 Liquid AI LFM3-1.7B 的修改形式（由 cortiq 转换为 CMF 并量化），仍受
LFM Open License v1.0 约束（副本见 [LICENSE](LICENSE)）；其中，年收入 1000 万美元
及以上的实体进行商业使用不在许可范围内。
