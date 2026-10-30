---
license: apache-2.0
library_name: cortiq
base_model: Apexintelligence-AI/Apex-35B-V1.0
base_model_relation: quantized
pipeline_tag: text-generation
tags:
  - cmf
  - cortiq
  - quantized
  - q4tp
  - moe
  - agent
  - reasoning
  - tool-calling
  - vulkan
language:
  - en
  - ru
  - zh
---

# Apex-35B-V1.0 — CMF

[Apex Intelligence Apex-35B-V1.0](https://huggingface.co/Apexintelligence-AI/Apex-35B-V1.0)
— an agentic research, coding and reasoning model fine-tuned from Qwen3.6-35B-A3B
(mixture of experts: 35.5B parameters, ~3B active per token, 256 routed experts
top-8 plus a shared expert; 30 Gated DeltaNet and 10 gated full-attention
layers; 262,144-token context; dynamic thinking; tool calling) — packaged as
single [CMF](https://github.com/infosave2007/cmf) files. They run with
`cortiq`, a Rust inference engine with no Python and no ML framework: NVIDIA,
AMD and Intel GPUs through Vulkan/DX12, Apple silicon through Metal, and the
CPU. The files carry the text model and its MTP layer; the checkpoint's vision
tower is not included.

[English](#quick-start) · [Русский](#документация-на-русском) · [中文](#中文说明)

## Quick start

```bash
cargo install cortiq-cli
hf download infosave/Apex-35B-V1.0-cmf apex-35b-v1.0-q4tp-v2.cmf --local-dir .
cortiq run apex-35b-v1.0-q4tp-v2.cmf --prompt "Write a Python function that merges two sorted lists." --max-tokens 4096
```

Use cortiq 0.8.48 or later; prebuilt binaries for Linux, macOS and Windows are
on the [releases page](https://github.com/infosave2007/cmf/releases). The
engine picks the GPU and its settings itself. Apex decides on each turn
whether to think and for how long, and the thinking counts toward
`--max-tokens` (default 256), so give it room; `--no-think` asks for a direct
answer.

## Files

| file | experts | size | wiki ppl | code ppl | chat ppl | top-1 vs bf16 (chat) | recommended for |
|---|---|---:|---:|---:|---:|---:|---|
| `apex-35b-v1.0-q4tp-v2.cmf` | 4-bit, GPTQ | 19,808,338,734 B (19.81 GB) | 25.83 | 1.926 | 4.071 | 95.2 % | **default**: GPUs with 24 GB+ |
| `apex-35b-v1.0-q4tp.cmf` | 4-bit | 19,827,917,614 B (19.83 GB) | 26.86 | 1.905 | 4.029 | 94.9 % | the plain conversion |
| `apex-35b-v1.0-q8_2f.cmf` | 8-bit | 35,840,111,406 B (35.84 GB) | 25.72 | 1.753 | 4.047 | 97.7 % | near-lossless; GPUs with 48 GB, large-memory CPUs |
| bf16 original (PyTorch reference) | — | 71.9 GB | 26.52 | 1.759 | 4.025 | — | — |

Attention, Gated DeltaNet projections, embeddings and the output head are
8-bit in every file; norms and the router stay 16-bit. `q4tp-v2` encodes every
routed and shared expert with GPTQ against 262,144 calibration tokens
(wikitext-2 train, UltraChat rendered in the model's template, Rust sources)
at the size of `q4tp`; per position it stays closer to the original on chat
(mean |ΔNLL| 0.138 against 0.160) and wiki text, and level on code.

Perplexity over the first 4096 tokens of wikitext-2 test and of Python
standard-library code, and over 8192 tokens of 12 held-out UltraChat
conversations in the chat template; *top-1 vs bf16*: positions of the chat
text where the file's top prediction matches the bf16 checkpoint run in
PyTorch over the same tokens. The fine-tune itself scores 26.5 on raw
wikitext (Qwen3.6-35B-A3B: 5.6): the number belongs to the model, not to the
quantization.

## Performance

`cortiq bench --core --ignore-eos`, one stream, cortiq 0.8.48, NVIDIA A40
(48 GB, Vulkan), the whole model on the card:

| file | decode, tok/s | decode at 4k / 16k | 4k / 16k-token prompt, tok/s | first token, 4k prompt |
|---|---:|---:|---:|---:|
| `q4tp-v2` | 80.3 | 73.3 / 59.5 | 390 / 368 | 10.5 s |
| `q4tp` | 80.5 | 73.7 / 61.1 | 383 / 367 | 10.7 s |
| `q8_2f` | 78.2 | 71.3 / 58.9 | 84 / 76 | 48.7 s |

Resident weights: 18.7 GB (`q4tp`, `q4tp-v2`), 34.3 GB (`q8_2f`). The `q8_2f`
file reads the prompt one position at a time on Vulkan; for long prompts use a
4-bit file.

## Usage

```bash
cortiq run apex-35b-v1.0-q4tp-v2.cmf                                  # interactive chat
cortiq run apex-35b-v1.0-q4tp-v2.cmf --prompt "..." --max-tokens 8192 # one answer, thinking as needed
cortiq run apex-35b-v1.0-q4tp-v2.cmf --prompt "..." --no-think        # direct answer
cortiq run apex-35b-v1.0-q4tp-v2.cmf --prompt "..." --greedy          # deterministic output
```

The files carry Apex's recommended sampling — temperature 1.0, top-p 0.95,
top-k 20 — as the `run` and `serve` defaults; flags and request fields
override it. The embedded chat template is the checkpoint's own (the
generation prompt ends with `<think>`).

### OpenAI-compatible server

```bash
cortiq serve apex-35b-v1.0-q4tp-v2.cmf --port 8080

curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "apex", "max_tokens": 4096,
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
the answer from its result) the call and the answer were correct with and
without thinking.

## Hardware

**NVIDIA / AMD / Intel (Vulkan, DX12).** The 4-bit files run whole on a 24 GB
card; a smaller card keeps as many leading layers as fit and the CPU runs the
rest. `cortiq gpu` lists the adapters; `CMF_GPU_ADAPTER=1` (index or name
substring) picks one. On headless Linux install
`libvulkan1 libglvnd0 libegl1 libgl1 libglx0` and set `XDG_RUNTIME_DIR=/tmp`.

**Apple silicon.** The native Metal backend is the default on macOS.

**CPU only.** `CMF_GPU=0`.

## Long context

The model supports 262,144 tokens; the KV cache holds 32,768 by default.
`CMF_MAX_SEQ=262144` raises it. The ten attention layers keep about 40 KB of
KV per token (1.3 GB at 32k, 10.5 GB at 262k); the Gated DeltaNet layers keep
a fixed state. At a 65,536-token prompt (`CMF_MAX_SEQ=131072`, A40,
`q4tp-v2`) the prompt reads at 254 tok/s and decoding runs at 28 tok/s.

## Verify the download

```bash
sha256sum -c apex-35b-v1.0-q4tp-v2.cmf.sha256
cortiq verify apex-35b-v1.0-q4tp-v2.cmf
cortiq info apex-35b-v1.0-q4tp-v2.cmf
```

| file | sha256 |
|---|---|
| `apex-35b-v1.0-q4tp-v2.cmf` | `9e5ff54dc7b11e9320d0d45f5b20cb7f7801318df6d35ecd525dfe7869d5ad8d` |
| `apex-35b-v1.0-q4tp.cmf` | `ee777dbc4e4e621a3fa47d11d56b5ee137cf718ad5dad4f7bc8a98089cf3b97b` |
| `apex-35b-v1.0-q8_2f.cmf` | `9e8d2794b6c2ec841e7ec347fbc3c67738e8780cb0951c65013929cf2f5f16a3` |

## Reproduce

```bash
cortiq convert --model Apexintelligence-AI/Apex-35B-V1.0 --quant q4tp \
  --output apex-35b-v1.0-q4tp.cmf --also-output q8_2f=apex-35b-v1.0-q8_2f.cmf \
  --sampling temperature=1.0,top_p=0.95,top_k=20,min_p=0,repetition_penalty=1.0
cortiq quantize-moe apex-35b-v1.0-q4tp.cmf --hf Apexintelligence-AI/Apex-35B-V1.0 \
  --calib wiki.train.raw=0.4 --calib chat.txt=0.4 --calib code.txt=0.2 \
  --tokens 262144 --codec q4tp --output apex-35b-v1.0-q4tp-v2.cmf
```

The conversion streams the checkpoint (peak disk: the outputs); `quantize-moe`
fetches the bf16 experts one layer at a time. The scripts used for these files
(reference perplexity, chat parity, benchmarks) are in
[scripts/apex](https://github.com/infosave2007/cmf/tree/master/scripts/apex).

Weights derive from Apex Intelligence's release and remain under its
Apache-2.0 terms; see the
[upstream card](https://huggingface.co/Apexintelligence-AI/Apex-35B-V1.0).

---

## Документация на русском

[Apex-35B-V1.0](https://huggingface.co/Apexintelligence-AI/Apex-35B-V1.0) от
Apex Intelligence — модель для агентной работы, исследований, кода и
рассуждений, дообученная из Qwen3.6-35B-A3B (смесь экспертов: 35.5 млрд
параметров, ~3 млрд активных на токен, 256 экспертов с выбором top-8 и общий
эксперт; 30 слоёв Gated DeltaNet и 10 слоёв полного внимания; контекст 262 144
токена; рассуждения по необходимости; вызов инструментов), в виде отдельных
файлов [CMF](https://github.com/infosave2007/cmf) для движка `cortiq` на Rust
без Python и ML-фреймворков. В файлах текстовая модель и её MTP-слой;
визуальной башни чекпойнта в них нет.

### Быстрый старт

```bash
cargo install cortiq-cli
hf download infosave/Apex-35B-V1.0-cmf apex-35b-v1.0-q4tp-v2.cmf --local-dir .
cortiq run apex-35b-v1.0-q4tp-v2.cmf --prompt "Напиши функцию на Python, которая сливает два отсортированных списка." --max-tokens 4096
```

Нужен cortiq 0.8.48 или новее. Модель сама решает, рассуждать ли и сколько;
рассуждение входит в `--max-tokens` (по умолчанию 256), поэтому дайте запас;
`--no-think` просит ответ сразу.

### Файлы

| файл | эксперты | размер | ppl wiki | ppl код | ppl чат | top-1 к bf16 (чат) | назначение |
|---|---|---:|---:|---:|---:|---:|---|
| `apex-35b-v1.0-q4tp-v2.cmf` | 4 бита, GPTQ | 19.81 ГБ | 25.83 | 1.926 | 4.071 | 95.2 % | **по умолчанию**: GPU от 24 ГБ |
| `apex-35b-v1.0-q4tp.cmf` | 4 бита | 19.83 ГБ | 26.86 | 1.905 | 4.029 | 94.9 % | простая конвертация |
| `apex-35b-v1.0-q8_2f.cmf` | 8 бит | 35.84 ГБ | 25.72 | 1.753 | 4.047 | 97.7 % | почти без потерь; GPU на 48 ГБ, CPU с большой памятью |
| bf16-оригинал (эталон PyTorch) | — | 71.9 ГБ | 26.52 | 1.759 | 4.025 | — | — |

Высокий ppl на wikitext — свойство самого дообучения (bf16-оригинал даёт
26.5, Qwen3.6-35B-A3B — 5.6), а не квантования.

### Скорость

NVIDIA A40 (48 ГБ, Vulkan), cortiq 0.8.48, модель целиком на карте:

| файл | генерация, ток/с | на глубине 4k / 16k | промпт 4k / 16k, ток/с | первый токен, промпт 4k |
|---|---:|---:|---:|---:|
| `q4tp-v2` | 80.3 | 73.3 / 59.5 | 390 / 368 | 10.5 с |
| `q4tp` | 80.5 | 73.7 / 61.1 | 383 / 367 | 10.7 с |
| `q8_2f` | 78.2 | 71.3 / 58.9 | 84 / 76 | 48.7 с |

Файл `q8_2f` читает промпт по одной позиции; для длинных промптов берите
4-битный файл. Сэмплирование по умолчанию — рекомендованное Apex
(temperature 1.0, top-p 0.95, top-k 20). `cortiq serve` даёт
OpenAI-совместимый API с вызовом инструментов (`message.tool_calls`).
Контекст до 262 144 токенов: `CMF_MAX_SEQ=262144` (по умолчанию 32 768).

## 中文说明

[Apex-35B-V1.0](https://huggingface.co/Apexintelligence-AI/Apex-35B-V1.0) 是
Apex Intelligence 基于 Qwen3.6-35B-A3B 微调的智能体、研究、编程与推理模型
（混合专家：35.5B 参数，每个 token 约 3B 激活，256 个路由专家 top-8 加共享专家；
30 层 Gated DeltaNet 与 10 层门控全注意力；262,144 token 上下文；按需思考；
工具调用）。本仓库提供可由 Rust 推理引擎 `cortiq`（0.8.48 及以上）加载的 CMF
文件，包含文本模型及其 MTP 层，不含视觉塔。

```bash
cargo install cortiq-cli
hf download infosave/Apex-35B-V1.0-cmf apex-35b-v1.0-q4tp-v2.cmf --local-dir .
cortiq run apex-35b-v1.0-q4tp-v2.cmf --prompt "你好" --max-tokens 4096
```

| 文件 | 专家 | 大小 | wiki ppl | 代码 ppl | 对话 ppl | 适用 |
|---|---|---:|---:|---:|---:|---|
| `apex-35b-v1.0-q4tp-v2.cmf` | 4 位 GPTQ | 19.81 GB | 25.83 | 1.926 | 4.071 | **默认**：24 GB 以上显卡 |
| `apex-35b-v1.0-q4tp.cmf` | 4 位 | 19.83 GB | 26.86 | 1.905 | 4.029 | 普通转换 |
| `apex-35b-v1.0-q8_2f.cmf` | 8 位 | 35.84 GB | 25.72 | 1.753 | 4.047 | 近无损；48 GB 显卡 |

bf16 原模型在 wikitext 上为 26.5（Qwen3.6-35B-A3B 为 5.6），较高的 wiki ppl 来自
微调本身而非量化。NVIDIA A40（48 GB，Vulkan）上 `q4tp-v2` 生成 80.3 token/s，
4k 提示词读取 390 token/s，首 token 10.5 s；`q8_2f` 逐位置读取提示词（84 token/s），
长提示词请使用 4 位文件。默认采样为 Apex 推荐的 temperature 1.0、top-p 0.95、
top-k 20。`cortiq serve` 提供 OpenAI 兼容接口与工具调用。
