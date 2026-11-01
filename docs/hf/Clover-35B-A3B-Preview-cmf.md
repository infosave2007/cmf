---
license: apache-2.0
library_name: cortiq
base_model: Diffuspace/Clover-35B-A3B-Preview
base_model_relation: quantized
pipeline_tag: text-generation
tags:
  - cmf
  - cortiq
  - quantized
  - q4tp
  - moe
  - diffusion
  - block-diffusion
  - vulkan
language:
  - en
  - ru
  - zh
---

# Clover-35B-A3B-Preview — CMF

[DiffuSpace Clover-35B-A3B-Preview](https://huggingface.co/Diffuspace/Clover-35B-A3B-Preview)
— a diffusion language model on the Qwen3.6-35B-A3B backbone (mixture of
experts: 35B parameters, ~3B active per token, 256 experts top-8 plus a shared
expert; 30 Gated DeltaNet and 10 gated full-attention layers; 128k context).
It is built to write in 32-token blocks by repeated denoising passes, and its
trunk also reads as an ordinary causal model — packaged as single
[CMF](https://github.com/infosave2007/cmf) files for `cortiq`, a Rust inference
engine with no Python and no ML framework. `cortiq` decodes causally by default,
the fastest on every machine we measured; block denoising and self-speculation
are one setting away.

[English](#quick-start) · [Русский](#документация-на-русском) · [中文](#中文说明)

## Quick start

```bash
cargo install cortiq-cli
hf download infosave/Clover-35B-A3B-Preview-cmf clover-35b-a3b-preview-q4tp-v2.cmf --local-dir .
cortiq run clover-35b-a3b-preview-q4tp-v2.cmf --prompt "Write a Python function that merges two sorted lists." --no-think --max-tokens 512
```

Use cortiq 0.8.52 or later; prebuilt binaries for Linux, macOS and Windows are on the [releases page](https://github.com/infosave2007/cmf/releases).
Without `--no-think` the model reasons in a `<think>` block first, so give it
`--max-tokens` in the thousands.

## Files

| file | experts | size | wiki ppl | first-pass top-1 | recommended for |
|---|---|---:|---:|---:|---|
| `clover-35b-a3b-preview-q4tp-v2.cmf` | 4-bit, GPTQ | 19.36 GB | 16.80 | 93/96 | **default**: GPUs with 24 GB+ |
| `clover-35b-a3b-preview-q4tp.cmf` | 4-bit | 19.38 GB | 18.22 | 93/96 | the plain conversion |
| `clover-35b-a3b-preview-q8_2f.cmf` | 8-bit | 35.00 GB | 15.86 | 62/64 | near-lossless; GPUs with 48 GB+, big-memory CPUs |

Attention, Gated DeltaNet projections, embeddings and the head are 8-bit in
every file. `q4tp-v2` calibrates the experts with GPTQ on 262k tokens (wiki,
chat and code text) at the same size as `q4tp`. *wiki ppl*: perplexity of the
model's causal path over 12 × 512 tokens of wikitext-2 test. *first-pass
top-1*: rows of the first denoising pass whose top token matches the official
PyTorch implementation (bf16), three chat prompts × 32 rows (two for
`q8_2f`); the misses are near-ties.

## Performance

Tokens per second after the prompt, `q4tp-v2`:

| hardware | prose | code | a 2048-token prompt |
|---|---:|---:|---:|
| RTX 5090 (32 GB), Vulkan | 120 | 121 | 1143 tok/s |
| 16 GB of VRAM, Vulkan + CPU | 54 | 53 | |
| Mac mini M4 (24 GB), prompt on Metal, decode on CPU | 13–18 | 14–18 | ~170 tok/s |
| CPU only, 15 Xeon cores | 13.8 | 10.9 | |

The whole model and its head take 18.7 GB on the card. `q8_2f` does not fit
32 GB whole and runs at 46–49 tok/s there.

### Decoding modes

`CMF_CLOVER_MODE` picks how a request uses the model. On the RTX 5090:

| mode | what it does | prose | code |
|---|---|---:|---:|
| `ar` (default) | causal decode with the release's response penalties | 120 | 121 |
| `draft` | self-speculation: a denoising pass proposes 16 tokens, the causal path verifies them | 42 | 104 |
| `denoise` | LowConfidence block denoising, the release's `generate` | 18–26 | 63–95 |

In `denoise` on an M4: 3.2 and 6.5–9 tok/s. `CMF_CLOVER_THRESHOLD` lowers the
denoising threshold, `CMF_CLOVER_DRAFT` sets the rows of a self-speculative round.

## Usage

```bash
cortiq run clover-35b-a3b-preview-q4tp-v2.cmf                                  # interactive chat
cortiq run clover-35b-a3b-preview-q4tp-v2.cmf --prompt "..." --max-tokens 4096 # with reasoning
cortiq run clover-35b-a3b-preview-q4tp-v2.cmf --prompt "..." --no-think        # direct answer
```

Every mode applies the release's response penalties (presence 0.9, repetition
1.05, a gradual push toward the stop token after 8192 answer tokens and toward
`</think>` after 4096). `--temperature` samples instead of greedy.

### OpenAI-compatible server

```bash
cortiq serve clover-35b-a3b-preview-q4tp-v2.cmf --port 8080

curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "clover", "enable_thinking": false,
       "messages": [{"role": "user", "content": "Explain how a hash map works."}]}'
```

`/v1/chat/completions` (streamed or not), `/v1/completions`, `/v1/models` and
a web dashboard on the same port.

## Hardware

**NVIDIA / AMD / Intel (Vulkan, DX12).** A card that holds the file runs the
whole model; a smaller one runs the first layers on the card and the rest on
the CPU. `cortiq gpu`
lists the adapters; on headless Linux install `libvulkan1 libglvnd0 libegl1
libgl1 libglx0` and set `XDG_RUNTIME_DIR=/tmp`.

**Apple silicon.** The prompt runs on Metal and the decode on the CPU, which
is the faster of the two for this file on a 24 GB Mac: Metal cannot keep the
19 GB of weights resident there. The weights stay in the file's mapping, so a
24 GB Mac runs the `q4tp` files.

**CPU only.** `CMF_GPU=0`.

## Verify the download

```bash
sha256sum -c clover-35b-a3b-preview-q4tp-v2.cmf.sha256
cortiq verify clover-35b-a3b-preview-q4tp-v2.cmf
cortiq info clover-35b-a3b-preview-q4tp-v2.cmf
```

## Reproduce

```bash
cortiq convert --model Diffuspace/Clover-35B-A3B-Preview --quant q4tp \
  --sampling temperature=0,top_p=1,top_k=0,min_p=0,repetition_penalty=1 \
  --output clover-35b-a3b-preview-q4tp.cmf
cortiq quantize-moe clover-35b-a3b-preview-q4tp.cmf --hf Diffuspace/Clover-35B-A3B-Preview \
  --calib wiki.train.txt=0.4 --calib chat.txt=0.4 --calib code.txt=0.2 \
  --tokens 262144 --codec q4tp --output clover-35b-a3b-preview-q4tp-v2.cmf
```

## Scope

Causal decoding, LowConfidence block diffusion and self-speculation. Weights derive from DiffuSpace's release (built on
Qwen3.6-35B-A3B) and remain under its Apache-2.0 terms; see the
[upstream card](https://huggingface.co/Diffuspace/Clover-35B-A3B-Preview) and
its attributions.

---

## Документация на русском

Clover-35B-A3B-Preview от DiffuSpace — диффузионная языковая модель на основе
Qwen3.6-35B-A3B (35 млрд параметров, ~3 млрд активных на токен). Она умеет
писать блоками по 32 токена, раскрывая их проходами денойзинга, а её основа
работает и как обычная каузальная модель. Здесь она в виде отдельных файлов CMF
для движка `cortiq` на Rust; по умолчанию `cortiq` декодирует каузально: это
быстрее всего на всех машинах, где мы мерили.

```bash
cargo install cortiq-cli
hf download infosave/Clover-35B-A3B-Preview-cmf clover-35b-a3b-preview-q4tp-v2.cmf --local-dir .
cortiq run clover-35b-a3b-preview-q4tp-v2.cmf --prompt "Объясни, как работает хеш-таблица." --no-think --max-tokens 512
```

Нужен cortiq 0.8.52 или новее. Файл по умолчанию — `q4tp-v2` (эксперты 4 бита
с GPTQ, 19.36 ГБ, wiki ppl 16.80); `q4tp` — простая конвертация (18.22),
`q8_2f` — почти без потерь (35 ГБ, 15.86).

Токены в секунду после промпта, `q4tp-v2`: RTX 5090 (Vulkan) — 120 на прозе и
121 на коде, промпт на 2048 токенов читается со скоростью 1143 ток/с; при 16 ГБ
видеопамяти — 54 и 53; Mac mini M4 (24 ГБ, промпт на Metal ~170 ток/с, декод на
CPU) — 13–18 и 14–18; только CPU, 15 ядер Xeon — 13.8 и 10.9.

`CMF_CLOVER_MODE` выбирает режим: `ar` (по умолчанию) — каузальный декод со
штрафами из релиза; `draft` — самоспекуляция, проход денойзинга предлагает
16 токенов, каузальный путь их проверяет (RTX 5090: 42 и 104 ток/с); `denoise` —
блочный денойзинг LowConfidence из релиза (RTX 5090: 18–26 и 63–95; M4: 3.2 и
6.5–9). `cortiq serve` даёт OpenAI-совместимый API с потоковой выдачей.

## 中文说明

Clover-35B-A3B-Preview 是 DiffuSpace 基于 Qwen3.6-35B-A3B 的扩散语言模型（35B
参数，每个 token 约 3B 激活）。它可以用 32 个 token 的块经多次去噪生成，其主干也可作
普通因果模型运行。本仓库提供可由 Rust 推理引擎 `cortiq`（0.8.52 及以上）加载的 CMF
文件；默认文件为 `q4tp-v2`（专家 4 位 GPTQ，19.36 GB）。`cortiq` 默认因果解码，这在
我们测过的所有机器上都最快。

每秒 token 数（提示之后，`q4tp-v2`）：RTX 5090（Vulkan）文本 120、代码 121，2048 个
token 的提示 1143 token/s；16 GB 显存 54 和 53；Mac mini M4（24 GB，提示走 Metal 约
170 token/s，解码走 CPU）13–18 和 14–18；仅 CPU（15 核 Xeon）13.8 和 10.9。

`CMF_CLOVER_MODE` 选择模式：`ar`（默认）为因果解码；`draft` 为自推测，一次去噪提出
16 个 token，由因果路径验证（RTX 5090：42 和 104）；`denoise` 为发布版的 LowConfidence
块去噪（RTX 5090：18–26 和 63–95；M4：3.2 和 6.5–9）。

```bash
hf download infosave/Clover-35B-A3B-Preview-cmf clover-35b-a3b-preview-q4tp-v2.cmf --local-dir .
cortiq run clover-35b-a3b-preview-q4tp-v2.cmf --prompt "你好" --no-think
```
