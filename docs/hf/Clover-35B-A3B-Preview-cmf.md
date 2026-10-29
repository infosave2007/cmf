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
---

# Clover-35B-A3B-Preview — CMF

[DiffuSpace Clover-35B-A3B-Preview](https://huggingface.co/Diffuspace/Clover-35B-A3B-Preview)
— a diffusion language model on the Qwen3.6-35B-A3B backbone (mixture of
experts: 35B parameters, ~3B active per token, 256 experts top-8 plus a shared
expert; 30 Gated DeltaNet and 10 gated full-attention layers; 128k context).
It writes the answer in 32-token blocks: each block starts as mask tokens and
is resolved by repeated denoising passes over the whole block, several tokens
at once — packaged as single [CMF](https://github.com/infosave2007/cmf) files
for `cortiq`, a Rust inference engine with no Python and no ML framework.

[English](#quick-start) · [Русский](#документация-на-русском) · [中文](#中文说明)

## Quick start

```bash
cargo install cortiq-cli
hf download infosave/Clover-35B-A3B-Preview-cmf clover-35b-a3b-preview-q4tp-v2.cmf --local-dir .
cortiq run clover-35b-a3b-preview-q4tp-v2.cmf --prompt "Write a Python function that merges two sorted lists." --no-think --max-tokens 512
```

Use cortiq 0.8.40 or later (block diffusion); prebuilt binaries for Linux,
macOS and Windows are on the [releases page](https://github.com/infosave2007/cmf/releases).
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

A *pass* is one denoising forward over the 32-row block. LowConfidence fixes
every row above the threshold after each pass (the release default 0.95), so
predictable text resolves several tokens a pass and free prose about one.
Tokens per second after the prompt, `q4tp-v2`, cortiq 0.8.40:

| hardware | backend | pass | prose | code |
|---|---|---:|---:|---:|
| RTX 5090 (32 GB) | Vulkan | 34 ms | 26 tok/s | 80–95 tok/s |
| Mac mini M4 (24 GB) | CPU, 10 cores | 1.1 s | 0.8 tok/s | 3 tok/s |

On the RTX 5090 the whole model and its head run on the card (18.8 GB). A
lower threshold trades care for speed: `CMF_CLOVER_THRESHOLD=0.8` measured
prose 25–30 and code 124–150 tok/s. `cortiq run` prints the passes, tokens
per pass and the rate after every answer.

## Usage

```bash
cortiq run clover-35b-a3b-preview-q4tp-v2.cmf                                  # interactive chat
cortiq run clover-35b-a3b-preview-q4tp-v2.cmf --prompt "..." --max-tokens 4096 # with reasoning
cortiq run clover-35b-a3b-preview-q4tp-v2.cmf --prompt "..." --no-think        # direct answer
```

Generation follows the release: greedy LowConfidence over 32-token blocks,
the block's experts restricted to the set that carries 90% of its routing,
and the release's response penalties (presence 0.9, repetition 1.05, a
gradual push toward the stop token after 8192 answer tokens and toward
`</think>` after 4096). Knobs: `CMF_CLOVER_THRESHOLD` (0–1),
`CMF_CLOVER_BLOCK` (8, 16 or 32). `--temperature` samples instead of greedy.

### OpenAI-compatible server

```bash
cortiq serve clover-35b-a3b-preview-q4tp-v2.cmf --port 8080

curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "clover", "enable_thinking": false,
       "messages": [{"role": "user", "content": "Explain how a hash map works."}]}'
```

`/v1/chat/completions` (streamed or not), `/v1/completions`, `/v1/models` and
a web dashboard on the same port. Streaming delivers each block as it
resolves.

## Hardware

**NVIDIA / AMD / Intel (Vulkan, DX12).** A card that holds the file runs every
pass as one GPU submission; a smaller one runs the first layers on the card
and the rest on the CPU (a 16 GB budget held 30 of 40 layers). `cortiq gpu`
lists the adapters; on headless Linux install `libvulkan1 libglvnd0 libegl1
libgl1 libglx0` and set `XDG_RUNTIME_DIR=/tmp`.

**Apple silicon.** Runs on the CPU; a 24 GB Mac holds the `q4tp` files.

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

LowConfidence block diffusion, the release's default decoding. The
self-speculative mode of DiffuSpace's SGLang and llama.cpp backends is not
part of this runtime. Weights derive from DiffuSpace's release (built on
Qwen3.6-35B-A3B) and remain under its Apache-2.0 terms; see the
[upstream card](https://huggingface.co/Diffuspace/Clover-35B-A3B-Preview) and
its attributions.

---

## Документация на русском

Clover-35B-A3B-Preview от DiffuSpace — диффузионная языковая модель на основе
Qwen3.6-35B-A3B (35 млрд параметров, ~3 млрд активных на токен). Ответ
пишется блоками по 32 токена: блок стартует из масок и раскрывается
повторными проходами денойзинга, по нескольку токенов за проход. Здесь она в
виде отдельных файлов CMF для движка `cortiq` на Rust.

```bash
cargo install cortiq-cli
hf download infosave/Clover-35B-A3B-Preview-cmf clover-35b-a3b-preview-q4tp-v2.cmf --local-dir .
cortiq run clover-35b-a3b-preview-q4tp-v2.cmf --prompt "Объясни, как работает хеш-таблица." --no-think --max-tokens 512
```

Нужен cortiq 0.8.40 или новее. Файл по умолчанию — `q4tp-v2` (эксперты 4 бита
с GPTQ, 19.36 ГБ, wiki ppl 16.80); `q4tp` — простая конвертация (18.22),
`q8_2f` — почти без потерь (35 ГБ, 15.86). На RTX 5090 проход по блоку
занимает 34 мс, модель целиком на карте (18.8 ГБ): проза ~26 ток/с, код
80–95 ток/с при пороге 0.95; `CMF_CLOVER_THRESHOLD=0.8` ускоряет до 25–30 и
124–150 ток/с. На Mac mini M4 (24 ГБ) модель работает на CPU (~3 ток/с на
коде). `cortiq serve` даёт OpenAI-совместимый API с потоковой выдачей.

## 中文说明

Clover-35B-A3B-Preview 是 DiffuSpace 基于 Qwen3.6-35B-A3B 的扩散语言模型（35B
参数，每个 token 约 3B 激活）。模型以 32 个 token 为一块生成：每块从掩码开始，
经多次去噪一次确定多个 token。本仓库提供可由 Rust 推理引擎 `cortiq`（0.8.40 及以上）
加载的 CMF 文件；默认文件为 `q4tp-v2`（专家 4 位 GPTQ，19.36 GB）。RTX 5090 上
每次去噪 34 ms，整模型驻留显存（18.8 GB）：阈值 0.95 时文本约 26 token/s，
代码 80–95 token/s。

```bash
hf download infosave/Clover-35B-A3B-Preview-cmf clover-35b-a3b-preview-q4tp-v2.cmf --local-dir .
cortiq run clover-35b-a3b-preview-q4tp-v2.cmf --prompt "你好" --no-think
```
