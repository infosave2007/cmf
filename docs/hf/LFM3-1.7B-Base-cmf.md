---
license: other
license_name: lfm1.0
license_link: LICENSE
library_name: cortiq
base_model: LiquidAI/LFM3-1.7B-Base
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
  - base-model
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

# LFM3-1.7B-Base — CMF

[Liquid AI LFM3-1.7B-Base](https://huggingface.co/LiquidAI/LFM3-1.7B-Base) — the
pre-trained checkpoint behind [LFM3-1.7B](https://huggingface.co/infosave/LFM3-1.7B-cmf)
(Liquid's tri-hybrid architecture: 24 layers, 13 gated short-convolution, 6
full-attention and 5 sliding-window attention; 262,144-token context), meant
as a starting point for fine-tuning and research — packaged as single
[CMF](https://github.com/infosave2007/cmf) files. They run with `cortiq`, a
Rust inference engine with no Python and no ML framework: NVIDIA, AMD and Intel
GPUs through Vulkan/DX12, and the CPU. For chat, tool calling and agents Liquid
recommends the instruct model, [LFM3-1.7B](https://huggingface.co/infosave/LFM3-1.7B-cmf).

[English](#quick-start) · [Русский](#документация-на-русском) · [中文](#中文说明)

## Quick start

```bash
cargo install cortiq-cli
hf download infosave/LFM3-1.7B-Base-cmf lfm3-1.7b-base-q8_2f.cmf --local-dir .
cortiq ppl lfm3-1.7b-base-q8_2f.cmf -f text.txt --tokens 4096
cortiq run lfm3-1.7b-base-q8_2f.cmf --prompt "What is the capital of France?" --max-tokens 512
```

Use cortiq 0.8.49 or later; prebuilt binaries for Linux, macOS and Windows are
on the [releases page](https://github.com/infosave2007/cmf/releases). The base
model saw chat data with and without thinking traces in mid-training, in the
same template as LFM3-1.7B, so `cortiq run` and `cortiq serve` work with it;
`--no-think` asks for a direct answer.

## Files

| file | weights | size | wiki ppl | top-1 vs original | recommended for |
|---|---|---:|---:|---:|---|
| `lfm3-1.7b-base-q8_2f.cmf` | 8-bit | 1,756,991,387 B (1.76 GB) | 9.556 | 98.6 % | **default**: near-lossless |
| `lfm3-1.7b-base-q4tp.cmf` | 4-bit | 927,275,931 B (0.93 GB) | 9.984 | 87.4 % | the smallest memory footprint |
| bf16 original (Transformers) | — | 3.46 GB | 9.556 | — | — |

Perplexity over the first 4096 tokens of wikitext-2 test (with BOS); *top-1 vs
original*: positions where the file's top prediction matches the bf16
checkpoint run in float32 by Transformers over the same tokens. Norms, the
short-convolution taps and the per-head attention gates stay 16-bit in both
files. The architecture path is the one checked exactly against Transformers on
LFM3-1.7B (an f16 conversion matched float32 position for position).

## Performance

`cortiq bench --core --ignore-eos`, one stream, cortiq 0.8.49, NVIDIA A40
(48 GB, Vulkan):

| file | decode, tok/s | decode at 1k / 4k | 1k / 4k-token prompt, tok/s | first token, 4k prompt |
|---|---:|---:|---:|---:|
| `q8_2f` | 166 | 133 / 131 | 663 / 657 | 7.0 s |
| `q4tp` | 228 | 160 / 166 | 624 / 622 | 8.2 s |

Resident weights: 1.73 GB (`q8_2f`), 0.90 GB (`q4tp`).

## Hardware and context

**NVIDIA / AMD / Intel (Vulkan, DX12).** Both files fit any card with 2 GB of
free memory; on headless Linux install
`libvulkan1 libglvnd0 libegl1 libgl1 libglx0` and set `XDG_RUNTIME_DIR=/tmp`.
**CPU only:** `CMF_GPU=0`. Apple silicon has not been measured with this model
yet.

The model supports 262,144 tokens; the KV cache holds 32,768 by default and
`CMF_MAX_SEQ=262144` raises it. KV grows by about 25 KB per token (the
short-convolution layers keep a fixed state, the sliding-window layers their
last 256 positions).

## Verify the download

```bash
sha256sum -c lfm3-1.7b-base-q8_2f.cmf.sha256
cortiq verify lfm3-1.7b-base-q8_2f.cmf
```

| file | sha256 |
|---|---|
| `lfm3-1.7b-base-q8_2f.cmf` | `612bc1f196da71ca3d0b65fc0cc5472500ebb05d5a5c9ab4cedc978e6f053647` |
| `lfm3-1.7b-base-q4tp.cmf` | `b2d2ad4217c58e2c283a4e5fc492f74bfb03e90239cd98cf61f1f1bbbd8910c5` |

## Reproduce

```bash
cortiq convert --model LiquidAI/LFM3-1.7B-Base --quant q8_2f \
  --output lfm3-1.7b-base-q8_2f.cmf --also-output q4tp=lfm3-1.7b-base-q4tp.cmf
```

Scripts: [scripts/lfm3](https://github.com/infosave2007/cmf/tree/master/scripts/lfm3).

## License

These files are a modified form of Liquid AI's LFM3-1.7B-Base: the weights were
converted to CMF and quantized (8-bit `q8_2f`, 4-bit `q4tp`) by cortiq. They
remain under the LFM Open License v1.0, a copy of which is in this repository
([LICENSE](LICENSE)); among its terms, commercial use by an entity with annual
revenue of US$10 million or more is not licensed. See the
[upstream card](https://huggingface.co/LiquidAI/LFM3-1.7B-Base).

---

## Документация на русском

[LFM3-1.7B-Base](https://huggingface.co/LiquidAI/LFM3-1.7B-Base) от Liquid AI —
предобученный чекпойнт, из которого сделана
[LFM3-1.7B](https://huggingface.co/infosave/LFM3-1.7B-cmf) (трёхгибридная
архитектура Liquid: 24 слоя — 13 короткой свёртки, 6 полного внимания, 5 со
скользящим окном; контекст 262 144 токена), — основа для дообучения и
исследований, в виде файлов [CMF](https://github.com/infosave2007/cmf) для
движка `cortiq` на Rust. Для чата, инструментов и агентов Liquid советует
инструктивную LFM3-1.7B.

```bash
cargo install cortiq-cli
hf download infosave/LFM3-1.7B-Base-cmf lfm3-1.7b-base-q8_2f.cmf --local-dir .
cortiq run lfm3-1.7b-base-q8_2f.cmf --prompt "Какая столица Франции?" --max-tokens 512
```

| файл | веса | размер | ppl wiki | top-1 к оригиналу | назначение |
|---|---|---:|---:|---:|---|
| `lfm3-1.7b-base-q8_2f.cmf` | 8 бит | 1.76 ГБ | 9.556 | 98.6 % | **по умолчанию**: почти без потерь |
| `lfm3-1.7b-base-q4tp.cmf` | 4 бита | 0.93 ГБ | 9.984 | 87.4 % | минимум памяти |
| bf16-оригинал (Transformers) | — | 3.46 ГБ | 9.556 | — | — |

Нужен cortiq 0.8.49 или новее. На NVIDIA A40 (Vulkan) `q8_2f` генерирует 166 ток/с
и читает промпт 4k со скоростью 657 ток/с, `q4tp` — 228 и 622 ток/с. Контекст до
262 144 токенов: `CMF_MAX_SEQ=262144`. Apple silicon с этой моделью пока не
измерялся. Файлы — изменённая форма LFM3-1.7B-Base от Liquid AI (конвертация в CMF
и квантование cortiq) по LFM Open License v1.0 (копия — [LICENSE](LICENSE));
коммерческое использование организацией с годовой выручкой от 10 млн долларов США
лицензией не покрывается.

## 中文说明

[LFM3-1.7B-Base](https://huggingface.co/LiquidAI/LFM3-1.7B-Base) 是 Liquid AI
的预训练检查点，[LFM3-1.7B](https://huggingface.co/infosave/LFM3-1.7B-cmf) 即由其
训练而来（三混合架构：24 层，13 层短卷积、6 层全注意力、5 层滑动窗口注意力；
262,144 token 上下文），适合作为微调与研究的起点。本仓库提供可由 Rust 推理引擎
`cortiq`（0.8.49 及以上）加载的 CMF 文件。对话、工具调用与智能体请使用
LFM3-1.7B。

```bash
cargo install cortiq-cli
hf download infosave/LFM3-1.7B-Base-cmf lfm3-1.7b-base-q8_2f.cmf --local-dir .
cortiq run lfm3-1.7b-base-q8_2f.cmf --prompt "你好" --max-tokens 512
```

| 文件 | 权重 | 大小 | wiki ppl | 与原模型 top-1 一致率 | 适用 |
|---|---|---:|---:|---:|---|
| `lfm3-1.7b-base-q8_2f.cmf` | 8 位 | 1.76 GB | 9.556 | 98.6 % | **默认**：近无损 |
| `lfm3-1.7b-base-q4tp.cmf` | 4 位 | 0.93 GB | 9.984 | 87.4 % | 内存最小 |
| bf16 原模型（Transformers） | — | 3.46 GB | 9.556 | — | — |

NVIDIA A40（Vulkan）上 `q8_2f` 生成 166 token/s、4k 提示词 657 token/s，`q4tp`
为 228 与 622 token/s。Apple silicon 尚未针对该模型测量。这些文件是 Liquid AI
LFM3-1.7B-Base 的修改形式（由 cortiq 转换为 CMF 并量化），仍受 LFM Open License
v1.0 约束（副本见 [LICENSE](LICENSE)）；年收入 1000 万美元及以上的实体进行商业
使用不在许可范围内。
