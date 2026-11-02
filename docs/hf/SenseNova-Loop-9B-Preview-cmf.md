---
license: apache-2.0
library_name: cortiq
base_model: sensenova/SenseNova-Loop-9B-Preview
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
  - cmf
  - cortiq
  - quantized
  - q4tp
  - looped-transformer
  - mhc
  - vision
  - vulkan
  - metal
language:
  - en
  - zh
  - ru
---

# SenseNova-Loop-9B-Preview — CMF

[SenseNova-Loop-9B-Preview](https://huggingface.co/sensenova/SenseNova-Loop-9B-Preview)
— a 9B multimodal model whose decoder runs a block of 24 layers twice with
shared weights (56 layer executions from 32 layers), two residual streams with
manifold-constrained hyper-connections (mHC) inside the loop, hybrid linear /
full attention, and a vision encoder for images at native resolution —
packaged as single [CMF](https://github.com/infosave2007/cmf) files. They run
with `cortiq`, a Rust inference engine with no Python and no ML framework:
NVIDIA, AMD and Intel GPUs through Vulkan/DX12, Apple silicon through Metal,
and the CPU.

[English](#quick-start) · [Русский](#документация-на-русском) · [中文](#中文说明)

## Quick start

```bash
cargo install cortiq-cli
hf download infosave/SenseNova-Loop-9B-Preview-cmf sensenova-loop-9b-q4tp.cmf --local-dir .
cortiq run sensenova-loop-9b-q4tp.cmf --image photo.jpg --prompt "Describe this image." --no-think
```

Use cortiq 0.8.51 or later; prebuilt binaries for Linux, macOS and Windows
are on the [releases page](https://github.com/infosave2007/cmf/releases).
`--image` takes a path, a data URL or an HTTP(S) URL and may be repeated. The
model thinks before it answers unless `--no-think` is given (thinking counts
toward `--max-tokens`).

## Files

| file | language model | vision encoder | size | wikitext-2 perplexity | greedy = original | recommended for |
|---|---|---|---:|---:|---|---|
| `sensenova-loop-9b-q4tp.cmf` | 4-bit | 16-bit | 5,622,731,597 B (5.62 GB) | 13.20 | no (same content, other words) | **default**: half the memory, no measurable loss |
| `sensenova-loop-9b-q8_2f.cmf` | 8-bit | 16-bit | 9,919,558,477 B (9.92 GB) | 13.44 | yes, all three | near-lossless |

Perplexity over the first 2048 tokens of wikitext-2.

The loop plan (layers 4–27 twice, each pass with its own cache), the mHC routers
(kept in float32, as the release ships them) and the image processor travel in
the file. The vision encoder stays 16-bit and runs in float32.
*Greedy = original*: a one-sentence image description and two text prompts,
48 greedy tokens each, compared with Transformers float32.

## Performance

cortiq 0.8.51, NVIDIA A40 (48 GB, Vulkan), `cortiq bench`:

| file | decode, tok/s | 2k-token prompt, tok/s | decode at 2k |
|---|---:|---:|---:|
| `q4tp` | 29.1 | 338 | 27.8 |
| `q8_2f` | 21.3 | 293 | 20.8 |

Every token passes 56 layers (24 of them twice). The vision encoder takes 5.9 s for a 2206×1048 screenshot (9108 patches,
2277 language tokens) on the A40; a running `cortiq serve` answers a 512×512
image in 1.9 s to the first token.

Apple M4 (Metal): `q4tp` decodes 11.1 tok/s and reads a 512-token prompt at
80 tok/s; the vision encoder takes 1.2 s for a 512×512 image.

## Usage

```bash
cortiq run sensenova-loop-9b-q4tp.cmf --prompt "Write a Python function that merges two sorted lists."
cortiq run sensenova-loop-9b-q4tp.cmf --image chart.png --prompt "Read the values of the chart." --no-think
cortiq run sensenova-loop-9b-q4tp.cmf --image a.png --image b.png --prompt "What changed between these two pictures?"
cortiq serve sensenova-loop-9b-q4tp.cmf --port 8080   # OpenAI-compatible, image_url parts
```

Images go through the release's processor (16×16 patches, 2×2 merged into one
token, up to 16.7 M pixels); `--image-max-pixels N` caps the resolution (fewer
tokens, a faster encoder). The files carry the release's sampling (temperature
1.0, top-p 0.95) as the `run`/`serve` defaults; `--greedy` decodes greedily.

## Hardware

**NVIDIA / AMD / Intel (Vulkan, DX12):** `q4tp` needs about 8 GB of GPU memory,
`q8_2f` about 12 GB; on headless Linux install
`libvulkan1 libglvnd0 libegl1 libgl1 libglx0` and set `XDG_RUNTIME_DIR=/tmp`.
**Apple silicon:** Metal is the default; 16 GB of unified memory for `q4tp`.
**CPU only:** `CMF_GPU=0`.

## Verify the download

```bash
sha256sum -c sensenova-loop-9b-q4tp.cmf.sha256
```

```
c781b08216d467ecbf68de2f8cc7d6c18e27e8cc0dafe6f95099b57d550fbcd1  sensenova-loop-9b-q4tp.cmf
ce1aaabcabae37e3e767078fa90a2329581708a73232cd5880b2caf53245142a  sensenova-loop-9b-q8_2f.cmf
```

## Reproduce

```bash
cortiq convert --model sensenova/SenseNova-Loop-9B-Preview --quant q4tp \
  --output sensenova-loop-9b-q4tp.cmf --also-output q8_2f=sensenova-loop-9b-q8_2f.cmf
```

Reference and parity scripts: [scripts/four](https://github.com/infosave2007/cmf/tree/master/scripts/four).

## License

Apache-2.0, as the original release; the weights were converted to CMF and the
language model quantized (`q4tp`, `q8_2f`) by cortiq.

---

## Документация на русском

[SenseNova-Loop-9B-Preview](https://huggingface.co/sensenova/SenseNova-Loop-9B-Preview)
— мультимодальная модель на 9 млрд параметров: блок из 24 слоёв декодера
выполняется дважды с общими весами (56 проходов слоёв из 32 слоёв), внутри
петли — два остаточных потока с mHC (manifold-constrained hyper-connections),
гибрид линейного и полного внимания, энкодер изображений в исходном
разрешении. Здесь — файлы [CMF](https://github.com/infosave2007/cmf) для
движка `cortiq` на Rust без Python: Vulkan/DX12, Metal, CPU.

```bash
cargo install cortiq-cli
hf download infosave/SenseNova-Loop-9B-Preview-cmf sensenova-loop-9b-q4tp.cmf --local-dir .
cortiq run sensenova-loop-9b-q4tp.cmf --image photo.jpg --prompt "Опиши это изображение." --no-think
```

Нужен cortiq 0.8.51 или новее. | файл | языковая модель | энкодер | размер | перплексия wikitext-2 | назначение |
|---|---|---|---:|---:|---|
| `sensenova-loop-9b-q4tp.cmf` | 4 бита | 16 бит | 5.62 ГБ | 13.20 | **по умолчанию** |
| `sensenova-loop-9b-q8_2f.cmf` | 8 бит | 16 бит | 9.92 ГБ | 13.44 | почти без потерь |

План петли, роутеры mHC (в float32, как в релизе) и процессор изображений
лежат в файле. У `q8_2f` 48 жадных токенов описания изображения и двух
текстовых промптов совпадают с Transformers float32. На NVIDIA A40 (Vulkan)
`q4tp` генерирует 29.1 ток/с, `q8_2f` — 21.3; на Apple M4 (Metal) `q4tp` —
11.1 ток/с. Каждый токен проходит 56 слоёв (24 из них дважды). `cortiq serve` принимает изображения в формате OpenAI
(`image_url`). Лицензия Apache-2.0, как у оригинала.

---

## 中文说明

[SenseNova-Loop-9B-Preview](https://huggingface.co/sensenova/SenseNova-Loop-9B-Preview)
是 90 亿参数的多模态模型：解码器中 24 层组成的循环块以共享权重执行两次（32 层、
56 次层计算），循环段内用两条残差流和 mHC（流形约束超连接）传递信息，线性注意力
与全注意力混合，并配有原生分辨率的视觉编码器。本仓库提供由 Rust 推理引擎 `cortiq`
（无 Python）加载的 CMF 文件，支持 Vulkan/DX12、Metal 与 CPU。

```bash
cargo install cortiq-cli
hf download infosave/SenseNova-Loop-9B-Preview-cmf sensenova-loop-9b-q4tp.cmf --local-dir .
cortiq run sensenova-loop-9b-q4tp.cmf --image photo.jpg --prompt "描述这张图片。" --no-think
```

需要 cortiq 0.8.51 或更高版本。| 文件 | 语言模型 | 视觉编码器 | 大小 | wikitext-2 困惑度 | 用途 |
|---|---|---|---:|---:|---|
| `sensenova-loop-9b-q4tp.cmf` | 4 位 | 16 位 | 5.62 GB | 13.20 | **默认** |
| `sensenova-loop-9b-q8_2f.cmf` | 8 位 | 16 位 | 9.92 GB | 13.44 | 近乎无损 |

循环计划、mHC 路由（按发布版本保持 float32）与图像预处理参数都存放在文件中。`q8_2f`
在一条图像描述和两条文本提示上的 48 个贪心 token 与 Transformers float32 一致。
NVIDIA A40（Vulkan）上 `q4tp` 生成 29.1 tok/s，`q8_2f` 21.3 tok/s；Apple M4（Metal）上
`q4tp` 11.1 tok/s。每个 token 经过 56 层（其中 24 层两次）。
`cortiq serve` 接受 OpenAI 格式的图像（`image_url`）。许可证与原模型相同，为 Apache-2.0。
