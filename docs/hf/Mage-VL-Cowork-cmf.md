---
license: apache-2.0
library_name: cortiq
base_model: microsoft/Mage-VL-Cowork
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
  - cmf
  - cortiq
  - quantized
  - q8_2f
  - q4tp
  - computer-use
  - gui-agent
  - vision
  - vulkan
  - metal
language:
  - en
  - ru
  - zh
---

# Mage-VL-Cowork — CMF

[Microsoft Mage-VL-Cowork](https://huggingface.co/microsoft/Mage-VL-Cowork) — a 4B
vision-language model for agents that work a computer through the screen and
the terminal (66.0 % on OSWorld-Verified): the OneVision vision encoder over
screenshots at native resolution on a Qwen3-4B language model that reasons in
`<think>` and calls `computer_use` / `bash` tools — packaged as single
[CMF](https://github.com/infosave2007/cmf) files. They run with `cortiq`, a Rust
inference engine with no Python and no ML framework: NVIDIA, AMD and Intel GPUs
through Vulkan/DX12, Apple silicon through Metal, and the CPU.

[English](#quick-start) · [Русский](#документация-на-русском) · [中文](#中文说明)

## Quick start

```bash
cargo install cortiq-cli
hf download infosave/Mage-VL-Cowork-cmf mage-vl-cowork-q8_2f.cmf --local-dir .
cortiq run mage-vl-cowork-q8_2f.cmf --image screenshot.png --prompt "Task: open the Downloads folder." --max-tokens 1024
```

Use cortiq 0.8.51 or later; prebuilt binaries for Linux, macOS and Windows
are on the [releases page](https://github.com/infosave2007/cmf/releases).
`--image` takes a path, a data URL or an HTTP(S) URL and may be repeated (the
images come before the text). The model thinks before it acts; the thinking
counts toward `--max-tokens`. For agent use, pass the release's system prompt
and tool schema (`MAGE_HYBRID_SYSTEM_PROMPT` on the
[upstream card](https://huggingface.co/microsoft/Mage-VL-Cowork)) with
`--system` or through the server; coordinates are on a 0–999 grid.

## Files

| file | language model | vision encoder | size | wikitext-2 perplexity | greedy = original | recommended for |
|---|---|---|---:|---:|---|---|
| `mage-vl-cowork-q8_2f.cmf` | 8-bit | 16-bit | 5,096,045,502 B (5.10 GB) | 125.6 | yes | **default**: near-lossless |
| `mage-vl-cowork-q4tp.cmf` | 4-bit | 16-bit | 2,977,659,838 B (2.98 GB) | 128.7 | no (same content, other words) | the smallest memory footprint |

Perplexity over the first 2048 tokens of wikitext-2; the original in Transformers
scores 121.8 (an agent model, not a language model of prose).

The vision encoder and the merger stay 16-bit in both files and run in
float32: the preprocessing and the rows handed to the language model match
Transformers float32 (cosine 1.000000 for every row on the CPU, above 0.999 with
the GPU's matrix units). *Greedy = original*: 48 greedy tokens of an image
description against Transformers 4.57.3 float32.

## Performance

cortiq 0.8.51, NVIDIA A40 (48 GB, Vulkan), `cortiq bench`:

| file | decode, tok/s | 2k-token prompt, tok/s | decode at 2k |
|---|---:|---:|---:|
| `q8_2f` | 66.8 | 855 | 53.2 |
| `q4tp` | 90.2 | 863 | 64.9 |

A 2206×1048 screenshot is 9108 patches and 2277 language tokens; the vision
encoder takes 4.8 s for it on the A40 (GEMMs and attention on the GPU). A
running `cortiq serve` answers a 512×512 image in 1.7 s to the first token.

Apple M4 (Metal): `q4tp` decodes 36 tok/s and reads a 2k-token prompt at
295 tok/s; the vision encoder takes 0.69 s for 512×512 and 10.8 s for the
2206×1048 screenshot.

## Usage

```bash
cortiq run mage-vl-cowork-q8_2f.cmf --system "$(cat mage_system_prompt.txt)" \
  --image screen.png --prompt "Task: merge cells A1:D1 in budget.xlsx and save it."
cortiq run mage-vl-cowork-q8_2f.cmf --image a.png --image b.png --prompt "What changed on the screen?"
cortiq run mage-vl-cowork-q8_2f.cmf --image screen.png --image-max-pixels 1000000 --prompt "Find the search box."
```

`--image-max-pixels N` caps the screenshot resolution before patching (fewer
tokens, a faster encoder); the release's processor allows up to 4 M pixels.

### OpenAI-compatible server

```bash
cortiq serve mage-vl-cowork-q8_2f.cmf --port 8080

curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "mage-vl", "max_tokens": 1024,
       "messages": [{"role": "system", "content": "<MAGE_HYBRID_SYSTEM_PROMPT>"},
                    {"role": "user", "content": [
         {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
         {"type": "text", "text": "Task: close the dialog."}]}]}'
```

Image parts take an HTTP(S) URL, a `data:` URL or a local path; a malformed
image is an HTTP 400. Tool calls come back in the model's `<tool_call>` JSON.

## Hardware

**NVIDIA / AMD / Intel (Vulkan, DX12):** `q8_2f` needs about 6 GB of GPU memory,
`q4tp` about 4 GB; on headless Linux install
`libvulkan1 libglvnd0 libegl1 libgl1 libglx0` and set `XDG_RUNTIME_DIR=/tmp`.
**Apple silicon:** Metal is the default. **CPU only:** `CMF_GPU=0`.

## Verify the download

```bash
sha256sum -c mage-vl-cowork-q8_2f.cmf.sha256
```

```
ca71b28811318523ff0b9742c0c1f0be5590f7bcfc10815cb0054f92dd170b88  mage-vl-cowork-q8_2f.cmf
aedc066d74bdb0f570541b2c53a74452b5f99e3fc39b364b223a8f0b351ead1f  mage-vl-cowork-q4tp.cmf
```

## Reproduce

```bash
cortiq convert --model microsoft/Mage-VL-Cowork --quant q8_2f \
  --output mage-vl-cowork-q8_2f.cmf --also-output q4tp=mage-vl-cowork-q4tp.cmf
```

Reference and parity scripts: [scripts/four](https://github.com/infosave2007/cmf/tree/master/scripts/four).

## License

Apache-2.0, as the original release; the weights were converted to CMF and the
language model quantized (`q8_2f`, `q4tp`) by cortiq.

---

## Документация на русском

[Mage-VL-Cowork](https://huggingface.co/microsoft/Mage-VL-Cowork) от Microsoft —
визуально-языковая модель на 4 млрд параметров для агентов, работающих с
компьютером через экран и терминал (66,0 % на OSWorld-Verified): энкодер
OneVision по скриншотам в исходном разрешении поверх языковой Qwen3-4B,
которая рассуждает в `<think>` и вызывает инструменты `computer_use` / `bash`.
Здесь — файлы [CMF](https://github.com/infosave2007/cmf) для движка `cortiq`
на Rust без Python: Vulkan/DX12, Metal, CPU.

```bash
cargo install cortiq-cli
hf download infosave/Mage-VL-Cowork-cmf mage-vl-cowork-q8_2f.cmf --local-dir .
cortiq run mage-vl-cowork-q8_2f.cmf --image screenshot.png --prompt "Task: open the Downloads folder." --max-tokens 1024
```

Нужен cortiq 0.8.51 или новее. | файл | языковая модель | энкодер | размер | перплексия wikitext-2 | назначение |
|---|---|---|---:|---:|---|
| `mage-vl-cowork-q8_2f.cmf` | 8 бит | 16 бит | 5.10 ГБ | 125.6 | **по умолчанию** |
| `mage-vl-cowork-q4tp.cmf` | 4 бита | 16 бит | 2.98 ГБ | 128.7 | минимум памяти |

Энкодер и мерджер хранятся в 16 битах и считаются в float32: строки,
передаваемые языковой модели, совпадают с Transformers float32 (косинус
1.000000 на CPU), 48 жадных токенов описания изображения у `q8_2f` — тоже.
На NVIDIA A40 (Vulkan) `q8_2f` генерирует 66.8 ток/с, `q4tp` — 90.2; скриншот
2206×1048 (2277 токенов) энкодер обрабатывает за 4.8 с. На Apple M4 (Metal)
`q4tp` генерирует 36 ток/с. Агенту нужен системный промпт и схема
инструментов релиза (`MAGE_HYBRID_SYSTEM_PROMPT` в
[исходной карточке](https://huggingface.co/microsoft/Mage-VL-Cowork)) через
`--system` или сервер; координаты — в сетке 0–999. `cortiq serve` принимает
изображения в формате OpenAI (`image_url`: URL, `data:` или локальный путь).
Лицензия Apache-2.0, как у оригинала.

---

## 中文说明

[Mage-VL-Cowork](https://huggingface.co/microsoft/Mage-VL-Cowork) 是 Microsoft 的 40 亿参数
视觉语言模型，面向通过屏幕与终端操作电脑的智能体（OSWorld-Verified 66.0%）：OneVision
视觉编码器以原生分辨率处理截图，语言部分为 Qwen3-4B，在 `<think>` 中推理并调用
`computer_use` / `bash` 工具。本仓库提供由 Rust 推理引擎 `cortiq`（无 Python）加载的
CMF 文件，支持 Vulkan/DX12、Metal 与 CPU。

```bash
cargo install cortiq-cli
hf download infosave/Mage-VL-Cowork-cmf mage-vl-cowork-q8_2f.cmf --local-dir .
cortiq run mage-vl-cowork-q8_2f.cmf --image screenshot.png --prompt "Task: open the Downloads folder." --max-tokens 1024
```

需要 cortiq 0.8.51 或更高版本。| 文件 | 语言模型 | 视觉编码器 | 大小 | wikitext-2 困惑度 | 用途 |
|---|---|---|---:|---:|---|
| `mage-vl-cowork-q8_2f.cmf` | 8 位 | 16 位 | 5.10 GB | 125.6 | **默认** |
| `mage-vl-cowork-q4tp.cmf` | 4 位 | 16 位 | 2.98 GB | 128.7 | 最小内存占用 |

视觉编码器与融合层以 16 位存储、按 float32 计算：交给语言模型的特征与 Transformers
float32 一致（CPU 上余弦 1.000000），`q8_2f` 的 48 个贪心解码 token 也与原模型一致。
NVIDIA A40（Vulkan）上 `q8_2f` 生成 66.8 tok/s，`q4tp` 90.2 tok/s；2206×1048 截图
（2277 个 token）编码耗时 4.8 s。Apple M4（Metal）上 `q4tp` 生成 36 tok/s。智能体用法需
通过 `--system` 或服务端传入发布方的系统提示与工具定义（见
[原始模型卡](https://huggingface.co/microsoft/Mage-VL-Cowork) 中的
`MAGE_HYBRID_SYSTEM_PROMPT`），坐标为 0–999 网格。`cortiq serve` 接受 OpenAI 格式的
图像（`image_url`：URL、`data:` 或本地路径）。许可证与原模型相同，为 Apache-2.0。
