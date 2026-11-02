---
license: apache-2.0
library_name: cortiq
base_model: openbmb/MiniCPM-OCR-1.0
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
  - cmf
  - cortiq
  - quantized
  - q8_2f
  - q4tp
  - ocr
  - document-parsing
  - vision
  - vulkan
language:
  - en
  - ru
  - zh
---

# MiniCPM-OCR-1.0 — CMF

[OpenBMB MiniCPM-OCR-1.0](https://huggingface.co/openbmb/MiniCPM-OCR-1.0) — a 1.3B
OCR model for text, tables, formulas, charts, seals and structured extraction:
a SigLIP vision encoder on a Qwen3.5 language model, trained on top of
MiniCPM-V 4.6. Packaged as single [CMF](https://github.com/infosave2007/cmf)
files for `cortiq`, a Rust inference engine with no Python and no ML framework.

[English](#quick-start) · [Русский](#документация-на-русском) · [中文](#中文说明)

## Quick start

```bash
cargo install cortiq-cli
hf download infosave/MiniCPM-OCR-1.0-cmf minicpm-ocr-1.0-q8_2f.cmf --local-dir .
cortiq run minicpm-ocr-1.0-q8_2f.cmf --image page.png --prompt "Text Recognition:" --max-tokens 2048
```

Use cortiq 0.8.53 or later; prebuilt binaries for Linux, macOS and Windows are
on the [releases page](https://github.com/infosave2007/cmf/releases). `--image`
takes a path, a data URL or an HTTP(S) URL.

## Prompts

| task | prompt | output |
|---|---|---|
| text, paragraph, title, page | `Text Recognition:` | plain text |
| table | `Table Recognition:` | table |
| formula | `Formula Recognition:` | LaTeX |
| seal | `Seal Recognition:` | plain text |
| chart | `Chart to Table:` | table |
| fields | `Extract the "field", .... Format the output as a JSON object with keys: "field", .... If an item is not in the image, use "###".` | JSON |

The upstream SDK adds layout detection (PP-DocLayoutV3) and reading order on top
of this model; these files cover the recognition step.

## Files

| file | language model | vision encoder | size | 300 greedy tokens against the original |
|---|---|---|---:|---|
| `minicpm-ocr-1.0-q8_2f.cmf` | 8-bit | 16-bit | 1.67 GB | identical; **default** |
| `minicpm-ocr-1.0-q4tp.cmf` | 4-bit | 16-bit | 1.43 GB | same text; its tokens run 48 characters short of the 300 |

The original is Transformers 5.21 float32 on a printed page. The preprocessing
is exact and the rows the encoder hands to the language model have a cosine of
at least 0.9999998.

## Performance

Measured on one A4 page at 144 dpi (an overview and nine slices, 2470 image
tokens), RTX PRO 4500 Blackwell (32 GB), Vulkan:

| file | decode | image and prompt to the first token |
|---|---:|---:|
| `q8_2f` | 254 tok/s | 2.0 s |
| `q4tp` | 267 tok/s | 1.9 s |

The encoder takes 0.13 s per view of those 1.3 s; the first request also uploads
the weights (4 s more). On a 10-core Xeon quota with `CMF_GPU=0` the language model
decodes at 61 tok/s.

## Server

```bash
cortiq serve minicpm-ocr-1.0-q8_2f.cmf --port 8080

curl http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "minicpm-ocr",
  "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "https://example.com/page.png"}},
    {"type": "text", "text": "Text Recognition:"}
  ]}]
}'
```

## Hardware

**NVIDIA / AMD / Intel (Vulkan, DX12).** The encoder and the language model run on
the card; 4 GB of VRAM holds either file. **CPU only.** `CMF_GPU=0`.

## Scope

Images only: the upstream video path, the `16x` downsampling mode and the DSpark
draft are not part of these files.

## License

Apache-2.0, as the [original](https://huggingface.co/openbmb/MiniCPM-OCR-1.0).

---

## Документация на русском

[MiniCPM-OCR-1.0](https://huggingface.co/openbmb/MiniCPM-OCR-1.0) от OpenBMB — OCR-модель
на 1.3B для текста, таблиц, формул, диаграмм, печатей и извлечения полей: энкодер
SigLIP и языковая модель Qwen3.5. Файлы [CMF](https://github.com/infosave2007/cmf)
для `cortiq`, движка на Rust без Python.

```bash
cargo install cortiq-cli
hf download infosave/MiniCPM-OCR-1.0-cmf minicpm-ocr-1.0-q8_2f.cmf --local-dir .
cortiq run minicpm-ocr-1.0-q8_2f.cmf --image page.png --prompt "Text Recognition:" --max-tokens 2048
```

Нужен cortiq 0.8.53 или новее. Подсказки: `Text Recognition:`, `Table Recognition:`,
`Formula Recognition:`, `Seal Recognition:`, `Chart to Table:`. Детектор разметки
и порядок чтения из SDK вендора в файлы не входят: здесь шаг распознавания.

`q8_2f` на странице A4 выдаёт те же 300 токенов, что и оригинал в float32; `q4tp`
меньше и даёт тот же текст на этой странице. Скорость на RTX PRO 4500: 254 и
267 ток/с. Только CPU: `CMF_GPU=0`. Лицензия Apache-2.0.

---

## 中文说明

[MiniCPM-OCR-1.0](https://huggingface.co/openbmb/MiniCPM-OCR-1.0) 是 OpenBMB 的 1.3B
OCR 模型，用于文本、表格、公式、图表、印章和字段提取：SigLIP 视觉编码器加 Qwen3.5
语言模型。这里是 `cortiq`（无 Python 的 Rust 推理引擎）使用的
[CMF](https://github.com/infosave2007/cmf) 文件。

```bash
cargo install cortiq-cli
hf download infosave/MiniCPM-OCR-1.0-cmf minicpm-ocr-1.0-q8_2f.cmf --local-dir .
cortiq run minicpm-ocr-1.0-q8_2f.cmf --image page.png --prompt "Text Recognition:" --max-tokens 2048
```

需要 cortiq 0.8.53 及以上。提示词：`Text Recognition:`、`Table Recognition:`、
`Formula Recognition:`、`Seal Recognition:`、`Chart to Table:`。上游 SDK 的版面检测和
阅读顺序不包含在内，这些文件负责识别这一步。

在一页 A4 上，`q8_2f` 的 300 个贪心 token 与 float32 原模型一致；`q4tp` 更小，这一页的文本
相同。RTX PRO 4500 上的速度：254 和 267 tok/s。仅 CPU：`CMF_GPU=0`。许可证
Apache-2.0。
