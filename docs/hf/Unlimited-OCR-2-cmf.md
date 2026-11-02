---
license: mit
library_name: cortiq
base_model: baidu/Unlimited-OCR-2
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
  - moe
  - vision
  - vulkan
language:
  - en
  - ru
  - zh
---

# Unlimited-OCR-2 — CMF

[Baidu Unlimited-OCR-2](https://huggingface.co/baidu/Unlimited-OCR-2) — a document
parser that reads many pages in one pass: a DeepEncoder (SAM ViT-B feeding
CLIP-L) on a 12-layer mixture-of-experts decoder whose attention reads at most
512 tokens per step, however long the document is. Packaged as single
[CMF](https://github.com/infosave2007/cmf) files for `cortiq`, a Rust inference
engine with no Python and no ML framework.

[English](#quick-start) · [Русский](#документация-на-русском) · [中文](#中文说明)

## Quick start

```bash
cargo install cortiq-cli
hf download infosave/Unlimited-OCR-2-cmf unlimited-ocr-2-q8_2f.cmf --local-dir .
cortiq run unlimited-ocr-2-q8_2f.cmf --image page1.png --image page2.png --prompt "Multi page parsing." --max-tokens 8192
```

Use cortiq 0.8.53 or later; prebuilt binaries for Linux, macOS and Windows are
on the [releases page](https://github.com/infosave2007/cmf/releases).

Pass the pages in order with one `--image` each. One image is tiled (up to 16
640×640 tiles and a 1024×1024 view); several take the 1024×1024 view only. The
answer has one `<PAGE>` per input page and marks every block:

```text
<PAGE><BLOCK>title<LEVEL1>[347, 134, 652, 154]<FORK>Unlimited OCR Works 2
<BLOCK>text[200, 161, 797, 181]<FORK>Pioneering the OCR-3.0 Era: ...
```

Coordinates are on a 0–999 grid. Generation stops when every page is out; give
`--max-tokens` room for the whole document.

## Files

| file | decoder | encoder | size | first 300 greedy tokens of one page against the original |
|---|---|---|---:|---|
| `unlimited-ocr-2-q8_2f.cmf` | 8-bit | 16-bit | 3.78 GB | identical; **default** |
| `unlimited-ocr-2-q4tp.cmf` | 4-bit | 16-bit | 2.37 GB | 98.5 % of the characters |

The original is Transformers 4.52 float32; preprocessing matches it (Pillow's
fixed-point bicubic). Over longer documents the coordinates and the order of
neighbouring blocks move at near-ties, as they do in the original itself between
float32 and its own bf16: on six pages both files take the same branch as the
bf16 build where it leaves float32.

## Performance

Measured on an RTX PRO 4500 Blackwell (32 GB), Vulkan:

| stage | |
|---|---:|
| encoder, one 640×640 tile | 0.10 s |
| encoder, one 1024×1024 view | 0.24 s |
| prompt ingest | about 2600 tok/s |
| decode, `q8_2f` | 323 tok/s |
| decode, `q4tp` | 319 tok/s |

Decode speed holds over a document: 316 tok/s at 24 pages. A tiled page reaches
its first token in 1.2 s once the weights are on the card (the first request adds
about 4 s). On a 10-core Xeon quota with `CMF_GPU=0` the decoder runs at 67 tok/s and the
encoder takes 16 s for a 1024×1024 view.

## Server

```bash
cortiq serve unlimited-ocr-2-q8_2f.cmf --port 8080

curl http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "unlimited-ocr",
  "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "https://example.com/page1.png"}},
    {"type": "image_url", "image_url": {"url": "https://example.com/page2.png"}},
    {"type": "text", "text": "Multi page parsing."}
  ]}]
}'
```

## Hardware

**NVIDIA / AMD / Intel (Vulkan, DX12).** The encoder and the decoder run on the
card; 8 GB of VRAM holds `q4tp`, 12 GB `q8_2f`. **CPU only.** `CMF_GPU=0`: a
page takes tens of seconds in the encoder.

## Scope

The model reads images: a request without one is refused. The trained
`<PAGE><image>` layout is always used; custom `<image>` placement is not exposed.

## License

MIT, as the [original](https://huggingface.co/baidu/Unlimited-OCR-2).

---

## Документация на русском

[Unlimited-OCR-2](https://huggingface.co/baidu/Unlimited-OCR-2) от Baidu разбирает
много страниц за один проход: DeepEncoder (SAM ViT-B и CLIP-L) и декодер на 12 слоёв со
смесью экспертов, внимание которого читает не больше 512 токенов на шаг при любой длине
документа. Файлы [CMF](https://github.com/infosave2007/cmf) для `cortiq`, движка на Rust
без Python.

```bash
cargo install cortiq-cli
hf download infosave/Unlimited-OCR-2-cmf unlimited-ocr-2-q8_2f.cmf --local-dir .
cortiq run unlimited-ocr-2-q8_2f.cmf --image page1.png --image page2.png --prompt "Multi page parsing." --max-tokens 8192
```

Нужен cortiq 0.8.53 или новее. Страницы передаются по порядку, по одному `--image` на
каждую. Одно изображение режется на тайлы, несколько идут только общим видом 1024×1024. В
ответе один `<PAGE>` на страницу и разметка блоков с координатами в сетке 0–999.

Скорость на RTX PRO 4500: декодирование 323 ток/с (`q8_2f`) и 319 ток/с (`q4tp`), и на 24 страницах
не падает; энкодер 0.10 с на тайл и 0.24 с на общий вид. Только CPU: `CMF_GPU=0`. Лицензия MIT.

---

## 中文说明

[Unlimited-OCR-2](https://huggingface.co/baidu/Unlimited-OCR-2) 是百度的文档解析模型，一次读取
多页：DeepEncoder（SAM ViT-B 接 CLIP-L）加 12 层混合专家解码器，无论文档多长，注意力每步最多读取
512 个 token。这里是 `cortiq`（无 Python 的 Rust 推理引擎）使用的
[CMF](https://github.com/infosave2007/cmf) 文件。

```bash
cargo install cortiq-cli
hf download infosave/Unlimited-OCR-2-cmf unlimited-ocr-2-q8_2f.cmf --local-dir .
cortiq run unlimited-ocr-2-q8_2f.cmf --image page1.png --image page2.png --prompt "Multi page parsing." --max-tokens 8192
```

需要 cortiq 0.8.53 及以上。按顺序传入各页，每页一个 `--image`。单张图片会切块，多张图片只用
1024×1024 的全局视图。输出每页一个 `<PAGE>`，并标注各区块，坐标范围 0–999。

RTX PRO 4500 上的速度：解码 323 tok/s（`q8_2f`）和 319 tok/s（`q4tp`），编码器每块 0.10 秒、
全局视图 0.24 秒。仅 CPU：`CMF_GPU=0`。许可证 MIT。
