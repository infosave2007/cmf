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
  - metal
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

Use cortiq 0.8.54 or later; prebuilt binaries for Linux, macOS and Windows are
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

| stage | RTX PRO 4500 Blackwell, Vulkan | M4, 10-core GPU, Metal |
|---|---:|---:|
| encoder, one 640×640 tile | 0.10 s | 0.35 s |
| encoder, one 1024×1024 view | 0.24 s | 1.15 s |
| prompt ingest | about 2600 tok/s | about 700 tok/s |
| decode, `q8_2f` | 323 tok/s | 95 tok/s |
| decode, `q4tp` | 319 tok/s | 112 tok/s |

Decode speed holds over a document: 316 tok/s at 24 pages on the card, 137 tok/s
on the M4 with `q4tp`. A tiled page reaches its first token in 1.2 s on the card once
the weights are loaded (the first request adds about 4 s), in 4.9 s on the M4. On a
10-core Xeon quota with `CMF_GPU=0` the decoder runs at 67 tok/s and the encoder takes
16 s for a 1024×1024 view.

**CUDA (RTX A5000 24 GB, cortiq 0.8.96).** With `CMF_GPU=cuda` on a build with the CUDA
backend (`cargo install cortiq-cli --features cuda`, the NVIDIA driver and the CUDA 12.x
libraries) `q4tp` decodes at 319–320 tok/s, a tiled page reaches its first token in 1.1 s
and one page takes 8–9 s end to end with the model load (Vulkan on the same card: 20–25 s).
On CUDA take `q4tp`: the `q8_2f` file's routed experts do not run on the card yet.

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

**NVIDIA (CUDA, Vulkan) / AMD / Intel (Vulkan, DX12).** The encoder and the decoder run
on the card; 8 GB of VRAM holds `q4tp`, 12 GB `q8_2f`. **Apple silicon (Metal).** Both run on
the GPU; a page takes 4.7 GB of memory with `q4tp`, 6.2 GB with `q8_2f`. **CPU only.**
`CMF_GPU=0`: a page takes tens of seconds in the encoder.

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

Нужен cortiq 0.8.54 или новее. Страницы передаются по порядку, по одному `--image` на
каждую. Одно изображение режется на тайлы, несколько идут только общим видом 1024×1024. В
ответе один `<PAGE>` на страницу и разметка блоков с координатами в сетке 0–999.

Скорость на RTX PRO 4500: декодирование 323 ток/с (`q8_2f`) и 319 ток/с (`q4tp`), и на 24 страницах
не падает; энкодер 0.10 с на тайл и 0.24 с на общий вид. На M4 с 10-ядерным GPU (Metal): 95 и 112 ток/с,
энкодер 0.35 с и 1.15 с; страница занимает 4.7 ГБ памяти с `q4tp` и 6.2 ГБ с `q8_2f`. Только CPU:
`CMF_GPU=0`. CUDA (RTX A5000, cortiq 0.8.96, сборка `--features cuda`, `CMF_GPU=cuda`): `q4tp`
декодирует 319–320 ток/с, первый токен страницы — через 1.1 с, страница целиком с загрузкой — 8–9 с
(Vulkan на той же карте — 20–25 с); на CUDA берите `q4tp`, эксперты `q8_2f` на карте пока не идут.
Лицензия MIT.

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

需要 cortiq 0.8.54 及以上。按顺序传入各页，每页一个 `--image`。单张图片会切块，多张图片只用
1024×1024 的全局视图。输出每页一个 `<PAGE>`，并标注各区块，坐标范围 0–999。

RTX PRO 4500 上的速度：解码 323 tok/s（`q8_2f`）和 319 tok/s（`q4tp`），编码器每块 0.10 秒、
全局视图 0.24 秒。10 核 GPU 的 M4（Metal）：解码 95 和 112 tok/s，编码器 0.35 秒和 1.15 秒；一页占用
内存 4.7 GB（`q4tp`）或 6.2 GB（`q8_2f`）。仅 CPU：`CMF_GPU=0`。CUDA（RTX A5000，cortiq 0.8.96，`--features cuda` 构建，`CMF_GPU=cuda`）：`q4tp` 解码 319–320 tok/s，分块页面 1.1 秒出首个 token，含加载的整页 8–9 秒（同卡 Vulkan 20–25 秒）；CUDA 上请用 `q4tp`，`q8_2f` 的路由专家暂不在显卡上运行。许可证 MIT。
