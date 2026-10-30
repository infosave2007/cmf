---
license: other
license_name: lfm1.0
license_link: LICENSE
library_name: cortiq
base_model: LiquidAI/LFM3-VL-2B
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
  - cmf
  - cortiq
  - quantized
  - q4tp
  - liquid
  - lfm3
  - vision
  - edge
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

# LFM3-VL-2B — CMF

[Liquid AI LFM3-VL-2B](https://huggingface.co/LiquidAI/LFM3-VL-2B) — a 2B
vision-language model: Liquid's 14-layer convolution-and-attention vision
encoder over images at native resolution (16×16 patches, each 2×2 block one
language token) on the LFM3-1.7B text model (short convolutions, full and
sliding-window attention; thinking on by default) — packaged as single
[CMF](https://github.com/infosave2007/cmf) files. They run with `cortiq`, a
Rust inference engine with no Python and no ML framework: NVIDIA, AMD and Intel
GPUs through Vulkan/DX12, and the CPU.

[English](#quick-start) · [Русский](#документация-на-русском) · [中文](#中文说明)

## Quick start

```bash
cargo install cortiq-cli
hf download infosave/LFM3-VL-2B-cmf lfm3-vl-2b-q8_2f.cmf --local-dir .
cortiq run lfm3-vl-2b-q8_2f.cmf --image photo.jpg --prompt "Describe this image." --max-tokens 1024
```

Use cortiq 0.8.49 or later; prebuilt binaries for Linux, macOS and Windows are
on the [releases page](https://github.com/infosave2007/cmf/releases). `--image`
takes a path, a data URL or an HTTP(S) URL and may be repeated; the images come
before the text in the prompt. The model thinks before it answers, and the
thinking counts toward `--max-tokens` (default 256); `--no-think` asks for a
direct answer. `--video` takes a local Y4M file or a directory of frames
(`--video-fps` gives the directory's frame rate).

## Files

| file | language model | vision encoder | size | greedy text = original | recommended for |
|---|---|---|---:|---:|---|
| `lfm3-vl-2b-q8_2f.cmf` | 8-bit | 16-bit | 2,135,818,786 B (2.14 GB) | 3 of 4 images | **default**: near-lossless |
| `lfm3-vl-2b-q4tp.cmf` | 4-bit | 16-bit | 1,306,104,354 B (1.31 GB) | 0 of 4 (same content, other words) | the smallest memory footprint |

The vision encoder and projector stay 16-bit in both files and run in float32:
on PNG images the preprocessing reproduces Transformers exactly and the rows
handed to the language model match the bf16 checkpoint run in float32 (cosine
above 0.9999 for every row). JPEG files decode slightly differently from PIL
(a few levels per pixel), which shifts single rows. *Greedy text = original*:
48 greedy tokens of a one-sentence description, compared with Transformers
float32 on four test images (640×480 and 960×686 JPEG, 1296×864 and 200×150
PNG).

## Performance

cortiq 0.8.49, NVIDIA A40 (48 GB, Vulkan), `cortiq bench --core --ignore-eos`
for text:

| file | decode, tok/s | decode at 4k | 4k-token prompt, tok/s | vision encoder, 640×480 / 1296×864 image |
|---|---:|---:|---:|---:|
| `q8_2f` | 167 | 127 | 670 | 3.0 s / 7.0 s |
| `q4tp` | 213 | 162 | 598 | 3.2 s / 6.6 s |

A 640×480 image is 300 language tokens, 1296×864 is 1080; the encoder runs on
the CPU in float32 (the attention on the GPU), which is what keeps it exact.

Apple M4 (Metal), cortiq 0.8.50: `q4tp` decodes 82 tok/s; the vision encoder
takes 0.46 s for a 640×480 image and 1.6 s for 1296×864 (Accelerate GEMMs,
Metal attention).
`--image-max-pixels N` caps the resolution before patching (fewer tokens, a
faster encoder).

## Usage

```bash
cortiq run lfm3-vl-2b-q8_2f.cmf --video clip.y4m --prompt "What happens in this video?"
cortiq run lfm3-vl-2b-q8_2f.cmf --video frames/ --video-fps 30 --prompt "Count the people who enter."
cortiq run lfm3-vl-2b-q8_2f.cmf --image a.png --image b.png --prompt "What changed between these two pictures?"
cortiq run lfm3-vl-2b-q8_2f.cmf --image chart.png --prompt "Read the values of the chart." --no-think
cortiq run lfm3-vl-2b-q8_2f.cmf --prompt "Write a haiku about rain."   # text only
```

Video is read the way Liquid's processor reads it: frames sampled at 2 per
second (4 to 768 of them), two consecutive frames per vision step, each step
preceded by its time (`<1.5 seconds>`); one pixel budget covers all frames, so
a longer clip gets smaller frames. On a 4-frame test clip the patches and the
vision rows match Transformers (cosine above 0.9999). The files carry Liquid's
recommended sampling — temperature 0.1, top-k 50, repetition penalty 1.1 — as
the `run` and `serve` defaults.

### OpenAI-compatible server

```bash
cortiq serve lfm3-vl-2b-q8_2f.cmf --port 8080

curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "lfm3-vl", "max_tokens": 1024,
       "messages": [{"role": "user", "content": [
         {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}},
         {"type": "text", "text": "What is in this picture?"}]}]}'
```

Image parts take an HTTP(S) URL, a `data:` URL or a local path; video parts
(`{"type": "video", "video": {"path": "clip.y4m"}}`, `fps` for a frame
directory) a local path. A malformed image or video is an HTTP 400. Text-only requests run as on
[LFM3-1.7B](https://huggingface.co/infosave/LFM3-1.7B-cmf), tool calls
included.

## Hardware

**NVIDIA / AMD / Intel (Vulkan, DX12).** Both files fit any card with 2 GB of
free memory; on headless Linux install
`libvulkan1 libglvnd0 libegl1 libgl1 libglx0` and set `XDG_RUNTIME_DIR=/tmp`.
**Apple silicon:** the Metal backend is the default. **CPU only:** `CMF_GPU=0`.

## Verify the download

```bash
sha256sum -c lfm3-vl-2b-q8_2f.cmf.sha256
cortiq verify lfm3-vl-2b-q8_2f.cmf
```

| file | sha256 |
|---|---|
| `lfm3-vl-2b-q8_2f.cmf` | `ee512f338d40c621a96d1c1dc0fa595a79d01fc17042a9f853098c4d04f73720` |
| `lfm3-vl-2b-q4tp.cmf` | `218d2ee4634039c287d763cc016880d63a7d0f121bfdc2dbd665e532b6eb46d8` |

## Reproduce

```bash
cortiq convert --model LiquidAI/LFM3-VL-2B --quant q8_2f \
  --output lfm3-vl-2b-q8_2f.cmf --also-output q4tp=lfm3-vl-2b-q4tp.cmf \
  --sampling temperature=0.1,top_k=50,repetition_penalty=1.1
```

Scripts (Transformers reference, image parity): [scripts/lfm3](https://github.com/infosave2007/cmf/tree/master/scripts/lfm3).

## License

These files are a modified form of Liquid AI's LFM3-VL-2B: the weights were
converted to CMF and the language model quantized (8-bit `q8_2f`, 4-bit `q4tp`)
by cortiq. They remain under the LFM Open License v1.0, a copy of which is in
this repository ([LICENSE](LICENSE)); among its terms, commercial use by an
entity with annual revenue of US$10 million or more is not licensed. See the
[upstream card](https://huggingface.co/LiquidAI/LFM3-VL-2B).

---

## Документация на русском

[LFM3-VL-2B](https://huggingface.co/LiquidAI/LFM3-VL-2B) от Liquid AI —
визуально-языковая модель на 2 млрд параметров: 14-слойный энкодер Liquid
(свёртки и внимание) по изображениям в исходном разрешении поверх текстовой
LFM3-1.7B, — в виде файлов [CMF](https://github.com/infosave2007/cmf) для
движка `cortiq` на Rust без Python.

```bash
cargo install cortiq-cli
hf download infosave/LFM3-VL-2B-cmf lfm3-vl-2b-q8_2f.cmf --local-dir .
cortiq run lfm3-vl-2b-q8_2f.cmf --image photo.jpg --prompt "Опиши это изображение." --max-tokens 1024
```

| файл | языковая модель | энкодер | размер | жадный текст = оригинал | назначение |
|---|---|---|---:|---:|---|
| `lfm3-vl-2b-q8_2f.cmf` | 8 бит | 16 бит | 2.14 ГБ | 3 из 4 изображений | **по умолчанию** |
| `lfm3-vl-2b-q4tp.cmf` | 4 бита | 16 бит | 1.31 ГБ | 0 из 4 (то же содержание) | минимум памяти |

Нужен cortiq 0.8.49 или новее. Энкодер и проектор хранятся в 16 битах и
считаются в float32: на PNG препроцессинг и строки, передаваемые языковой
модели, совпадают с Transformers (косинус выше 0.9999 для каждой строки).
На NVIDIA A40 (Vulkan) `q8_2f` генерирует 167 ток/с, энкодер обрабатывает
изображение 640×480 (300 токенов) за 3 с. `--image` можно повторять;
`cortiq serve` принимает изображения в формате OpenAI (`image_url`: URL, `data:`
или локальный путь). Видео: `--video` (файл Y4M или каталог кадров с
`--video-fps`), 2 кадра в секунду, как у процессора Liquid. На Apple M4 (Metal,
cortiq 0.8.50) `q4tp` генерирует 82 ток/с, энкодер обрабатывает 640×480 за 0.46 с. Файлы — изменённая форма LFM3-VL-2B от Liquid AI по
LFM Open License v1.0 (копия — [LICENSE](LICENSE)); коммерческое использование
организацией с годовой выручкой от 10 млн долларов США лицензией не покрывается.

## 中文说明

[LFM3-VL-2B](https://huggingface.co/LiquidAI/LFM3-VL-2B) 是 Liquid AI 的 2B
视觉语言模型：14 层卷积加注意力的视觉编码器以原生分辨率处理图像，接在
LFM3-1.7B 文本模型之上。本仓库提供可由 Rust 推理引擎 `cortiq`（0.8.49 及以上）
加载的 CMF 文件。

```bash
cargo install cortiq-cli
hf download infosave/LFM3-VL-2B-cmf lfm3-vl-2b-q8_2f.cmf --local-dir .
cortiq run lfm3-vl-2b-q8_2f.cmf --image photo.jpg --prompt "描述这张图片。" --max-tokens 1024
```

| 文件 | 语言模型 | 视觉编码器 | 大小 | 贪心输出与原模型一致 | 适用 |
|---|---|---|---:|---:|---|
| `lfm3-vl-2b-q8_2f.cmf` | 8 位 | 16 位 | 2.14 GB | 4 张图中 3 张 | **默认** |
| `lfm3-vl-2b-q4tp.cmf` | 4 位 | 16 位 | 1.31 GB | 0 张（内容相同、措辞不同） | 内存最小 |

视觉编码器与投影层以 16 位存储、以 float32 计算：PNG 输入的预处理以及送入语言模型
的向量与 Transformers 一致（每行余弦相似度高于 0.9999）。NVIDIA A40（Vulkan）上
`q8_2f` 生成 167 token/s，640×480 图像（300 个 token）编码约 3 s。`cortiq serve`
支持 OpenAI 格式的图像输入；视频用 `--video`（Y4M 文件或帧目录加 `--video-fps`），
按每秒 2 帧采样。Apple M4（Metal，cortiq 0.8.50）上 `q4tp` 生成 82 token/s，
640×480 图像编码 0.46 s。这些文件是
Liquid AI LFM3-VL-2B 的修改形式，仍受 LFM Open License v1.0 约束（副本见
[LICENSE](LICENSE)）；年收入 1000 万美元及以上的实体进行商业使用不在许可范围内。
