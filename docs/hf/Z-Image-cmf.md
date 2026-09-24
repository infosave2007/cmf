---
license: apache-2.0
base_model: Tongyi-MAI/Z-Image
tags:
  - cmf
  - cortiq
  - text-to-image
  - quantized
language:
  - en
  - ru
  - zh
---

# Z-Image — CMF

[Z-Image](https://huggingface.co/Tongyi-MAI/Z-Image) by Tongyi-MAI, packaged as a single
[CMF](https://github.com/infosave2007/cmf) file: the 6B image transformer, the Qwen3-4B
text encoder and the VAE together, with the generation recipe stored inside. It runs
with `cortiq`, a Rust engine with no Python: NVIDIA GPUs through Vulkan
(tensor cores), Apple silicon through Metal, and a CPU fallback everywhere.

[English](#quick-start) · [Русский](#документация-на-русском) · [中文](#中文文档)

## Quick start

```bash
cargo install cortiq-cli          # 0.7.5 or later; prebuilt binaries: github.com/infosave2007/cmf/releases
hf download infosave/Z-Image-cmf z-image.cmf --local-dir .
cortiq imagine z-image.cmf --prompt "A cat sitting on a windowsill at sunset, photorealistic" --negative-prompt "blurry, low quality"
```

No flags needed: the file carries the base recipe (1024×1024, 28 steps, guidance 4 with CFG, shift 6). Set `XDG_RUNTIME_DIR=/tmp` on a headless Linux box.

## File

| file | size | inside |
|---|---:|---|
| `z-image.cmf` | 10.5 GB | DiT 8-bit (`q8_2f`), Qwen3-4B text encoder 8-bit, VAE |

sha256 `d7138e6fe1b6af76ef7c7eea9e8524cd1e25bd01dcb2d002bc7abda26b058156` (`z-image.cmf.sha256` is next to it). 8-bit was chosen by measurement: the 4-bit
transformer loses 3–15 dB of image PSNR, the 8-bit one is comparable to bf16 inference.

## Performance

RTX 3090, Vulkan, one image per process, no flags (cortiq 0.7.5):

| size | time to image | of which DiT steps | VAE |
|---|---:|---:|---:|
| 512×512 | 16.2 s | 13.6 s (28 steps, CFG) | 0.12 s |
| 1024×1024 | 60 s | 57 s (28 steps, CFG) | 0.49 s |

Mac mini M4 (24 GB), Metal, no flags:

| size | time to image | of which DiT steps | VAE |
|---|---:|---:|---:|
| 512×512 | 3.9 min | 3.9 min (28 steps, CFG) | 1.3 s |
| 1024×1024 | 21 min | 21 min (28 steps, CFG) | 5.0 s |

About 8 GB of memory; the M4's GPU is compute-bound, so this is its ceiling —
use fewer `--steps` and 512² for drafts, or Z-Image-Turbo.

RTX 3090: peak VRAM 14–15 GB (16 GB cards fit). Per image, the DiT runs at the speed of diffusers
bf16 on the same card and the whole process starts about four times faster.

## Options

| option | meaning |
|---|---|
| `--width`, `--height` | multiples of 16 (default 1024×1024) |
| `--steps`, `--seed`, `--num-images N` | image i uses seed+i |
| `--cfg G`, `--negative-prompt` | guidance scale and negative prompt (defaults: 4 and "") |
| `--cfg-normalization [C]`, `--cfg-truncation T` | the diffusers CFG extras |
| `--shift`, `--max-sequence-length` | scheduler shift, prompt token cap (512) |
| `--out` | `.png` (default `out.png`), `.jpg`, `.ppm` |

`CMF_ZIMAGE_PROF=1` prints stage times. `CMF_GPU=0` runs everything on the CPU (minutes per image).

The base model answers to negative prompts and guidance 3–5 (28–50 steps); for the fastest results use [Z-Image-Turbo](https://huggingface.co/infosave/Z-Image-Turbo-cmf) (8 steps).

## Samples

1024², seed 7, default settings: [samples/](https://huggingface.co/infosave/Z-Image-cmf/tree/main/samples).

## Verify

```bash
sha256sum -c z-image.cmf.sha256
cortiq info z-image.cmf
```

---

## Документация на русском

[Z-Image](https://huggingface.co/Tongyi-MAI/Z-Image) от Tongyi-MAI одним файлом
[CMF](https://github.com/infosave2007/cmf): трансформер 6B, текстовый энкодер Qwen3-4B и VAE
вместе, рецепт генерации записан в файл. Запускается движком `cortiq` на Rust без Python:
NVIDIA через Vulkan (тензорные ядра), Apple silicon через Metal, везде есть путь на CPU.

```bash
cargo install cortiq-cli          # 0.7.5 или новее; готовые бинарники: github.com/infosave2007/cmf/releases
hf download infosave/Z-Image-cmf z-image.cmf --local-dir .
cortiq imagine z-image.cmf --prompt "Кот на подоконнике на закате, фотореализм" --negative-prompt "blurry, low quality"
```

Флаги не нужны: в файле записан рецепт базовой модели (1024×1024, 28 шагов, guidance 4 с CFG, сдвиг 6). На Linux без дисплея задайте `XDG_RUNTIME_DIR=/tmp`.

**Файл:** `z-image.cmf`, 10.5 ГБ — DiT 8 бит (`q8_2f`), энкодер Qwen3-4B 8 бит, VAE;
sha256 `d7138e6fe1b6af76…` (полностью в `z-image.cmf.sha256`). 8 бит выбраны по замеру: 4-битный трансформер
теряет 3–15 дБ PSNR картинки, 8-битный сравним с выводом в bf16.

**Скорость** (RTX 3090, Vulkan, одна картинка на процесс, без флагов, cortiq 0.7.5):

| размер | картинка целиком | из них шаги DiT | VAE |
|---|---:|---:|---:|
| 512×512 | 16.2 с | 13.6 с (28 шагов, CFG) | 0.12 с |
| 1024×1024 | 60 с | 57 с (28 шагов, CFG) | 0.49 с |

Mac mini M4 (24 ГБ), Metal, без флагов: 512² за 3.9 мин, 1024² за 21 мин (28 шагов с
CFG); память около 8 ГБ. GPU M4 упирается в вычисления — для черновиков берите
меньше `--steps` и 512², либо Z-Image-Turbo.

RTX 3090: пик видеопамяти 14–15 ГБ (хватает карты на 16 ГБ). DiT идёт со скоростью diffusers bf16 на той же
карте, а старт процесса примерно в четыре раза быстрее.

**Опции:** `--width`/`--height` (кратны 16, по умолчанию 1024×1024), `--steps`, `--seed`,
`--num-images N` (картинка i берёт seed+i), `--cfg G` и `--negative-prompt` (по умолчанию 4 и ""),
`--cfg-normalization [C]`, `--cfg-truncation T`, `--shift`, `--max-sequence-length` (512),
`--out` (`.png` по умолчанию, `.jpg`, `.ppm`). `CMF_ZIMAGE_PROF=1` печатает время стадий,
`CMF_GPU=0` считает всё на CPU (минуты на картинку).

Базовая модель слушается негативных промптов и guidance 3–5 (28–50 шагов); для скорости есть [Z-Image-Turbo](https://huggingface.co/infosave/Z-Image-Turbo-cmf) (8 шагов).

**Примеры:** 1024², seed 7, настройки по умолчанию — [samples/](https://huggingface.co/infosave/Z-Image-cmf/tree/main/samples).
**Проверка:** `sha256sum -c z-image.cmf.sha256`, `cortiq info z-image.cmf`.

---

## 中文文档

Tongyi-MAI 的 [Z-Image](https://huggingface.co/Tongyi-MAI/Z-Image) 打包为单个
[CMF](https://github.com/infosave2007/cmf) 文件：6B 图像 Transformer、Qwen3-4B 文本编码器和
VAE 合在一起，生成配方也写在文件里。由 Rust 引擎 `cortiq` 运行，无需 Python：NVIDIA 显卡走
Vulkan（张量核心），Apple silicon 走 Metal，任何机器都有 CPU 后备路径。

```bash
cargo install cortiq-cli          # 0.7.5 或更新；预编译二进制：github.com/infosave2007/cmf/releases
hf download infosave/Z-Image-cmf z-image.cmf --local-dir .
cortiq imagine z-image.cmf --prompt "夕阳下坐在窗台上的猫，照片级写实" --negative-prompt "blurry, low quality"
```

无需任何参数：文件内含基础模型配方（1024×1024，28 步，CFG guidance 4，shift 6）。 无显示器的 Linux 机器请设置 `XDG_RUNTIME_DIR=/tmp`。

**文件：** `z-image.cmf`，10.5 GB —— DiT 8 位（`q8_2f`）、Qwen3-4B 文本编码器 8 位、VAE；
sha256 `d7138e6fe1b6af76…`（完整值见 `z-image.cmf.sha256`）。8 位是实测的选择：4 位 Transformer 会让图像
PSNR 下降 3–15 dB，8 位则与 bf16 推理相当。

**性能**（RTX 3090，Vulkan，每进程一张图，无参数，cortiq 0.7.5）：

| 尺寸 | 整张图 | 其中 DiT 步骤 | VAE |
|---|---:|---:|---:|
| 512×512 | 16.2 s | 13.6 s（28 步，CFG） | 0.12 s |
| 1024×1024 | 60 s | 57 s（28 步，CFG） | 0.49 s |

Mac mini M4（24 GB），Metal，无参数：512² 用时 3.9 分钟，1024² 用时 21 分钟（28 步，CFG）；
内存约 8 GB。M4 的 GPU 受算力限制——草稿请减少 `--steps` 并使用 512²，或改用 Z-Image-Turbo。

RTX 3090：显存峰值 14–15 GB（16 GB 显卡可用）。DiT 与同一显卡上的 diffusers bf16 速度相同，进程启动快约四倍。

**选项：** `--width`/`--height`（16 的倍数，默认 1024×1024）、`--steps`、`--seed`、
`--num-images N`（第 i 张用 seed+i）、`--cfg G` 与 `--negative-prompt`（默认 4 与 ""）、
`--cfg-normalization [C]`、`--cfg-truncation T`、`--shift`、`--max-sequence-length`（512）、
`--out`（默认 `.png`，也可 `.jpg`、`.ppm`）。`CMF_ZIMAGE_PROF=1` 打印各阶段耗时，
`CMF_GPU=0` 全部在 CPU 上运行（每张图需数分钟）。

基础模型支持负面提示词和 guidance 3–5（28–50 步）；追求速度请用 [Z-Image-Turbo](https://huggingface.co/infosave/Z-Image-Turbo-cmf)（8 步）。

**示例：** 1024²，seed 7，默认设置 —— [samples/](https://huggingface.co/infosave/Z-Image-cmf/tree/main/samples)。
**校验：** `sha256sum -c z-image.cmf.sha256`，`cortiq info z-image.cmf`。
