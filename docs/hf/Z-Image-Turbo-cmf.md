---
license: apache-2.0
base_model: Tongyi-MAI/Z-Image-Turbo
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

# Z-Image-Turbo — CMF

[Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) by Tongyi-MAI, packaged as a single
[CMF](https://github.com/infosave2007/cmf) file: the 6B image transformer, the Qwen3-4B
text encoder and the VAE together, with the generation recipe stored inside. It runs
with `cortiq`, a Rust engine with no Python: NVIDIA GPUs through Vulkan today
(tensor cores), CPU fallback everywhere; Apple silicon (Metal) is next.

[English](#quick-start) · [Русский](#документация-на-русском) · [中文](#中文文档)

## Quick start

```bash
cargo install cortiq-cli          # 0.7.4 or later; prebuilt binaries: github.com/infosave2007/cmf/releases
hf download infosave/Z-Image-Turbo-cmf z-image-turbo.cmf --local-dir .
cortiq imagine z-image-turbo.cmf --prompt "A cat sitting on a windowsill at sunset, photorealistic"
```

No flags needed: the file carries the Turbo recipe (1024×1024, 8 steps, no CFG). Set `XDG_RUNTIME_DIR=/tmp` on a headless Linux box.

## File

| file | size | inside |
|---|---:|---|
| `z-image-turbo.cmf` | 10.5 GB | DiT 8-bit (`q8_2f`), Qwen3-4B text encoder 8-bit, VAE |

sha256 `c0aaffaf887c595431cddfb0dce369d04eba23dac7a69ca162a051486564f1f8` (`z-image-turbo.cmf.sha256` is next to it). 8-bit was chosen by measurement: the 4-bit
transformer loses 3–15 dB of image PSNR, the 8-bit one is comparable to bf16 inference.

## Performance

RTX 3090, Vulkan, one image per process, no flags (cortiq 0.7.4):

| size | time to image | of which DiT steps | VAE |
|---|---:|---:|---:|
| 512×512 | 4.6 s | 2.1 s (8 steps) | 0.13 s |
| 1024×1024 | 11.4 s | 8.3 s (8 steps) | 0.50 s |

Peak VRAM 14–15 GB (16 GB cards fit). Per image, the DiT runs at the speed of diffusers
bf16 on the same card and the whole process starts about four times faster.

## Options

| option | meaning |
|---|---|
| `--width`, `--height` | multiples of 16 (default 1024×1024) |
| `--steps`, `--seed`, `--num-images N` | image i uses seed+i |
| `--cfg G`, `--negative-prompt` | guidance scale and negative prompt (Turbo: 0, no CFG) |
| `--cfg-normalization [C]`, `--cfg-truncation T` | the diffusers CFG extras |
| `--shift`, `--max-sequence-length` | scheduler shift, prompt token cap (512) |
| `--out` | `.png` (default `out.png`), `.jpg`, `.ppm` |

`CMF_ZIMAGE_PROF=1` prints stage times. `CMF_GPU=0` runs everything on the CPU (minutes per image).

Turbo is distilled: guidance stays at 0 and 8 steps are the recipe; negative prompts have no effect on it — use [Z-Image](https://huggingface.co/infosave/Z-Image-cmf) for CFG and negative prompting.

## Samples

1024², seed 7, default settings: [samples/](https://huggingface.co/infosave/Z-Image-Turbo-cmf/tree/main/samples).

## Verify

```bash
sha256sum -c z-image-turbo.cmf.sha256
cortiq info z-image-turbo.cmf
```

---

## Документация на русском

[Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) от Tongyi-MAI одним файлом
[CMF](https://github.com/infosave2007/cmf): трансформер 6B, текстовый энкодер Qwen3-4B и VAE
вместе, рецепт генерации записан в файл. Запускается движком `cortiq` на Rust без Python:
NVIDIA через Vulkan (тензорные ядра), везде есть путь на CPU; Apple silicon (Metal) — следующий.

```bash
cargo install cortiq-cli          # 0.7.4 или новее; готовые бинарники: github.com/infosave2007/cmf/releases
hf download infosave/Z-Image-Turbo-cmf z-image-turbo.cmf --local-dir .
cortiq imagine z-image-turbo.cmf --prompt "Кот на подоконнике на закате, фотореализм"
```

Флаги не нужны: в файле записан рецепт Turbo (1024×1024, 8 шагов, без CFG). На Linux без дисплея задайте `XDG_RUNTIME_DIR=/tmp`.

**Файл:** `z-image-turbo.cmf`, 10.5 ГБ — DiT 8 бит (`q8_2f`), энкодер Qwen3-4B 8 бит, VAE;
sha256 `c0aaffaf887c5954…` (полностью в `z-image-turbo.cmf.sha256`). 8 бит выбраны по замеру: 4-битный трансформер
теряет 3–15 дБ PSNR картинки, 8-битный сравним с выводом в bf16.

**Скорость** (RTX 3090, Vulkan, одна картинка на процесс, без флагов, cortiq 0.7.4):

| размер | картинка целиком | из них шаги DiT | VAE |
|---|---:|---:|---:|
| 512×512 | 4.6 с | 2.1 с (8 шагов) | 0.13 с |
| 1024×1024 | 11.4 с | 8.3 с (8 шагов) | 0.50 с |

Пик видеопамяти 14–15 ГБ (хватает карты на 16 ГБ). DiT идёт со скоростью diffusers bf16 на той же
карте, а старт процесса примерно в четыре раза быстрее.

**Опции:** `--width`/`--height` (кратны 16, по умолчанию 1024×1024), `--steps`, `--seed`,
`--num-images N` (картинка i берёт seed+i), `--cfg G` и `--negative-prompt` (Turbo: 0, без CFG),
`--cfg-normalization [C]`, `--cfg-truncation T`, `--shift`, `--max-sequence-length` (512),
`--out` (`.png` по умолчанию, `.jpg`, `.ppm`). `CMF_ZIMAGE_PROF=1` печатает время стадий,
`CMF_GPU=0` считает всё на CPU (минуты на картинку).

Turbo дистиллирована: guidance 0 и 8 шагов — это её рецепт, негативный промпт на неё не действует; для CFG и негативных промптов есть [Z-Image](https://huggingface.co/infosave/Z-Image-cmf).

**Примеры:** 1024², seed 7, настройки по умолчанию — [samples/](https://huggingface.co/infosave/Z-Image-Turbo-cmf/tree/main/samples).
**Проверка:** `sha256sum -c z-image-turbo.cmf.sha256`, `cortiq info z-image-turbo.cmf`.

---

## 中文文档

Tongyi-MAI 的 [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) 打包为单个
[CMF](https://github.com/infosave2007/cmf) 文件：6B 图像 Transformer、Qwen3-4B 文本编码器和
VAE 合在一起，生成配方也写在文件里。由 Rust 引擎 `cortiq` 运行，无需 Python：NVIDIA 显卡走
Vulkan（张量核心），任何机器都有 CPU 后备路径；Apple silicon（Metal）即将支持。

```bash
cargo install cortiq-cli          # 0.7.4 或更新；预编译二进制：github.com/infosave2007/cmf/releases
hf download infosave/Z-Image-Turbo-cmf z-image-turbo.cmf --local-dir .
cortiq imagine z-image-turbo.cmf --prompt "夕阳下坐在窗台上的猫，照片级写实"
```

无需任何参数：文件内含 Turbo 配方（1024×1024，8 步，无 CFG）。 无显示器的 Linux 机器请设置 `XDG_RUNTIME_DIR=/tmp`。

**文件：** `z-image-turbo.cmf`，10.5 GB —— DiT 8 位（`q8_2f`）、Qwen3-4B 文本编码器 8 位、VAE；
sha256 `c0aaffaf887c5954…`（完整值见 `z-image-turbo.cmf.sha256`）。8 位是实测的选择：4 位 Transformer 会让图像
PSNR 下降 3–15 dB，8 位则与 bf16 推理相当。

**性能**（RTX 3090，Vulkan，每进程一张图，无参数，cortiq 0.7.4）：

| 尺寸 | 整张图 | 其中 DiT 步骤 | VAE |
|---|---:|---:|---:|
| 512×512 | 4.6 s | 2.1 s（8 步） | 0.13 s |
| 1024×1024 | 11.4 s | 8.3 s（8 步） | 0.50 s |

显存峰值 14–15 GB（16 GB 显卡可用）。DiT 与同一显卡上的 diffusers bf16 速度相同，进程启动快约四倍。

**选项：** `--width`/`--height`（16 的倍数，默认 1024×1024）、`--steps`、`--seed`、
`--num-images N`（第 i 张用 seed+i）、`--cfg G` 与 `--negative-prompt`（Turbo：0，无 CFG）、
`--cfg-normalization [C]`、`--cfg-truncation T`、`--shift`、`--max-sequence-length`（512）、
`--out`（默认 `.png`，也可 `.jpg`、`.ppm`）。`CMF_ZIMAGE_PROF=1` 打印各阶段耗时，
`CMF_GPU=0` 全部在 CPU 上运行（每张图需数分钟）。

Turbo 是蒸馏模型：guidance 保持 0、8 步即为其配方，负面提示词对它无效；需要 CFG 和负面提示词请用 [Z-Image](https://huggingface.co/infosave/Z-Image-cmf)。

**示例：** 1024²，seed 7，默认设置 —— [samples/](https://huggingface.co/infosave/Z-Image-Turbo-cmf/tree/main/samples)。
**校验：** `sha256sum -c z-image-turbo.cmf.sha256`，`cortiq info z-image-turbo.cmf`。
