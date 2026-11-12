---
license: other
license_name: minimax-h3-image-community-license-agreement
license_link: LICENSE
library_name: cortiq
base_model:
  - MiniMaxAI/MiniMax-H3-Image
base_model_relation: quantized
pipeline_tag: text-to-image
tags:
  - cmf
  - cortiq
  - quantized
  - q4tp
  - cuda
  - nvidia
  - vulkan
  - text-to-image
  - image-to-image
  - image-editing
language:
  - en
  - zh
  - ru
---

# MiniMax-H3-Image — CMF

[MiniMax-H3-Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image), text to image and image editing at 4 to
16 megapixels in six steps, as one [CMF](https://github.com/infosave2007/cmf) file of 24.6 GB for `cortiq`, a
Rust inference engine with no Python and no ML framework. The diffusion transformer and the prompt encoder are
in 4 bits, the VAE in 8. On a 16 GB RTX 2000 Ada the native CUDA backend renders 2048 x 2048 in 50.3 s,
and under a ceiling of 4 GB on the card's memory in 60.2 s, the same picture. The same file runs over
Vulkan on any vendor's card.

[English](#examples) · [Русский](#документация-на-русском) · [中文](#中文说明)

## Examples

Text to image: 2048 x 2048, seed 42, six steps, this file on CUDA. Click for the full size; the prompts are in
[examples/prompts](https://huggingface.co/infosave/MiniMax-H3-Image-cmf/tree/main/examples/prompts).

| | |
|---|---|
| [![A lighthouse on a headland](https://huggingface.co/infosave/MiniMax-H3-Image-cmf/resolve/main/examples/t2i_lighthouse_1024.jpg)](https://huggingface.co/infosave/MiniMax-H3-Image-cmf/resolve/main/examples/t2i_lighthouse.jpg) | [![A bakery shopfront with a bicycle](https://huggingface.co/infosave/MiniMax-H3-Image-cmf/resolve/main/examples/t2i_cafe_1024.jpg)](https://huggingface.co/infosave/MiniMax-H3-Image-cmf/resolve/main/examples/t2i_cafe.jpg) |
| [![A fox](https://huggingface.co/infosave/MiniMax-H3-Image-cmf/resolve/main/examples/t2i_fox_1024.jpg)](https://huggingface.co/infosave/MiniMax-H3-Image-cmf/resolve/main/examples/t2i_fox.jpg) | [![A portrait of an old fisherman](https://huggingface.co/infosave/MiniMax-H3-Image-cmf/resolve/main/examples/t2i_portrait_1024.jpg)](https://huggingface.co/infosave/MiniMax-H3-Image-cmf/resolve/main/examples/t2i_portrait.jpg) |

At 4096 x 4096 (16 megapixels, 283.7 s on the same card): [the full file](https://huggingface.co/infosave/MiniMax-H3-Image-cmf/resolve/main/examples/t2i_feast_4096.jpg), the
prompt of the model authors' example.

Image editing. Left: the picture given with `--image`. Right: the result.

| reference | edit |
|---|---|
| ![Two podcast hosts, the man in headphones](https://huggingface.co/infosave/MiniMax-H3-Image-cmf/resolve/main/examples/edit_minimax_reference.jpg) | [![The headphones are gone, the hosts bump fists](https://huggingface.co/infosave/MiniMax-H3-Image-cmf/resolve/main/examples/edit_minimax_result_1024.jpg)](https://huggingface.co/infosave/MiniMax-H3-Image-cmf/resolve/main/examples/edit_minimax_result.jpg) |
| the model authors' example picture | their edit brief ([prompt](https://huggingface.co/infosave/MiniMax-H3-Image-cmf/resolve/main/examples/prompts/edit_minimax.txt)), 2048 x 2048, 95.2 s: headphones removed, a fist bump, every sign and mug text kept |
| ![The bakery shopfront with a red bicycle](https://huggingface.co/infosave/MiniMax-H3-Image-cmf/resolve/main/examples/edit_cafe_reference_1024.jpg) | ![The same shopfront, the bicycle pastel blue](https://huggingface.co/infosave/MiniMax-H3-Image-cmf/resolve/main/examples/edit_cafe_result_1024.jpg) |
| the picture generated above | "repaint the bicycle pastel blue" ([prompt](https://huggingface.co/infosave/MiniMax-H3-Image-cmf/resolve/main/examples/prompts/edit_min_cafe.txt)), 1024 x 1024, 42.8 s |

## Quick start

```bash
cargo install --git https://github.com/infosave2007/cmf --branch h3-image cortiq-cli --features cuda
hf download infosave/MiniMax-H3-Image-cmf minimax-h3-image-q4tp.cmf --local-dir .
# text to image
CMF_GPU=cuda cortiq imagine minimax-h3-image-q4tp.cmf --prompt @prompt.txt --out out.png
# image editing: <Picture 1> in the prompt is the file given with --image
CMF_GPU=cuda cortiq imagine minimax-h3-image-q4tp.cmf --image photo.png --prompt @edit.txt --out edit.png
# a card with 6 GB
CMF_GPU=cuda CMF_GPU_VRAM_MB=6000 cortiq imagine minimax-h3-image-q4tp.cmf --prompt @prompt.txt --out out.png
# Vulkan, any vendor's card
cortiq imagine minimax-h3-image-q4tp.cmf --prompt @prompt.txt --out out.png
```

| what it needs | |
|---|---|
| card | CUDA: NVIDIA with its driver, and CUDA 12.x libraries `libnvrtc.so.12` and `libcublas.so.12` on the loader's path. Vulkan: any card |
| memory | 16 GB of the card without a ceiling; from 4 GB with `CMF_GPU_VRAM_MB`. About 28 GB of RAM or page cache for the file |
| engine | `cortiq` built from the branch above (stable Rust; nothing is linked against CUDA at build time); the 0.8.72 release does not include the model |

`--width` and `--height` are multiples of 32 and default to 2048 x 2048. The model works from about 4
megapixels to 20; a smaller size is generated on a 2048 x 2048-area canvas and resized back (`--min-area 1024`
turns that off). There is no guidance and no negative prompt. The model expects a detailed English description
of the whole image; its authors expand a short request with a separate model, which this file does not
include. Full guide: [docs/MINIMAX_H3_IMAGE.md](https://github.com/infosave2007/cmf/blob/h3-image/docs/MINIMAX_H3_IMAGE.md).

## Speed

NVIDIA RTX 2000 Ada, 16 GB, one image, seed 42, single runs (2048 x 2048: the median of three).

| run | prompt, s | six steps, s | a step, s | decode, s | total, s | peak memory of the card, MiB |
|---|---:|---:|---:|---:|---:|---:|
| CUDA, 1024 x 1024 (`--min-area 1024`) | 2.4 | 11.7 | 1.95 | 0.5 | **16.1** | 12 116 |
| CUDA, 2048 x 2048 | 2.6 | 44.3 | 7.38 | 1.7 | **50.3** | 12 654 |
| CUDA, 4096 x 4096 | 9.2 | 264.5 | 43.90 | 8.3 | **283.7** | 13 414 |
| CUDA, edit at 2048 x 2048, one 1024 x 1024 picture | 9.8 | 67.9 | 11.37 | 1.7 | **80.8** | 12 854 |
| Vulkan, 1024 x 1024 (`--min-area 1024`) | 13.0 | 70.0 | 10.99 | 4.0 | **88.7** | 12 529 |

## Small cards

`CMF_GPU_VRAM_MB` is a ceiling on what the process holds on the card. The blocks of the transformer that do
not fit under it pass through the card every step. 2048 x 2048 on CUDA:

| ceiling | total, s | a step, s | peak memory of the card, MiB |
|---|---:|---:|---:|
| none | 50.3 | 7.38 | 12 654 |
| `CMF_GPU_VRAM_MB=8000` | 53.5 | 8.05 | 7 634 |
| `CMF_GPU_VRAM_MB=6000` | 54.4 | 8.18 | 5 720 |
| `CMF_GPU_VRAM_MB=4000` | 60.2 | 9.15 | 3 710 |

The picture under the 4 GB ceiling is the same picture: 49.6 dB PSNR against the run without one.

## Image editing

`--image FILE` makes the run an edit; repeat it for several pictures (`<Picture 1>`, `<Picture 2>`, ... in the
prompt, the first is the one to edit). The model's own edit prompts are briefs that name every subject and say
what stays; a short instruction also works, and text that the prompt does not name tends to change.

Known limit: **an edit of a densely detailed, high-contrast picture comes out over-processed at 2048 x 2048**
(oversaturated, posterized); the bakery picture above is one. The same edit at 1024 x 1024
(`--width 1024 --height 1024 --min-area 1024`) is clean, and so are edits of softer pictures at 2048 x 2048.
An 8-bit encoder, an 8-bit transformer and the f32 path give the same result; the cause is not found.

## What is in the file

| part | source | codec |
|---|---|---|
| diffusion transformer, 20.1 B parameters, 50 blocks | `Comfy-Org/MiniMax-H3-Image`, the pruned BF16 checkpoint | q4tp, 4 bits |
| prompt encoder, Qwen3-VL-32B up to layer 50, with the vision tower | `Comfy-Org/MiniMax-H3` | q4tp, 4 bits |
| image VAE (CNN encoder, 2-D ViT decoder) | `Comfy-Org/MiniMax-H3-Image` | q8_2f, 8 bits |

`minimax-h3-image-q4tp.cmf`: 24 562 634 323 bytes, sha256
`f4af395ada66213cd401a455fcc7fd7252d2d4c4813f8d1437ff58a89740f309`. The weights are quantized from the
originals; nothing was retrained.

Checked: the CUDA chain against a per-operation f32 path (54.0 dB PSNR on a 2048 x 2048 edit), CUDA against
Vulkan (47.0 dB), the 4 GB ceiling against none (49.6 dB). Not checked: the reference implementation was
not run on the same card, so there is no picture-to-picture comparison with it.

## License

MiniMax-H3-Image is licensed under the MiniMax-H3-Image Community License Agreement, Copyright © 2026 MiniMax.
All Rights Reserved. A copy is in [LICENSE](LICENSE), the notice in [NOTICE](NOTICE). This repository holds a
modified form of the model: the weights are quantized and repacked into the CMF container. The Agreement
applies to this file and to what it generates. In particular it grants rights only within its Applicable
Territory, which excludes the European Union, the United Kingdom, the Republic of Korea and the United States
of America; it requires a separate authorization from MiniMax above 1 million USD of annual revenue or for a
hosted service; and it carries an Acceptable Use Policy. Read it before use. The `cortiq` engine itself is
Apache-2.0.

## Документация на русском

[MiniMax-H3-Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image), генерация и правка изображений от 4 до
16 мегапикселей за шесть шагов, в одном файле [CMF](https://github.com/infosave2007/cmf) на 24.6 ГБ для
`cortiq`, движка на Rust без Python и без ML-фреймворка. Трансформер и энкодер промпта в 4 битах, VAE в 8.
На RTX 2000 Ada с 16 ГБ собственный бэкенд CUDA рисует 2048 x 2048 за 50,3 с, а под потолком памяти карты
4 ГБ за 60,2 с, ту же картинку. Тот же файл работает через Vulkan на карте любого производителя.
Примеры генерации и правки — [в начале страницы](#examples).

```bash
cargo install --git https://github.com/infosave2007/cmf --branch h3-image cortiq-cli --features cuda
hf download infosave/MiniMax-H3-Image-cmf minimax-h3-image-q4tp.cmf --local-dir .
# текст в картинку
CMF_GPU=cuda cortiq imagine minimax-h3-image-q4tp.cmf --prompt @prompt.txt --out out.png
# правка: <Picture 1> в промпте — файл из --image
CMF_GPU=cuda cortiq imagine minimax-h3-image-q4tp.cmf --image photo.png --prompt @edit.txt --out edit.png
# карта с 6 ГБ
CMF_GPU=cuda CMF_GPU_VRAM_MB=6000 cortiq imagine minimax-h3-image-q4tp.cmf --prompt @prompt.txt --out out.png
# Vulkan, карта любого производителя
cortiq imagine minimax-h3-image-q4tp.cmf --prompt @prompt.txt --out out.png
```

Для CUDA нужны драйвер NVIDIA и библиотеки CUDA 12.x (`libnvrtc.so.12`, `libcublas.so.12`); без потолка
требуется 16 ГБ памяти карты, с `CMF_GPU_VRAM_MB` достаточно 4 ГБ. `--width` и `--height` кратны 32, по
умолчанию 2048 x 2048. Рабочий диапазон модели от 4 до 20 мегапикселей; меньший размер генерируется на холсте
площадью 2048 x 2048 и уменьшается (`--min-area 1024` это отключает). Guidance и негативного промпта нет.
Модель ждёт подробное английское описание всей картинки; расширителя промпта в файле нет. Полное руководство:
[docs/MINIMAX_H3_IMAGE.ru.md](https://github.com/infosave2007/cmf/blob/h3-image/docs/MINIMAX_H3_IMAGE.ru.md).

Скорость, NVIDIA RTX 2000 Ada, 16 GB, одна картинка, сид 42, одиночные прогоны (2048 x 2048 — медиана трёх):

| прогон | промпт, с | шесть шагов, с | шаг, с | декодер, с | всего, с | пик памяти карты, МиБ |
|---|---:|---:|---:|---:|---:|---:|
| CUDA, 1024 x 1024 (`--min-area 1024`) | 2,4 | 11,7 | 1,95 | 0,5 | **16,1** | 12 116 |
| CUDA, 2048 x 2048 | 2,6 | 44,3 | 7,38 | 1,7 | **50,3** | 12 654 |
| CUDA, 4096 x 4096 | 9,2 | 264,5 | 43,90 | 8,3 | **283,7** | 13 414 |
| CUDA, правка 2048 x 2048, одна картинка 1024 x 1024 | 9,8 | 67,9 | 11,37 | 1,7 | **80,8** | 12 854 |
| Vulkan, 1024 x 1024 (`--min-area 1024`) | 13,0 | 70,0 | 10,99 | 4,0 | **88,7** | 12 529 |

Потолок памяти карты, 2048 x 2048 на CUDA. Блоки трансформера, не поместившиеся под потолок, проходят через
карту каждый шаг; под потолком 4 ГБ получается та же картинка (PSNR 49,6 дБ):

| потолок | всего, с | шаг, с | пик памяти карты, МиБ |
|---|---:|---:|---:|
| нет | 50,3 | 7,38 | 12 654 |
| `CMF_GPU_VRAM_MB=8000` | 53,5 | 8,05 | 7 634 |
| `CMF_GPU_VRAM_MB=6000` | 54,4 | 8,18 | 5 720 |
| `CMF_GPU_VRAM_MB=4000` | 60,2 | 9,15 | 3 710 |

Правка: `--image FILE`, опцию можно повторять (`<Picture 1>`, `<Picture 2>`, ... в промпте, первая картинка
редактируется). Известное ограничение: **правка плотно детализированной контрастной картинки на холсте
2048 x 2048 выходит пережжённой** (перенасыщенной, с постеризацией). Та же правка на 1024 x 1024
(`--width 1024 --height 1024 --min-area 1024`) чистая, как и правки более мягких картинок на 2048 x 2048.
8-битный энкодер, 8-битный трансформер и путь в f32 дают тот же результат; причина не найдена.

Лицензия: MiniMax-H3-Image Community License Agreement, копия в [LICENSE](LICENSE), уведомление в
[NOTICE](NOTICE). В репозитории изменённая форма модели: веса квантованы и упакованы в контейнер CMF.
Соглашение действует только на своей территории применения, из которой исключены Европейский союз,
Великобритания, Республика Корея и США; при годовой выручке выше 1 млн долларов США и для хостинга нужно
отдельное разрешение MiniMax. Прочитайте соглашение до использования.

## 中文说明

[MiniMax-H3-Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image)：六步生成和编辑 4 到 16 百万像素的
图片，打包成一个 24.6 GB 的 [CMF](https://github.com/infosave2007/cmf) 文件，由 `cortiq` 运行。`cortiq` 是
用 Rust 写的推理引擎，不需要 Python，也不依赖任何机器学习框架。Transformer 和提示词编码器为 4 位，VAE
为 8 位。在 16 GB 的 RTX 2000 Ada 上，原生 CUDA 后端生成 2048 x 2048 用时 50.3 秒；在 4 GB 显存上限下用时
60.2 秒，得到同一张图片。同一个文件可以通过 Vulkan 在任何厂商的显卡上运行。生成和编辑的示例见
[页面开头](#examples)。

```bash
cargo install --git https://github.com/infosave2007/cmf --branch h3-image cortiq-cli --features cuda
hf download infosave/MiniMax-H3-Image-cmf minimax-h3-image-q4tp.cmf --local-dir .
# 文生图
CMF_GPU=cuda cortiq imagine minimax-h3-image-q4tp.cmf --prompt @prompt.txt --out out.png
# 图片编辑：提示词里的 <Picture 1> 就是 --image 给出的文件
CMF_GPU=cuda cortiq imagine minimax-h3-image-q4tp.cmf --image photo.png --prompt @edit.txt --out edit.png
# 6 GB 显卡
CMF_GPU=cuda CMF_GPU_VRAM_MB=6000 cortiq imagine minimax-h3-image-q4tp.cmf --prompt @prompt.txt --out out.png
# Vulkan，任何厂商的显卡
cortiq imagine minimax-h3-image-q4tp.cmf --prompt @prompt.txt --out out.png
```

CUDA 需要 NVIDIA 驱动和 CUDA 12.x 库（`libnvrtc.so.12`、`libcublas.so.12`）；不设上限时需要 16 GB 显存，
设置 `CMF_GPU_VRAM_MB` 后 4 GB 即可。`--width` 和 `--height` 是 32 的倍数，默认 2048 x 2048。模型的工作
范围是 4 到 20 百万像素；更小的尺寸会在面积为 2048 x 2048 的画布上生成再缩小（`--min-area 1024` 关闭此
行为）。没有 guidance，也没有负面提示词。模型需要对整张图片的详细英文描述；文件不包含提示词扩展器。
完整指南：[docs/MINIMAX_H3_IMAGE.zh.md](https://github.com/infosave2007/cmf/blob/h3-image/docs/MINIMAX_H3_IMAGE.zh.md)。

速度，NVIDIA RTX 2000 Ada, 16 GB，一张图片，种子 42，单次运行（2048 x 2048 为三次的中位数）：

| 运行 | 提示词，秒 | 六步，秒 | 每步，秒 | 解码，秒 | 总计，秒 | 显存峰值，MiB |
|---|---:|---:|---:|---:|---:|---:|
| CUDA，1024 x 1024（`--min-area 1024`） | 2.4 | 11.7 | 1.95 | 0.5 | **16.1** | 12 116 |
| CUDA，2048 x 2048 | 2.6 | 44.3 | 7.38 | 1.7 | **50.3** | 12 654 |
| CUDA，4096 x 4096 | 9.2 | 264.5 | 43.90 | 8.3 | **283.7** | 13 414 |
| CUDA，2048 x 2048 编辑，一张 1024 x 1024 参考图 | 9.8 | 67.9 | 11.37 | 1.7 | **80.8** | 12 854 |
| Vulkan，1024 x 1024（`--min-area 1024`） | 13.0 | 70.0 | 10.99 | 4.0 | **88.7** | 12 529 |

显存上限，CUDA 上 2048 x 2048。放不进上限的 Transformer 块每一步都流经显卡；4 GB 上限下得到的是同一张
图片（PSNR 49.6 dB）：

| 上限 | 总计，秒 | 每步，秒 | 显存峰值，MiB |
|---|---:|---:|---:|
| 无 | 50.3 | 7.38 | 12 654 |
| `CMF_GPU_VRAM_MB=8000` | 53.5 | 8.05 | 7 634 |
| `CMF_GPU_VRAM_MB=6000` | 54.4 | 8.18 | 5 720 |
| `CMF_GPU_VRAM_MB=4000` | 60.2 | 9.15 | 3 710 |

编辑：`--image FILE`，可以重复（提示词中为 `<Picture 1>`、`<Picture 2>` 等，第一张是被编辑的图片）。
已知限制：**在 2048 x 2048 上编辑细节密集、对比强烈的图片，结果会过度处理**（过饱和、色调分离）。同样的
编辑在 1024 x 1024 上（`--width 1024 --height 1024 --min-area 1024`）是干净的，较柔和的图片在
2048 x 2048 上的编辑也是干净的。8 位编码器、8 位 Transformer 和 f32 路径给出相同结果；原因尚未找到。

许可：MiniMax-H3-Image Community License Agreement，副本见 [LICENSE](LICENSE)，声明见 [NOTICE](NOTICE)。
本仓库中是模型的修改形式：权重经过量化并打包进 CMF 容器。该协议只在其适用地区内授予权利，欧盟、英国、
韩国和美国不在其内；年收入超过 100 万美元或提供托管服务需要 MiniMax 的单独授权。使用前请阅读协议。
