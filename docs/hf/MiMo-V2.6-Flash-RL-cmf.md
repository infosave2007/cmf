---
license: mit
library_name: cortiq
base_model: XiaomiMiMo/MiMo-V2.6-Flash-RL
base_model_relation: quantized
pipeline_tag: text-generation
tags: [cmf, cortiq, quantized, multimodal]
language: [en, ru, zh]
---

# MiMo-V2.6-Flash-RL · CMF q4tp

Native CMF weights for the [Cortiq inference engine](https://github.com/infosave2007/cmf), converted from [Xiaomi MiMo-V2.6-Flash-RL](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL).

**[English](#choose-your-edition) · [Русский](#русский) · [中文](#中文)**

> **Release candidate.** All three weight files are uploaded and SHA-256 verified.
> Text and greedy MTP checks pass; strict multimodal acceptance is still incomplete.
> **Engine: Cortiq 0.7.7** (release build/publication in progress). Older published versions do not include this MiMo integration.

## Choose your edition

| Edition | Capabilities | Download | Total size |
|---|---|---|---:|
| **q4tp Text-only** | Text generation + MTP | **[Download Text-only](packages/text-only/README.md)** | **164.851 GB** |
| **q4tp Full** | Text + MTP + image, video and audio input | **[Download Full](packages/full/README.md)** | **166.093 GB** |

**These are two complete file sets, not two copies of the 164-GB backbone.**
Full includes the same text weights plus a **1.241-GB multimodal companion**.
To upgrade from Text-only, download only the `.mm.cmf` file into the same folder.
Text-only does not download or load the media towers.

| File | Text-only | Full | Size |
|---|:---:|:---:|---:|
| [Backbone · q4tp.cmf](MiMo-V2.6-Flash-RL-q4tp.cmf) | ✓ | ✓ | 163.863 GB |
| [MTP · q4tp.mtp.cmf](MiMo-V2.6-Flash-RL-q4tp.mtp.cmf) | ✓ | ✓ | 0.989 GB |
| [Multimodal · q4tp.mm.cmf](MiMo-V2.6-Flash-RL-q4tp.mm.cmf) | — | ✓ | 1.241 GB |

Sizes are decimal GB. Download guides include exact files and checksum commands.

## Run

Keep the companions beside the backbone; they are discovered automatically.
GPU placement adapts to the available weight budget using resident layers and a
dynamic expert cache. Host RAM must accommodate the mapped model and runtime.

```bash
# Both editions
cortiq run MiMo-V2.6-Flash-RL-q4tp.cmf --prompt "Explain why the sky is blue."
cortiq serve MiMo-V2.6-Flash-RL-q4tp.cmf --host 127.0.0.1 --port 8080

# Full edition
cortiq run MiMo-V2.6-Flash-RL-q4tp.cmf --prompt "Read the text." --image page.png
cortiq run MiMo-V2.6-Flash-RL-q4tp.cmf --prompt "Describe the clip." --video frames --video-fps 1
cortiq run MiMo-V2.6-Flash-RL-q4tp.cmf --prompt "Transcribe the speech." --audio speech.wav
```

MTP uses three checkpoint draft layers and GPU speculative verification for
**greedy decoding**. Nonzero-temperature sampling does not use this greedy MTP
path. MTP can accelerate some prompts but is not universally faster.

Media support: images; WAV audio, including resampling/stereo; silent video from
Y4M or a frame directory. MP4, compressed audio and interleaved video/audio are
not supported. Full is for **media understanding**, not image/audio generation.

## Performance

RTX PRO 6000 Blackwell **96 GB**, Vulkan, one stream:

```bash
cortiq bench MiMo-V2.6-Flash-RL-q4tp.cmf --core --tokens 128 --ignore-eos --json
```

Default-setting runs: **32.20 / 41.40 / 40.85 tok/s · median 40.85 tok/s**.
This is a warmed synthetic core benchmark, not a guarantee for every prompt.
Natural 128-token prompts: warm plain **29.90–36.05 tok/s**; MTP **28.16–33.62 tok/s**.

| Weight budget (`CMF_GPU_VRAM_MB`, MiB) | Median tok/s, 3 runs | Highest sampled VRAM, MiB |
|---:|---:|---:|
| 16000 | 10.96 | 20417 |
| 24000 | 17.27 | 33111 |
| 32000 | 29.66 | 46585 |
| 48000 | 36.47 | 58102 |
| 64000 | **44.48** | 70164 |
| 80000 | 42.36 | 85497 |
| Automatic, 96-GB GPU | **40.85** | — |

The ladder simulates weight budgets on **one 96-GB card**, not different physical
cards. Temporary upload buffers exceeded the budgets in these baseline runs;
a bounded-upload fix is under test. Do not interpret this table as physical
16–80 GB card qualification. See [validation details](VALIDATION.md).

## Precision and validation

- Experts: **upstream MXFP4 recoded to q4tp**, not a direct BF16 quantization.
- Text attention, embedding/head and initial dense MLP: **q8_2f**; small tensors
  retain their required float formats.
- Full companion: **q8_2f vision and audio tokenizer**, GPTQ-q4tp audio patch encoder.
  The package name does not mean every tensor is four-bit.
- Recorded Wikitext PPL: **4.478**, source **4.361**, same four 512-token windows.
- CPU/full-GPU/24-GB-budget PPL128: **3.647**. EN/RU/code: all 128 token IDs match
  between plain/MTP and repeated runs. Separate 456-token prompt + 64-token
  CPU/GPU greedy continuation: exact IDs.
- Full passes OCR, shapes, chart reading and five ASR clips (**WER 0%**).
  **Open gates:** strict vision row cosine, precise video timestamp answer,
  with quantized tower weights. Exact-weight default-GPU tower precision now passes all six reference fixtures (maximum relative error 3.663e-5).

These open checks prevent claiming fully qualified multimodal inference.
[Detailed results and limitations](VALIDATION.md) are kept separate from the quick start.

## Verify the download

Every weight file has a matching `.sha256` file. All three also passed
`cortiq verify`: envelope, tensor directory and per-tensor hashes.

```bash
sha256sum -c MiMo-V2.6-Flash-RL-q4tp.cmf.sha256
sha256sum -c MiMo-V2.6-Flash-RL-q4tp.mtp.cmf.sha256
# Full only:
sha256sum -c MiMo-V2.6-Flash-RL-q4tp.mm.cmf.sha256
cortiq verify MiMo-V2.6-Flash-RL-q4tp.cmf
```

## Generated example

[Download the interactive aquarium HTML](examples/aquarium/aquarium.html) ·
[Original prompt](examples/aquarium/prompt.md) · [Generation and QA notes](examples/aquarium/QA.md)

![MiMo-generated Three.js aquarium](examples/aquarium/desktop.png)

Browser-tested at 1280×800 and 390×844, with no captured console warnings/errors.
The original run reached 16,000 tokens; a model continuation supplied the ending.
The downloadable example includes documented, minimal manual QA fixes for
waypoints, tail geometry and bubble timing. The [unaltered assembled model output](examples/aquarium/aquarium.model.html)
is also provided. This is not presented as a successful default one-shot result.

---

## Русский

**Две комплектации q4tp:**

- **[Только текст — 164.851 ГБ](packages/text-only/README.md):** основа + MTP.
- **[Полная — 166.093 ГБ](packages/full/README.md):** та же основа + MTP + MM
  для изображений, видео и аудио. Все файлы уже загружены.

Для перехода к полной достаточно добавить **1.241 ГБ `.mm.cmf`** рядом с основой.
Повторно скачивать 164 ГБ не нужно. Текстовые запросы не загружают медиабашни.

```bash
cortiq run MiMo-V2.6-Flash-RL-q4tp.cmf --prompt "Почему небо голубое?"
```

Нужна сборка движка с поддержкой MiMo. Размещение на GPU и обнаружение
компаньонов автоматические. GPU-MTP работает при жадном декодировании;
для обычной генерации с ненулевой температурой этот путь не используется.

На RTX PRO 6000 96 ГБ медиана core-бенчмарка — **40.85 ток/с**; при бюджете
64000 MiB — **44.48 ток/с**. Таблица выше — бюджеты на одной карте, не тесты
физических карт всех размеров. Скорость на естественных промптах отличается.

Это **кандидат, не полностью проверенный релиз**. Текстовые проверки проходят;
OCR, фигуры, диаграмма и пять ASR-клипов проходят. Строгие проверки vision,
и временной отметки видео ещё не закрыты. Точность GPU-башни с точными весами проверена: все шесть тестов прошли. Подробности —
в [отчёте](VALIDATION.md). Лицензия исходных весов — **MIT**.

---

## 中文

**提供两种 q4tp 套件：**

- **[纯文本版 · 164.851 GB](packages/text-only/README.md)：**主模型 + MTP。
- **[完整多模态版 · 166.093 GB](packages/full/README.md)：**相同主模型 + MTP +
  图像、视频和音频输入编码器。所有权重文件均已上传。

升级只需将 **1.241 GB `.mm.cmf`** 放在主模型旁，无需重新下载 164 GB 主干。
纯文本请求不加载媒体编码器。使用支持 MiMo 的候选引擎：

```bash
cortiq run MiMo-V2.6-Flash-RL-q4tp.cmf --prompt "天空为什么是蓝色的？"
```

GPU 放置和伴随文件发现均为自动。GPU MTP 推测验证用于贪心解码，
不用于非零温度采样，也不保证所有提示均加速。

RTX PRO 6000 96 GB 上 core 基准中位数 **40.85 token/s**；64000 MiB
权重预算下为 **44.48 token/s**。上表为同一显卡的预算测试，不代表已验证
所有容量的实体显卡。自然语言提示的速度有所不同。

本版本仍是**候选版本**：文本、OCR、形状、图表和五段 ASR 测试通过；
严格量化视觉精度及视频时间戳验收尚未完成；默认 GPU 精确权重编码器的六项参考测试均已通过。
完整限制见[验证报告](VALIDATION.md)。权重沿用 **MIT** 许可证。
