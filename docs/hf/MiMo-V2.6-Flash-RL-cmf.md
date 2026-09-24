---
license: mit
library_name: cortiq
base_model: XiaomiMiMo/MiMo-V2.6-Flash-RL
base_model_relation: quantized
pipeline_tag: text-generation
tags:
  - cmf
  - cortiq
  - quantized
  - multimodal
language:
  - en
  - ru
  - zh
---

# MiMo-V2.6-Flash-RL — CMF q4tp

> **Release draft — do not publish yet.** The reference-layer drift audit and aquarium example are not complete.
> The VRAM ladder exposed staging-memory peaks; a bounded-upload fix is under test.
> Strict vision cosine and video timestamp gates remain open. These packages
> are candidates; the MiMo-enabled engine has not been released as 0.7.7.

[Xiaomi MiMo-V2.6-Flash-RL](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL)
in [CMF](https://github.com/infosave2007/cmf), for the Cortiq Rust inference
engine. Two q4tp packages share the same text weights:

| Package | Files | Total size (decimal GB) |
|---|---|---:|
| **Text-only** | backbone + MTP | 164.851 |
| **Full** | backbone + MTP + multimodal companion | 166.093 |

Text-only needs no multimodal download. Upgrade to full by adding only the
`.mm.cmf` file beside the backbone. Text requests do not load media towers.

[English](#quick-start) · [Русский](#документация-на-русском) · [中文](#中文文档)

## Quick start

Commands below target the **candidate MiMo-enabled build**, not the current
published engine. Final installation/download commands are pending release.

```bash
cortiq run MiMo-V2.6-Flash-RL-q4tp.cmf --prompt "Explain why the sky is blue."
cortiq serve MiMo-V2.6-Flash-RL-q4tp.cmf --host 127.0.0.1 --port 8080
```

GPU placement is automatic: resident layer prefix, dynamic expert cache or a
hybrid, according to available memory. The MTP companion is discovered by
filename. GPU speculative verification is used for greedy decoding; ordinary
nonzero-temperature sampling does not use this greedy MTP path.

## Files and precision

| File | Bytes | Role |
|---|---:|---|
| `MiMo-V2.6-Flash-RL-q4tp.cmf` | 163862777935 | shared text backbone |
| `MiMo-V2.6-Flash-RL-q4tp.mtp.cmf` | 988712960 | three checkpoint draft layers |
| `MiMo-V2.6-Flash-RL-q4tp.mm.cmf` | 1241235456 | image/video/audio input towers |

The text backbone is unchanged between packages. **Experts are recoded from
upstream MXFP4 to q4tp**, not claimed to be a direct BF16 quantization.
Attention projections, embedding/head and the initial dense MLP use q8_2f;
routers/norms and special small tensors retain their required float formats.
The current companion uses q8_2f vision and audio-tokenizer matrices and a
GPTQ-q4tp audio patch encoder. Gather tables and RVQ codebooks retain their
required precision. “q4tp” names the package profile, not every tensor's dtype.

Recorded text quality: Wikitext perplexity **4.478**, against **4.361** for
source weights on the same four 512-token windows. These are not the separate
128-token smoke-test results: that corpus scores **3.647** on CPU, full-budget
GPU and both repeated 24-GB-budget checks after the worker-placement fix.

## Measured performance

RTX PRO 6000 Blackwell 96 GB, Vulkan, one stream, current candidate build:

```bash
cortiq bench MiMo-V2.6-Flash-RL-q4tp.cmf --core --tokens 128 --ignore-eos --json
```

Three default-setting runs: **32.20 / 41.40 / 40.85 tok/s; median 40.85**.
No steady-window weight uploads were recorded. This is a warmed synthetic
core benchmark, not a promise of 40 tok/s for every prompt or sampling mode.

Natural 128-token greedy prompts, one loaded model:

| Prompt | Warm plain, two repetitions | MTP |
|---|---:|---:|
| English | 34.24 / 33.05 | 28.16 |
| Russian | 36.05 / 35.29 | 32.62 |
| Code | 29.90 / 30.12 | 33.62 |

All four arms (cold plain, warm plain, MTP, repeated plain) emitted identical
IDs for each prompt. A separate exact-float CPU/GPU check used a 456-token
natural prompt and matched all 64 subsequent greedy token IDs, beyond the
sliding-window boundary. MTP is not faster on every text. The following are **weight budgets on the
same 96-GB card**, not measurements on six different physical cards:

| `CMF_GPU_VRAM_MB` (MiB) | Auto placement | Median tok/s, 3 runs | Largest sampled VRAM peak (MiB) |
|---:|---|---:|---:|
| 16000 | dynamic | 10.96 | 20417 |
| 24000 | dynamic | 17.27 | 33111 |
| 32000 | dynamic | 29.66 | 46585 |
| 48000 | dynamic | 36.47 | 58102 |
| 64000 | dynamic | 44.48 | 70164 |
| 80000 | hybrid, 8 prefix layers | 42.36 | 85497 |

These runs expose temporary allocations above the configured weight budget;
**they do not qualify operation on physical 16–80-GB cards**. A bounded staging
candidate is under test. Host RAM must also accommodate mapped text weights
and execution state. No unverified low-memory compatibility is claimed.

## Full package

```bash
cortiq run MiMo-V2.6-Flash-RL-q4tp.cmf --prompt "Read the text." --image page.png
cortiq run MiMo-V2.6-Flash-RL-q4tp.cmf --prompt "Describe the clip." --video frames --video-fps 1
cortiq run MiMo-V2.6-Flash-RL-q4tp.cmf --prompt "Transcribe the speech." --audio speech.wav
```

OpenAI chat ingress preserves ordered `image_url`, `input_audio`, `audio_url`
and `video` content blocks. Audio input currently supports WAV; silent video
supports a local frame directory or Y4M. MP4, compressed audio and interleaved
video audio tracks are not implemented. This is input understanding, not
image/video/speech generation.

Candidate acceptance results: OCR `CORTIQ 7392`, colored-shape order, a chart
question and five speech clips pass (WER 0%, including resampled/stereo WAV).
**Not passed:** strict vision row cosine (mean 0.99774, minimum 0.85990;
6/1488 rows below 0.98), and the digit-at-00:04 video fixture (6 instead of 5,
also with exact BF16 towers). These limitations are not hidden by the aggregate
scores and prevent claiming full qualification.

## Verify the download

Checksum files must be generated/verified before publication. After release:

```bash
sha256sum -c MiMo-V2.6-Flash-RL-q4tp.cmf.sha256
sha256sum -c MiMo-V2.6-Flash-RL-q4tp.mtp.cmf.sha256
# Full package only:
sha256sum -c MiMo-V2.6-Flash-RL-q4tp.mm.cmf.sha256
cortiq verify MiMo-V2.6-Flash-RL-q4tp.cmf
```

Weights retain the upstream MIT license. The interactive aquarium HTML and
browser screenshot will be included only after generation and browser QA.

---

## Документация на русском

**Черновик релиза; не публиковать до завершения проверок.** Две комплектации
q4tp: **только текст** (основа + MTP, 164.851 ГБ) и **полная** (те же файлы +
`.mm.cmf`, 166.093 ГБ). Для перехода к полной достаточно скачать компаньон;
повторная загрузка основы не нужна. Текстовые запросы не загружают MM-башни.

Запуск кандидата движка:

```bash
cortiq run MiMo-V2.6-Flash-RL-q4tp.cmf --prompt "Почему небо голубое?"
```

Размещение на GPU и обнаружение MTP автоматические. Спекулятивная проверка
работает для жадного декодирования, не для обычной генерации с ненулевой
температурой. Эксперты перекодированы **из MXFP4 в q4tp**; скелет — q8_2f.
В полной комплектации применены описанные выше более точные кодеки башен.

RTX PRO 6000 96 ГБ: медиана трёх `bench --core --tokens 128 --ignore-eos`
равна **40.85 ток/с** (32.20 / 41.40 / 40.85). На естественных промптах скорость
другая; MTP не всегда быстрее. На EN/RU/коде все 128 ID совпали в четырёх руках.
PPL128 = 3.647 на CPU, полном GPU и в двух повторах с бюджетом 24 ГБ.
Лестница бюджетов приведена выше: при 64000 MiB медиана 44.48 ток/с.
Пики VRAM превышали бюджет весов, поэтому это не подтверждение работы
на физических картах 16–80 ГБ; исправление временных аллокаций проверяется.

Полная версия читает изображения, WAV и немые видео из Y4M/каталога кадров.
OCR, фигуры, диаграмма и пять ASR-клипов прошли; строгий минимум cosine для
vision и точный timestamp видео пока **не прошли**. Это не законченный релиз.
Размеры файлов и команды SHA-256 приведены выше. Лицензия — MIT.

---

## 中文文档

**发布草稿：验收完成前不得公开发布。** 提供两个 q4tp 套件：
**纯文本版**（主模型 + MTP，164.851 GB）与**完整多模态版**
（相同文件 + `.mm.cmf`，166.093 GB）。升级仅需添加多模态伴随文件，
无需重新下载主模型；纯文本请求不会加载多模态编码器。

候选引擎用法：

```bash
cortiq run MiMo-V2.6-Flash-RL-q4tp.cmf --prompt "天空为什么是蓝色的？"
```

GPU 放置和 MTP 文件发现均为自动。当前 MTP 推测验证用于贪心解码，
不用于非零温度采样。专家权重由上游 **MXFP4 重新编码为 q4tp**；
注意力等主干矩阵采用 q8_2f。完整套件的编码器精度配置见上表。

RTX PRO 6000 96 GB 上，三次 128-token core 基准为
32.20 / 41.40 / 40.85 token/s，中位数 **40.85**。这不代表所有自然语言
请求都达到该速度。英、俄、代码提示的四种执行均输出相同的 128 个 ID。
CPU、完整 GPU 预算及两次 24-GB 预算测试的 PPL128 均为 3.647。
上表列出同一张卡的预算测试；64000 MiB 预算中位数为 44.48 token/s。
显存峰值超过权重预算，不能据此声称已验证实体 16–80 GB 显卡；
限制临时上传缓冲区的修复仍在测试中。

完整套件接受图像、WAV，以及 Y4M 或帧目录形式的无声视频。
OCR、形状、图表和五段语音测试已通过；严格视觉行余弦与精确视频时间戳
测试仍未通过。因此不能宣称已完成发布验收。文件大小与 SHA-256
验证命令见上文。权重沿用 MIT 许可证。
