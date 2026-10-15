---
license: apache-2.0
library_name: cortiq
base_model: openbmb/MiniCPM-V-4.7-35B-A3B
base_model_relation: quantized
pipeline_tag: text-generation
tags:
  - cmf
  - cortiq
  - quantized
  - q4tp
  - q8_2f
  - text-only
---

# MiniCPM-V 4.7 35B-A3B — CMF text backbone

Two [CMF](https://github.com/infosave2007/cmf) files for the **text
backbone** of [OpenBMB MiniCPM-V 4.7 35B-A3B](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B).
CMF packages quantized weights, tokenizer, and chat template in one
memory-mapped file for `cortiq`.

> **Important — text only.** This is not a full MiniCPM-V release. It does
> **not** accept or process images or video. The source vision tower, visual
> merger/projector, visual-token injection, and canvas M-RoPE path are not
> present in this CMF release. Do not send image/video content or rely on
> `<|image_pad|>` / `<|video_pad|>` tokens: they do not enable vision here.

The retained language backbone is a 40-layer Qwen3.5-style hybrid MoE model:
30 Gated DeltaNet linear-attention layers and 10 full-attention layers, with
256 routed experts (top 8) and a shared expert. The source checkpoint is
published in BF16; both files below are direct quantizations of that text
backbone. Although its config advertises one MTP layer, the checkpoint ships
no physical `mtp.*` tensors; this release therefore has no MTP/speculative
head.

## Quick start

```bash
REPO=infosave/MiniCPM-V-4.7-35B-A3B-cmf
MODEL=MiniCPM-V-4.7-35B-A3B-q4tp.cmf

hf download "$REPO" "$MODEL" --local-dir .
cortiq verify "$MODEL"
cortiq run "$MODEL" --prompt "Explain quicksort in three sentences." --max-tokens 256
```

Use the `q4tp` file first for a compact deployment. Select `q8_2f` when
fidelity is more important than storage and memory use.

## Files and precision policy

| file | profile | intended use |
|---|---|---|
| `MiniCPM-V-4.7-35B-A3B-q4tp.cmf` | q4tp routed and shared-expert matrices; `q8_2f` always-active text skeleton (embedding, LM head and attention/GDN); F16 sensitive controls | compact operational profile; validate on the target Mac before relying on Metal performance |
| `MiniCPM-V-4.7-35B-A3B-q8_2f.cmf` | `q8_2f` for ordinary 2-D language matrices; the same F16 controls | higher-fidelity reference / machines with more storage and unified memory |

The F16 controls include the MoE router and shared-expert gate, Gated
DeltaNet `in_proj_a` / `in_proj_b`, convolution and scalar/norm tensors.
Keeping these small recurrent and routing paths out of the low-bit body is
part of the profile, not an accidental fallback. The q4tp file therefore has
mixed per-tensor dtypes; inspect them with, for example,
`cortiq info "$MODEL" --tensors model.layers.`.

The release manifest records exact sizes and hashes. The scoped Vulkan
measurement below is neither a perplexity/quality result nor a general
throughput guarantee. In particular, the source name “35B-A3B” does not make
the full stored expert bank fit in 3 GB: plan for the complete CMF file,
runtime buffers, context/KV state, and macOS unified-memory headroom.

## Measured Vulkan profile (RTX 4090; opt-in)

Starting with `cortiq` **0.8.19**, the q4tp file has an opt-in Vulkan dynamic
MoE-tail path. It preserves the source router, top-8 routing weights, and
shared expert. A bounded GPU expert bank serves resident winners; cold routed
experts complete on CPU. An admission, slot-map, or GPU-completion failure
uses the ordinary CPU MoE path rather than returning a partial result.

```bash
CMF_GPU=1 WGPU_BACKEND=vulkan CMF_SDOT=0 \
CMF_MINICPM_DYNAMIC_TAIL=1 \
  cortiq run MiniCPM-V-4.7-35B-A3B-q4tp.cmf \
  --raw --prompt "The capital of France is" --greedy -n 512
```

On one RTX 4090 under Vulkan, that profile measured **23.8, 24.3, and 23.8
 tok/s** for 512 greedy generated tokens; first-token latency was **18.1,
17.9, and 17.9 s**, respectively. A separate 16-token greedy regression gate
matched the response text and all **16/16 argmaxes** against the ordinary
path; max `|Δlogit|` was `4.00543212890625e-05`.

These are one-machine measurements, not a promise for other GPUs, drivers,
prompts, context lengths, or Macs. They are not directly comparable with a
short CPU-only control run, so no universal percentage speedup is claimed.

## Release manifest

| file | exact bytes | SHA-256 |
|---|---:|---|
| `MiniCPM-V-4.7-35B-A3B-q4tp.cmf` | 19,372,423,311 | `b25db0effc41bda0175f15c41ec8ac29317d9105a42879ebbca5669a7f6ab1e5` |
| `MiniCPM-V-4.7-35B-A3B-q8_2f.cmf` | 34,994,075,791 | `c5f8e51903206ef0c42e4a93d2210899882c8f4782d195924430d6175abf1d91` |

Each artifact and its matching `.sha256` file was verified with `cortiq verify`
before publication. The table records artifact identity, not a quality or
throughput claim.

## Apple silicon / Metal

`q4tp` is the profile to evaluate first for a Metal deployment. `q8_2f` is a
higher-fidelity alternative, not a claim of faster Metal decoding. Actual
fit and throughput depend on the Mac, free unified memory, context length,
and the installed `cortiq` build.

```bash
CMF_GPU=1 RUST_LOG=info \
  cortiq run MiniCPM-V-4.7-35B-A3B-q4tp.cmf \
  --prompt "Summarize the purpose of a hash table." --max-tokens 128
```

Use the log to confirm the active backend, then measure the exact release on
the target machine. A CPU smoke test or a successful file conversion is not a
Metal performance result. `CMF_METAL_SPARSE_MOE=1` is an experimental,
explicit q4tp selected-view path; it preserves CPU routing and declines to the
ordinary path on admission/completion failure. It is not a whole-model Metal
throughput claim. For long prompts, account for the full-attention layers' KV
cache in addition to model weights.

## Scope and known limitations

- **Supported:** ordinary text prompts and text chat through the embedded
  tokenizer/template.
- **Not supported in this release:** image understanding, video understanding,
  visual crops/slices, the vision encoder, merger/projector, and canvas
  M-RoPE position construction for multimodal spans.
- A prompt containing visual placeholder tokens is still only text to this
  runtime; it is not a substitute for image embeddings.
- The source model's visual capability must be used through an implementation
  that explicitly supports its processor, vision tower, visual token injection,
  and canvas M-RoPE.

## Verify a release

Each published artifact should be accompanied by its matching checksum file.

```bash
# Linux
sha256sum -c MiniCPM-V-4.7-35B-A3B-q4tp.cmf.sha256
# macOS
shasum -a 256 -c MiniCPM-V-4.7-35B-A3B-q4tp.cmf.sha256
cortiq verify MiniCPM-V-4.7-35B-A3B-q4tp.cmf
cortiq info MiniCPM-V-4.7-35B-A3B-q4tp.cmf

# Repeat with MiniCPM-V-4.7-35B-A3B-q8_2f.cmf when downloaded.
```

Weights derive from OpenBMB's Apache-2.0 release. This card describes only
the CMF text-backbone packaging; see the upstream model card for the complete
multimodal model and its license, usage, and safety information.

---

## Документация на русском

Это две [CMF](https://github.com/infosave2007/cmf)-сборки **только текстовой
части** [OpenBMB MiniCPM-V 4.7 35B-A3B](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B).
В одном memory-mapped файле CMF хранятся квантизованные веса, токенизатор и
chat template для запуска через `cortiq`.

> **Важно: это не полная MiniCPM-V.** Сборка не принимает и не обрабатывает
> изображения или видео. В ней нет vision tower, merger/projector, вставки
> visual tokens и canvas M-RoPE пути исходной модели. Токены
> `<|image_pad|>` и `<|video_pad|>` сами по себе не добавляют поддержку зрения.

Сохранённый языковой backbone — гибридная MoE-модель в стиле Qwen3.5: 40
слоёв, из них 30 Gated DeltaNet linear-attention и 10 full-attention; 256
маршрутизируемых экспертов, top-8 и shared expert. Исходный checkpoint
опубликован в BF16; оба файла ниже являются прямой квантизацией этой
текстовой части. Хотя config объявляет один MTP-слой, физических тензоров
`mtp.*` в checkpoint нет; поэтому MTP/speculative head в релиз не входит.

### Быстрый старт

```bash
REPO=infosave/MiniCPM-V-4.7-35B-A3B-cmf
MODEL=MiniCPM-V-4.7-35B-A3B-q4tp.cmf

hf download "$REPO" "$MODEL" --local-dir .
cortiq verify "$MODEL"
cortiq run "$MODEL" --prompt "Объясни быструю сортировку в трёх предложениях." --max-tokens 256
```

Для компактного запуска сначала выбирайте `q4tp`. `q8_2f` нужен, когда
приоритетнее точность, чем место на диске и в unified memory.

### Файлы и политика точности

| файл | профиль | назначение |
|---|---|---|
| `MiniCPM-V-4.7-35B-A3B-q4tp.cmf` | q4tp для routed и shared-expert матриц; `q8_2f` для always-active text skeleton (embedding, LM head и attention/GDN); F16 для чувствительных управляющих тензоров | компактный рабочий профиль; перед эксплуатацией измерьте его на целевом Mac |
| `MiniCPM-V-4.7-35B-A3B-q8_2f.cmf` | `q8_2f` для обычных двумерных матриц языка; те же F16-управляющие тензоры | более точный эталон для машин с большим запасом места и unified memory |

К F16 относятся router MoE, shared-expert gate, `in_proj_a` / `in_proj_b`
Gated DeltaNet, свёртка, скалярные тензоры и нормы. Это сознательно выбранная смешанная
точность, а не случайный fallback: тип каждого тензора можно посмотреть
командой `cortiq info "$MODEL" --tensors model.layers.`.

В release manifest указаны точные размеры и хеши. Приведённый ниже
Vulkan-замер не является результатом perplexity/качества и не даёт общего
обещания tok/s. Метка исходника «35B-A3B» не означает, что весь банк экспертов
поместится в 3 ГБ: учитывайте полный CMF, буферы runtime, контекст/KV и запас
памяти macOS.

### Измеренный Vulkan-профиль (RTX 4090; opt-in)

Начиная с `cortiq` **0.8.19**, для q4tp доступен opt-in динамический
Vulkan MoE-tail. Он сохраняет исходные router, веса top-8 route и shared
expert. Ограниченный GPU-bank обслуживает resident winners, а cold routed
experts досчитываются на CPU. При ошибке admission, карты slot или completion
GPU выполняется обычный CPU MoE-путь, а не возвращается частичный результат.

```bash
CMF_GPU=1 WGPU_BACKEND=vulkan CMF_SDOT=0 \
CMF_MINICPM_DYNAMIC_TAIL=1 \
  cortiq run MiniCPM-V-4.7-35B-A3B-q4tp.cmf \
  --raw --prompt "The capital of France is" --greedy -n 512
```

На одной RTX 4090/Vulkan этот профиль показал **23.8, 24.3 и 23.8 tok/s**
на 512 greedy generated tokens; first-token был **18.1, 17.9 и 17.9 с**.
Отдельный strict regression gate на 16 токенах дал одинаковое тело ответа и
все **16/16 argmax** против обычного пути; max `|Δlogit|` —
`4.00543212890625e-05`.

Это замеры одной машины, а не обещание результата на других GPU, драйверах,
prompt, длинах контекста или Mac. Они не сопоставимы напрямую с коротким
CPU-only control run, поэтому универсальный процент ускорения не заявляется.

### Apple silicon / Metal

Для Metal сначала проверяйте `q4tp`; `q8_2f` — вариант с большей точностью,
а не обещание большей скорости. Реальная совместимость и скорость зависят от
конкретного Mac, свободной unified memory, длины контекста и версии `cortiq`.

```bash
CMF_GPU=1 RUST_LOG=info \
  cortiq run MiniCPM-V-4.7-35B-A3B-q4tp.cmf \
  --prompt "Кратко объясни назначение хеш-таблицы." --max-tokens 128
```

По логу убедитесь, какой backend включился, затем измерьте именно
опубликованный файл на целевой машине. Успешная CPU-проверка или конвертация
не является бенчмарком Metal. `CMF_METAL_SPARSE_MOE=1` — экспериментальный
явный q4tp selected-view путь: router остаётся на CPU, а при admission/completion
ошибке выполняется обычный путь. Это не заявление о whole-model скорости
Metal. Для длинных промптов дополнительно учитывайте KV-cache full-attention
слоёв.

### Границы поддержки

- Поддерживаются обычные текстовые запросы и текстовый чат через встроенные
  токенизатор и template.
- Не поддерживаются изображения, видео, crops/slices, vision encoder,
  merger/projector и canvas M-RoPE позиции мультимодальных фрагментов.
- Визуальный placeholder в текстовом prompt остаётся обычным текстом и не
  заменяет image embedding.
- Для исходных визуальных возможностей MiniCPM-V нужен runtime, который
  реализует processor, vision tower, вставку visual tokens и canvas M-RoPE.

### Проверка релиза

```bash
# Linux
sha256sum -c MiniCPM-V-4.7-35B-A3B-q4tp.cmf.sha256
# macOS
shasum -a 256 -c MiniCPM-V-4.7-35B-A3B-q4tp.cmf.sha256
cortiq verify MiniCPM-V-4.7-35B-A3B-q4tp.cmf
cortiq info MiniCPM-V-4.7-35B-A3B-q4tp.cmf

# Повторите команды для MiniCPM-V-4.7-35B-A3B-q8_2f.cmf.
```

Веса происходят из Apache-2.0 релиза OpenBMB. Эта карточка описывает только
текстовую CMF-упаковку; полное мультимодальное описание, лицензию, сценарии
использования и сведения о безопасности смотрите в карточке исходной модели.
