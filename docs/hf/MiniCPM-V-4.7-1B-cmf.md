---
license: apache-2.0
library_name: cortiq
base_model: openbmb/MiniCPM-V-4.7-1B
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

# MiniCPM-V 4.7 1B — CMF text backbone

Two [CMF](https://github.com/infosave2007/cmf) files for the **text
backbone** of [OpenBMB MiniCPM-V 4.7 1B](https://huggingface.co/openbmb/MiniCPM-V-4.7-1B).
Each file bundles the quantized language weights, tokenizer, and chat template
for `cortiq`.

> **Important — text only.** This is not a full MiniCPM-V distribution. It
> does **not** accept or process images or video. The source vision tower,
> visual merger/projector, visual-token injection, and canvas M-RoPE path are
> intentionally absent. Do not send image/video content or rely on
> `<|image_pad|>` / `<|video_pad|>`: those text tokens do not enable vision in
> this release.

The upstream checkpoint is a `minicpmv4_7` multimodal wrapper around a dense
`qwen3_5_text` language decoder: 24 layers (18 Gated DeltaNet
linear-attention layers and 6 full-attention layers), hidden size 1024,
3584-wide MLP, and a 248,094-token vocabulary. Its config advertises a
262,144-token maximum position range. The source checkpoint contains 453
vision-tower tensors and 6 merger tensors; this text-only packaging omits
those tensors instead of storing unreadable payloads. It also contains no
physical `mtp.*` tensors, so this release does not ship an MTP/speculative
head even though the upstream config advertises one layer.

## Quick start

```bash
REPO=infosave/MiniCPM-V-4.7-1B-cmf
MODEL=MiniCPM-V-4.7-1B-q4tp.cmf

hf download "$REPO" "$MODEL" --local-dir .
cortiq verify "$MODEL"
cortiq run "$MODEL" --prompt "Explain quicksort in three sentences." --no-think
```

Use `q4tp` first when storage and memory are limited. Use `q8_2f` when
fidelity matters more than footprint. The CMF profile name is **`q8_2f`**;
`q8_2` in informal shorthand refers to this file, not to a separate profile.

## Files and precision policy

| file | profile | intended use |
|---|---|---|
| `MiniCPM-V-4.7-1B-q4tp.cmf` | q4tp decoder body; q8_2f embedding and LM head; protected recurrent/control tensors retain their required higher-precision representation | compact quality-balanced profile; validate on the target device before deployment |
| `MiniCPM-V-4.7-1B-q8_2f.cmf` | `q8_2f` for ordinary 2-D language matrices; the same protected controls | higher-fidelity reference |

The 248k-token embedding and LM head remain q8_2f in the q4tp file to avoid
placing a second four-bit error at both vocabulary boundaries. The Gated
DeltaNet control path and norms are likewise not silently forced into the
low-bit matrix plane. Inspect the actual tensor directory with, for example,
`cortiq info "$MODEL" --tensors model.layers.`; a CMF profile can contain mixed
per-tensor dtypes by design.

The release manifest records exact sizes and hashes, but this card makes no
unmeasured perplexity, quality, or throughput claim. Publish those values only
after device-specific benchmark checks on the exact artifacts.

## Release manifest

| file | exact bytes | SHA-256 |
|---|---:|---|
| `MiniCPM-V-4.7-1B-q4tp.cmf` | 798,400,417 | `72e9222f9c5482f70371b34b39391a6bddb8173814abe501ede54a6baeab3106` |
| `MiniCPM-V-4.7-1B-q8_2f.cmf` | 1,037,030,305 | `b5bbf9eaa9644c199475f8d5ac99821873ced4fe30831fab0b2c945bfc06079c` |

Each artifact and its matching `.sha256` file was verified with `cortiq verify`
before publication. The table records artifact identity, not a quality or
throughput claim.

## Apple silicon / Metal

`q4tp` is the profile to evaluate first on a memory-constrained Mac. `q8_2f`
is a fidelity alternative, not a promise of higher Metal throughput. Confirm
the selected backend and measure the released file on the target machine:

```bash
CMF_GPU=1 RUST_LOG=info \
  cortiq run MiniCPM-V-4.7-1B-q4tp.cmf \
  --prompt "Summarize the purpose of a hash table." --no-think --max-tokens 128
```

A successful conversion or CPU smoke test is not a Metal performance result.
For long prompts, account for context/KV memory in addition to the mapped
model file.

## Scope and known limitations

- **Supported:** ordinary text prompts and text chat through the embedded
  tokenizer/template.
- **Not supported in this release:** image understanding, video understanding,
  visual crops/slices, vision encoder, merger/projector, visual-token
  injection, or canvas M-RoPE positions for multimodal spans.
- A visual placeholder in a text prompt remains text; it is not a replacement
  for image embeddings.
- Use an implementation with an explicit MiniCPM-V processor, vision tower,
  projector, and canvas M-RoPE runtime for the original model's visual
  features.

## Verify a release

Every published artifact should have its own checksum file.

```bash
# Linux
sha256sum -c MiniCPM-V-4.7-1B-q4tp.cmf.sha256
# macOS
shasum -a 256 -c MiniCPM-V-4.7-1B-q4tp.cmf.sha256
cortiq verify MiniCPM-V-4.7-1B-q4tp.cmf
cortiq info MiniCPM-V-4.7-1B-q4tp.cmf

# Repeat with MiniCPM-V-4.7-1B-q8_2f.cmf when downloaded.
```

Weights derive from OpenBMB's Apache-2.0 release. This card describes only the
CMF text-backbone packaging; see the upstream card for the complete multimodal
model, license, usage, and safety information.

---

## Документация на русском

Это две [CMF](https://github.com/infosave2007/cmf)-сборки **только текстовой
части** [OpenBMB MiniCPM-V 4.7 1B](https://huggingface.co/openbmb/MiniCPM-V-4.7-1B).
Каждый файл содержит квантизованные языковые веса, токенизатор и chat template
для запуска через `cortiq`.

> **Важно: это не полная MiniCPM-V.** Сборка не принимает и не обрабатывает
> изображения или видео. В ней намеренно нет vision tower, merger/projector,
> вставки visual tokens и canvas M-RoPE пути исходной модели. Токены
> `<|image_pad|>` и `<|video_pad|>` сами по себе не добавляют поддержку зрения.

В исходном checkpoint `minicpmv4_7` оборачивает плотный языковой декодер
`qwen3_5_text`: 24 слоя (18 Gated DeltaNet linear-attention и 6
full-attention), hidden size 1024, MLP 3584 и словарь 248 094 токена. В
конфигурации указан максимум 262 144 позиций. В checkpoint есть 453 тензора
vision tower и 6 тензоров merger; в этой text-only упаковке они не сохраняются
как бесполезный payload. Физических тензоров `mtp.*` в checkpoint нет, поэтому
MTP/speculative head в релиз не входит, хотя конфигурация объявляет один слой.

### Быстрый старт

```bash
REPO=infosave/MiniCPM-V-4.7-1B-cmf
MODEL=MiniCPM-V-4.7-1B-q4tp.cmf

hf download "$REPO" "$MODEL" --local-dir .
cortiq verify "$MODEL"
cortiq run "$MODEL" --prompt "Объясни быструю сортировку в трёх предложениях." --no-think
```

При ограниченной памяти и месте сначала используйте `q4tp`. Для большей
точности выбирайте `q8_2f`. Официальное имя профиля CMF — **`q8_2f`**:
сокращение `q8_2` означает именно этот файл, а не отдельный формат.

### Файлы и политика точности

| файл | профиль | назначение |
|---|---|---|
| `MiniCPM-V-4.7-1B-q4tp.cmf` | q4tp для decoder body; q8_2f для embedding и LM head; защищённые рекуррентные/управляющие тензоры сохраняют требуемую повышенную точность | компактный quality-balanced профиль; проверьте его на целевом устройстве до эксплуатации |
| `MiniCPM-V-4.7-1B-q8_2f.cmf` | `q8_2f` для обычных 2-D матриц языка; те же защищённые управляющие тензоры | более точный эталон |

Embedding и LM head для словаря из 248 тыс. токенов остаются q8_2f даже в
q4tp-файле, чтобы не вносить четырёхбитную ошибку на обеих границах словаря.
Управляющий путь Gated DeltaNet и нормы также не загоняются молча в low-bit
матрицы. Фактический каталог тензоров смотрите, например, через
`cortiq info "$MODEL" --tensors model.layers.`: смешанные типы тензоров внутри
одного профиля CMF могут быть намеренными.

В release manifest указаны точные размеры и хеши, но в карточке нет
неизмеренных perplexity, оценок качества или tok/s. Такие значения можно
добавлять только после отдельных измерений на конкретном устройстве.

### Apple silicon / Metal

На Mac с ограниченной памятью первым проверяйте `q4tp`. `q8_2f` — вариант с
большей точностью, но не обещание большей скорости Metal. Убедитесь по логу,
что выбран нужный backend, и измерьте именно опубликованный файл:

```bash
CMF_GPU=1 RUST_LOG=info \
  cortiq run MiniCPM-V-4.7-1B-q4tp.cmf \
  --prompt "Кратко объясни назначение хеш-таблицы." --no-think --max-tokens 128
```

Успешная конвертация или CPU smoke test не являются результатом Metal
benchmark. Для длинных prompt учитывайте память контекста/KV помимо
memory-mapped файла модели.

### Границы поддержки

- Поддерживаются обычные текстовые запросы и текстовый чат через встроенные
  токенизатор/template.
- Не поддерживаются изображения, видео, crops/slices, vision encoder,
  merger/projector, вставка visual tokens и canvas M-RoPE позиции
  мультимодальных фрагментов.
- Visual placeholder в текстовом prompt остаётся текстом и не заменяет image
  embedding.
- Для исходных визуальных возможностей MiniCPM-V нужен runtime, который
  реализует processor, vision tower, projector и canvas M-RoPE.

### Проверка релиза

Каждому опубликованному artifact должен соответствовать свой checksum-файл:

```bash
# Linux
sha256sum -c MiniCPM-V-4.7-1B-q4tp.cmf.sha256
# macOS
shasum -a 256 -c MiniCPM-V-4.7-1B-q4tp.cmf.sha256
cortiq verify MiniCPM-V-4.7-1B-q4tp.cmf
cortiq info MiniCPM-V-4.7-1B-q4tp.cmf

# Повторите команды для MiniCPM-V-4.7-1B-q8_2f.cmf.
```

Веса происходят из Apache-2.0 релиза OpenBMB. Эта карточка описывает только
text-backbone упаковку CMF; полное мультимодальное описание, лицензию,
сценарии использования и сведения о безопасности смотрите в карточке исходной
модели.
