English: [README.md](README.md) · 中文: [README.zh.md](README.zh.md)

# CMF — Cortiq Model Format

**Один самоописываемый файл для квантованной LLM — веса, токенизатор, шаблон чата, маски задач и оверлеи под каждый навык — с портативным рантаймом на Rust без зависимостей, работающим на CPU и GPU (Vulkan · Metal · DX12).**

[![CI](https://github.com/infosave2007/cmf/actions/workflows/ci.yml/badge.svg)](https://github.com/infosave2007/cmf/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/cortiq-core.svg)](https://crates.io/crates/cortiq-core)
[![docs.rs](https://img.shields.io/docsrs/cortiq-core)](https://docs.rs/cortiq-core)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/infosave2007/cmf/blob/master/LICENSE)

---

## Зачем нужен CMF

- **Один файл.** Веса, токенизатор, шаблон чата, маски под каждую задачу и delta-записи под каждый навык поставляются как единый контейнер `.cmf` — одна единица распространения, без сопутствующих файлов.
- **Самоописываемость.** Фиксированный 128-байтный конверт (envelope) адресует каждую секцию; бинарный каталог тензоров — единственный источник истины о раскладке. Файлы проверяются, а не угадываются: `.cmf` либо валиден, либо `open()` возвращает ошибку.
- **mmap, zero-copy, работает где угодно.** Блок весов выровнен по странице, а каждый тензор выровнен по 64 байтам под SIMD. Рантайм отображает файл в память (memory-map) и читает веса на месте — «холодные» (замаскированные) веса не тратят RSS. CPU-ядру без зависимостей больше ничего не нужно; опциональный GPU-бэкенд нацелен на Vulkan, Metal и DX12.
- **Квантование.** Типы данных на уровне тензора включают `q8_row`, `q4_block`, двухполевой `q8_2f` и переменную битность `vbit` (3–8 бит), свободно смешиваемые в пределах одной модели.
- **Оверлей нескольких навыков.** Единый общий backbone плюс тензоры полной формы, заменяющие его под каждый навык. На проходе вперёд рантайм читает тензоры выбранного навыка *вместо* тензоров backbone — не материализуя отдельную модель. Объём хранения растёт как `|backbone| + Σ|deltas|`.

## Раскладка контейнера

```
 .cmf file
 ┌──────────────────────────────────────────────────────────┐
 │ Envelope        128 bytes, fixed                          │
 │   magic "CMF\x01" · version · feature bits · section       │
 │   offsets+lengths (header, dir, data, masks, vocab, index)│
 ├──────────────────────────────────────────────────────────┤
 │ Header JSON     arch, quant defaults, chat bundle,        │
 │                 skill registry, provenance                │
 ├──────────────────────────────────────────────────────────┤
 │ Tensor directory   binary 56-byte records:                │
 │                 name · dtype · shape · offset · nbytes ·  │
 │                 hash64  (read without parsing)            │
 ├──────────────────────────────────────────────────────────┤
 │ Weight blob     page-aligned (4096); every tensor 64-byte │
 │                 aligned; quantized; mmap zero-copy        │
 ├──────────────────────────────────────────────────────────┤
 │ Masks / Skills  bit-packed per-task masks (1 bit/neuron)  │
 │                 + per-skill replacement tensors           │
 ├──────────────────────────────────────────────────────────┤
 │ Tokenizer       HF tokenizer.json, verbatim               │
 ├──────────────────────────────────────────────────────────┤
 │ Sparse index    precomputed mask → active groups/heads    │
 └──────────────────────────────────────────────────────────┘
```

Читатель адресует секции **только** через конверт — никогда не полагаясь на предполагаемый порядок.

## Возможности

- Однофайловый, отображаемый в память, самопроверяемый бинарный контейнер.
- Бинарный каталог тензоров с именами тензоров 1:1 к исходной модели и 64-битным хешем на каждый тензор для обнаружения повреждений.
- Смешанное квантование на уровне тензора: `f32`, `f16`, `bf16`, `q8_row`, `q4_block`, `q8_2f`, `vbit`.
- Встроенный токенизатор (паритет с HF byte-level BPE) и шаблон чата (Jinja, семантика HF) — поведение чата задаёт файл, а не бинарник.
- Маски под каждую задачу (bit-packed) и предвычисленный разреженный индекс.
- Рой нескольких навыков (multi-skill swarm): один backbone + тензоры полной формы, заменяющие его под каждый навык, накладываемые на проходе вперёд через индирекцию источника тензора; рост только добавлением (append-only) и уплотнение.
- Опциональная голова multi-token-prediction (MTP) и слои FFN типа mixture-of-experts (MoE).
- Шардирование: модель, разбитая на `N` самостоятельно валидных файлов `.cmf`.
- Рантайм на Rust без зависимостей, работающий на **CPU и GPU** (опциональная фича `gpu`: wgpu → Vulkan / DX12 / Metal).
- Референсные реализации на Rust (читатель + рантайм) и Python (писатель + читатель на stdlib+numpy).

## Установка

Установить инструмент командной строки:

```sh
cargo install cortiq-cli
```

Использовать формат из своего Rust-проекта:

```sh
cargo add cortiq-core
```

## Быстрый старт

Осмотреть `.cmf` — архитектуру, тензоры, квантование, маски и навыки:

```sh
cortiq info model.cmf
cortiq masks model.cmf
```

Сконвертировать чекпоинт Hugging Face в `.cmf`:

```sh
python converter/convert_dtgma_to_cmf.py \
    --model  ./my-hf-checkpoint \
    --quant  Q8_ROW \
    --output model.cmf
```

Импортировать модель GGUF:

```sh
python converter/import_gguf.py --input model.gguf --output model.cmf
```

Запустить инференс:

```sh
# Интерактивный чат
cortiq run model.cmf

# Один промпт, жадное декодирование
cortiq run model.cmf --prompt "Write a haiku about memory-mapped files." --greedy

# Наложить конкретный навык (заменяющие тензоры читаются вместо backbone)
cortiq run model.cmf --prompt "SELECT ..." --skill sql
```

## Обзор формата

Полная нормативная спецификация — конверт, header JSON, каталог тензоров, раскладки квантования, маски, комплект токенизатора, разреженный индекс, `hash64`, навыки и шардирование — находится в [docs/CMF_V2_SPEC.md](docs/CMF_V2_SPEC.md).

## Сравнение

Как CMF соотносится с safetensors, GGUF и «сырыми» чекпоинтами HF — компромиссы одного файла, самоописываемости, mmap-квантования и мультинавыковости — описано в [docs/COMPARISON.md](docs/COMPARISON.md).

## Сборка из исходников

```sh
cargo build --release --workspace
```

Опциональный кроссплатформенный GPU-бэкенд (wgpu → Vulkan / DX12 / Metal):

```sh
cargo build --release --workspace --features gpu
```

## Структура проекта

```
crates/
  cortiq-core     format reader: envelope, directory, quant, masks, mmap
  cortiq-engine   portable CPU/GPU inference runtime, tokenizer, chat, skill overlay
  cortiq-server   OpenAI-compatible HTTP serving
  cortiq-cli      the `cortiq` command-line tool (inspect/convert/run/serve)
converter/        Python writers: HF → .cmf, GGUF → .cmf
python/           dependency-free reader (stdlib + numpy)
docs/             format specification and comparison
```

## Лицензия и патенты

Распространяется под лицензией **Apache License, Version 2.0** — см. [LICENSE](LICENSE).

Это ПО реализует методы, являющиеся предметом трёх патентных заявок США; подробности — в [PATENTS.md](PATENTS.md). Патентный грант из раздела 3 Apache-2.0 распространяется на эти три упомянутые заявки, предоставляя каждому пользователю безвозмездную (royalty-free) лицензию на патентные притязания, неизбежно нарушаемые данным ПО в том виде, в каком оно распространяется.
