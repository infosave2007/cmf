English: [README.md](README.md) · 中文: [README.zh.md](README.zh.md)

# CMF — Cortiq Model Format

**Обслуживайте множество специализированных LLM из одной общей модели — в единственном файле, на CPU или GPU.**

[![CI](https://github.com/infosave2007/cmf/actions/workflows/ci.yml/badge.svg)](https://github.com/infosave2007/cmf/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/cortiq-core.svg)](https://crates.io/crates/cortiq-core)
[![downloads](https://img.shields.io/crates/d/cortiq-cli.svg)](https://crates.io/crates/cortiq-cli)
[![docs.rs](https://img.shields.io/docsrs/cortiq-core)](https://docs.rs/cortiq-core)
[![stars](https://img.shields.io/github/stars/infosave2007/cmf?style=flat)](https://github.com/infosave2007/cmf)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/infosave2007/cmf/blob/master/LICENSE)

---

## Проблема

Выкатывать целый парк моделей, заточенных под отдельные задачи, — дорого. *N* дообученных версий обычно означают *N* полных копий на диске и в оперативной памяти — плюс разрозненные файлы-спутники `config.json` / `tokenizer.json` / адаптеры, которые нужно держать синхронизированными, и никакого встроенного способа отличить повреждённый файл от исправного.

## Идея

**CMF хранит один backbone и накладывает поверх него лёгкие оверлеи под каждый навык.** Навык хранит только те тензоры, которые он реально меняет; на инференсе рантайм читает тензоры выбранного навыка *вместо* тензоров backbone — отдельная модель никогда не собирается. Так целый набор специалистов умещается в **один самоописываемый файл** и запускается с ноутбука: веса читаются прямо с диска (`mmap`, zero-copy), а неиспользуемые навыки не стоят ни байта оперативной памяти.

И специалист не просто дешевле — он *лучше на своей задаче*: по замерам на отложенных данных навык, наложенный на свой backbone, снижает **перплексию на задаче на 24.9%** относительно одного лишь backbone (см. [спецификацию §9](docs/CMF_V2_SPEC.md)).

## Кому это нужно

- **Тем, кто строит агентов и плагины** — одна модель, несущая 20 навыков (SQL, код, перевод…), вместо 20 моделей, которые нужно хранить, загружать и между которыми нужно маршрутизировать.
- **Для edge- и локального развёртывания** — уместите маршрутизируемую многонавыковую модель в бюджет оперативной памяти одной модели; веса подгружаются с диска по требованию.
- **Всем, кто выкатывает квантованные LLM** — один файл с проверкой целостности несёт веса **+ токенизатор + шаблон чата**, так что терять нечего — спутников нет, а повреждения ловятся потензорными хешами.

## Посмотрите в деле

```console
$ cortiq run model.cmf --prompt "What is the capital of France?" --greedy
Ready: qwen2 | Task: general | Sparsity: 0%
Prompt: What is the capital of France?
 The capital of France is Paris.
[8 tokens, 33.6 tok/s, finish: stop]
```

## Почему CMF — что вы получаете

- **Добавляйте навык, не копируя модель.** Один backbone + небольшие дельты под каждый навык: объём хранения — это `|backbone| + Σ|deltas|`, а не `N × |model|`.
- **Мгновенный старт, лёгкость по RAM.** Веса отображаются в память и читаются на месте; отмаскированные или неиспользуемые веса вообще не попадают в оперативную память.
- **Меньше на диске — честно.** Смешивайте квантизации потензорно — `q8`, `q4`, двухполевой `q8_2f`, с переменной битностью (3–8 bit) — вплоть до ~1 byte/param и ниже. Двухполевой и переменнобитный кодеки восстанавливают большую часть разрыва в качестве int8→fp16 при том же размере файла, а компромисс по точности *измерен*, а не заявлен.
- **Один файл, никаких спутников.** Токенизатор HF (byte-level BPE) и шаблон чата (Jinja) путешествуют внутри модели — поведение в чате задаёт сам файл, а не ваш рантайм-бинарник.
- **Доверяйте файлу.** Фиксированный 128-byte конверт плюс 64-bit хеш на каждый тензор означают, что `.cmf` либо валиден, либо `open()` возвращает ошибку; `cortiq verify` проверяет всю цепочку.
- **Работает где угодно.** Ядро на Rust без зависимостей на CPU, плюс опциональный GPU-бэкенд (wgpu → Vulkan · Metal · DX12).
- **Конвертация одной командой.** `cortiq convert --model <hf-repo>` — нативно на Rust, без Python/numpy/torch; модель скачивается (параллельно) и квантуется за один шаг.

## Как это соотносится с другими подходами

Обслуживание **N специалистов под задачи**:

| | N полных дообучений | База + N внешних LoRA | **CMF — один backbone + N навыков** |
|---|---|---|---|
| На диске | N × полная модель | база + N адаптеров (спутники) | один backbone + N небольших дельт, **один файл** |
| Токенизатор + шаблон чата | на каждую копию / спутник | спутник | **встроены** |
| Потензорный хеш целостности | — | — | **да** |
| Холодный / неиспользуемый навык в RAM | загружен | загружен | **0** (подгружается по мере использования) |

Полное, честное сравнение формат-за-форматом — GGUF, safetensors, ONNX, PyTorch, GGML, TensorRT, с явно расписанными компромиссами — в [docs/COMPARISON.md](docs/COMPARISON.md).

## Установка

Установите инструмент командной строки:

```sh
cargo install cortiq-cli
```

Используйте формат из своего проекта на Rust:

```sh
cargo add cortiq-core
```

## Быстрый старт

Осмотрите `.cmf` — архитектура, тензоры, квантизация, маски и навыки:

```sh
cortiq info  model.cmf
cortiq masks model.cmf
cortiq verify model.cmf     # envelope, sections, per-tensor hashes
```

Сконвертируйте модель в `.cmf` — **нативно на Rust, без Python/numpy/torch**.
Укажите id репозитория Hugging Face (скачается параллельно) или локальную папку модели:

```sh
cortiq convert --model Qwen/Qwen2.5-0.5B-Instruct --quant q8    --output model.cmf
cortiq convert --model ./my-hf-checkpoint         --quant q8_2f --output model.cmf
```

Или импортируйте GGUF напрямую — локальный файл или **id GGUF-репозитория**
Hugging Face (лучший `.gguf` выбирается и скачивается). Все распространённые
ggml-кванты декодируются нативно (`Q4_0/1`, `Q5_0/1`, `Q8_0`, `Q2_K`…`Q6_K`,
`IQ4_NL/XS`, `BF16`) — без Python:

```sh
cortiq import-gguf Qwen/Qwen2.5-0.5B-Instruct-GGUF --output model.cmf --quant q8
cortiq import-gguf model.gguf                      --output model.cmf --quant q8
```

Квантизация: `q8` · `q8_2f` (двухполевая, лучшее качество/размер) · `q4` · `f16` ·
`vbit` (переменная 3–8 бит, ~4.25 в среднем).
Dense-, **MoE-** и **GatedDeltaNet-модели** (qwen2 / qwen3 / qwen3.5 / llama /
mistral / qwen-moe) конвертируются нативно — включая слитую раскладку
qwen3_next / AgentWorld. Встроенный Python-конвертер (`converter/`) теперь нужен
лишь для **GPTQ-калиброванного** v-bit (требует гессиан активаций) — базовый
v-bit по весам нативный.

Запустите инференс:

```sh
# Interactive chat
cortiq run model.cmf

# Single prompt, greedy decoding, capped length
cortiq run model.cmf --prompt "Write a haiku about memory-mapped files." --greedy --max-tokens 64

# Overlay a specific skill — its replacement tensors are read in place of the backbone
cortiq run model.cmf --prompt "SELECT ..." --skill sql
```

## Устройство контейнера

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

Читатель адресует секции **только** через конверт — никогда не полагаясь на их порядок.

## Возможности

- Однофайловый, отображаемый в память, самопроверяемый бинарный контейнер.
- Бинарный каталог тензоров с именами тензоров 1:1 к исходной модели и 64-битным хешем на каждый тензор для обнаружения повреждений.
- Смешанная квантизация потензорно: `f32`, `f16`, `bf16`, `q8_row`, `q4_block`, `q8_2f`, `vbit`.
- Встроенный токенизатор (паритет с HF byte-level BPE) и шаблон чата (Jinja, семантика HF).
- Маски под каждую задачу (bit-packed) и предвычисленный разреженный индекс.
- Рой из множества навыков: один backbone + полноформенные замещающие тензоры под каждый навык, накладываемые на этапе прямого прохода; рост только добавлением (append-only) и уплотнение.
- Опциональная голова многотокенного предсказания (MTP) и FFN-слои со смесью экспертов (MoE).
- Шардинг: модель, разбитая на `N` самостоятельно валидных файлов `.cmf`.
- Рантайм на Rust без зависимостей на **CPU и GPU** (опциональная фича `gpu`: wgpu → Vulkan / DX12 / Metal).
- Референсные реализации на Rust (читатель + рантайм) и Python (писатель + читатель на stdlib+numpy).

## Обзор формата

Полная нормативная спецификация — конверт, header JSON, каталог тензоров, раскладки квантизации, маски, комплект токенизатора, разреженный индекс, `hash64`, навыки и шардинг — находится в [docs/CMF_V2_SPEC.md](docs/CMF_V2_SPEC.md).

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
converter/        Python converters for exotic archs (MoE / linear-attention)
python/           dependency-free reader (stdlib + numpy)
docs/             format specification and comparison
```

## Лицензия и патенты

Распространяется под лицензией **Apache License, Version 2.0** — см. [LICENSE](LICENSE).

Это программное обеспечение реализует методы, являющиеся предметом трёх патентных заявок в Соединённых Штатах; подробности — в [PATENTS.md](PATENTS.md). Патентный грант из раздела 3 Apache-2.0 распространяется на эти три упомянутые заявки, предоставляя каждому пользователю безвозмездную (royalty-free) лицензию на патентные притязания, неизбежно нарушаемые данным программным обеспечением в том виде, в каком оно распространяется.
