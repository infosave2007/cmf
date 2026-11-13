---
license: other
license_name: qwen-community-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE
library_name: cortiq
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: text-generation
tags:
  - cmf
  - cortiq
  - quantized
  - q4tp
  - moe
  - code
  - expert-pruning
  - speculative-decoding
  - vulkan
  - cuda
language:
  - en
  - ru
  - zh
---

# Qwen3.8-Flash-Next Fast Code — CMF

[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) cut for code: 256 of the 512 routed experts of
every layer at 4 bits, with the shared expert, the draft (MTP) head and the n-gram table, in one 66.8 GB
[CMF](https://github.com/infosave2007/cmf) file. Runs with `cortiq`, a Rust inference engine, on two backends:

- **CUDA, NVIDIA cards from 12 GB.** Code at 145 / 175 / 207 tok/s within 12 / 16 / 24 GB of card memory: 94 / 91 /
  97 % of the speed of the 2-bit Strata Coder on the same card, with the experts kept at 4 bits and the attention
  at 8. Measured on an RTX 5090 with its memory held to 12, 16 and 24 GB.
- **Vulkan, one card with 48 GB or more.** The whole model on the card: prompt reading at ~840 tok/s, 4× the full
  512-expert file; code at 136–166 tok/s; 41 GB of card memory instead of 60. Measured on an RTX PRO 6000 96 GB.
- **Accuracy.** HumanEval pass@1 93.3 %; perplexity on fresh code 3.337, against 3.436 for the 2-bit Strata Coder.

[English](#which-backend) · [Русский](#документация-на-русском) · [中文](#中文文档)

## Which backend

| your card | backend | what it needs | measured on |
|---|---|---|---|
| NVIDIA, 12 GB or more | [CUDA](#cuda-nvidia-cards-from-12-gb) | cortiq 0.8.78+ built with `--features cuda`, CUDA 12.8+ libraries, 64 GB of RAM, a CPU with AVX-512 VNNI | RTX 5090, memory held to 12 / 16 / 24 GB |
| 48 GB or more, with a Vulkan driver | [Vulkan](#vulkan-one-card-with-48-gb-or-more) | cortiq 0.8.65+ (prebuilt binaries), 32 GB of RAM, any CPU | RTX PRO 6000 96 GB |

The file, the profile and the server API are the same for both. The two backends were measured on different cards
with different tasks: compare the numbers inside one section, not between the two.

## Download

The same files for both backends.

```bash
pip install -U huggingface_hub
hf download infosave/Qwen3.8-Flash-Next-Fast-Code-cmf qwen38-flash-next-fast-code.cmf fast-code.profile \
  qwen38-flash-next-fast-code.cmf.sha256 --local-dir .
sha256sum -c qwen38-flash-next-fast-code.cmf.sha256
```

## CUDA: NVIDIA cards from 12 GB

The card holds the experts that fit in its memory; the host's cores compute the rest.

### CUDA · Requirements

| component | requirement |
|---|---|
| GPU | NVIDIA, 12 GB or more; driver 570 or later |
| CUDA libraries | `libnvrtc` and `libcublas` of the CUDA 12.8+ toolkit, on `LD_LIBRARY_PATH` |
| CPU | x86-64 with AVX-512 VNNI (AMD Zen 4/5, Intel Sapphire Rapids or newer), 8 threads or more |
| RAM | 64 GB |
| disk | 67 GB |
| OS | Linux |
| cortiq | 0.8.78 or later, built with `--features cuda`; the prebuilt binaries have no CUDA |

### CUDA · Install

```bash
# Rust 1.88 or later; the build itself needs no CUDA toolkit
cargo install cortiq-cli --locked --features cuda
```

### CUDA · Run

```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH   # where libnvrtc and libcublas are
cortiq gpu   # "CUDA: <your card> …" means ready; "CUDA: unavailable — <reason>" says what is missing

CMF_GPU=cuda CMF_QWEN_PROFILE=fast-code.profile CMF_QWEN_ROUTE_TAIL=0.05 \
  CMF_SERVE_SLOTS=1 CMF_QWEN_CUDA_REG=1 CMF_QWEN_CUDA_REG_ALL=1 \
  cortiq serve qwen38-flash-next-fast-code.cmf --host 127.0.0.1 --port 8080
```

The server is ready when it answers on `/v1/models`; the first start reads the 66.8 GB file and takes a minute or
more. The line `qwen4 cuda: arena prefilled with …` in its log means the card is in use. Requests:
[Using the server](#using-the-server).

### CUDA · Settings

| variable | what it does |
|---|---|
| `CMF_GPU=cuda` | selects the CUDA backend |
| `CMF_QWEN_PROFILE=fast-code.profile` | the experts' order on the card; always set it |
| `CMF_QWEN_ROUTE_TAIL=0.05` | skip the experts with under 5 % of a token's routing weight (the measured setting) |
| `CMF_QWEN_CUDA_REG=1 CMF_QWEN_CUDA_REG_ALL=1` | keep every expert pinned in RAM, so nothing waits on the disk; needs 64 GB of RAM |
| `CMF_SERVE_SLOTS=1` | one request at a time, at full speed |
| `CMF_CUDA_DEVICE=<n>` | which NVIDIA card, by its number (default 0) |
| `CMF_GPU_VRAM_MB=<MiB>` | cap the card memory the model takes (default: what is free) |
| `CMF_QWEN_CUDA_GROW_MB=<MiB>` | card memory kept free for a growing context (default 1024) |
| `CMF_QWEN_CPU_EXPERT_THREADS=<n>` | host threads for the experts outside the card (default: every core but three; every core but one under 16 cores) |

### CUDA · Speed

Stand: one RTX 5090 with its memory held by `CMF_GPU_VRAM_MB` to 10 732 / 16 384 / 24 576 MiB; cortiq 0.8.78, the
run command above, a fresh server for every run; host 2× EPYC 9554 (31 threads, 62 GB of RAM). Task: a program
written from a 2 151-token spec, answered to the end; greedy, thinking off; mean of two runs.

Strata Coder is ISTA-DASLab's 2-bit cut of the same model. It ran on its own engine, Strata, on the same card, with
an expert cache of 1 500 / 3 000 / 5 000 experts (10 732 / 14 542 / 19 622 MiB of the card).

| card memory | Fast Code on cortiq, 4-bit experts | Strata Coder on Strata, 2-bit experts | Fast Code against Strata Coder | HumanEval, first 40 tasks (Fast Code) |
|---|---:|---:|---:|---:|
| 12 GB | **145.0 tok/s** | 154.1 tok/s | 94 % | 38 / 40 |
| 16 GB | **174.6 tok/s** | 192.5 tok/s | 91 % | 39 / 40 |
| 24 GB | **206.7 tok/s** | 212.0 tok/s | 97 % | 39 / 40 |

![Code generation on NVIDIA cards with 12–24 GB, CUDA backend](https://huggingface.co/infosave/Qwen3.8-Flash-Next-Fast-Code-cmf/resolve/main/assets/fc_cuda_en.png)

### CUDA · Troubleshooting

- `cortiq gpu` prints `CUDA: unavailable`, or the server log has `libnvrtc (CUDA toolkit) was not found` or
  `qwen4 cuda: …; staying on the host path`: the model is running on the CPU. Install the CUDA 12.8+ toolkit and put
  its `lib64` on `LD_LIBRARY_PATH`.
- **Out of memory on the card**: close other GPU programs or set `CMF_GPU_VRAM_MB` a little lower.
- **Less than 64 GB of RAM**: drop `CMF_QWEN_CUDA_REG_ALL=1` (and if needed `CMF_QWEN_CUDA_REG=1`); it runs slower.
- **A host without AVX-512 VNNI**: every expert outside the card is copied to it instead; it runs, slower.
- **Several NVIDIA cards**: pick one with `CMF_CUDA_DEVICE`.
- More in the [CUDA guide](https://github.com/infosave2007/cmf/blob/master/docs/CUDA.md).

## Vulkan: one card with 48 GB or more

The whole model is on the card.

### Vulkan · Requirements

| component | requirement |
|---|---|
| GPU | one card with 48 GB or more and a Vulkan driver; tested on an RTX PRO 6000 96 GB |
| CPU | any x86-64 or ARM64 |
| RAM | 32 GB (the n-gram table, 28 GB, stays in RAM) |
| disk | 67 GB |
| OS | Linux |
| cortiq | 0.8.65 or later: a [prebuilt binary](https://github.com/infosave2007/cmf/releases) or `cargo install` |

### Vulkan · Install

A prebuilt binary from the [releases page](https://github.com/infosave2007/cmf/releases), or:

```bash
cargo install cortiq-cli --locked
```

In a container, add the Vulkan loader:

```bash
apt-get install -y --no-install-recommends vulkan-tools libglvnd0 libegl1 libgl1 libglx0
```

### Vulkan · Run

```bash
export XDG_RUNTIME_DIR=/tmp   # only on Linux without a display
cortiq gpu   # lists the cards Vulkan sees

CMF_QWEN_PROFILE=fast-code.profile \
  cortiq serve qwen38-flash-next-fast-code.cmf --host 127.0.0.1 --port 8080
```

The server is ready when it answers on `/v1/models`; the first start reads the 66.8 GB file and takes a minute or
more. Requests: [Using the server](#using-the-server).

### Vulkan · Settings

| variable | what it does |
|---|---|
| `CMF_QWEN_PROFILE=fast-code.profile` | the experts' order on the card; always set it |
| `CMF_GPU_ADAPTER=<n>` | which card, by its number in `cortiq gpu` |
| `CMF_SERVE_SLOTS=<n>` | requests served at once (default: cores / 4, from 1 to 4); 1 gives one request the full speed |
| `XDG_RUNTIME_DIR=/tmp` | Vulkan on Linux without a display |

### Vulkan · Speed

Stand: one RTX PRO 6000 96 GB; cortiq on the Vulkan backend, the run command above. One stream, greedy, thinking
off, draft tokens on; a new session tag before every prompt, so nothing comes from a cache; Fast Code is the mean of
two runs.

The full file is the same model with all 512 experts of a layer at 4 bits
([Qwen3.8-Flash-Next-cmf](https://huggingface.co/infosave/Qwen3.8-Flash-Next-cmf)), on the same engine and the same
card.

| | Fast Code | full file |
|---|---:|---:|
| prompt reading at 1K / 4K / 16K / 32K tokens, tok/s | **798 / 827 / 844 / 842** | 171 / 208 / 206 / 210 |
| code generation after a 4K / 16K / 32K prompt, tok/s | **161 / 166 / 136** | 118 / 138 / 116 |
| peak card memory (32K context) | **40.9 GB** | 59.6 GB |
| file on disk | **66.8 GB** | 98.6 GB |

![Fast Code on one card with 48 GB or more, Vulkan backend](https://huggingface.co/infosave/Qwen3.8-Flash-Next-Fast-Code-cmf/resolve/main/assets/fc_vulkan_en.png)

### Vulkan · Troubleshooting

- **`cortiq gpu` lists no card**: on Linux without a display set `XDG_RUNTIME_DIR=/tmp`; in a container install the
  packages above.
- **Several cards**: pick one with `CMF_GPU_ADAPTER`.
- **Out of memory on the card**: close other GPU programs; the model takes 41 GB at a 32K context.

## Using the server

The same on both backends.

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "fast-code", "enable_thinking": false, "temperature": 0,
       "messages": [{"role": "user", "content": "Write a Rust function that parses a semantic version, with tests."}]}'
```

Endpoints: `/v1/chat/completions` (streaming, `tools`, `enable_thinking`), `/v1/completions`, `/v1/models`,
`/healthz`, a web dashboard on `/`. Draft tokens are used under `temperature: 0`. Any OpenAI-compatible editor or
agent can point at `http://127.0.0.1:8080/v1`.

One prompt without a server (on CUDA, with the variables of the run command):

```bash
CMF_QWEN_PROFILE=fast-code.profile cortiq run qwen38-flash-next-fast-code.cmf --prompt-file task.txt --greedy --no-think
```

- **The answer starts with reasoning**: send `"enable_thinking": false` (`--no-think` for `cortiq run`).
- **Slow first request**: the weights are still coming from the disk; keep the server running between requests.
- **Context**: 32 768 tokens by default; `CMF_MAX_SEQ=<tokens>` changes the limit.
- **Network**: the server checks no key; keep `--host 127.0.0.1` unless the network is trusted.

## Accuracy

Accuracy is a property of the file, the same on both backends.

| | Fast Code | Strata Coder | full file |
|---|---:|---:|---:|
| expert weights | **4 bits** | 2 bits | 4 bits |
| code perplexity, code written after 20.10.2026 (8192 tokens) | **3.337** | 3.436¹ | 3.366 |
| code perplexity, Rust (1536 tokens) | 4.900 | 4.690¹ | 4.111 |
| HumanEval pass@1 (164 tasks, thinking off) | 93.3 % | 94.5 % | 94.5 % |
| file on disk | 66.8 GB | 58.4 GB + 5.2 GB MTP | 98.6 GB |

¹ Strata Coder's weights imported into CMF and run by cortiq: the same text and tokenizer for all three.

![Accuracy on code](https://huggingface.co/infosave/Qwen3.8-Flash-Next-Fast-Code-cmf/resolve/main/assets/fc_quality_en.png)

## Files

| file | contents | size |
|---|---|---:|
| `qwen38-flash-next-fast-code.cmf` | 256 of 512 routed experts a layer (4-bit q4tp) and the shared expert; attention, embeddings and the output head 8-bit; the n-gram table 4-bit; the draft (MTP) head | 66.76 GB |
| `fast-code.profile` | routing profile (not weights): the experts' order on the card | 49 KB |
| `qwen38-flash-next-fast-code.cmf.sha256` | checksum | |

The kept experts are the ones code routes to most. For general chat use the
[full file](https://huggingface.co/infosave/Qwen3.8-Flash-Next-cmf). Attention is exact up to 262 144 tokens.
Weights derive from Qwen's release and remain under its license. Review generated code before running it.

---

## Документация на русском

[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), урезанная под код: 256 из 512 экспертов
каждого слоя в 4 битах, с общим экспертом, головой черновиков (MTP) и таблицей n-грамм, в одном файле
[CMF](https://github.com/infosave2007/cmf) на 66.8 ГБ. Запускается движком `cortiq` на Rust, на двух бэкендах:

- **CUDA, карты NVIDIA от 12 ГБ.** Код 145 / 175 / 207 ток/с в 12 / 16 / 24 ГБ памяти карты: 94 / 91 / 97 %
  скорости 2-битной Strata Coder на той же карте, при экспертах в 4 битах и внимании в 8. Измерено на RTX 5090 с
  памятью, ограниченной до 12, 16 и 24 ГБ.
- **Vulkan, одна карта от 48 ГБ.** Вся модель на карте: чтение промпта ~840 ток/с, в 4 раза быстрее полного файла
  на 512 экспертов; код 136–166 ток/с; 41 ГБ памяти карты вместо 60. Измерено на RTX PRO 6000 96 ГБ.
- **Точность.** HumanEval pass@1 93.3 %; перплексия на свежем коде 3.337 против 3.436 у 2-битной Strata Coder.

### Какой бэкенд

| ваша карта | бэкенд | что нужно | на чём измерено |
|---|---|---|---|
| NVIDIA от 12 ГБ | [CUDA](#cuda-карты-nvidia-от-12-гб) | cortiq 0.8.78+ со сборкой `--features cuda`, библиотеки CUDA 12.8+, 64 ГБ ОЗУ, процессор с AVX-512 VNNI | RTX 5090, память ограничена до 12 / 16 / 24 ГБ |
| от 48 ГБ, с драйвером Vulkan | [Vulkan](#vulkan-одна-карта-от-48-гб) | cortiq 0.8.65+ (готовые сборки), 32 ГБ ОЗУ, любой процессор | RTX PRO 6000 96 ГБ |

Файл, профиль и API сервера одни и те же. Бэкенды измерены на разных картах и разных задачах: сравнивайте числа
внутри одного раздела, а не между разделами.

### Загрузка

Файлы одни для обоих бэкендов.

```bash
pip install -U huggingface_hub
hf download infosave/Qwen3.8-Flash-Next-Fast-Code-cmf qwen38-flash-next-fast-code.cmf fast-code.profile \
  qwen38-flash-next-fast-code.cmf.sha256 --local-dir .
sha256sum -c qwen38-flash-next-fast-code.cmf.sha256
```

### CUDA: карты NVIDIA от 12 ГБ

Карта держит экспертов, которые помещаются в её память; остальных считают ядра процессора.

#### CUDA · Требования

| компонент | требование |
|---|---|
| видеокарта | NVIDIA от 12 ГБ; драйвер 570 или новее |
| библиотеки CUDA | `libnvrtc` и `libcublas` из CUDA 12.8+, в `LD_LIBRARY_PATH` |
| процессор | x86-64 с AVX-512 VNNI (AMD Zen 4/5, Intel Sapphire Rapids и новее), от 8 потоков |
| ОЗУ | 64 ГБ |
| диск | 67 ГБ |
| ОС | Linux |
| cortiq | 0.8.78 или новее, сборка с `--features cuda`; в готовых сборках CUDA нет |

#### CUDA · Установка

```bash
# Rust 1.88 или новее; для самой сборки CUDA не нужна
cargo install cortiq-cli --locked --features cuda
```

#### CUDA · Запуск

```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH   # где лежат libnvrtc и libcublas
cortiq gpu   # "CUDA: <ваша карта> …" — готово; "CUDA: unavailable — <причина>" — чего не хватает

CMF_GPU=cuda CMF_QWEN_PROFILE=fast-code.profile CMF_QWEN_ROUTE_TAIL=0.05 \
  CMF_SERVE_SLOTS=1 CMF_QWEN_CUDA_REG=1 CMF_QWEN_CUDA_REG_ALL=1 \
  cortiq serve qwen38-flash-next-fast-code.cmf --host 127.0.0.1 --port 8080
```

Сервер готов, когда отвечает на `/v1/models`; первый запуск читает файл на 66.8 ГБ, это минута и дольше. Строка
`qwen4 cuda: arena prefilled with …` в логе означает, что карта работает. Запросы — в разделе
[Работа с сервером](#работа-с-сервером).

#### CUDA · Настройки

| переменная | что делает |
|---|---|
| `CMF_GPU=cuda` | включает бэкенд CUDA |
| `CMF_QWEN_PROFILE=fast-code.profile` | порядок экспертов на карте; задавайте всегда |
| `CMF_QWEN_ROUTE_TAIL=0.05` | пропускать экспертов с долей маршрута токена меньше 5 % (так измерено) |
| `CMF_QWEN_CUDA_REG=1 CMF_QWEN_CUDA_REG_ALL=1` | все эксперты закреплены в ОЗУ, ничто не ждёт диск; нужно 64 ГБ ОЗУ |
| `CMF_SERVE_SLOTS=1` | один запрос за раз, на полной скорости |
| `CMF_CUDA_DEVICE=<n>` | какая карта NVIDIA, по номеру (по умолчанию 0) |
| `CMF_GPU_VRAM_MB=<МиБ>` | сколько памяти карты брать (по умолчанию — свободную) |
| `CMF_QWEN_CUDA_GROW_MB=<МиБ>` | запас памяти карты под растущий контекст (по умолчанию 1024) |
| `CMF_QWEN_CPU_EXPERT_THREADS=<n>` | потоки процессора для экспертов вне карты (по умолчанию все ядра, кроме трёх; при менее чем 16 ядрах — кроме одного) |

#### CUDA · Скорость

Стенд: одна RTX 5090, память ограничена через `CMF_GPU_VRAM_MB` до 10 732 / 16 384 / 24 576 МиБ; cortiq 0.8.78,
команда запуска выше, новый сервер на каждый прогон; хост 2× EPYC 9554 (31 поток, 62 ГБ ОЗУ). Задача: программа по
ТЗ на 2 151 токен, ответ до конца; жадный выбор, без рассуждений; среднее двух прогонов.

Strata Coder — 2-битная сборка той же модели от ISTA-DASLab. Она работала на собственном движке Strata, на той же
карте, с кешем на 1 500 / 3 000 / 5 000 экспертов (10 732 / 14 542 / 19 622 МиБ карты).

| память карты | Fast Code на cortiq, эксперты 4 бита | Strata Coder на Strata, эксперты 2 бита | Fast Code к Strata Coder | HumanEval, первые 40 задач (Fast Code) |
|---|---:|---:|---:|---:|
| 12 ГБ | **145.0 ток/с** | 154.1 ток/с | 94 % | 38 / 40 |
| 16 ГБ | **174.6 ток/с** | 192.5 ток/с | 91 % | 39 / 40 |
| 24 ГБ | **206.7 ток/с** | 212.0 ток/с | 97 % | 39 / 40 |

![Генерация кода на картах NVIDIA 12–24 ГБ, бэкенд CUDA](https://huggingface.co/infosave/Qwen3.8-Flash-Next-Fast-Code-cmf/resolve/main/assets/fc_cuda_ru.png)

#### CUDA · Если не работает

- `cortiq gpu` пишет `CUDA: unavailable`, либо в логе сервера `libnvrtc (CUDA toolkit) was not found` или
  `qwen4 cuda: …; staying on the host path`: модель считается на процессоре. Поставьте CUDA 12.8+ и добавьте её
  `lib64` в `LD_LIBRARY_PATH`.
- **Не хватает памяти карты**: закройте другие программы на GPU или задайте `CMF_GPU_VRAM_MB` чуть меньше.
- **ОЗУ меньше 64 ГБ**: уберите `CMF_QWEN_CUDA_REG_ALL=1` (при нехватке и `CMF_QWEN_CUDA_REG=1`); работает медленнее.
- **Процессор без AVX-512 VNNI**: каждый эксперт вне карты копируется на неё; работает, медленнее.
- **Несколько карт NVIDIA**: выберите одну через `CMF_CUDA_DEVICE`.
- Подробнее — в [руководстве по CUDA](https://github.com/infosave2007/cmf/blob/master/docs/CUDA.md).

### Vulkan: одна карта от 48 ГБ

Вся модель лежит на карте.

#### Vulkan · Требования

| компонент | требование |
|---|---|
| видеокарта | одна карта от 48 ГБ с драйвером Vulkan; проверено на RTX PRO 6000 96 ГБ |
| процессор | любой x86-64 или ARM64 |
| ОЗУ | 32 ГБ (таблица n-грамм, 28 ГБ, живёт в ОЗУ) |
| диск | 67 ГБ |
| ОС | Linux |
| cortiq | 0.8.65 или новее: [готовая сборка](https://github.com/infosave2007/cmf/releases) или `cargo install` |

#### Vulkan · Установка

Готовая сборка со [страницы релизов](https://github.com/infosave2007/cmf/releases), либо:

```bash
cargo install cortiq-cli --locked
```

В контейнере добавьте загрузчик Vulkan:

```bash
apt-get install -y --no-install-recommends vulkan-tools libglvnd0 libegl1 libgl1 libglx0
```

#### Vulkan · Запуск

```bash
export XDG_RUNTIME_DIR=/tmp   # только на Linux без экрана
cortiq gpu   # список карт, которые видит Vulkan

CMF_QWEN_PROFILE=fast-code.profile \
  cortiq serve qwen38-flash-next-fast-code.cmf --host 127.0.0.1 --port 8080
```

Сервер готов, когда отвечает на `/v1/models`; первый запуск читает файл на 66.8 ГБ, это минута и дольше. Запросы —
в разделе [Работа с сервером](#работа-с-сервером).

#### Vulkan · Настройки

| переменная | что делает |
|---|---|
| `CMF_QWEN_PROFILE=fast-code.profile` | порядок экспертов на карте; задавайте всегда |
| `CMF_GPU_ADAPTER=<n>` | какая карта, по номеру из `cortiq gpu` |
| `CMF_SERVE_SLOTS=<n>` | сколько запросов обслуживать сразу (по умолчанию ядра / 4, от 1 до 4); 1 отдаёт одному запросу всю скорость |
| `XDG_RUNTIME_DIR=/tmp` | Vulkan на Linux без экрана |

#### Vulkan · Скорость

Стенд: одна RTX PRO 6000 96 ГБ; cortiq на бэкенде Vulkan, команда запуска выше. Один поток, жадный выбор, без
рассуждений, черновики включены; перед каждым промптом новая метка сессии, поэтому кеш не участвует; Fast Code —
среднее двух прогонов.

Полный файл — та же модель со всеми 512 экспертами слоя в 4 битах
([Qwen3.8-Flash-Next-cmf](https://huggingface.co/infosave/Qwen3.8-Flash-Next-cmf)), тот же движок и та же карта.

| | Fast Code | полный файл |
|---|---:|---:|
| чтение промпта на 1K / 4K / 16K / 32K токенов, ток/с | **798 / 827 / 844 / 842** | 171 / 208 / 206 / 210 |
| генерация кода после промпта 4K / 16K / 32K, ток/с | **161 / 166 / 136** | 118 / 138 / 116 |
| пик памяти карты (контекст 32K) | **40.9 ГБ** | 59.6 ГБ |
| файл на диске | **66.8 ГБ** | 98.6 ГБ |

![Fast Code на одной карте от 48 ГБ, бэкенд Vulkan](https://huggingface.co/infosave/Qwen3.8-Flash-Next-Fast-Code-cmf/resolve/main/assets/fc_vulkan_ru.png)

#### Vulkan · Если не работает

- **`cortiq gpu` не показывает карту**: на Linux без экрана задайте `XDG_RUNTIME_DIR=/tmp`; в контейнере поставьте
  пакеты из раздела установки.
- **Несколько карт**: выберите одну через `CMF_GPU_ADAPTER`.
- **Не хватает памяти карты**: закройте другие программы на GPU; модель занимает 41 ГБ при контексте 32K.

### Работа с сервером

Одинаково на обоих бэкендах.

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "fast-code", "enable_thinking": false, "temperature": 0,
       "messages": [{"role": "user", "content": "Напиши функцию на Rust, которая разбирает semver, с тестами."}]}'
```

Точки входа: `/v1/chat/completions` (потоковый вывод, `tools`, `enable_thinking`), `/v1/completions`, `/v1/models`,
`/healthz`, веб-панель на `/`. Черновики работают при `temperature: 0`. Любой редактор или агент с OpenAI API
подключается к `http://127.0.0.1:8080/v1`.

Один запрос без сервера (на CUDA — с переменными из команды запуска):

```bash
CMF_QWEN_PROFILE=fast-code.profile cortiq run qwen38-flash-next-fast-code.cmf --prompt-file task.txt --greedy --no-think
```

- **Ответ начинается с рассуждений**: передайте `"enable_thinking": false` (`--no-think` для `cortiq run`).
- **Медленный первый запрос**: веса ещё читаются с диска; держите сервер запущенным.
- **Контекст**: по умолчанию 32 768 токенов; предел меняет `CMF_MAX_SEQ=<токены>`.
- **Сеть**: сервер не проверяет ключ; оставляйте `--host 127.0.0.1`, если сеть не доверенная.

### Точность

Точность — свойство файла, она одна на обоих бэкендах.

| | Fast Code | Strata Coder | полный файл |
|---|---:|---:|---:|
| веса экспертов | **4 бита** | 2 бита | 4 бита |
| перплексия на коде, написанном после 20.10.2026 (8192 токена) | **3.337** | 3.436¹ | 3.366 |
| перплексия на коде Rust (1536 токенов) | 4.900 | 4.690¹ | 4.111 |
| HumanEval pass@1 (164 задачи, без рассуждений) | 93.3 % | 94.5 % | 94.5 % |
| файл на диске | 66.8 ГБ | 58.4 ГБ + 5.2 ГБ MTP | 98.6 ГБ |

¹ Веса Strata Coder, импортированные в CMF и запущенные cortiq: один текст и один токенизатор для всех трёх.

![Точность на коде](https://huggingface.co/infosave/Qwen3.8-Flash-Next-Fast-Code-cmf/resolve/main/assets/fc_quality_ru.png)

### Файлы

| файл | содержимое | размер |
|---|---|---:|
| `qwen38-flash-next-fast-code.cmf` | 256 из 512 экспертов слоя (4 бита, q4tp) и общий эксперт; внимание, эмбеддинги и выходная голова в 8 битах; таблица n-грамм в 4 битах; голова черновиков (MTP) | 66.76 ГБ |
| `fast-code.profile` | профиль маршрутизации (не веса): порядок экспертов на карте | 49 КБ |
| `qwen38-flash-next-fast-code.cmf.sha256` | контрольная сумма | |

Оставлены эксперты, к которым код обращается чаще всего. Для общего чата —
[полный файл](https://huggingface.co/infosave/Qwen3.8-Flash-Next-cmf). Внимание точное до 262 144 токенов.
Веса получены из релиза Qwen и остаются под его лицензией. Проверяйте сгенерированный код перед запуском.

---

## 中文文档

[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) 的代码专用裁剪版：每层 512 个路由专家保留 256 个（4 位），
连同共享专家、草稿（MTP）头和 n-gram 表，打包为一个 66.8 GB 的 [CMF](https://github.com/infosave2007/cmf) 文件，由 Rust 推理引擎
`cortiq` 运行，支持两种后端：

- **CUDA，12 GB 起的 NVIDIA 显卡。** 在 12 / 16 / 24 GB 显存内生成代码 145 / 175 / 207 token/s，为同一张显卡上 2 位
  Strata Coder 速度的 94 / 91 / 97 %，专家保持 4 位、注意力保持 8 位。测试平台：显存限制为 12、16、24 GB 的 RTX 5090。
- **Vulkan，一张 48 GB 以上的显卡。** 整个模型都在显卡上：读取提示词约 840 token/s，是 512 专家完整文件的 4 倍；生成代码
  136–166 token/s；显存 41 GB，完整文件为 60 GB。测试平台：RTX PRO 6000 96 GB。
- **准确率。** HumanEval pass@1 93.3 %；新代码困惑度 3.337，2 位 Strata Coder 为 3.436。

### 选择后端

| 你的显卡 | 后端 | 需要 | 测试平台 |
|---|---|---|---|
| NVIDIA，12 GB 以上 | [CUDA](#cuda-后端：12-gb-起的-nvidia-显卡) | cortiq 0.8.78+，以 `--features cuda` 构建；CUDA 12.8+ 库；64 GB 内存；支持 AVX-512 VNNI 的 CPU | RTX 5090，显存限制为 12 / 16 / 24 GB |
| 48 GB 以上，带 Vulkan 驱动 | [Vulkan](#vulkan-后端：一张-48-gb-以上的显卡) | cortiq 0.8.65+（预编译版本）；32 GB 内存；任意 CPU | RTX PRO 6000 96 GB |

文件、画像和服务器 API 对两种后端相同。两种后端在不同的显卡和不同的任务上测得：数字只在同一节内比较，不要跨节比较。

### 下载

两种后端使用相同的文件。

```bash
pip install -U huggingface_hub
hf download infosave/Qwen3.8-Flash-Next-Fast-Code-cmf qwen38-flash-next-fast-code.cmf fast-code.profile \
  qwen38-flash-next-fast-code.cmf.sha256 --local-dir .
sha256sum -c qwen38-flash-next-fast-code.cmf.sha256
```

### CUDA 后端：12 GB 起的 NVIDIA 显卡

显卡保存放得下的专家，其余专家由主机 CPU 计算。

#### CUDA · 环境要求

| 组件 | 要求 |
|---|---|
| 显卡 | NVIDIA，12 GB 以上；驱动 570 或更高 |
| CUDA 库 | CUDA 12.8+ 的 `libnvrtc` 与 `libcublas`，位于 `LD_LIBRARY_PATH` |
| CPU | 支持 AVX-512 VNNI 的 x86-64（AMD Zen 4/5、Intel Sapphire Rapids 或更新），8 线程以上 |
| 内存 | 64 GB |
| 磁盘 | 67 GB |
| 系统 | Linux |
| cortiq | 0.8.78 或更高，以 `--features cuda` 构建；预编译版本不含 CUDA |

#### CUDA · 安装

```bash
# Rust 1.88 或更高；构建本身不需要 CUDA
cargo install cortiq-cli --locked --features cuda
```

#### CUDA · 运行

```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH   # libnvrtc 与 libcublas 所在目录
cortiq gpu   # "CUDA: <你的显卡> …" 表示就绪；"CUDA: unavailable — <原因>" 说明缺少什么

CMF_GPU=cuda CMF_QWEN_PROFILE=fast-code.profile CMF_QWEN_ROUTE_TAIL=0.05 \
  CMF_SERVE_SLOTS=1 CMF_QWEN_CUDA_REG=1 CMF_QWEN_CUDA_REG_ALL=1 \
  cortiq serve qwen38-flash-next-fast-code.cmf --host 127.0.0.1 --port 8080
```

服务器在 `/v1/models` 有响应即就绪；首次启动需读取 66.8 GB 文件，约一分钟或更久。日志中出现
`qwen4 cuda: arena prefilled with …` 表示显卡已在工作。请求方式见[使用服务器](#使用服务器)。

#### CUDA · 设置

| 变量 | 作用 |
|---|---|
| `CMF_GPU=cuda` | 选择 CUDA 后端 |
| `CMF_QWEN_PROFILE=fast-code.profile` | 专家在显卡上的顺序；务必设置 |
| `CMF_QWEN_ROUTE_TAIL=0.05` | 跳过路由权重低于 5 % 的专家（测试所用设置） |
| `CMF_QWEN_CUDA_REG=1 CMF_QWEN_CUDA_REG_ALL=1` | 全部专家锁定在内存中，不等待磁盘；需要 64 GB 内存 |
| `CMF_SERVE_SLOTS=1` | 一次一个请求，全速 |
| `CMF_CUDA_DEVICE=<n>` | 使用第几张 NVIDIA 显卡（默认 0） |
| `CMF_GPU_VRAM_MB=<MiB>` | 模型可用的显存上限（默认：全部空闲显存） |
| `CMF_QWEN_CUDA_GROW_MB=<MiB>` | 为增长的上下文预留的显存（默认 1024） |
| `CMF_QWEN_CPU_EXPERT_THREADS=<n>` | 计算显卡外专家的主机线程数（默认：全部核心减三；不足 16 核时减一） |

#### CUDA · 速度

测试平台：一张 RTX 5090，用 `CMF_GPU_VRAM_MB` 把显存限制为 10 732 / 16 384 / 24 576 MiB；cortiq 0.8.78，上面的运行命令，
每次运行都重新启动服务器；主机 2× EPYC 9554（31 线程，62 GB 内存）。任务：按 2 151 token 的需求写出完整程序，直到回答结束；
贪心解码，关闭思考；两次运行的平均值。

Strata Coder 是 ISTA-DASLab 对同一模型的 2 位裁剪版，运行于其自有引擎 Strata，在同一张显卡上，专家缓存为
1 500 / 3 000 / 5 000 个（占显卡 10 732 / 14 542 / 19 622 MiB）。

| 显存 | Fast Code（cortiq），专家 4 位 | Strata Coder（Strata），专家 2 位 | Fast Code 相对 Strata Coder | HumanEval 前 40 题（Fast Code） |
|---|---:|---:|---:|---:|
| 12 GB | **145.0 token/s** | 154.1 token/s | 94 % | 38 / 40 |
| 16 GB | **174.6 token/s** | 192.5 token/s | 91 % | 39 / 40 |
| 24 GB | **206.7 token/s** | 212.0 token/s | 97 % | 39 / 40 |

![在 12–24 GB 的 NVIDIA 显卡上生成代码，CUDA 后端](https://huggingface.co/infosave/Qwen3.8-Flash-Next-Fast-Code-cmf/resolve/main/assets/fc_cuda_zh.png)

#### CUDA · 常见问题

- `cortiq gpu` 显示 `CUDA: unavailable`，或服务器日志出现 `libnvrtc (CUDA toolkit) was not found`、
  `qwen4 cuda: …; staying on the host path`：模型正在 CPU 上运行。安装 CUDA 12.8+，并把其 `lib64` 加入 `LD_LIBRARY_PATH`。
- **显存不足**：关闭其他使用 GPU 的程序，或把 `CMF_GPU_VRAM_MB` 设小一些。
- **内存不足 64 GB**：去掉 `CMF_QWEN_CUDA_REG_ALL=1`（必要时也去掉 `CMF_QWEN_CUDA_REG=1`），速度会变慢。
- **CPU 不支持 AVX-512 VNNI**：显卡外的专家改为复制到显卡；可以运行，但更慢。
- **多张 NVIDIA 显卡**：用 `CMF_CUDA_DEVICE` 选择其中一张。
- 更多内容见 [CUDA 指南](https://github.com/infosave2007/cmf/blob/master/docs/CUDA.md)。

### Vulkan 后端：一张 48 GB 以上的显卡

整个模型都在显卡上。

#### Vulkan · 环境要求

| 组件 | 要求 |
|---|---|
| 显卡 | 一张 48 GB 以上、带 Vulkan 驱动的显卡；已在 RTX PRO 6000 96 GB 上验证 |
| CPU | 任意 x86-64 或 ARM64 |
| 内存 | 32 GB（n-gram 表 28 GB 常驻内存） |
| 磁盘 | 67 GB |
| 系统 | Linux |
| cortiq | 0.8.65 或更高：[预编译版本](https://github.com/infosave2007/cmf/releases)或 `cargo install` |

#### Vulkan · 安装

从[发布页](https://github.com/infosave2007/cmf/releases)下载预编译版本，或：

```bash
cargo install cortiq-cli --locked
```

在容器中还需安装 Vulkan 加载器：

```bash
apt-get install -y --no-install-recommends vulkan-tools libglvnd0 libegl1 libgl1 libglx0
```

#### Vulkan · 运行

```bash
export XDG_RUNTIME_DIR=/tmp   # 仅用于无显示器的 Linux
cortiq gpu   # 列出 Vulkan 能看到的显卡

CMF_QWEN_PROFILE=fast-code.profile \
  cortiq serve qwen38-flash-next-fast-code.cmf --host 127.0.0.1 --port 8080
```

服务器在 `/v1/models` 有响应即就绪；首次启动需读取 66.8 GB 文件，约一分钟或更久。请求方式见[使用服务器](#使用服务器)。

#### Vulkan · 设置

| 变量 | 作用 |
|---|---|
| `CMF_QWEN_PROFILE=fast-code.profile` | 专家在显卡上的顺序；务必设置 |
| `CMF_GPU_ADAPTER=<n>` | 使用哪张显卡，取 `cortiq gpu` 中的编号 |
| `CMF_SERVE_SLOTS=<n>` | 同时处理的请求数（默认：核心数 / 4，范围 1 到 4）；设为 1 时单个请求全速 |
| `XDG_RUNTIME_DIR=/tmp` | 无显示器的 Linux 上使用 Vulkan |

#### Vulkan · 速度

测试平台：一张 RTX PRO 6000 96 GB；cortiq 的 Vulkan 后端，上面的运行命令。单路，贪心解码，关闭思考，开启草稿；
每个提示词前加新的会话标记，因此不使用缓存；Fast Code 为两次运行的平均值。

完整文件是同一模型，保留每层全部 512 个专家，4 位
（[Qwen3.8-Flash-Next-cmf](https://huggingface.co/infosave/Qwen3.8-Flash-Next-cmf)），同一引擎，同一张显卡。

| | Fast Code | 完整文件 |
|---|---:|---:|
| 读取 1K / 4K / 16K / 32K token 的提示词，token/s | **798 / 827 / 844 / 842** | 171 / 208 / 206 / 210 |
| 4K / 16K / 32K 提示词后生成代码，token/s | **161 / 166 / 136** | 118 / 138 / 116 |
| 显存峰值（32K 上下文） | **40.9 GB** | 59.6 GB |
| 磁盘文件 | **66.8 GB** | 98.6 GB |

![Fast Code：一张 48 GB 以上的显卡，Vulkan 后端](https://huggingface.co/infosave/Qwen3.8-Flash-Next-Fast-Code-cmf/resolve/main/assets/fc_vulkan_zh.png)

#### Vulkan · 常见问题

- `cortiq gpu` 没有列出显卡：在无显示器的 Linux 上设置 `XDG_RUNTIME_DIR=/tmp`；在容器中安装上面的软件包。
- **多张显卡**：用 `CMF_GPU_ADAPTER` 选择其中一张。
- **显存不足**：关闭其他使用 GPU 的程序；32K 上下文时模型占用 41 GB。

### 使用服务器

两种后端用法相同。

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "fast-code", "enable_thinking": false, "temperature": 0,
       "messages": [{"role": "user", "content": "用 Rust 写一个解析语义化版本号的函数，并附测试。"}]}'
```

接口：`/v1/chat/completions`（流式、`tools`、`enable_thinking`）、`/v1/completions`、`/v1/models`、`/healthz`，`/` 为网页面板。
`temperature: 0` 时启用草稿。任何兼容 OpenAI 的编辑器或智能体都可连接 `http://127.0.0.1:8080/v1`。

不启动服务器、单次提示（CUDA 需带上运行命令中的变量）：

```bash
CMF_QWEN_PROFILE=fast-code.profile cortiq run qwen38-flash-next-fast-code.cmf --prompt-file task.txt --greedy --no-think
```

- **回答以推理开头**：发送 `"enable_thinking": false`（`cortiq run` 用 `--no-think`）。
- **首个请求较慢**：权重仍在从磁盘读取；请保持服务器运行。
- **上下文**：默认 32 768 token；用 `CMF_MAX_SEQ=<token 数>` 修改上限。
- **网络**：服务器不校验密钥；除非网络可信，请保留 `--host 127.0.0.1`。

### 准确率

准确率取决于文件，两种后端相同。

| | Fast Code | Strata Coder | 完整文件 |
|---|---:|---:|---:|
| 专家权重 | **4 位** | 2 位 | 4 位 |
| 代码困惑度，2026-10-20 之后写的代码（8192 token） | **3.337** | 3.436¹ | 3.366 |
| 代码困惑度，Rust（1536 token） | 4.900 | 4.690¹ | 4.111 |
| HumanEval pass@1（164 题，关闭思考） | 93.3 % | 94.5 % | 94.5 % |
| 磁盘文件 | 66.8 GB | 58.4 GB + 5.2 GB MTP | 98.6 GB |

¹ Strata Coder 的权重导入 CMF 后由 cortiq 运行：三者使用相同的文本与分词器。

![代码准确率](https://huggingface.co/infosave/Qwen3.8-Flash-Next-Fast-Code-cmf/resolve/main/assets/fc_quality_zh.png)

### 文件

| 文件 | 内容 | 大小 |
|---|---|---:|
| `qwen38-flash-next-fast-code.cmf` | 每层 512 个路由专家中的 256 个（4 位 q4tp）与共享专家；注意力、嵌入与输出头 8 位；n-gram 表 4 位；草稿（MTP）头 | 66.76 GB |
| `fast-code.profile` | 路由画像（非权重）：专家在显卡上的顺序 | 49 KB |
| `qwen38-flash-next-fast-code.cmf.sha256` | 校验和 | |

保留的是代码最常路由到的专家。通用对话请使用[完整文件](https://huggingface.co/infosave/Qwen3.8-Flash-Next-cmf)。注意力在 262 144 token 内为精确计算。
权重源自 Qwen 的发布，仍受其许可约束。运行生成的代码前请先审阅。
