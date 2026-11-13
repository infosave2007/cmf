---
license: other
license_name: edge0-community-model-1.0
license_link: LICENSE
library_name: cortiq
base_model: Edge0/Edge0-120B-NVFP4
base_model_relation: quantized
pipeline_tag: text-generation
tags:
  - cmf
  - cortiq
  - moe
  - latent-moe
  - nvfp4
  - code
  - speculative-decoding
  - vulkan
  - cuda
language:
  - en
  - ru
  - zh
---

# Edge0-120B — CMF

[Edge0-120B](https://huggingface.co/Edge0/Edge0-120B-NVFP4) by Edge0, a latent mixture-of-experts model built on
[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), in one 95.6 GB
[CMF](https://github.com/infosave2007/cmf) file for `cortiq`, a Rust inference engine. The 512 experts of every layer
work at 1280 dimensions and keep Edge0's 4-bit training: their NVFP4 weights are copied bit for bit.

- **CUDA:** code at 149–178 tok/s, prompt reading at 4 797–5 924 tok/s (4K–32K tokens).
- **Vulkan:** code at 108–122 tok/s, prompt reading at 721–725 tok/s.
- **24 GB card (CUDA):** code at 100–118 tok/s, prompt reading at 2 394–2 823 tok/s; 12–32 GB cards [below](#cuda--smaller-cards).
- **Accuracy:** HumanEval pass@1 93.3 % (164 tasks, thinking off).

Measured on one RTX PRO 6000 96 GB, with draft tokens (MTP) on.

[English](#which-backend) · [Русский](#документация-на-русском) · [中文](#中文文档)

## Which backend

| your card | backend | what it needs |
|---|---|---|
| NVIDIA, 64 GB or more | [CUDA](#cuda-one-nvidia-card), the whole model on the card | cortiq 0.8.94+ built with `--features cuda`, CUDA 12.8+ libraries |
| NVIDIA, 12 GB to 64 GB | [CUDA](#cuda--smaller-cards), part of the experts in RAM | the same, and 96 GB of RAM |
| 64 GB or more, with a Vulkan driver | [Vulkan](#vulkan-one-card) | cortiq 0.8.94+ (prebuilt binaries) |
| 24 GB to 64 GB, with a Vulkan driver | [Vulkan](#vulkan--smaller-cards), part of the experts in RAM | the same, and 96 GB of RAM |

The file and the server API are the same for both backends; an NVIDIA card can run either. The larger model with the same setup: [Edge0-145B](https://huggingface.co/infosave/Edge0-145B-cmf).

## Quick start

Linux, one card. Five steps from nothing to an answer.

**1. Download** the model, its checksum and its routing profile (96 GB) into one folder:

```bash
pip install -U huggingface_hub
hf download infosave/Edge0-120B-cmf edge0-120b.cmf edge0-120b.cmf.sha256 edge0-120b.profile --local-dir .
sha256sum -c edge0-120b.cmf.sha256
```

**2. Install cortiq:**

| your card | install |
|---|---|
| NVIDIA | `cargo install cortiq-cli --locked --features cuda`, and the CUDA 12.8+ libraries `libnvrtc` and `libcublas` (the CUDA toolkit has them) |
| any card with a Vulkan driver | a [prebuilt binary](https://github.com/infosave2007/cmf/releases), or `cargo install cortiq-cli --locked` |

**3. Start the server** in that folder, with the line for your card:

```bash
# NVIDIA card (CUDA)
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
CMF_GPU=cuda CMF_SERVE_SLOTS=1 cortiq serve edge0-120b.cmf --host 127.0.0.1 --port 8080

# any card (Vulkan)
CMF_SERVE_SLOTS=1 cortiq serve edge0-120b.cmf --host 127.0.0.1 --port 8080
```

The first start reads the file and loads the card: a minute or more. The server is ready when this answers:

```bash
curl http://127.0.0.1:8080/v1/models
```

**4. Ask a question:**

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "edge0", "enable_thinking": false, "temperature": 0,
       "messages": [{"role": "user", "content": "Write a Python function that merges two sorted lists, with tests."}]}'
```

**5. Connect your editor or agent.** Anything that speaks the OpenAI API works: base URL
`http://127.0.0.1:8080/v1`, any API key, any model name (`/v1/models` lists `qwen4_exp-cortiq`). A chat page is at
`http://127.0.0.1:8080/`.

Nothing else to configure: at start cortiq looks at the card's free memory and decides how much of the model goes
on it. What each backend needs, its settings and speeds are below; server options are in
[Using the server](#using-the-server).

## CUDA: one NVIDIA card

### CUDA · Requirements

| component | requirement |
|---|---|
| GPU | NVIDIA; 64 GB or more holds the whole model (46.3 GB in use at a 32K context), from 12 GB with [part of the experts in RAM](#cuda--smaller-cards); driver 570 or later |
| CUDA libraries | `libnvrtc` and `libcublas` of the CUDA 12.8+ toolkit, on `LD_LIBRARY_PATH` |
| CPU | x86-64, 8 threads or more |
| RAM | 96 GB (the n-gram table, 52 GB, is pinned in RAM) |
| disk | 96 GB |
| OS | Linux |
| cortiq | 0.8.94 or later, built with `--features cuda`; the prebuilt binaries have no CUDA |

### CUDA · Install

```bash
# Rust 1.88 or later; the build itself needs no CUDA toolkit
cargo install cortiq-cli --locked --features cuda
```

### CUDA · Run

```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH   # where libnvrtc and libcublas are
cortiq gpu   # "CUDA: <your card> …" means ready

CMF_GPU=cuda CMF_SERVE_SLOTS=1 cortiq serve edge0-120b.cmf --host 127.0.0.1 --port 8080
```

The server is ready when it answers on `/v1/models`. The first start reads the file and loads the card: a minute or
more. Requests: [Using the server](#using-the-server).

### CUDA · Settings

| variable | what it does |
|---|---|
| `CMF_GPU=cuda` | selects the CUDA backend |
| `CMF_SERVE_SLOTS=1` | one request at a time, at full speed |
| `CMF_CUDA_DEVICE=<n>` | which NVIDIA card, by its number (default 0) |
| `CMF_QWEN_MTP=0` | turns the draft tokens off |
| `CMF_MAX_SEQ=<tokens>` | the context limit (default 32 768) |

### CUDA · Speed

Stand: one RTX PRO 6000 96 GB; cortiq on the CUDA backend, the run command above. One stream, greedy, thinking off,
draft tokens on; a new session tag before every prompt, so nothing comes from a cache; mean of two runs.

| prompt | prompt reading, tok/s | code generation, tok/s |
|---|---:|---:|
| 1K | **4 208** | **188** |
| 4K | **5 924** | **178** |
| 16K | **5 264** | **174** |
| 32K | **4 797** | **149** |

![Edge0-120B on CUDA](https://huggingface.co/infosave/Edge0-120B-cmf/resolve/main/assets/e0_120_cuda_en.png)

### CUDA · Smaller cards

A card that cannot hold every expert keeps the ones used most and leaves the rest in RAM: a decode step computes
those on the CPU, a prompt carries them up in batches. Which experts stay on the card comes from the routing profile
`edge0-120b.profile`, which has to sit next to the model file. Nothing to set: cortiq takes what the card has free
(`CMF_GPU_VRAM_MB=<MiB>` lowers it).

Stand: the same RTX PRO 6000 with all but the named amount of its memory held by another process; 12 CPU threads
compute the experts off the card. One run per size, the protocol above. Draft tokens are off on 12 GB (their head does not fit).

| card memory | experts on the card | prompt reading, 4K | prompt reading, 16K | code generation |
|---|---:|---:|---:|---:|
| 32 GB | 53 % | **3 352** tok/s | **3 135** tok/s | **130–145** tok/s |
| 24 GB | 30 % | **2 823** tok/s | **2 394** tok/s | **100–118** tok/s |
| 16 GB | 8 % | **1 571** tok/s | **1 571** tok/s | **71–73** tok/s |
| 12 GB | 2 % | **904** tok/s | **928** tok/s | **42** tok/s |



![Edge0-120B on smaller cards, CUDA](https://huggingface.co/infosave/Edge0-120B-cmf/resolve/main/assets/e0_120_cuda_cards_en.png)

### CUDA · Troubleshooting

- `cortiq gpu` prints `CUDA: unavailable`, or the log says `libnvrtc (CUDA toolkit) was not found`: the model runs
  on the CPU. Install the CUDA 12.8+ toolkit and put its `lib64` on `LD_LIBRARY_PATH`.
- **Out of memory on the card**: close other GPU programs, or lower what cortiq takes with `CMF_GPU_VRAM_MB`.
- **Slow on a smaller card**: `edge0-120b.profile` is not next to the model (the log then has no "hottest first" line).
- **Several NVIDIA cards**: pick one with `CMF_CUDA_DEVICE`.

## Vulkan: one card

### Vulkan · Requirements

| component | requirement |
|---|---|
| GPU | one card with a Vulkan driver; 64 GB or more holds the whole model (44.7 GB in use at a 32K context), from 24 GB with [part of the experts in RAM](#vulkan--smaller-cards); tested on an RTX PRO 6000 96 GB |
| CPU | any x86-64 or ARM64 |
| RAM | 64 GB (the n-gram table, 52 GB, stays in RAM); 96 GB on a smaller card |
| disk | 96 GB |
| OS | Linux |
| cortiq | 0.8.94 or later: a [prebuilt binary](https://github.com/infosave2007/cmf/releases) or `cargo install` |

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

CMF_SERVE_SLOTS=1 cortiq serve edge0-120b.cmf --host 127.0.0.1 --port 8080
```

The server is ready when it answers on `/v1/models`; the first start loads the card, a minute or more.

### Vulkan · Settings

| variable | what it does |
|---|---|
| `CMF_GPU_ADAPTER=<n>` | which card, by its number in `cortiq gpu` |
| `CMF_SERVE_SLOTS=1` | one request at a time, at full speed |
| `CMF_QWEN_MTP=0` | turns the draft tokens off |
| `XDG_RUNTIME_DIR=/tmp` | Vulkan on Linux without a display |

### Vulkan · Speed

Stand: one RTX PRO 6000 96 GB; cortiq on the Vulkan backend, the run command above; the same protocol as on CUDA.

| prompt | prompt reading, tok/s | code generation, tok/s |
|---|---:|---:|
| 1K | **757** | **115** |
| 4K | **725** | **108** |
| 16K | **724** | **112** |
| 32K | **721** | **122** |

![Edge0-120B on Vulkan](https://huggingface.co/infosave/Edge0-120B-cmf/resolve/main/assets/e0_120_vulkan_en.png)

### Vulkan · Smaller cards

The card takes the expert banks of as many layers as it has room for; the banks of the other layers stay in system
memory and the card reads them over the bus (Linux). Nothing to set. On the same RTX PRO 6000 limited to
24 GB (34 % of the experts on the card): code at **51 tok/s**, prompt reading at **44 tok/s**. Reading
over the bus is slow for prompts: on an NVIDIA card use [CUDA](#cuda--smaller-cards).

### Vulkan · Troubleshooting

- **`cortiq gpu` lists no card**: on Linux without a display set `XDG_RUNTIME_DIR=/tmp`; in a container install the
  packages above.
- **Out of memory on the card**: close other GPU programs, or lower the budget with `CMF_GPU_VRAM_MB=<MiB>`.
- **Several cards**: pick one with `CMF_GPU_ADAPTER`.

## Using the server

The same on both backends. The server speaks the OpenAI API:

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "edge0", "enable_thinking": false, "temperature": 0,
       "messages": [{"role": "user", "content": "Write a Rust function that parses a semantic version, with tests."}]}'
```

| what you want | how |
|---|---|
| another port | `--port <n>` |
| reach the server from other machines | `--host 0.0.0.0`; the server checks no key, so only on a network you trust |
| answers without the reasoning part | `"enable_thinking": false` in the request |
| streaming | `"stream": true` in the request |
| a longer context | `CMF_MAX_SEQ=<tokens>` before `cortiq serve` (default 32 768) |
| the fastest single stream | `CMF_SERVE_SLOTS=1` (as in the commands here) |
| the model name in requests | any name is accepted; `/v1/models` lists `qwen4_exp-cortiq` |

Endpoints: `/v1/chat/completions` (streaming, `tools`, `enable_thinking`), `/v1/completions`, `/v1/models`,
`/healthz`, and a chat page on `/`. One prompt without a server (on CUDA with `CMF_GPU=cuda` in front):

```bash
cortiq run edge0-120b.cmf --prompt-file task.txt --greedy --no-think
```

## Accuracy

HumanEval pass@1: **93.3 %** (164 tasks, greedy, thinking off, through this file on cortiq's server). Edge0's own
results on agent and reasoning benchmarks are on [Edge0's card](https://huggingface.co/Edge0/Edge0-120B-NVFP4).

## Files

| file | contents | size |
|---|---|---:|
| `edge0-120b.cmf` | 512 routed experts a layer at 1280 dimensions, NVFP4 as Edge0 trained them; the eight D/U projections of every layer, the shared expert, attention, embeddings and the output head at 8 bits; the router at 16 bits; the n-gram table at 8 bits; the draft (MTP) head | 95.64 GB |
| `edge0-120b.cmf.sha256` | checksum | |
| `edge0-120b.profile` | routing profile: how often each expert was picked on code and chat prompts; a smaller card keeps the experts in this order | 98 KB |
| `LICENSE`, `NOTICE`, `third-party/Qwen-LICENSE.txt` | the licenses that apply | |

## License and changes

Edge0-120B is Edge0's model under the [Edge0 Community Model License 1.0](LICENSE) and is built on Qwen3.8-Flash-Next under
the [Qwen Community License 1.0](third-party/Qwen-LICENSE.txt); [NOTICE](NOTICE) says which parts each covers, and both
apply to this file. This page is not made or endorsed by Edge0 or Qwen. Commercial use beyond the licenses: Edge0
(info@edge0.ai) and Qwen's own terms.

Changes made by infosave (cortiq), 13.11.2026, from [Edge0/Edge0-120B-NVFP4](https://huggingface.co/Edge0/Edge0-120B-NVFP4) revision 291b1ee0: the
routed experts copied bit for bit into CMF's `nvfp4`; the D/U projections and the shared expert re-encoded from Q8_0 to
q8_2f; the router stored as f16 (from bf16); the n-gram table re-encoded from FP8 to q8_2f; attention, GDN,
hyper-connections, embeddings, the output head and the draft head taken from Qwen3.8-Flash-Next's own weights in our
CMF of it (they match Edge0's to within Q8_0's rounding; the draft head's experts at 4 bits).

---

## Документация на русском

[Edge0-120B](https://huggingface.co/Edge0/Edge0-120B-NVFP4) от Edge0 — латентная модель-смесь экспертов на основе
[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) — в одном файле
[CMF](https://github.com/infosave2007/cmf) на 95.6 ГБ для движка `cortiq` на Rust. 512 экспертов каждого слоя работают
в 1280 измерениях и сохраняют 4-битное обучение Edge0: их веса NVFP4 скопированы бит в бит.

- **CUDA:** код 149–178 ток/с, чтение промпта 4 797–5 924 ток/с (4K–32K токенов).
- **Vulkan:** код 108–122 ток/с, чтение промпта 721–725 ток/с.
- **Карта 24 ГБ (CUDA):** код 100–118 ток/с, чтение промпта 2 394–2 823 ток/с; карты 12–32 ГБ — [ниже](#cuda--карты-поменьше).
- **Точность:** HumanEval pass@1 93.3 % (164 задачи, без рассуждений).

Измерено на одной RTX PRO 6000 96 ГБ, с черновыми токенами (MTP).

### Какой бэкенд

| ваша карта | бэкенд | что нужно |
|---|---|---|
| NVIDIA от 64 ГБ | [CUDA](#cuda-одна-карта-nvidia), вся модель на карте | cortiq 0.8.94+ со сборкой `--features cuda`, библиотеки CUDA 12.8+ |
| NVIDIA от 12 до 64 ГБ | [CUDA](#cuda--карты-поменьше), часть экспертов в ОЗУ | то же и 96 ГБ ОЗУ |
| от 64 ГБ, с драйвером Vulkan | [Vulkan](#vulkan-одна-карта) | cortiq 0.8.94+ (готовые сборки) |
| от 24 до 64 ГБ, с драйвером Vulkan | [Vulkan](#vulkan--карты-поменьше), часть экспертов в ОЗУ | то же и 96 ГБ ОЗУ |

Файл и API сервера одни для обоих бэкендов; карта NVIDIA работает с любым из них. Модель крупнее с той же настройкой — [Edge0-145B](https://huggingface.co/infosave/Edge0-145B-cmf).

### Быстрый старт

Linux, одна карта. Пять шагов от нуля до ответа.

**1. Скачайте** модель, контрольную сумму и профиль маршрутизации (96 ГБ) в одну папку:

```bash
pip install -U huggingface_hub
hf download infosave/Edge0-120B-cmf edge0-120b.cmf edge0-120b.cmf.sha256 edge0-120b.profile --local-dir .
sha256sum -c edge0-120b.cmf.sha256
```

**2. Установите cortiq:**

| ваша карта | установка |
|---|---|
| NVIDIA | `cargo install cortiq-cli --locked --features cuda` и библиотеки CUDA 12.8+ `libnvrtc` и `libcublas` (они есть в CUDA toolkit) |
| любая карта с драйвером Vulkan | [готовая сборка](https://github.com/infosave2007/cmf/releases) или `cargo install cortiq-cli --locked` |

**3. Запустите сервер** в этой папке — строкой для вашей карты:

```bash
# карта NVIDIA (CUDA)
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
CMF_GPU=cuda CMF_SERVE_SLOTS=1 cortiq serve edge0-120b.cmf --host 127.0.0.1 --port 8080

# любая карта (Vulkan)
CMF_SERVE_SLOTS=1 cortiq serve edge0-120b.cmf --host 127.0.0.1 --port 8080
```

Первый запуск читает файл и загружает карту — минута и дольше. Сервер готов, когда отвечает на:

```bash
curl http://127.0.0.1:8080/v1/models
```

**4. Задайте вопрос:**

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "edge0", "enable_thinking": false, "temperature": 0,
       "messages": [{"role": "user", "content": "Напиши функцию на Python, которая сливает два отсортированных списка, с тестами."}]}'
```

**5. Подключите редактор или агента.** Подойдёт всё, что работает с OpenAI API: базовый адрес
`http://127.0.0.1:8080/v1`, ключ любой, имя модели любое (`/v1/models` показывает `qwen4_exp-cortiq`). Страница
чата — `http://127.0.0.1:8080/`.

Больше ничего настраивать не нужно: при запуске cortiq смотрит на свободную память карты и сам решает, какая часть
модели на неё пойдёт. Требования, настройки и скорости каждого бэкенда — ниже; параметры сервера — в разделе
[Работа с сервером](#работа-с-сервером).

### CUDA: одна карта NVIDIA

#### CUDA · Требования

| компонент | требование |
|---|---|
| видеокарта | NVIDIA; от 64 ГБ вся модель на карте (46.3 ГБ занято при контексте 32K), от 12 ГБ — [часть экспертов в ОЗУ](#cuda--карты-поменьше); драйвер 570 или новее |
| библиотеки CUDA | `libnvrtc` и `libcublas` из CUDA 12.8+, в `LD_LIBRARY_PATH` |
| процессор | x86-64, от 8 потоков |
| ОЗУ | 96 ГБ (таблица n-грамм, 52 ГБ, закреплена в ОЗУ) |
| диск | 96 ГБ |
| ОС | Linux |
| cortiq | 0.8.94 или новее, сборка с `--features cuda`; в готовых сборках CUDA нет |

#### CUDA · Установка

```bash
# Rust 1.88 или новее; для самой сборки CUDA не нужна
cargo install cortiq-cli --locked --features cuda
```

#### CUDA · Запуск

```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH   # где лежат libnvrtc и libcublas
cortiq gpu   # "CUDA: <ваша карта> …" — готово

CMF_GPU=cuda CMF_SERVE_SLOTS=1 cortiq serve edge0-120b.cmf --host 127.0.0.1 --port 8080
```

Сервер готов, когда отвечает на `/v1/models`. Первый запуск читает файл и загружает карту — минута и дольше.
Запросы — в разделе [Работа с сервером](#работа-с-сервером).

#### CUDA · Настройки

| переменная | что делает |
|---|---|
| `CMF_GPU=cuda` | включает бэкенд CUDA |
| `CMF_SERVE_SLOTS=1` | один запрос за раз, на полной скорости |
| `CMF_CUDA_DEVICE=<n>` | какая карта NVIDIA, по номеру (по умолчанию 0) |
| `CMF_QWEN_MTP=0` | выключает черновые токены |
| `CMF_MAX_SEQ=<токены>` | предел контекста (по умолчанию 32 768) |

#### CUDA · Скорость

Стенд: одна RTX PRO 6000 96 ГБ; cortiq на бэкенде CUDA, команда запуска выше. Один поток, жадный выбор, без рассуждений,
черновики включены; перед каждым промптом новая метка сессии, поэтому кеш не участвует; среднее двух прогонов.

| промпт | чтение промпта, ток/с | генерация кода, ток/с |
|---|---:|---:|
| 1K | **4 208** | **188** |
| 4K | **5 924** | **178** |
| 16K | **5 264** | **174** |
| 32K | **4 797** | **149** |

![Edge0-120B на CUDA](https://huggingface.co/infosave/Edge0-120B-cmf/resolve/main/assets/e0_120_cuda_ru.png)

#### CUDA · Карты поменьше

Карта, на которую не помещаются все эксперты, держит самых используемых, остальные остаются в ОЗУ: шаг генерации
считает их на процессоре, промпт подгружает пакетами. Каких экспертов оставить на карте, говорит профиль
маршрутизации `edge0-120b.profile` — он должен лежать рядом с файлом модели. Настраивать ничего не нужно: cortiq берёт
свободную память карты (`CMF_GPU_VRAM_MB=<МиБ>` уменьшает её).

Стенд: та же RTX PRO 6000, у которой вся память сверх названного объёма занята другим процессом; экспертов вне
карты считают 12 потоков процессора. Один запуск на объём, методика та же. На 12 ГБ черновые токены выключены (их голова не помещается).

| память карты | экспертов на карте | чтение промпта, 4K | чтение промпта, 16K | генерация кода |
|---|---:|---:|---:|---:|
| 32 ГБ | 53 % | **3 352** ток/с | **3 135** ток/с | **130–145** ток/с |
| 24 ГБ | 30 % | **2 823** ток/с | **2 394** ток/с | **100–118** ток/с |
| 16 ГБ | 8 % | **1 571** ток/с | **1 571** ток/с | **71–73** ток/с |
| 12 ГБ | 2 % | **904** ток/с | **928** ток/с | **42** ток/с |



![Edge0-120B на картах поменьше, CUDA](https://huggingface.co/infosave/Edge0-120B-cmf/resolve/main/assets/e0_120_cuda_cards_ru.png)

#### CUDA · Если не работает

- `cortiq gpu` пишет `CUDA: unavailable`, либо в логе `libnvrtc (CUDA toolkit) was not found`: модель считается на
  процессоре. Поставьте CUDA 12.8+ и добавьте её `lib64` в `LD_LIBRARY_PATH`.
- **Не хватает памяти карты**: закройте другие программы на GPU или уменьшите долю cortiq через `CMF_GPU_VRAM_MB`.
- **Медленно на карте поменьше**: рядом с моделью нет `edge0-120b.profile` (в логе тогда нет строки «hottest first»).
- **Несколько карт NVIDIA**: выберите одну через `CMF_CUDA_DEVICE`.

### Vulkan: одна карта

#### Vulkan · Требования

| компонент | требование |
|---|---|
| видеокарта | одна карта с драйвером Vulkan; от 64 ГБ вся модель на карте (44.7 ГБ занято при контексте 32K), от 24 ГБ — [часть экспертов в ОЗУ](#vulkan--карты-поменьше); проверено на RTX PRO 6000 96 ГБ |
| процессор | любой x86-64 или ARM64 |
| ОЗУ | 64 ГБ (таблица n-грамм, 52 ГБ, живёт в ОЗУ); 96 ГБ на карте поменьше |
| диск | 96 ГБ |
| ОС | Linux |
| cortiq | 0.8.94 или новее: [готовая сборка](https://github.com/infosave2007/cmf/releases) или `cargo install` |

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

CMF_SERVE_SLOTS=1 cortiq serve edge0-120b.cmf --host 127.0.0.1 --port 8080
```

Сервер готов, когда отвечает на `/v1/models`; первый запуск загружает карту — минута и дольше.

#### Vulkan · Настройки

| переменная | что делает |
|---|---|
| `CMF_GPU_ADAPTER=<n>` | какая карта, по номеру из `cortiq gpu` |
| `CMF_SERVE_SLOTS=1` | один запрос за раз, на полной скорости |
| `CMF_QWEN_MTP=0` | выключает черновые токены |
| `XDG_RUNTIME_DIR=/tmp` | Vulkan на Linux без экрана |

#### Vulkan · Скорость

Стенд: одна RTX PRO 6000 96 ГБ; cortiq на бэкенде Vulkan, команда запуска выше; та же методика, что на CUDA.

| промпт | чтение промпта, ток/с | генерация кода, ток/с |
|---|---:|---:|
| 1K | **757** | **115** |
| 4K | **725** | **108** |
| 16K | **724** | **112** |
| 32K | **721** | **122** |

![Edge0-120B на Vulkan](https://huggingface.co/infosave/Edge0-120B-cmf/resolve/main/assets/e0_120_vulkan_ru.png)

#### Vulkan · Карты поменьше

Карта берёт банки экспертов стольких слоёв, сколько помещается; банки остальных слоёв остаются в системной памяти, и
карта читает их по шине (Linux). Настраивать ничего не нужно. На той же RTX PRO 6000, ограниченной до 24 ГБ
(34 % экспертов на карте): код **51 ток/с**, чтение промпта **44 ток/с**. Чтение по шине медленно для
промптов: на карте NVIDIA берите [CUDA](#cuda--карты-поменьше).

#### Vulkan · Если не работает

- **`cortiq gpu` не показывает карту**: на Linux без экрана задайте `XDG_RUNTIME_DIR=/tmp`; в контейнере поставьте
  пакеты из раздела установки.
- **Не хватает памяти карты**: закройте другие программы на GPU или уменьшите бюджет через `CMF_GPU_VRAM_MB=<МиБ>`.
- **Несколько карт**: выберите одну через `CMF_GPU_ADAPTER`.

### Работа с сервером

Одинаково на обоих бэкендах. Сервер говорит на OpenAI API:

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "edge0", "enable_thinking": false, "temperature": 0,
       "messages": [{"role": "user", "content": "Напиши функцию на Rust, которая разбирает semver, с тестами."}]}'
```

| что нужно | как |
|---|---|
| другой порт | `--port <n>` |
| доступ к серверу с других машин | `--host 0.0.0.0`; сервер не проверяет ключ, поэтому только в доверенной сети |
| ответы без рассуждений | `"enable_thinking": false` в запросе |
| потоковый вывод | `"stream": true` в запросе |
| контекст длиннее | `CMF_MAX_SEQ=<токены>` перед `cortiq serve` (по умолчанию 32 768) |
| самый быстрый одиночный поток | `CMF_SERVE_SLOTS=1` (как в командах здесь) |
| имя модели в запросах | принимается любое; `/v1/models` показывает `qwen4_exp-cortiq` |

Точки входа: `/v1/chat/completions` (потоковый вывод, `tools`, `enable_thinking`), `/v1/completions`, `/v1/models`,
`/healthz` и страница чата на `/`. Один запрос без сервера (на CUDA — с `CMF_GPU=cuda` впереди):

```bash
cortiq run edge0-120b.cmf --prompt-file task.txt --greedy --no-think
```

### Точность

HumanEval pass@1: **93.3 %** (164 задачи, жадный выбор, без рассуждений, этот файл на сервере cortiq). Результаты
Edge0 на агентных и логических тестах — на [странице Edge0](https://huggingface.co/Edge0/Edge0-120B-NVFP4).

### Файлы

| файл | содержимое | размер |
|---|---|---:|
| `edge0-120b.cmf` | 512 экспертов слоя в 1280 измерениях, NVFP4 как их обучил Edge0; восемь проекций D/U каждого слоя, общий эксперт, внимание, эмбеддинги и выходная голова в 8 битах; роутер в 16 битах; таблица n-грамм в 8 битах; голова черновиков (MTP) | 95.64 ГБ |
| `edge0-120b.cmf.sha256` | контрольная сумма | |
| `edge0-120b.profile` | профиль маршрутизации: как часто выбирался каждый эксперт на промптах с кодом и диалогами; в этом порядке карта поменьше оставляет экспертов у себя | 98 КБ |
| `LICENSE`, `NOTICE`, `third-party/Qwen-LICENSE.txt` | действующие лицензии | |

### Лицензия и изменения

Edge0-120B — модель Edge0 под [Edge0 Community Model License 1.0](LICENSE), построенная на Qwen3.8-Flash-Next под
[Qwen Community License 1.0](third-party/Qwen-LICENSE.txt); [NOTICE](NOTICE) объясняет, что покрывает каждая, и обе
действуют для этого файла. Страница не сделана и не одобрена Edge0 или Qwen. Коммерческое использование сверх лицензий:
Edge0 (info@edge0.ai) и собственные условия Qwen.

Изменения, внесённые infosave (cortiq) 13.11.2026, из [Edge0/Edge0-120B-NVFP4](https://huggingface.co/Edge0/Edge0-120B-NVFP4) ревизии 291b1ee0:
эксперты скопированы бит в бит в `nvfp4` формата CMF; проекции D/U и общий эксперт перекодированы из Q8_0 в q8_2f;
роутер хранится в f16 (из bf16); таблица n-грамм перекодирована из FP8 в q8_2f; внимание, GDN, hyper-connections,
эмбеддинги, выходная голова и голова черновиков взяты из собственных весов Qwen3.8-Flash-Next в нашем CMF (совпадают с
весами Edge0 в пределах округления Q8_0; эксперты головы черновиков в 4 битах).

---

## 中文文档

Edge0 的 [Edge0-120B](https://huggingface.co/Edge0/Edge0-120B-NVFP4) 是基于 [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
的潜空间混合专家模型，这里打包为一个 95.6 GB 的 [CMF](https://github.com/infosave2007/cmf) 文件，由 Rust 推理引擎
`cortiq` 运行。每层 512 个专家在 1280 维上工作，保留 Edge0 的 4 位训练：其 NVFP4 权重逐位复制。

- **CUDA：** 生成代码 149–178 token/s，读取提示词 4 797–5 924 token/s（4K–32K token）。
- **Vulkan：** 生成代码 108–122 token/s，读取提示词 721–725 token/s。
- **24 GB 显卡（CUDA）：** 生成代码 100–118 token/s，读取提示词 2 394–2 823 token/s；12–32 GB 显卡见[下文](#cuda--较小的显卡)。
- **准确率：** HumanEval pass@1 93.3 %（164 题，关闭思考）。

测试平台：一张 RTX PRO 6000 96 GB，开启草稿（MTP）。

### 选择后端

| 你的显卡 | 后端 | 需要 |
|---|---|---|
| NVIDIA，64 GB 以上 | [CUDA](#cuda-后端一张-nvidia-显卡)，整个模型在显卡上 | cortiq 0.8.94+，以 `--features cuda` 构建；CUDA 12.8+ 库 |
| NVIDIA，12 GB 至 64 GB | [CUDA](#cuda--较小的显卡)，部分专家在内存中 | 同上，另需 96 GB 内存 |
| 64 GB 以上，带 Vulkan 驱动 | [Vulkan](#vulkan-后端一张显卡) | cortiq 0.8.94+（预编译版本） |
| 24 GB 至 64 GB，带 Vulkan 驱动 | [Vulkan](#vulkan--较小的显卡)，部分专家在内存中 | 同上，另需 96 GB 内存 |

两种后端使用相同的文件和服务器 API；NVIDIA 显卡两种都能用。同样设置的更大模型：[Edge0-145B](https://huggingface.co/infosave/Edge0-145B-cmf)。

### 快速开始

Linux，一张显卡。五步即可得到回答。

**1. 下载**模型、校验和与路由统计文件（96 GB），放在同一目录：

```bash
pip install -U huggingface_hub
hf download infosave/Edge0-120B-cmf edge0-120b.cmf edge0-120b.cmf.sha256 edge0-120b.profile --local-dir .
sha256sum -c edge0-120b.cmf.sha256
```

**2. 安装 cortiq：**

| 你的显卡 | 安装 |
|---|---|
| NVIDIA | `cargo install cortiq-cli --locked --features cuda`，以及 CUDA 12.8+ 的 `libnvrtc` 和 `libcublas`（CUDA toolkit 自带） |
| 任何带 Vulkan 驱动的显卡 | [预编译版本](https://github.com/infosave2007/cmf/releases)，或 `cargo install cortiq-cli --locked` |

**3. 在该目录启动服务器**，按显卡选择对应的一行：

```bash
# NVIDIA 显卡（CUDA）
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
CMF_GPU=cuda CMF_SERVE_SLOTS=1 cortiq serve edge0-120b.cmf --host 127.0.0.1 --port 8080

# 任意显卡（Vulkan）
CMF_SERVE_SLOTS=1 cortiq serve edge0-120b.cmf --host 127.0.0.1 --port 8080
```

首次启动需读取文件并装载显卡，约一分钟或更久。下面的命令有响应即表示就绪：

```bash
curl http://127.0.0.1:8080/v1/models
```

**4. 提问：**

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "edge0", "enable_thinking": false, "temperature": 0,
       "messages": [{"role": "user", "content": "用 Python 写一个合并两个有序列表的函数，并附测试。"}]}'
```

**5. 连接编辑器或智能体。** 任何支持 OpenAI API 的工具都可以：base URL 为 `http://127.0.0.1:8080/v1`，API key 任意，
模型名任意（`/v1/models` 列出 `qwen4_exp-cortiq`）。聊天页面在 `http://127.0.0.1:8080/`。

无需其他设置：启动时 cortiq 根据显卡的空闲显存决定把模型的多少放到显卡上。各后端的要求、设置和速度见下文；服务器选项见
[使用服务器](#使用服务器)。

### CUDA 后端：一张 NVIDIA 显卡

#### CUDA · 环境要求

| 组件 | 要求 |
|---|---|
| 显卡 | NVIDIA；64 GB 以上可放下整个模型（32K 上下文时占用 46.3 GB），12 GB 起[部分专家留在内存](#cuda--较小的显卡)；驱动 570 或更高 |
| CUDA 库 | CUDA 12.8+ 的 `libnvrtc` 与 `libcublas`，位于 `LD_LIBRARY_PATH` |
| CPU | x86-64，8 线程以上 |
| 内存 | 96 GB（n-gram 表 52 GB 锁定在内存中） |
| 磁盘 | 96 GB |
| 系统 | Linux |
| cortiq | 0.8.94 或更高，以 `--features cuda` 构建；预编译版本不含 CUDA |

#### CUDA · 安装

```bash
# Rust 1.88 或更高；构建本身不需要 CUDA
cargo install cortiq-cli --locked --features cuda
```

#### CUDA · 运行

```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH   # libnvrtc 与 libcublas 所在目录
cortiq gpu   # "CUDA: <你的显卡> …" 表示就绪

CMF_GPU=cuda CMF_SERVE_SLOTS=1 cortiq serve edge0-120b.cmf --host 127.0.0.1 --port 8080
```

服务器在 `/v1/models` 有响应即就绪。首次启动需读取文件并装载显卡，约一分钟或更久。请求方式见[使用服务器](#使用服务器)。

#### CUDA · 设置

| 变量 | 作用 |
|---|---|
| `CMF_GPU=cuda` | 选择 CUDA 后端 |
| `CMF_SERVE_SLOTS=1` | 一次一个请求，全速 |
| `CMF_CUDA_DEVICE=<n>` | 使用第几张 NVIDIA 显卡（默认 0） |
| `CMF_QWEN_MTP=0` | 关闭草稿 |
| `CMF_MAX_SEQ=<token 数>` | 上下文上限（默认 32 768） |

#### CUDA · 速度

测试平台：一张 RTX PRO 6000 96 GB；cortiq 的 CUDA 后端，上面的运行命令。单路，贪心解码，关闭思考，开启草稿；每个提示词前加
新的会话标记，因此不使用缓存；两次运行的平均值。

| 提示词 | 读取提示词，token/s | 生成代码，token/s |
|---|---:|---:|
| 1K | **4 208** | **188** |
| 4K | **5 924** | **178** |
| 16K | **5 264** | **174** |
| 32K | **4 797** | **149** |

![Edge0-120B · CUDA](https://huggingface.co/infosave/Edge0-120B-cmf/resolve/main/assets/e0_120_cuda_zh.png)

#### CUDA · 较小的显卡

放不下全部专家的显卡只保留最常用的专家，其余留在内存：生成时由 CPU 计算，读取提示词时分批送入显卡。保留哪些专家由路由
统计文件 `edge0-120b.profile` 决定，它必须与模型文件放在同一目录。无需设置：cortiq 使用显卡的空闲显存（`CMF_GPU_VRAM_MB=<MiB>`
可调低）。

测试平台：同一张 RTX PRO 6000，所示显存以外的部分由另一进程占用；显卡外的专家由 12 个 CPU 线程计算。每种显存运行一次，
方法同上。12 GB 时关闭草稿（其头部放不下）。

| 显存 | 显卡上的专家 | 读取提示词，4K | 读取提示词，16K | 生成代码 |
|---|---:|---:|---:|---:|
| 32 GB | 53 % | **3 352** token/s | **3 135** token/s | **130–145** token/s |
| 24 GB | 30 % | **2 823** token/s | **2 394** token/s | **100–118** token/s |
| 16 GB | 8 % | **1 571** token/s | **1 571** token/s | **71–73** token/s |
| 12 GB | 2 % | **904** token/s | **928** token/s | **42** token/s |



![Edge0-120B · 较小的显卡，CUDA](https://huggingface.co/infosave/Edge0-120B-cmf/resolve/main/assets/e0_120_cuda_cards_zh.png)

#### CUDA · 常见问题

- `cortiq gpu` 显示 `CUDA: unavailable`，或日志出现 `libnvrtc (CUDA toolkit) was not found`：模型正在 CPU 上运行。
  安装 CUDA 12.8+，并把其 `lib64` 加入 `LD_LIBRARY_PATH`。
- **显存不足**：关闭其他使用 GPU 的程序，或用 `CMF_GPU_VRAM_MB` 调低 cortiq 的占用。
- **较小的显卡上很慢**：模型旁边没有 `edge0-120b.profile`（日志中没有 “hottest first” 一行）。
- **多张 NVIDIA 显卡**：用 `CMF_CUDA_DEVICE` 选择其中一张。

### Vulkan 后端：一张显卡

#### Vulkan · 环境要求

| 组件 | 要求 |
|---|---|
| 显卡 | 一张带 Vulkan 驱动的显卡；64 GB 以上可放下整个模型（32K 上下文时占用 44.7 GB），24 GB 起[部分专家留在内存](#vulkan--较小的显卡)；已在 RTX PRO 6000 96 GB 上验证 |
| CPU | 任意 x86-64 或 ARM64 |
| 内存 | 64 GB（n-gram 表 52 GB 常驻内存）；较小的显卡需 96 GB |
| 磁盘 | 96 GB |
| 系统 | Linux |
| cortiq | 0.8.94 或更高：[预编译版本](https://github.com/infosave2007/cmf/releases)或 `cargo install` |

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

CMF_SERVE_SLOTS=1 cortiq serve edge0-120b.cmf --host 127.0.0.1 --port 8080
```

服务器在 `/v1/models` 有响应即就绪；首次启动需装载显卡，约一分钟或更久。

#### Vulkan · 设置

| 变量 | 作用 |
|---|---|
| `CMF_GPU_ADAPTER=<n>` | 使用哪张显卡，取 `cortiq gpu` 中的编号 |
| `CMF_SERVE_SLOTS=1` | 一次一个请求，全速 |
| `CMF_QWEN_MTP=0` | 关闭草稿 |
| `XDG_RUNTIME_DIR=/tmp` | 无显示器的 Linux 上使用 Vulkan |

#### Vulkan · 速度

测试平台：一张 RTX PRO 6000 96 GB；cortiq 的 Vulkan 后端，上面的运行命令；测试方法与 CUDA 相同。

| 提示词 | 读取提示词，token/s | 生成代码，token/s |
|---|---:|---:|
| 1K | **757** | **115** |
| 4K | **725** | **108** |
| 16K | **724** | **112** |
| 32K | **721** | **122** |

![Edge0-120B · Vulkan](https://huggingface.co/infosave/Edge0-120B-cmf/resolve/main/assets/e0_120_vulkan_zh.png)

#### Vulkan · 较小的显卡

显卡放下尽可能多的层的专家库；其余层的专家库留在系统内存，由显卡经总线读取（Linux）。无需设置。同一张 RTX PRO 6000 限制为
24 GB 时（34 % 的专家在显卡上）：生成代码 **51 token/s**，读取提示词 **44 token/s**。经总线读取提示词较慢：
NVIDIA 显卡请用 [CUDA](#cuda--较小的显卡)。

#### Vulkan · 常见问题

- `cortiq gpu` 没有列出显卡：在无显示器的 Linux 上设置 `XDG_RUNTIME_DIR=/tmp`；在容器中安装上面的软件包。
- **显存不足**：关闭其他使用 GPU 的程序，或用 `CMF_GPU_VRAM_MB=<MiB>` 调低预算。
- **多张显卡**：用 `CMF_GPU_ADAPTER` 选择其中一张。

### 使用服务器

两种后端用法相同。服务器使用 OpenAI API：

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "edge0", "enable_thinking": false, "temperature": 0,
       "messages": [{"role": "user", "content": "用 Rust 写一个解析语义化版本号的函数，并附测试。"}]}'
```

| 需求 | 做法 |
|---|---|
| 换端口 | `--port <n>` |
| 让其他机器访问 | `--host 0.0.0.0`；服务器不校验密钥，仅限可信网络 |
| 回答不带推理部分 | 请求中加 `"enable_thinking": false` |
| 流式输出 | 请求中加 `"stream": true` |
| 更长的上下文 | 在 `cortiq serve` 前设置 `CMF_MAX_SEQ=<token 数>`（默认 32 768） |
| 单路最快 | `CMF_SERVE_SLOTS=1`（本页命令已包含） |
| 请求中的模型名 | 任意名称均可；`/v1/models` 列出 `qwen4_exp-cortiq` |

接口：`/v1/chat/completions`（流式、`tools`、`enable_thinking`）、`/v1/completions`、`/v1/models`、`/healthz`，`/` 为聊天页面。
不启动服务器、单次提示（CUDA 需在前面加 `CMF_GPU=cuda`）：

```bash
cortiq run edge0-120b.cmf --prompt-file task.txt --greedy --no-think
```

### 准确率

HumanEval pass@1：**93.3 %**（164 题，贪心解码，关闭思考，本文件在 cortiq 服务器上运行）。Edge0 在智能体与推理基准上的结果见
[Edge0 的页面](https://huggingface.co/Edge0/Edge0-120B-NVFP4)。

### 文件

| 文件 | 内容 | 大小 |
|---|---|---:|
| `edge0-120b.cmf` | 每层 512 个专家，1280 维，按 Edge0 训练的 NVFP4；每层八组 D/U 投影、共享专家、注意力、嵌入与输出头 8 位；路由器 16 位；n-gram 表 8 位；草稿（MTP）头 | 95.64 GB |
| `edge0-120b.cmf.sha256` | 校验和 | |
| `edge0-120b.profile` | 路由统计：代码与对话提示词中每个专家被选中的次数；较小的显卡按此顺序保留专家 | 98 KB |
| `LICENSE`、`NOTICE`、`third-party/Qwen-LICENSE.txt` | 适用的许可证 | |

### 许可证与修改

Edge0-120B 是 Edge0 的模型，适用 [Edge0 Community Model License 1.0](LICENSE)，并基于适用 [Qwen Community License 1.0](third-party/Qwen-LICENSE.txt)
的 Qwen3.8-Flash-Next；[NOTICE](NOTICE) 说明两者各自覆盖的部分，两者都适用于本文件。本页面并非由 Edge0 或 Qwen 制作或认可。超出许可证
范围的商业使用：Edge0（info@edge0.ai）及 Qwen 自身的条款。

infosave（cortiq）于 2026-11-13 基于 [Edge0/Edge0-120B-NVFP4](https://huggingface.co/Edge0/Edge0-120B-NVFP4) 修订版 291b1ee0 所做的修改：路由专家逐位复制为 CMF 的
`nvfp4`；D/U 投影与共享专家由 Q8_0 重新编码为 q8_2f；路由器以 f16 存储（来自 bf16）；n-gram 表由 FP8 重新编码为 q8_2f；注意力、GDN、
hyper-connections、嵌入、输出头与草稿头取自我们 CMF 中 Qwen3.8-Flash-Next 自己的权重（与 Edge0 的权重在 Q8_0 舍入范围内一致；草稿头
的专家为 4 位）。
