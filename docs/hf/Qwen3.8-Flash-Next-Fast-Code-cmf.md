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
[CMF](https://github.com/infosave2007/cmf) file. Runs with `cortiq`, a Rust inference engine.

- **One card with 48 GB or more:** prompt ~840 tok/s, code 136–166 tok/s, 41 GB of VRAM.
- **NVIDIA cards from 12 GB (CUDA):** code 145 / 175 / 207 tok/s within 12 / 16 / 24 GB — 94 / 91 / 97 % of the
  2-bit Strata Coder's speed, with the experts at 4 bits and the attention at 8.
- **HumanEval pass@1:** 93.3 % (164 tasks).

[English](#requirements) · [Русский](#документация-на-русском) · [中文](#中文文档)

## Requirements

| | one card with 48 GB or more | NVIDIA card from 12 GB |
|---|---|---|
| GPU | 48 GB of VRAM or more, Vulkan or CUDA | NVIDIA, 12 GB or more; driver 570+, CUDA 12.8+ libraries (`libnvrtc`, `libcublas`) |
| CPU | any x86-64 or ARM64 | x86-64 with AVX-512 VNNI (AMD Zen 4/5, Intel Sapphire Rapids or newer), 8 threads or more |
| RAM | 32 GB (the n-gram table, 28 GB, stays in RAM) | 64 GB |
| disk | 67 GB | 67 GB |
| cortiq | 0.8.65 or later, [prebuilt binaries](https://github.com/infosave2007/cmf/releases) or `cargo install` | 0.8.78 or later, built with `--features cuda` (the prebuilt binaries have no CUDA) |

Tested on Linux.

## Install

```bash
# a card with 48 GB or more (Vulkan); or download a prebuilt binary from the releases page
cargo install cortiq-cli --locked

# an NVIDIA card from 12 GB (CUDA): Rust 1.88+, then
cargo install cortiq-cli --locked --features cuda
```

## Download

```bash
pip install -U huggingface_hub
hf download infosave/Qwen3.8-Flash-Next-Fast-Code-cmf qwen38-flash-next-fast-code.cmf fast-code.profile \
  qwen38-flash-next-fast-code.cmf.sha256 --local-dir .
sha256sum -c qwen38-flash-next-fast-code.cmf.sha256
```

## Run

A card with 48 GB or more:

```bash
CMF_QWEN_PROFILE=fast-code.profile \
  cortiq serve qwen38-flash-next-fast-code.cmf --host 127.0.0.1 --port 8080
```

An NVIDIA card from 12 GB:

```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH   # where libnvrtc and libcublas are
CMF_GPU=cuda CMF_QWEN_PROFILE=fast-code.profile CMF_QWEN_ROUTE_TAIL=0.05 \
  CMF_SERVE_SLOTS=1 CMF_QWEN_CUDA_REG=1 CMF_QWEN_CUDA_REG_ALL=1 \
  cortiq serve qwen38-flash-next-fast-code.cmf --host 127.0.0.1 --port 8080
```

The server is ready when it answers on `/v1/models`; the first start reads the 66.8 GB file and takes a minute or
more. Then:

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "fast-code", "enable_thinking": false, "temperature": 0,
       "messages": [{"role": "user", "content": "Write a Rust function that parses a semantic version, with tests."}]}'
```

Endpoints: `/v1/chat/completions` (streaming, `tools`, `enable_thinking`), `/v1/completions`, `/v1/models`,
`/healthz`, a web dashboard on `/`. Draft tokens are used under `temperature: 0`. Any OpenAI-compatible editor or
agent can point at `http://127.0.0.1:8080/v1`.

One prompt without a server:

```bash
CMF_QWEN_PROFILE=fast-code.profile cortiq run qwen38-flash-next-fast-code.cmf --prompt-file task.txt --greedy --no-think
```

## Settings

| variable | what it does |
|---|---|
| `CMF_QWEN_PROFILE=fast-code.profile` | the experts' order on the card; always set it |
| `CMF_GPU=cuda` | the CUDA backend (a `--features cuda` build) |
| `CMF_QWEN_CUDA_REG=1 CMF_QWEN_CUDA_REG_ALL=1` | keep every expert pinned in RAM, so nothing waits on the disk; needs 64 GB of RAM |
| `CMF_QWEN_ROUTE_TAIL=0.05` | skip the experts with under 5 % of a token's routing weight (the measured setting) |
| `CMF_SERVE_SLOTS=1` | one request at a time, at full speed |
| `CMF_GPU_VRAM_MB=<MiB>` | cap the card memory the model takes (default: what is free) |
| `CMF_QWEN_CUDA_GROW_MB=<MiB>` | card memory kept free for a growing context (default 1024) |
| `CMF_QWEN_CPU_EXPERT_THREADS=<n>` | host threads for the experts outside the card (default: all but three) |
| `CMF_GPU_ADAPTER=<n>` | which GPU (`cortiq gpu` lists them) |
| `XDG_RUNTIME_DIR=/tmp` | Vulkan on headless Linux |

## Troubleshooting

- **`libnvrtc (CUDA toolkit) was not found`** or **`qwen4 cuda: …; staying on the host path`**: install the CUDA
  12.8+ toolkit and put its `lib64` on `LD_LIBRARY_PATH`. The server then logs `qwen4 cuda: arena prefilled with …`.
- **Out of memory on the card**: close other GPU programs or set `CMF_GPU_VRAM_MB` a little lower.
- **Less than 64 GB of RAM**: drop `CMF_QWEN_CUDA_REG_ALL=1` (and if needed `CMF_QWEN_CUDA_REG=1`); it runs slower.
- **A host without AVX-512 VNNI**: every expert outside the card is copied to it instead; it runs, slower.
- **The answer starts with reasoning**: send `"enable_thinking": false` (`--no-think` for `cortiq run`).
- **Slow first request**: the weights are still coming from the disk; keep the server running between requests.

## Speed and accuracy

One RTX PRO 6000 (96 GB, Vulkan), one stream, greedy, thinking off, draft tokens on. Strata Coder is ISTA-DASLab's
2-bit cut of the same model on its own engine Strata 0.1.58 (CUDA); the full file is the 512-expert
[q4tp file](https://huggingface.co/infosave/Qwen3.8-Flash-Next-cmf) on cortiq.

| | Fast Code | Strata Coder (Strata) | full file (cortiq) |
|---|---:|---:|---:|
| HumanEval pass@1 (164 tasks) | 93.3 % | 94.5 % | 94.5 % |
| code perplexity, code written after 20.10.2026 (8192 tokens) | **3.337** | 3.436¹ | 3.366 |
| code perplexity, Rust (1536 tokens) | 4.900 | 4.690¹ | 4.111 |
| prompt 1K / 4K / 16K / 32K tokens, tok/s | 798 / 827 / 844 / 842 | 1916 / 5492 / 7229 / 7659 | 171 / 208 / 206 / 210 |
| generation after a 4K / 16K / 32K prompt, tok/s | 161 / 166 / 136 | 220 / 223 / 222 | 118 / 138 / 116 |
| expert weights | 4 bits | 2 bits | 4 bits |
| file on disk | 66.8 GB | 58.4 GB + 5.2 GB MTP | 98.6 GB |
| peak VRAM (32K context) | 40.9 GB | 31.1 GB | 59.6 GB |

¹ Strata Coder's weights imported into CMF and run by cortiq: the same text and tokenizer for both.

NVIDIA cards from 12 GB: one RTX 5090 held to the memory shown, host 2× EPYC 9554 (31 threads, 62 GB of RAM), the
command above, code generation on a 2 151-token spec, mean of two runs. Strata Coder ran on the same machine with an
expert cache of 1 500 / 3 000 / 5 000 experts (10.7 / 14.5 / 19.6 GB of the card).

| card memory | Fast Code (cortiq), tok/s | Strata Coder (Strata), tok/s | | HumanEval, first 40 |
|---|---:|---:|---:|---:|
| 12 GB | **145.0** | 154.1 | 94 % | 38 / 40 |
| 16 GB | **174.6** | 192.5 | 91 % | 39 / 40 |
| 24 GB | **206.7** | 212.0 | 97 % | 39 / 40 |

![Speed](https://huggingface.co/infosave/Qwen3.8-Flash-Next-Fast-Code-cmf/resolve/main/assets/fc_speed_en.png)

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
[CMF](https://github.com/infosave2007/cmf) на 66.8 ГБ. Запускается движком `cortiq` на Rust.

- **Одна карта от 48 ГБ:** промпт ~840 ток/с, код 136–166 ток/с, 41 ГБ видеопамяти.
- **Карты NVIDIA от 12 ГБ (CUDA):** код 145 / 175 / 207 ток/с в 12 / 16 / 24 ГБ — 94 / 91 / 97 % скорости
  2-битной Strata Coder, при экспертах в 4 битах и внимании в 8.
- **HumanEval pass@1:** 93.3 % (164 задачи).

### Требования

| | одна карта от 48 ГБ | карта NVIDIA от 12 ГБ |
|---|---|---|
| видеокарта | от 48 ГБ, Vulkan или CUDA | NVIDIA от 12 ГБ; драйвер 570+, библиотеки CUDA 12.8+ (`libnvrtc`, `libcublas`) |
| процессор | любой x86-64 или ARM64 | x86-64 с AVX-512 VNNI (AMD Zen 4/5, Intel Sapphire Rapids и новее), от 8 потоков |
| ОЗУ | 32 ГБ (таблица n-грамм, 28 ГБ, живёт в ОЗУ) | 64 ГБ |
| диск | 67 ГБ | 67 ГБ |
| cortiq | 0.8.65 или новее, [готовые сборки](https://github.com/infosave2007/cmf/releases) или `cargo install` | 0.8.78 или новее, сборка с `--features cuda` (в готовых сборках CUDA нет) |

Проверено на Linux.

### Установка

```bash
# карта от 48 ГБ (Vulkan); либо готовая сборка со страницы релизов
cargo install cortiq-cli --locked

# карта NVIDIA от 12 ГБ (CUDA): Rust 1.88+, затем
cargo install cortiq-cli --locked --features cuda
```

### Загрузка

```bash
pip install -U huggingface_hub
hf download infosave/Qwen3.8-Flash-Next-Fast-Code-cmf qwen38-flash-next-fast-code.cmf fast-code.profile \
  qwen38-flash-next-fast-code.cmf.sha256 --local-dir .
sha256sum -c qwen38-flash-next-fast-code.cmf.sha256
```

### Запуск

Карта от 48 ГБ:

```bash
CMF_QWEN_PROFILE=fast-code.profile \
  cortiq serve qwen38-flash-next-fast-code.cmf --host 127.0.0.1 --port 8080
```

Карта NVIDIA от 12 ГБ:

```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH   # где лежат libnvrtc и libcublas
CMF_GPU=cuda CMF_QWEN_PROFILE=fast-code.profile CMF_QWEN_ROUTE_TAIL=0.05 \
  CMF_SERVE_SLOTS=1 CMF_QWEN_CUDA_REG=1 CMF_QWEN_CUDA_REG_ALL=1 \
  cortiq serve qwen38-flash-next-fast-code.cmf --host 127.0.0.1 --port 8080
```

Сервер готов, когда отвечает на `/v1/models`; первый запуск читает файл на 66.8 ГБ, это минута и дольше. Затем:

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "fast-code", "enable_thinking": false, "temperature": 0,
       "messages": [{"role": "user", "content": "Напиши функцию на Rust, которая разбирает semver, с тестами."}]}'
```

Точки входа: `/v1/chat/completions` (потоковый вывод, `tools`, `enable_thinking`), `/v1/completions`, `/v1/models`,
`/healthz`, веб-панель на `/`. Черновики работают при `temperature: 0`. Любой редактор или агент с OpenAI API
подключается к `http://127.0.0.1:8080/v1`.

Один запрос без сервера:

```bash
CMF_QWEN_PROFILE=fast-code.profile cortiq run qwen38-flash-next-fast-code.cmf --prompt-file task.txt --greedy --no-think
```

### Настройки

| переменная | что делает |
|---|---|
| `CMF_QWEN_PROFILE=fast-code.profile` | порядок экспертов на карте; задавайте всегда |
| `CMF_GPU=cuda` | CUDA-бэкенд (сборка с `--features cuda`) |
| `CMF_QWEN_CUDA_REG=1 CMF_QWEN_CUDA_REG_ALL=1` | все эксперты закреплены в ОЗУ, ничто не ждёт диск; нужно 64 ГБ ОЗУ |
| `CMF_QWEN_ROUTE_TAIL=0.05` | пропускать экспертов с долей маршрута токена меньше 5 % (так измерено) |
| `CMF_SERVE_SLOTS=1` | один запрос за раз, на полной скорости |
| `CMF_GPU_VRAM_MB=<МиБ>` | сколько памяти карты брать (по умолчанию — свободную) |
| `CMF_QWEN_CUDA_GROW_MB=<МиБ>` | запас памяти карты под растущий контекст (по умолчанию 1024) |
| `CMF_QWEN_CPU_EXPERT_THREADS=<n>` | потоки хоста для экспертов вне карты (по умолчанию все, кроме трёх) |
| `CMF_GPU_ADAPTER=<n>` | какая видеокарта (`cortiq gpu` выводит список) |
| `XDG_RUNTIME_DIR=/tmp` | Vulkan на Linux без экрана |

### Если не работает

- **`libnvrtc (CUDA toolkit) was not found`** или **`qwen4 cuda: …; staying on the host path`**: поставьте CUDA
  12.8+ и добавьте её `lib64` в `LD_LIBRARY_PATH`. После этого в логе есть `qwen4 cuda: arena prefilled with …`.
- **Не хватает памяти карты**: закройте другие программы на GPU или задайте `CMF_GPU_VRAM_MB` чуть меньше.
- **ОЗУ меньше 64 ГБ**: уберите `CMF_QWEN_CUDA_REG_ALL=1` (при нехватке и `CMF_QWEN_CUDA_REG=1`); работает медленнее.
- **Процессор без AVX-512 VNNI**: каждый эксперт вне карты копируется на неё; работает, медленнее.
- **Ответ начинается с рассуждений**: передайте `"enable_thinking": false` (`--no-think` для `cortiq run`).
- **Медленный первый запрос**: веса ещё читаются с диска; держите сервер запущенным.

### Скорость и точность

Одна RTX PRO 6000 (96 ГБ, Vulkan), один поток, жадный выбор, без рассуждений, черновики включены. Strata Coder —
2-битная сборка ISTA-DASLab той же модели на собственном движке Strata 0.1.58 (CUDA); полный файл — 512 экспертов,
[q4tp](https://huggingface.co/infosave/Qwen3.8-Flash-Next-cmf), cortiq.

| | Fast Code | Strata Coder (Strata) | полный файл (cortiq) |
|---|---:|---:|---:|
| HumanEval pass@1 (164 задачи) | 93.3 % | 94.5 % | 94.5 % |
| перплексия на коде, написанном после 20.10.2026 (8192 токена) | **3.337** | 3.436¹ | 3.366 |
| перплексия на коде Rust (1536 токенов) | 4.900 | 4.690¹ | 4.111 |
| промпт 1K / 4K / 16K / 32K токенов, ток/с | 798 / 827 / 844 / 842 | 1916 / 5492 / 7229 / 7659 | 171 / 208 / 206 / 210 |
| генерация после промпта 4K / 16K / 32K, ток/с | 161 / 166 / 136 | 220 / 223 / 222 | 118 / 138 / 116 |
| веса экспертов | 4 бита | 2 бита | 4 бита |
| файл на диске | 66.8 ГБ | 58.4 ГБ + 5.2 ГБ MTP | 98.6 ГБ |
| пик видеопамяти (контекст 32K) | 40.9 ГБ | 31.1 ГБ | 59.6 ГБ |

¹ Веса Strata Coder, импортированные в CMF и запущенные cortiq: один текст и один токенизатор.

Карты NVIDIA от 12 ГБ: одна RTX 5090, ограниченная указанной памятью, хост 2× EPYC 9554 (31 поток, 62 ГБ ОЗУ),
команда выше, генерация кода по ТЗ на 2 151 токен, среднее двух запусков. Strata Coder — на той же машине с кешем на
1 500 / 3 000 / 5 000 экспертов (10,7 / 14,5 / 19,6 ГБ карты).

| память карты | Fast Code (cortiq), ток/с | Strata Coder (Strata), ток/с | | HumanEval, первые 40 |
|---|---:|---:|---:|---:|
| 12 ГБ | **145.0** | 154.1 | 94 % | 38 / 40 |
| 16 ГБ | **174.6** | 192.5 | 91 % | 39 / 40 |
| 24 ГБ | **206.7** | 212.0 | 97 % | 39 / 40 |

![Скорость](https://huggingface.co/infosave/Qwen3.8-Flash-Next-Fast-Code-cmf/resolve/main/assets/fc_speed_ru.png)

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
连同共享专家、草稿（MTP）头和 n-gram 表，打包为一个 66.8 GB 的 [CMF](https://github.com/infosave2007/cmf) 文件，由 Rust 推理引擎 `cortiq` 运行。

- **一张 48 GB 以上的显卡：** 读取提示约 840 token/s，生成代码 136–166 token/s，显存 41 GB。
- **12 GB 起的 NVIDIA 显卡（CUDA）：** 在 12 / 16 / 24 GB 内生成代码 145 / 175 / 207 token/s，为 2 位 Strata Coder
  速度的 94 / 91 / 97 %，专家保持 4 位、注意力保持 8 位。
- **HumanEval pass@1：** 93.3 %（164 题）。

### 环境要求

| | 一张 48 GB 以上的显卡 | 12 GB 起的 NVIDIA 显卡 |
|---|---|---|
| 显卡 | 48 GB 以上，Vulkan 或 CUDA | NVIDIA 12 GB 以上；驱动 570+，CUDA 12.8+ 库（`libnvrtc`、`libcublas`） |
| CPU | 任意 x86-64 或 ARM64 | 支持 AVX-512 VNNI 的 x86-64（AMD Zen 4/5、Intel Sapphire Rapids 或更新），8 线程以上 |
| 内存 | 32 GB（n-gram 表 28 GB 常驻内存） | 64 GB |
| 磁盘 | 67 GB | 67 GB |
| cortiq | 0.8.65 或更高，[预编译版本](https://github.com/infosave2007/cmf/releases)或 `cargo install` | 0.8.78 或更高，以 `--features cuda` 构建（预编译版本不含 CUDA） |

已在 Linux 上验证。

### 安装

```bash
# 48 GB 以上的显卡（Vulkan）；或从发布页下载预编译版本
cargo install cortiq-cli --locked

# 12 GB 起的 NVIDIA 显卡（CUDA）：Rust 1.88+，然后
cargo install cortiq-cli --locked --features cuda
```

### 下载

```bash
pip install -U huggingface_hub
hf download infosave/Qwen3.8-Flash-Next-Fast-Code-cmf qwen38-flash-next-fast-code.cmf fast-code.profile \
  qwen38-flash-next-fast-code.cmf.sha256 --local-dir .
sha256sum -c qwen38-flash-next-fast-code.cmf.sha256
```

### 运行

48 GB 以上的显卡：

```bash
CMF_QWEN_PROFILE=fast-code.profile \
  cortiq serve qwen38-flash-next-fast-code.cmf --host 127.0.0.1 --port 8080
```

12 GB 起的 NVIDIA 显卡：

```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH   # libnvrtc 与 libcublas 所在目录
CMF_GPU=cuda CMF_QWEN_PROFILE=fast-code.profile CMF_QWEN_ROUTE_TAIL=0.05 \
  CMF_SERVE_SLOTS=1 CMF_QWEN_CUDA_REG=1 CMF_QWEN_CUDA_REG_ALL=1 \
  cortiq serve qwen38-flash-next-fast-code.cmf --host 127.0.0.1 --port 8080
```

服务器在 `/v1/models` 有响应即就绪；首次启动需读取 66.8 GB 文件，约一分钟或更久。然后：

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "fast-code", "enable_thinking": false, "temperature": 0,
       "messages": [{"role": "user", "content": "用 Rust 写一个解析语义化版本号的函数，并附测试。"}]}'
```

接口：`/v1/chat/completions`（流式、`tools`、`enable_thinking`）、`/v1/completions`、`/v1/models`、`/healthz`，`/` 为网页面板。
`temperature: 0` 时启用草稿。任何兼容 OpenAI 的编辑器或智能体都可连接 `http://127.0.0.1:8080/v1`。

不启动服务器、单次提示：

```bash
CMF_QWEN_PROFILE=fast-code.profile cortiq run qwen38-flash-next-fast-code.cmf --prompt-file task.txt --greedy --no-think
```

### 设置

| 变量 | 作用 |
|---|---|
| `CMF_QWEN_PROFILE=fast-code.profile` | 专家在显卡上的顺序；务必设置 |
| `CMF_GPU=cuda` | CUDA 后端（需 `--features cuda` 构建） |
| `CMF_QWEN_CUDA_REG=1 CMF_QWEN_CUDA_REG_ALL=1` | 全部专家锁定在内存中，不等待磁盘；需要 64 GB 内存 |
| `CMF_QWEN_ROUTE_TAIL=0.05` | 跳过路由权重低于 5 % 的专家（测试所用设置） |
| `CMF_SERVE_SLOTS=1` | 一次一个请求，全速 |
| `CMF_GPU_VRAM_MB=<MiB>` | 模型可用的显存上限（默认：全部空闲显存） |
| `CMF_QWEN_CUDA_GROW_MB=<MiB>` | 为增长的上下文预留的显存（默认 1024） |
| `CMF_QWEN_CPU_EXPERT_THREADS=<n>` | 计算显卡外专家的主机线程数（默认：除三个外的全部） |
| `CMF_GPU_ADAPTER=<n>` | 选择显卡（`cortiq gpu` 列出） |
| `XDG_RUNTIME_DIR=/tmp` | 无显示器的 Linux 上使用 Vulkan |

### 常见问题

- 日志出现 `libnvrtc (CUDA toolkit) was not found` 或 `qwen4 cuda: …; staying on the host path`：安装 CUDA 12.8+，并把其
  `lib64` 加入 `LD_LIBRARY_PATH`。之后日志中会出现 `qwen4 cuda: arena prefilled with …`。
- **显存不足**：关闭其他使用 GPU 的程序，或把 `CMF_GPU_VRAM_MB` 设小一些。
- **内存不足 64 GB**：去掉 `CMF_QWEN_CUDA_REG_ALL=1`（必要时也去掉 `CMF_QWEN_CUDA_REG=1`），速度会变慢。
- **CPU 不支持 AVX-512 VNNI**：显卡外的专家改为复制到显卡；可以运行，但更慢。
- **回答以推理开头**：发送 `"enable_thinking": false`（`cortiq run` 用 `--no-think`）。
- **首个请求较慢**：权重仍在从磁盘读取；请保持服务器运行。

### 速度与准确率

一张 RTX PRO 6000（96 GB，Vulkan），单路，贪心解码，关闭思考，开启草稿。Strata Coder 是 ISTA-DASLab 对同一模型的 2 位裁剪版，
运行于其自有引擎 Strata 0.1.58（CUDA）；完整文件为 512 专家的 [q4tp 文件](https://huggingface.co/infosave/Qwen3.8-Flash-Next-cmf)，由 cortiq 运行。

| | Fast Code | Strata Coder（Strata） | 完整文件（cortiq） |
|---|---:|---:|---:|
| HumanEval pass@1（164 题） | 93.3 % | 94.5 % | 94.5 % |
| 代码困惑度，2026-10-20 之后写的代码（8192 token） | **3.337** | 3.436¹ | 3.366 |
| 代码困惑度，Rust（1536 token） | 4.900 | 4.690¹ | 4.111 |
| 提示 1K / 4K / 16K / 32K token，token/s | 798 / 827 / 844 / 842 | 1916 / 5492 / 7229 / 7659 | 171 / 208 / 206 / 210 |
| 4K / 16K / 32K 提示后的生成，token/s | 161 / 166 / 136 | 220 / 223 / 222 | 118 / 138 / 116 |
| 专家权重 | 4 位 | 2 位 | 4 位 |
| 磁盘文件 | 66.8 GB | 58.4 GB + 5.2 GB MTP | 98.6 GB |
| 显存峰值（32K 上下文） | 40.9 GB | 31.1 GB | 59.6 GB |

¹ Strata Coder 的权重导入 CMF 后由 cortiq 运行：两者使用相同文本与分词器。

12 GB 起的 NVIDIA 显卡：一张 RTX 5090，限制为表中显存，主机 2× EPYC 9554（31 线程，62 GB 内存），上面的命令，按 2 151 token 的需求生成代码，
两次运行的平均值。Strata Coder 在同一台机器上运行，专家缓存 1 500 / 3 000 / 5 000 个（占显卡 10.7 / 14.5 / 19.6 GB）。

| 显存 | Fast Code（cortiq），token/s | Strata Coder（Strata），token/s | | HumanEval 前 40 题 |
|---|---:|---:|---:|---:|
| 12 GB | **145.0** | 154.1 | 94 % | 38 / 40 |
| 16 GB | **174.6** | 192.5 | 91 % | 39 / 40 |
| 24 GB | **206.7** | 212.0 | 97 % | 39 / 40 |

![速度](https://huggingface.co/infosave/Qwen3.8-Flash-Next-Fast-Code-cmf/resolve/main/assets/fc_speed_zh.png)

![代码准确率](https://huggingface.co/infosave/Qwen3.8-Flash-Next-Fast-Code-cmf/resolve/main/assets/fc_quality_zh.png)

### 文件

| 文件 | 内容 | 大小 |
|---|---|---:|
| `qwen38-flash-next-fast-code.cmf` | 每层 512 个路由专家中的 256 个（4 位 q4tp）与共享专家；注意力、嵌入与输出头 8 位；n-gram 表 4 位；草稿（MTP）头 | 66.76 GB |
| `fast-code.profile` | 路由画像（非权重）：专家在显卡上的顺序 | 49 KB |
| `qwen38-flash-next-fast-code.cmf.sha256` | 校验和 | |

保留的是代码最常路由到的专家。通用对话请使用[完整文件](https://huggingface.co/infosave/Qwen3.8-Flash-Next-cmf)。注意力在 262 144 token 内为精确计算。
权重源自 Qwen 的发布，仍受其许可约束。运行生成的代码前请先审阅。
