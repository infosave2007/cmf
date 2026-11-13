#!/usr/bin/env python3
"""The Edge0-145B / Edge0-120B CMF model cards (EN, RU, ZH) from the measured numbers.

    python3 docs/hf/edge0_cards.py docs/hf/figures/edge0/figures_data.json docs/hf

Writes Edge0-145B-cmf.md and Edge0-120B-cmf.md. Every number comes from DATA.json (the journal
scripts/edge0/JOURNAL_EDGE0_2026-11-13.md says where each was measured).
"""
import json, sys

REPO = {"m145": "infosave/Edge0-145B-cmf", "m120": "infosave/Edge0-120B-cmf"}
SRC = {"m145": "Edge0/Edge0-145B-NVFP4", "m120": "Edge0/Edge0-120B-NVFP4"}
FILE = {"m145": "edge0-145b.cmf", "m120": "edge0-120b.cmf"}
NAME = {"m145": "Edge0-145B", "m120": "Edge0-120B"}


def rng(vals, digits=0):
    a, b = min(vals), max(vals)
    f = (lambda v: f"{v:,.0f}".replace(",", " ")) if digits == 0 else (lambda v: f"{v:.1f}")
    return f"{f(a)}–{f(b)}"


def card_gb(mib):
    """The smallest common card size that holds `mib` with a margin."""
    gb = mib / 1024
    for size in (24, 32, 48, 64, 80, 96, 141):
        if gb <= size * 0.92:
            return size
    return 192


def tbl_rows(d, m, lang):
    head = {"en": ("prompt", "prompt reading, tok/s", "code generation, tok/s"),
            "ru": ("промпт", "чтение промпта, ток/с", "генерация кода, ток/с"),
            "zh": ("提示词", "读取提示词，token/s", "生成代码，token/s")}[lang]
    out = f"| {head[0]} | {head[1]} | {head[2]} |\n|---|---:|---:|\n"
    for p, r, g in zip(d["prompts"], d["prefill"][m], d["decode"][m]):
        out += f"| {p} | **{r:,.0f}** | **{g:.0f}** |\n".replace(",", " ")
    return out


def card(data, m):
    D = data[m]
    cu, vk = data["cuda"], data["vulkan"]
    he = data["humaneval"][m]
    name, fn, repo, src = NAME[m], FILE[m], REPO[m], SRC[m]
    rd_c = [v for p, v in zip(cu["prompts"], cu["prefill"][m]) if p != "1K"]
    gn_c = [v for p, v in zip(cu["prompts"], cu["decode"][m]) if p != "1K"]
    rd_v = [v for p, v in zip(vk["prompts"], vk["prefill"][m]) if p != "1K"]
    gn_v = [v for p, v in zip(vk["prompts"], vk["decode"][m]) if p != "1K"]
    cg = card_gb(cu["card_mib"][m])
    vg = card_gb(vk["card_mib"][m])
    cu_mem = cu["card_mib"][m] / 1024
    vk_mem = vk["card_mib"][m] / 1024
    size = D["file_gb"]
    disk = int(size) + 1
    img = lambda b, lang: f"https://huggingface.co/{repo}/resolve/main/assets/e0_{m[1:]}_{b}_{lang}.png"
    serve = f"cortiq serve {fn} --host 127.0.0.1 --port 8080"
    dl = (f"pip install -U huggingface_hub\nhf download {repo} {fn} {fn}.sha256 --local-dir .\n"
          f"sha256sum -c {fn}.sha256")
    cu_run = ("export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH   # {lib}\n"
              "cortiq gpu   # {chk}\n\n"
              f"CMF_GPU=cuda CMF_SERVE_SLOTS=1 {serve}")
    vk_run = ("export XDG_RUNTIME_DIR=/tmp   # {hl}\n"
              "cortiq gpu   # {lst}\n\n"
              f"CMF_SERVE_SLOTS=1 {serve}")
    apt = "apt-get install -y --no-install-recommends vulkan-tools libglvnd0 libegl1 libgl1 libglx0"
    curl = lambda content: ('curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" \\\n'
                            '  -d \'{"model": "edge0", "enable_thinking": false, "temperature": 0,\n'
                            f'       "messages": [{{"role": "user", "content": "{content}"}}]}}\'')
    run1 = f"cortiq run {fn} --prompt-file task.txt --greedy --no-think"
    other = "m120" if m == "m145" else "m145"
    oom = {
        "en": f"take [{NAME[other]}](https://huggingface.co/{REPO[other]})" if m == "m145" else "use a card with more memory",
        "ru": f"возьмите [{NAME[other]}](https://huggingface.co/{REPO[other]})" if m == "m145" else "используйте карту с большей памятью",
        "zh": f"换用 [{NAME[other]}](https://huggingface.co/{REPO[other]})" if m == "m145" else "使用显存更大的显卡",
    }
    alt = {
        "en": f"For a smaller card, take [{NAME[other]}](https://huggingface.co/{REPO[other]})." if m == "m145"
        else f"The larger model with the same setup: [{NAME[other]}](https://huggingface.co/{REPO[other]}).",
        "ru": f"Для карты поменьше — [{NAME[other]}](https://huggingface.co/{REPO[other]})." if m == "m145"
        else f"Модель крупнее с той же настройкой — [{NAME[other]}](https://huggingface.co/{REPO[other]}).",
        "zh": f"显存较小时请用 [{NAME[other]}](https://huggingface.co/{REPO[other]})。" if m == "m145"
        else f"同样设置的更大模型：[{NAME[other]}](https://huggingface.co/{REPO[other]})。",
    }

    en = f"""# {name} — CMF

[{name}](https://huggingface.co/{src}) by Edge0, a latent mixture-of-experts model built on
[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), in one {size:.1f} GB
[CMF](https://github.com/infosave2007/cmf) file for `cortiq`, a Rust inference engine. The 512 experts of every layer
work at {D["latent"]} dimensions and keep Edge0's 4-bit training: their NVFP4 weights are copied bit for bit.

- **CUDA:** code at {rng(gn_c)} tok/s, prompt reading at {rng(rd_c)} tok/s (4K–32K tokens).
- **Vulkan:** code at {rng(gn_v)} tok/s, prompt reading at {rng(rd_v)} tok/s.
- **Accuracy:** HumanEval pass@1 {he:.1f} % (164 tasks, thinking off).

Measured on one RTX PRO 6000 96 GB, with draft tokens (MTP) on.

[English](#which-backend) · [Русский](#документация-на-русском) · [中文](#中文文档)

## Which backend

| your card | backend | what it needs |
|---|---|---|
| NVIDIA, {cg} GB or more | [CUDA](#cuda-one-nvidia-card-with-{cg}-gb-or-more) | cortiq 0.8.90+ built with `--features cuda`, CUDA 12.8+ libraries |
| {vg} GB or more, with a Vulkan driver | [Vulkan](#vulkan-one-card-with-{vg}-gb-or-more) | cortiq 0.8.90+ (prebuilt binaries) |

The whole model sits on one card. The file and the server API are the same for both backends; an NVIDIA card can run
either. {alt["en"]}

## Download

```bash
{dl}
```

## CUDA: one NVIDIA card with {cg} GB or more

### CUDA · Requirements

| component | requirement |
|---|---|
| GPU | NVIDIA, {cg} GB or more ({cu_mem:.1f} GB in use at a 32K context); driver 570 or later |
| CUDA libraries | `libnvrtc` and `libcublas` of the CUDA 12.8+ toolkit, on `LD_LIBRARY_PATH` |
| CPU | x86-64, 8 threads or more |
| RAM | 64 GB (the n-gram table, 52 GB, stays in RAM) |
| disk | {disk} GB |
| OS | Linux |
| cortiq | 0.8.90 or later, built with `--features cuda`; the prebuilt binaries have no CUDA |

### CUDA · Install

```bash
# Rust 1.88 or later; the build itself needs no CUDA toolkit
cargo install cortiq-cli --locked --features cuda
```

### CUDA · Run

```bash
{cu_run.format(lib="where libnvrtc and libcublas are", chk='"CUDA: <your card> …" means ready')}
```

The server is ready when it answers on `/v1/models`. The first start reads the file and puts every expert on the card:
a minute or more. Requests: [Using the server](#using-the-server).

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

{tbl_rows(cu, m, "en")}
![{name} on CUDA]({img("cuda", "en")})

### CUDA · Troubleshooting

- `cortiq gpu` prints `CUDA: unavailable`, or the log says `libnvrtc (CUDA toolkit) was not found`: the model runs
  on the CPU. Install the CUDA 12.8+ toolkit and put its `lib64` on `LD_LIBRARY_PATH`.
- **Out of memory on the card**: every expert has to fit; close other GPU programs, or {oom["en"]}.
- **Several NVIDIA cards**: pick one with `CMF_CUDA_DEVICE`.

## Vulkan: one card with {vg} GB or more

### Vulkan · Requirements

| component | requirement |
|---|---|
| GPU | one card with {vg} GB or more and a Vulkan driver ({vk_mem:.1f} GB in use at a 32K context); tested on an RTX PRO 6000 96 GB |
| CPU | any x86-64 or ARM64 |
| RAM | 64 GB (the n-gram table, 52 GB, stays in RAM) |
| disk | {disk} GB |
| OS | Linux |
| cortiq | 0.8.90 or later: a [prebuilt binary](https://github.com/infosave2007/cmf/releases) or `cargo install` |

### Vulkan · Install

A prebuilt binary from the [releases page](https://github.com/infosave2007/cmf/releases), or:

```bash
cargo install cortiq-cli --locked
```

In a container, add the Vulkan loader:

```bash
{apt}
```

### Vulkan · Run

```bash
{vk_run.format(hl="only on Linux without a display", lst="lists the cards Vulkan sees")}
```

The server is ready when it answers on `/v1/models`; the first start puts every expert on the card, a minute or more.

### Vulkan · Settings

| variable | what it does |
|---|---|
| `CMF_GPU_ADAPTER=<n>` | which card, by its number in `cortiq gpu` |
| `CMF_SERVE_SLOTS=1` | one request at a time, at full speed |
| `CMF_QWEN_MTP=0` | turns the draft tokens off |
| `XDG_RUNTIME_DIR=/tmp` | Vulkan on Linux without a display |

### Vulkan · Speed

Stand: one RTX PRO 6000 96 GB; cortiq on the Vulkan backend, the run command above; the same protocol as on CUDA.

{tbl_rows(vk, m, "en")}
![{name} on Vulkan]({img("vulkan", "en")})

### Vulkan · Troubleshooting

- **`cortiq gpu` lists no card**: on Linux without a display set `XDG_RUNTIME_DIR=/tmp`; in a container install the
  packages above.
- **Out of memory on the card**: every expert has to fit; close other GPU programs, or {oom["en"]}.
- **Several cards**: pick one with `CMF_GPU_ADAPTER`.

## Using the server

The same on both backends.

```bash
{curl("Write a Rust function that parses a semantic version, with tests.")}
```

Endpoints: `/v1/chat/completions` (streaming, `tools`, `enable_thinking`), `/v1/completions`, `/v1/models`,
`/healthz`, a web dashboard on `/`. Any OpenAI-compatible editor or agent can point at `http://127.0.0.1:8080/v1`.
One prompt without a server (on CUDA with `CMF_GPU=cuda`):

```bash
{run1}
```

- **The answer starts with reasoning**: send `"enable_thinking": false` (`--no-think` for `cortiq run`).
- **Network**: the server checks no key; keep `--host 127.0.0.1` unless the network is trusted.

## Accuracy

HumanEval pass@1: **{he:.1f} %** (164 tasks, greedy, thinking off, through this file on cortiq's server). Edge0's own
results on agent and reasoning benchmarks are on [Edge0's card](https://huggingface.co/{src}).

## Files

| file | contents | size |
|---|---|---:|
| `{fn}` | 512 routed experts a layer at {D["latent"]} dimensions, NVFP4 as Edge0 trained them; the eight D/U projections of every layer, the shared expert, attention, embeddings and the output head at 8 bits; the router at 16 bits; the n-gram table at 8 bits; the draft (MTP) head | {size:.2f} GB |
| `{fn}.sha256` | checksum | |
| `LICENSE`, `NOTICE`, `third-party/Qwen-LICENSE.txt` | the licenses that apply | |

## License and changes

{name} is Edge0's model under the [Edge0 Community Model License 1.0](LICENSE) and is built on Qwen3.8-Flash-Next under
the [Qwen Community License 1.0](third-party/Qwen-LICENSE.txt); [NOTICE](NOTICE) says which parts each covers, and both
apply to this file. This page is not made or endorsed by Edge0 or Qwen. Commercial use beyond the licenses: Edge0
(info@edge0.ai) and Qwen's own terms.

Changes made by infosave (cortiq), 13.11.2026, from [{src}](https://huggingface.co/{src}) revision {D["rev"][:8]}: the
routed experts copied bit for bit into CMF's `nvfp4`; the D/U projections and the shared expert re-encoded from Q8_0 to
q8_2f; the router stored as f16 (from bf16); the n-gram table re-encoded from FP8 to q8_2f; attention, GDN,
hyper-connections, embeddings, the output head and the draft head taken from Qwen3.8-Flash-Next's own weights in our
CMF of it (they match Edge0's to within Q8_0's rounding; the draft head's experts at 4 bits).
"""

    ru = f"""## Документация на русском

[{name}](https://huggingface.co/{src}) от Edge0 — латентная модель-смесь экспертов на основе
[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) — в одном файле
[CMF](https://github.com/infosave2007/cmf) на {size:.1f} ГБ для движка `cortiq` на Rust. 512 экспертов каждого слоя работают
в {D["latent"]} измерениях и сохраняют 4-битное обучение Edge0: их веса NVFP4 скопированы бит в бит.

- **CUDA:** код {rng(gn_c)} ток/с, чтение промпта {rng(rd_c)} ток/с (4K–32K токенов).
- **Vulkan:** код {rng(gn_v)} ток/с, чтение промпта {rng(rd_v)} ток/с.
- **Точность:** HumanEval pass@1 {he:.1f} % (164 задачи, без рассуждений).

Измерено на одной RTX PRO 6000 96 ГБ, с черновыми токенами (MTP).

### Какой бэкенд

| ваша карта | бэкенд | что нужно |
|---|---|---|
| NVIDIA от {cg} ГБ | [CUDA](#cuda-одна-карта-nvidia-от-{cg}-гб) | cortiq 0.8.90+ со сборкой `--features cuda`, библиотеки CUDA 12.8+ |
| от {vg} ГБ, с драйвером Vulkan | [Vulkan](#vulkan-одна-карта-от-{vg}-гб) | cortiq 0.8.90+ (готовые сборки) |

Вся модель лежит на одной карте. Файл и API сервера одни для обоих бэкендов; карта NVIDIA работает с любым из них.
{alt["ru"]}

### Загрузка

```bash
{dl}
```

### CUDA: одна карта NVIDIA от {cg} ГБ

#### CUDA · Требования

| компонент | требование |
|---|---|
| видеокарта | NVIDIA от {cg} ГБ ({cu_mem:.1f} ГБ занято при контексте 32K); драйвер 570 или новее |
| библиотеки CUDA | `libnvrtc` и `libcublas` из CUDA 12.8+, в `LD_LIBRARY_PATH` |
| процессор | x86-64, от 8 потоков |
| ОЗУ | 64 ГБ (таблица n-грамм, 52 ГБ, живёт в ОЗУ) |
| диск | {disk} ГБ |
| ОС | Linux |
| cortiq | 0.8.90 или новее, сборка с `--features cuda`; в готовых сборках CUDA нет |

#### CUDA · Установка

```bash
# Rust 1.88 или новее; для самой сборки CUDA не нужна
cargo install cortiq-cli --locked --features cuda
```

#### CUDA · Запуск

```bash
{cu_run.format(lib="где лежат libnvrtc и libcublas", chk='"CUDA: <ваша карта> …" — готово')}
```

Сервер готов, когда отвечает на `/v1/models`. Первый запуск читает файл и кладёт всех экспертов на карту — минута и
дольше. Запросы — в разделе [Работа с сервером](#работа-с-сервером).

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

{tbl_rows(cu, m, "ru")}
![{name} на CUDA]({img("cuda", "ru")})

#### CUDA · Если не работает

- `cortiq gpu` пишет `CUDA: unavailable`, либо в логе `libnvrtc (CUDA toolkit) was not found`: модель считается на
  процессоре. Поставьте CUDA 12.8+ и добавьте её `lib64` в `LD_LIBRARY_PATH`.
- **Не хватает памяти карты**: все эксперты должны поместиться; закройте другие программы на GPU или
  {oom["ru"]}.
- **Несколько карт NVIDIA**: выберите одну через `CMF_CUDA_DEVICE`.

### Vulkan: одна карта от {vg} ГБ

#### Vulkan · Требования

| компонент | требование |
|---|---|
| видеокарта | одна карта от {vg} ГБ с драйвером Vulkan ({vk_mem:.1f} ГБ занято при контексте 32K); проверено на RTX PRO 6000 96 ГБ |
| процессор | любой x86-64 или ARM64 |
| ОЗУ | 64 ГБ (таблица n-грамм, 52 ГБ, живёт в ОЗУ) |
| диск | {disk} ГБ |
| ОС | Linux |
| cortiq | 0.8.90 или новее: [готовая сборка](https://github.com/infosave2007/cmf/releases) или `cargo install` |

#### Vulkan · Установка

Готовая сборка со [страницы релизов](https://github.com/infosave2007/cmf/releases), либо:

```bash
cargo install cortiq-cli --locked
```

В контейнере добавьте загрузчик Vulkan:

```bash
{apt}
```

#### Vulkan · Запуск

```bash
{vk_run.format(hl="только на Linux без экрана", lst="список карт, которые видит Vulkan")}
```

Сервер готов, когда отвечает на `/v1/models`; первый запуск кладёт всех экспертов на карту — минута и дольше.

#### Vulkan · Настройки

| переменная | что делает |
|---|---|
| `CMF_GPU_ADAPTER=<n>` | какая карта, по номеру из `cortiq gpu` |
| `CMF_SERVE_SLOTS=1` | один запрос за раз, на полной скорости |
| `CMF_QWEN_MTP=0` | выключает черновые токены |
| `XDG_RUNTIME_DIR=/tmp` | Vulkan на Linux без экрана |

#### Vulkan · Скорость

Стенд: одна RTX PRO 6000 96 ГБ; cortiq на бэкенде Vulkan, команда запуска выше; та же методика, что на CUDA.

{tbl_rows(vk, m, "ru")}
![{name} на Vulkan]({img("vulkan", "ru")})

#### Vulkan · Если не работает

- **`cortiq gpu` не показывает карту**: на Linux без экрана задайте `XDG_RUNTIME_DIR=/tmp`; в контейнере поставьте
  пакеты из раздела установки.
- **Не хватает памяти карты**: все эксперты должны поместиться; закройте другие программы на GPU или
  {oom["ru"]}.
- **Несколько карт**: выберите одну через `CMF_GPU_ADAPTER`.

### Работа с сервером

Одинаково на обоих бэкендах.

```bash
{curl("Напиши функцию на Rust, которая разбирает semver, с тестами.")}
```

Точки входа: `/v1/chat/completions` (потоковый вывод, `tools`, `enable_thinking`), `/v1/completions`, `/v1/models`,
`/healthz`, веб-панель на `/`. Любой редактор или агент с OpenAI API подключается к `http://127.0.0.1:8080/v1`. Один
запрос без сервера (на CUDA — с `CMF_GPU=cuda`):

```bash
{run1}
```

- **Ответ начинается с рассуждений**: передайте `"enable_thinking": false` (`--no-think` для `cortiq run`).
- **Сеть**: сервер не проверяет ключ; оставляйте `--host 127.0.0.1`, если сеть не доверенная.

### Точность

HumanEval pass@1: **{he:.1f} %** (164 задачи, жадный выбор, без рассуждений, этот файл на сервере cortiq). Результаты
Edge0 на агентных и логических тестах — на [странице Edge0](https://huggingface.co/{src}).

### Файлы

| файл | содержимое | размер |
|---|---|---:|
| `{fn}` | 512 экспертов слоя в {D["latent"]} измерениях, NVFP4 как их обучил Edge0; восемь проекций D/U каждого слоя, общий эксперт, внимание, эмбеддинги и выходная голова в 8 битах; роутер в 16 битах; таблица n-грамм в 8 битах; голова черновиков (MTP) | {size:.2f} ГБ |
| `{fn}.sha256` | контрольная сумма | |
| `LICENSE`, `NOTICE`, `third-party/Qwen-LICENSE.txt` | действующие лицензии | |

### Лицензия и изменения

{name} — модель Edge0 под [Edge0 Community Model License 1.0](LICENSE), построенная на Qwen3.8-Flash-Next под
[Qwen Community License 1.0](third-party/Qwen-LICENSE.txt); [NOTICE](NOTICE) объясняет, что покрывает каждая, и обе
действуют для этого файла. Страница не сделана и не одобрена Edge0 или Qwen. Коммерческое использование сверх лицензий:
Edge0 (info@edge0.ai) и собственные условия Qwen.

Изменения, внесённые infosave (cortiq) 13.11.2026, из [{src}](https://huggingface.co/{src}) ревизии {D["rev"][:8]}:
эксперты скопированы бит в бит в `nvfp4` формата CMF; проекции D/U и общий эксперт перекодированы из Q8_0 в q8_2f;
роутер хранится в f16 (из bf16); таблица n-грамм перекодирована из FP8 в q8_2f; внимание, GDN, hyper-connections,
эмбеддинги, выходная голова и голова черновиков взяты из собственных весов Qwen3.8-Flash-Next в нашем CMF (совпадают с
весами Edge0 в пределах округления Q8_0; эксперты головы черновиков в 4 битах).
"""

    zh = f"""## 中文文档

Edge0 的 [{name}](https://huggingface.co/{src}) 是基于 [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
的潜空间混合专家模型，这里打包为一个 {size:.1f} GB 的 [CMF](https://github.com/infosave2007/cmf) 文件，由 Rust 推理引擎
`cortiq` 运行。每层 512 个专家在 {D["latent"]} 维上工作，保留 Edge0 的 4 位训练：其 NVFP4 权重逐位复制。

- **CUDA：** 生成代码 {rng(gn_c)} token/s，读取提示词 {rng(rd_c)} token/s（4K–32K token）。
- **Vulkan：** 生成代码 {rng(gn_v)} token/s，读取提示词 {rng(rd_v)} token/s。
- **准确率：** HumanEval pass@1 {he:.1f} %（164 题，关闭思考）。

测试平台：一张 RTX PRO 6000 96 GB，开启草稿（MTP）。

### 选择后端

| 你的显卡 | 后端 | 需要 |
|---|---|---|
| NVIDIA，{cg} GB 以上 | [CUDA](#cuda-后端一张-{cg}-gb-以上的-nvidia-显卡) | cortiq 0.8.90+，以 `--features cuda` 构建；CUDA 12.8+ 库 |
| {vg} GB 以上，带 Vulkan 驱动 | [Vulkan](#vulkan-后端一张-{vg}-gb-以上的显卡) | cortiq 0.8.90+（预编译版本） |

整个模型放在一张显卡上。两种后端使用相同的文件和服务器 API；NVIDIA 显卡两种都能用。{alt["zh"]}

### 下载

```bash
{dl}
```

### CUDA 后端：一张 {cg} GB 以上的 NVIDIA 显卡

#### CUDA · 环境要求

| 组件 | 要求 |
|---|---|
| 显卡 | NVIDIA，{cg} GB 以上（32K 上下文时占用 {cu_mem:.1f} GB）；驱动 570 或更高 |
| CUDA 库 | CUDA 12.8+ 的 `libnvrtc` 与 `libcublas`，位于 `LD_LIBRARY_PATH` |
| CPU | x86-64，8 线程以上 |
| 内存 | 64 GB（n-gram 表 52 GB 常驻内存） |
| 磁盘 | {disk} GB |
| 系统 | Linux |
| cortiq | 0.8.90 或更高，以 `--features cuda` 构建；预编译版本不含 CUDA |

#### CUDA · 安装

```bash
# Rust 1.88 或更高；构建本身不需要 CUDA
cargo install cortiq-cli --locked --features cuda
```

#### CUDA · 运行

```bash
{cu_run.format(lib="libnvrtc 与 libcublas 所在目录", chk='"CUDA: <你的显卡> …" 表示就绪')}
```

服务器在 `/v1/models` 有响应即就绪。首次启动需读取文件并把所有专家放到显卡上，约一分钟或更久。请求方式见[使用服务器](#使用服务器)。

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

{tbl_rows(cu, m, "zh")}
![{name} · CUDA]({img("cuda", "zh")})

#### CUDA · 常见问题

- `cortiq gpu` 显示 `CUDA: unavailable`，或日志出现 `libnvrtc (CUDA toolkit) was not found`：模型正在 CPU 上运行。
  安装 CUDA 12.8+，并把其 `lib64` 加入 `LD_LIBRARY_PATH`。
- **显存不足**：所有专家都必须放进显卡；关闭其他使用 GPU 的程序，或{oom["zh"]}。
- **多张 NVIDIA 显卡**：用 `CMF_CUDA_DEVICE` 选择其中一张。

### Vulkan 后端：一张 {vg} GB 以上的显卡

#### Vulkan · 环境要求

| 组件 | 要求 |
|---|---|
| 显卡 | 一张 {vg} GB 以上、带 Vulkan 驱动的显卡（32K 上下文时占用 {vk_mem:.1f} GB）；已在 RTX PRO 6000 96 GB 上验证 |
| CPU | 任意 x86-64 或 ARM64 |
| 内存 | 64 GB（n-gram 表 52 GB 常驻内存） |
| 磁盘 | {disk} GB |
| 系统 | Linux |
| cortiq | 0.8.90 或更高：[预编译版本](https://github.com/infosave2007/cmf/releases)或 `cargo install` |

#### Vulkan · 安装

从[发布页](https://github.com/infosave2007/cmf/releases)下载预编译版本，或：

```bash
cargo install cortiq-cli --locked
```

在容器中还需安装 Vulkan 加载器：

```bash
{apt}
```

#### Vulkan · 运行

```bash
{vk_run.format(hl="仅用于无显示器的 Linux", lst="列出 Vulkan 能看到的显卡")}
```

服务器在 `/v1/models` 有响应即就绪；首次启动需把所有专家放到显卡上，约一分钟或更久。

#### Vulkan · 设置

| 变量 | 作用 |
|---|---|
| `CMF_GPU_ADAPTER=<n>` | 使用哪张显卡，取 `cortiq gpu` 中的编号 |
| `CMF_SERVE_SLOTS=1` | 一次一个请求，全速 |
| `CMF_QWEN_MTP=0` | 关闭草稿 |
| `XDG_RUNTIME_DIR=/tmp` | 无显示器的 Linux 上使用 Vulkan |

#### Vulkan · 速度

测试平台：一张 RTX PRO 6000 96 GB；cortiq 的 Vulkan 后端，上面的运行命令；测试方法与 CUDA 相同。

{tbl_rows(vk, m, "zh")}
![{name} · Vulkan]({img("vulkan", "zh")})

#### Vulkan · 常见问题

- `cortiq gpu` 没有列出显卡：在无显示器的 Linux 上设置 `XDG_RUNTIME_DIR=/tmp`；在容器中安装上面的软件包。
- **显存不足**：所有专家都必须放进显卡；关闭其他使用 GPU 的程序，或{oom["zh"]}。
- **多张显卡**：用 `CMF_GPU_ADAPTER` 选择其中一张。

### 使用服务器

两种后端用法相同。

```bash
{curl("用 Rust 写一个解析语义化版本号的函数，并附测试。")}
```

接口：`/v1/chat/completions`（流式、`tools`、`enable_thinking`）、`/v1/completions`、`/v1/models`、`/healthz`，`/` 为网页面板。
任何兼容 OpenAI 的编辑器或智能体都可连接 `http://127.0.0.1:8080/v1`。不启动服务器、单次提示（CUDA 需加 `CMF_GPU=cuda`）：

```bash
{run1}
```

- **回答以推理开头**：发送 `"enable_thinking": false`（`cortiq run` 用 `--no-think`）。
- **网络**：服务器不校验密钥；除非网络可信，请保留 `--host 127.0.0.1`。

### 准确率

HumanEval pass@1：**{he:.1f} %**（164 题，贪心解码，关闭思考，本文件在 cortiq 服务器上运行）。Edge0 在智能体与推理基准上的结果见
[Edge0 的页面](https://huggingface.co/{src})。

### 文件

| 文件 | 内容 | 大小 |
|---|---|---:|
| `{fn}` | 每层 512 个专家，{D["latent"]} 维，按 Edge0 训练的 NVFP4；每层八组 D/U 投影、共享专家、注意力、嵌入与输出头 8 位；路由器 16 位；n-gram 表 8 位；草稿（MTP）头 | {size:.2f} GB |
| `{fn}.sha256` | 校验和 | |
| `LICENSE`、`NOTICE`、`third-party/Qwen-LICENSE.txt` | 适用的许可证 | |

### 许可证与修改

{name} 是 Edge0 的模型，适用 [Edge0 Community Model License 1.0](LICENSE)，并基于适用 [Qwen Community License 1.0](third-party/Qwen-LICENSE.txt)
的 Qwen3.8-Flash-Next；[NOTICE](NOTICE) 说明两者各自覆盖的部分，两者都适用于本文件。本页面并非由 Edge0 或 Qwen 制作或认可。超出许可证
范围的商业使用：Edge0（info@edge0.ai）及 Qwen 自身的条款。

infosave（cortiq）于 2026-11-13 基于 [{src}](https://huggingface.co/{src}) 修订版 {D["rev"][:8]} 所做的修改：路由专家逐位复制为 CMF 的
`nvfp4`；D/U 投影与共享专家由 Q8_0 重新编码为 q8_2f；路由器以 f16 存储（来自 bf16）；n-gram 表由 FP8 重新编码为 q8_2f；注意力、GDN、
hyper-connections、嵌入、输出头与草稿头取自我们 CMF 中 Qwen3.8-Flash-Next 自己的权重（与 Edge0 的权重在 Q8_0 舍入范围内一致；草稿头
的专家为 4 位）。
"""
    front = f"""---
license: other
license_name: edge0-community-model-1.0
license_link: LICENSE
library_name: cortiq
base_model: {src}
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

"""
    return front + en + "\n---\n\n" + ru + "\n---\n\n" + zh


def main():
    data = json.load(open(sys.argv[1]))
    out = sys.argv[2]
    for m in ("m145", "m120"):
        open(f"{out}/{NAME[m]}-cmf.md", "w").write(card(data, m))


if __name__ == "__main__":
    main()
