---
license: mit
library_name: cortiq
base_model: microsoft/Aion-1.0-Instruct-2.3B
base_model_relation: quantized
pipeline_tag: text-generation
tags:
  - cmf
  - cortiq
  - quantized
  - q4tp
  - long-context
  - low-vram
  - cuda
  - vulkan
language:
  - en
  - de
  - es
  - pt
  - it
  - fr
  - zh
  - ja
---

# Aion-1.0-Instruct-2.3B Turbo — CMF

The 4-bit weights of [Aion-1.0-Instruct-2.3B](https://huggingface.co/microsoft/Aion-1.0-Instruct-2.3B) under a
header that holds the card's memory and the decode speed in place as the prompt grows.

- **NVIDIA L4 (24 GB), Vulkan:** the card holds 3.5 GB from an 8 192-token prompt to a 30 000-token one
  (4.6 GB for the standard 4-bit file at 30 000) and decodes at 80.3–80.6 tok/s against 52.6; the first token of
  30 000 tokens comes in 21.0 s against 27.3.
- **NVIDIA L4, CUDA:** decode holds at 89.0–90.7 tok/s from 8 192 to 30 000 tokens (106.3 → 79.7 for the standard
  file); with `CMF_O1_WHOLE=0` a 30 000-token prompt takes 2.8 GB of the card and decodes at 112.4 tok/s.
- **MacBook Air M4, Metal:** 16 384 tokens in 0.83 GB of keys and values against 1.62 GB, decode 44.7 tok/s against 42.2.
- **Planted facts** in long Wikipedia text, 12 prompts a length: on CUDA 12/12 at about 13 000 tokens and 12/12 at
  about 27 000 (Transformers bf16: 12/12 and 11/12); on Vulkan's fixed-memory reading 9/12 and 7/12.

It is one [CMF](https://github.com/infosave2007/cmf) file for `cortiq`, a Rust inference engine with no Python and
no ML framework. Prompts under 8192 tokens run exact attention, as in the
[standard files](https://huggingface.co/infosave/Aion-1.0-Instruct-2.3B-cmf).

[English](#quick-start) · [Русский](#документация-на-русском) · [中文](#中文文档)

## Quick start

```bash
cargo install cortiq-cli
hf download infosave/Aion-1.0-Instruct-2.3B-Turbo-cmf Aion-1.0-Instruct-2.3B-turbo.cmf --local-dir .
cortiq run Aion-1.0-Instruct-2.3B-turbo.cmf --prompt-file report.txt --greedy
```

With CUDA: `cargo install cortiq-cli --features cuda`, then `CMF_GPU=cuda cortiq run …`. Use cortiq 0.9.10 or
later; prebuilt binaries are on the [releases page](https://github.com/infosave2007/cmf/releases).

## Server

`cortiq serve` puts the file behind an OpenAI-compatible HTTP API, with a web dashboard on the same port.

```bash
CMF_GPU=cuda cortiq serve Aion-1.0-Instruct-2.3B-turbo.cmf --port 8080 --host 127.0.0.1   # NVIDIA through CUDA
cortiq serve Aion-1.0-Instruct-2.3B-turbo.cmf --port 8080 --host 127.0.0.1                # Vulkan, DX12 or Metal
```

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "aion",
  "messages": [{"role": "user", "content": "Summarize the report below in five bullet points.\n\n..."}],
  "max_tokens": 512, "temperature": 0, "stream": true
}'
```

Endpoints: `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/healthz`. Tools go in `tools` in the OpenAI
format and come back as `message.tool_calls`. `--host 127.0.0.1` keeps the server local; no API key is checked.

## What the file does

**A fixed ceiling.** 21 of Aion's 28 layers attend over a 2048-token window and keep only those rows on the card.
The other 7 attend over the whole prompt. In this file, once a prompt reaches 8192 tokens, those 7 layers keep the
first 4 rows and the last 1800 exact and fold the rest into 32 landmarks per head, a far field of fixed size.
Card memory and the cost of a token stop growing with the text. Patent application No. 19/738,763.

**Exact rows.** A far field alone loses a password. Each row is scored by what the layers do with it (for each of
the 7 layers' 8 KV heads: how far the key lies from the span of the landmarks, and the value's norm; a linear fit
over the 112 ranks, made on planted facts in Wikipedia: left-out prompts rank the fact's rows within the top
0.02 %), and the 248 best stay exact. The question at the end of the prompt reads them before it is answered.

## Performance

`cortiq bench --core`, cortiq 0.9.9, 64 tokens after a prompt of that length; card memory is the peak
`nvidia-smi` shows during the run. The standard file is `Aion-1.0-Instruct-2.3B-q4tp.cmf`, the same weights. Both
backends ran on the same card; read each section on its own.

### CUDA · NVIDIA L4

Stand: NVIDIA L4 (24 GB), Google Colab, the CUDA backend (`CMF_GPU=cuda`).

![Standard and Turbo on CUDA](https://huggingface.co/infosave/Aion-1.0-Instruct-2.3B-Turbo-cmf/resolve/main/assets/turbo_cuda_en.png)

| prompt, tokens | first token, s (standard) | first token, s (**Turbo**) | decode, tok/s (standard) | decode, tok/s (**Turbo**) | peak card memory, GB (standard) | peak card memory, GB (**Turbo**) |
|---:|---:|---:|---:|---:|---:|---:|
| 2 048 | 0.32 | **0.39** | 116.9 | **117.3** | 2.2 | **2.2** |
| 4 096 | 0.74 | **0.80** | 112.4 | **112.6** | 2.3 | **2.3** |
| 8 192 | 1.72 | **2.80** | 106.3 | **90.7** | 2.5 | **2.6** |
| 12 288 | 2.86 | **4.44** | 100.1 | **90.0** | 2.7 | **2.8** |
| 16 384 | 4.15 | **6.30** | 95.7 | **89.9** | 2.9 | **3.0** |
| 24 576 | 7.30 | **10.2** | 86.3 | **89.7** | 3.3 | **3.4** |
| 30 000 | 9.92 | **13.2** | 79.7 | **89.0** | 3.5 | **3.6** |

On CUDA the prompt is read whole and exact, so the card's memory grows as the standard file's does; decode holds at
89.0–90.7 tok/s from 8 192 tokens on and is ahead of the standard file from 24 576 tokens. `CMF_O1_WHOLE=0`
reads the prompt in chunks instead: 30 000 tokens in 2.8 GB, the first token in 11.5 s, then 112.4 tok/s.

### Vulkan · NVIDIA L4

Stand: the same NVIDIA L4 (24 GB), over Vulkan (the default backend).

![Standard and Turbo over Vulkan](https://huggingface.co/infosave/Aion-1.0-Instruct-2.3B-Turbo-cmf/resolve/main/assets/turbo_vulkan_en.png)

| prompt, tokens | first token, s (standard) | first token, s (**Turbo**) | decode, tok/s (standard) | decode, tok/s (**Turbo**) | peak card memory, GB (standard) | peak card memory, GB (**Turbo**) |
|---:|---:|---:|---:|---:|---:|---:|
| 2 048 | 0.62 | **0.84** | 82.8 | **82.4** | 3.1 | **3.0** |
| 4 096 | 1.54 | **1.73** | 79.8 | **79.8** | 3.2 | **3.2** |
| 8 192 | 3.83 | **5.71** | 73.7 | **80.4** | 3.4 | **3.5** |
| 12 288 | 6.78 | **9.75** | 68.2 | **80.5** | 3.6 | **3.5** |
| 16 384 | 10.3 | **12.2** | 63.5 | **80.5** | 3.9 | **3.5** |
| 24 576 | 19.7 | **17.8** | 56.9 | **80.6** | 4.3 | **3.5** |
| 30 000 | 27.3 | **21.0** | 52.6 | **80.3** | 4.6 | **3.5** |

Over Vulkan the card holds 3.5 GB from 8 192 tokens on and decode holds at 80.3–80.6 tok/s; at 30 000 tokens the
standard file takes 4.6 GB and decodes at 52.6 tok/s. `CMF_O1_WHOLE=1` reads the prompt whole: 30 000 tokens in
4.5 GB, the first token in 34.1 s, then 64.8 tok/s.

### Metal · MacBook Air M4

Stand: MacBook Air M4 (24 GB), fanless, Metal.

| prompt, tokens | file | first token | decode, tok/s | keys and values |
|---|---|---:|---:|---:|
| 16 384 | Turbo | 43 s | 44.7 | 0.83 GB |
| 16 384 | standard `q4tp` | 44 s | 42.2 | 1.62 GB |

### Two ways to read a long prompt

On Vulkan the prompt past 8192 tokens is read through the far field, chunk by chunk, and the rows leave
the card as they are read: memory is fixed, and what those rows compute is an approximation. On CUDA the whole
prompt is read exact and the far field is built once before its last 512 rows: memory grows with the prompt as the
standard file's does, the answer starts from exact rows. `CMF_O1_WHOLE=1` reads the prompt whole on Vulkan too
(the 30 000-token row above), and `CMF_O1_WHOLE=0` reads it in chunks on CUDA: 2.6–2.8 GB from 16 000 to 30 000
tokens and 112 tok/s, with fewer facts kept (3/4 and 2/4 on the first four prompts of each length).

## Quality

Perplexity is the 4-bit file's below 8192 tokens (WikiText-2 14.61, Rust 7.83, Python 7.43; bf16 14.24 / 7.66 /
7.26). Past the seal the answer depends on what the far field and the exact rows keep: planted facts in long Wikipedia text (four kinds: a pass key, an access code, a password, a
room number), greedy:

| prompt | Transformers bf16 | standard `q4tp` | Turbo, CUDA | Turbo, Vulkan | Turbo, Vulkan, `CMF_O1_WHOLE=1` |
|---|---:|---:|---:|---:|---:|
| about 13 000 tokens | 12/12 | 12/12 | 12/12 | 9/12 | 12/12 |
| about 27 000 tokens | 11/12 | 11/12 | 12/12 | 7/12 | 12/12 |

Twelve prompts a length; the standard file scores the same on CUDA and Vulkan. Read whole, the Turbo file keeps
the facts as the exact files do; the fixed-memory reading on Vulkan trades some of them for the flat 3.5 GB.

## Verify the download

| file | sha256 |
|---|---|
| `Aion-1.0-Instruct-2.3B-turbo.cmf` | `fddd9d0c18423f76c96e31e875eb54d46a0bec1ffe81d3fe93ade44752971015` |
| `keep_aion.bin` (the exact-row weights, already inside the file) | `b1ef2fb742c664f115931e3106e3ef525fea16063fa7c669ce4fb8b12acaf529` |

Weights: [MIT](https://huggingface.co/microsoft/Aion-1.0-Instruct-2.3B/blob/main/LICENSE), from
[microsoft/Aion-1.0-Instruct-2.3B](https://huggingface.co/microsoft/Aion-1.0-Instruct-2.3B).

---

## Документация на русском

4-битные веса [Aion-1.0-Instruct-2.3B](https://huggingface.co/microsoft/Aion-1.0-Instruct-2.3B) с заголовком,
который держит память карты и скорость генерации на месте по мере роста промпта.

- **NVIDIA L4 (24 ГБ), Vulkan:** карта держит 3,5 ГБ от промпта в 8 192 токена до 30 000 (4,6 ГБ у
  стандартного 4-битного файла на 30 000), генерация 80,3–80,6 ток/с против 52,6; первый токен промпта из 30 000
  токенов — за 21,0 с против 27,3.
- **NVIDIA L4, CUDA:** генерация держится на 89,0–90,7 ток/с от 8 192 до 30 000 токенов (у стандартного файла
  106,3 → 79,7); с `CMF_O1_WHOLE=0` промпт из 30 000 токенов занимает 2,8 ГБ карты, генерация 112,4 ток/с.
- **MacBook Air M4, Metal:** 16 384 токена в 0,83 ГБ ключей и значений против 1,62 ГБ, генерация 44,7 ток/с против 42,2.
- **Спрятанные факты** в длинном тексте Википедии, по 12 промптов на длину: на CUDA 12/12 примерно на 13 000
  токенов и 12/12 примерно на 27 000 (Transformers bf16 — 12/12 и 11/12); на Vulkan при чтении с фиксированной
  памятью — 9/12 и 7/12.

Один файл [CMF](https://github.com/infosave2007/cmf) для `cortiq` — движка на Rust без Python и ML-фреймворков.
Промпты короче 8192 токенов идут точным вниманием, как в
[стандартных файлах](https://huggingface.co/infosave/Aion-1.0-Instruct-2.3B-cmf).

### Быстрый старт

```bash
cargo install cortiq-cli
hf download infosave/Aion-1.0-Instruct-2.3B-Turbo-cmf Aion-1.0-Instruct-2.3B-turbo.cmf --local-dir .
cortiq run Aion-1.0-Instruct-2.3B-turbo.cmf --prompt-file report.txt --greedy
```

С CUDA: установка с `--features cuda`, запуск с `CMF_GPU=cuda`. Нужен cortiq 0.9.10 или новее; готовые сборки — на
[странице релизов](https://github.com/infosave2007/cmf/releases).

### Сервер

`cortiq serve` поднимает файл за HTTP API, совместимым с OpenAI, с веб-панелью на том же порту.

```bash
CMF_GPU=cuda cortiq serve Aion-1.0-Instruct-2.3B-turbo.cmf --port 8080 --host 127.0.0.1   # NVIDIA через CUDA
cortiq serve Aion-1.0-Instruct-2.3B-turbo.cmf --port 8080 --host 127.0.0.1                # Vulkan, DX12 или Metal
```

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "aion",
  "messages": [{"role": "user", "content": "Изложи отчёт ниже в пяти пунктах.\n\n..."}],
  "max_tokens": 512, "temperature": 0, "stream": true
}'
```

Адреса: `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/healthz`. Инструменты передаются в `tools` в
формате OpenAI и возвращаются как `message.tool_calls`. `--host 127.0.0.1` оставляет сервер локальным; ключ API не
проверяется.

### Что делает файл

**Фиксированный потолок.** 21 из 28 слоёв Aion смотрит в окно из 2048 токенов и держит на карте только эти строки.
Остальные 7 смотрят на весь промпт. В этом файле, как только промпт доходит до 8192 токенов, эти 7 слоёв держат
точными первые 4 строки и последние 1800, а остальное сворачивают в 32 опорные точки на голову — дальнее поле
фиксированного размера. Память карты и цена токена перестают расти с текстом. Заявка на патент № 19/738,763.

**Точные строки.** Одно дальнее поле теряет пароль. Каждая строка получает оценку по тому, что с ней делают слои
(для каждой из 8 KV-голов 7 слоёв: как далеко ключ от линейной оболочки опорных точек и норма значения; линейная
модель над 112 рангами, подобранная на спрятанных фактах в Википедии: на отложенных промптах строки факта попадают
в верхние 0,02 %), и 248 лучших остаются точными. Вопрос в конце промпта читает их перед ответом.

### Скорость

`cortiq bench --core`, cortiq 0.9.9, 64 токена после промпта такой длины; память карты — пик по `nvidia-smi` за
прогон. Стандартный файл — `Aion-1.0-Instruct-2.3B-q4tp.cmf`, те же веса. Оба бэкенда измерены на одной карте;
каждый раздел читается сам по себе.

**CUDA · NVIDIA L4**

Стенд: NVIDIA L4 (24 ГБ), Google Colab, бэкенд CUDA (`CMF_GPU=cuda`).

![Стандартный файл и Turbo на CUDA](https://huggingface.co/infosave/Aion-1.0-Instruct-2.3B-Turbo-cmf/resolve/main/assets/turbo_cuda_ru.png)

| промпт, токенов | первый токен, с (стандартный) | первый токен, с (**Turbo**) | генерация, ток/с (стандартный) | генерация, ток/с (**Turbo**) | пик памяти карты, ГБ (стандартный) | пик памяти карты, ГБ (**Turbo**) |
|---:|---:|---:|---:|---:|---:|---:|
| 2 048 | 0,32 | **0,39** | 116,9 | **117,3** | 2,2 | **2,2** |
| 4 096 | 0,74 | **0,80** | 112,4 | **112,6** | 2,3 | **2,3** |
| 8 192 | 1,72 | **2,80** | 106,3 | **90,7** | 2,5 | **2,6** |
| 12 288 | 2,86 | **4,44** | 100,1 | **90,0** | 2,7 | **2,8** |
| 16 384 | 4,15 | **6,30** | 95,7 | **89,9** | 2,9 | **3,0** |
| 24 576 | 7,30 | **10,2** | 86,3 | **89,7** | 3,3 | **3,4** |
| 30 000 | 9,92 | **13,2** | 79,7 | **89,0** | 3,5 | **3,6** |

На CUDA промпт читается целиком и точно, поэтому память карты растёт, как у стандартного файла; генерация
держится на 89,0–90,7 ток/с начиная с 8 192 токенов и обгоняет стандартный файл с 24 576 токенов. `CMF_O1_WHOLE=0`
читает промпт кусками: 30 000 токенов в 2,8 ГБ, первый токен за 11,5 с, затем 112,4 ток/с.

**Vulkan · NVIDIA L4**

Стенд: та же NVIDIA L4 (24 ГБ), через Vulkan (бэкенд по умолчанию).

![Стандартный файл и Turbo на Vulkan](https://huggingface.co/infosave/Aion-1.0-Instruct-2.3B-Turbo-cmf/resolve/main/assets/turbo_vulkan_ru.png)

| промпт, токенов | первый токен, с (стандартный) | первый токен, с (**Turbo**) | генерация, ток/с (стандартный) | генерация, ток/с (**Turbo**) | пик памяти карты, ГБ (стандартный) | пик памяти карты, ГБ (**Turbo**) |
|---:|---:|---:|---:|---:|---:|---:|
| 2 048 | 0,62 | **0,84** | 82,8 | **82,4** | 3,1 | **3,0** |
| 4 096 | 1,54 | **1,73** | 79,8 | **79,8** | 3,2 | **3,2** |
| 8 192 | 3,83 | **5,71** | 73,7 | **80,4** | 3,4 | **3,5** |
| 12 288 | 6,78 | **9,75** | 68,2 | **80,5** | 3,6 | **3,5** |
| 16 384 | 10,3 | **12,2** | 63,5 | **80,5** | 3,9 | **3,5** |
| 24 576 | 19,7 | **17,8** | 56,9 | **80,6** | 4,3 | **3,5** |
| 30 000 | 27,3 | **21,0** | 52,6 | **80,3** | 4,6 | **3,5** |

На Vulkan карта держит 3,5 ГБ начиная с 8 192 токенов, генерация — 80,3–80,6 ток/с; на 30 000 токенов
стандартный файл занимает 4,6 ГБ и пишет 52,6 ток/с. `CMF_O1_WHOLE=1` читает промпт целиком: 30 000 токенов в
4,5 ГБ, первый токен за 34,1 с, затем 64,8 ток/с.

**Metal · MacBook Air M4**

Стенд: MacBook Air M4 (24 ГБ), без вентилятора, Metal.

| промпт, токенов | файл | первый токен | генерация, ток/с | ключи и значения |
|---|---|---:|---:|---:|
| 16 384 | Turbo | 43 с | 44,7 | 0,83 ГБ |
| 16 384 | стандартный `q4tp` | 44 с | 42,2 | 1,62 ГБ |

**Два способа читать длинный промпт.** На Vulkan промпт после 8192 токенов читается через дальнее поле,
кусками, и строки уходят с карты по мере чтения: память фиксирована, а вычисленное по этим строкам — приближение.
На CUDA весь промпт читается точно, и дальнее поле строится один раз перед последними 512 строками: память растёт
с промптом, как у стандартного файла, зато ответ опирается на точные строки. `CMF_O1_WHOLE=1` читает промпт целиком
и на Vulkan (строка на 30 000 токенов выше), а `CMF_O1_WHOLE=0` читает его кусками на CUDA: 2,6–2,8 ГБ от 16 000 до
30 000 токенов и 112 ток/с, фактов удерживается меньше (3/4 и 2/4 на первых четырёх промптах каждой длины).

### Качество

До 8192 токенов перплексия — как у 4-битного файла (WikiText-2 14,61, Rust 7,83, Python 7,43; bf16 14,24 / 7,66 /
7,26). Дальше ответ зависит от того, что сохранят дальнее поле и точные строки. Спрятанные факты (ключ, код
доступа, пароль, номер комнаты), жадно:

| промпт | Transformers bf16 | стандартный `q4tp` | Turbo, CUDA | Turbo, Vulkan | Turbo, Vulkan, `CMF_O1_WHOLE=1` |
|---|---:|---:|---:|---:|---:|
| около 13 000 токенов | 12/12 | 12/12 | 12/12 | 9/12 | 12/12 |
| около 27 000 токенов | 11/12 | 11/12 | 12/12 | 7/12 | 12/12 |

По двенадцать промптов на длину; у стандартного файла результат одинаков на CUDA и Vulkan. При чтении целиком
Turbo-файл удерживает факты так же, как точные файлы; чтение с фиксированной памятью на Vulkan меняет часть из них
на неизменные 3,5 ГБ.

Суммы sha256 — в таблице английского раздела. Веса:
[MIT](https://huggingface.co/microsoft/Aion-1.0-Instruct-2.3B/blob/main/LICENSE), от
[microsoft/Aion-1.0-Instruct-2.3B](https://huggingface.co/microsoft/Aion-1.0-Instruct-2.3B).

---

## 中文文档

[Aion-1.0-Instruct-2.3B](https://huggingface.co/microsoft/Aion-1.0-Instruct-2.3B) 的 4 位权重，配合一个让显存和
生成速度不随提示词增长而变化的文件头。

- **NVIDIA L4（24 GB），Vulkan：** 从 8 192 token 到 30 000 token 的提示词，显存保持 3.5 GB（标准 4 位文件在
  30 000 token 时为 4.6 GB），生成 80.3–80.6 tok/s（标准 52.6）；30 000 token 的首 token 为 21.0 秒（标准 27.3）。
- **NVIDIA L4，CUDA：** 从 8 192 到 30 000 token 生成保持 89.0–90.7 tok/s（标准文件 106.3 → 79.7）；设
  `CMF_O1_WHOLE=0` 时 30 000 token 的提示词占用 2.8 GB 显存，生成 112.4 tok/s。
- **MacBook Air M4，Metal：** 16 384 token 的键值占 0.83 GB（标准 1.62 GB），生成 44.7 tok/s（标准 42.2）。
- **埋入事实**（长篇维基百科文本，每个长度 12 个提示词）：CUDA 约 13 000 token 时 12/12，约 27 000 token 时 12/12
  （Transformers bf16 为 12/12 和 11/12）；Vulkan 固定显存读取为 9/12 和 7/12。

这是一个供 `cortiq` 使用的 [CMF](https://github.com/infosave2007/cmf) 文件（不依赖 Python 和机器学习框架的 Rust
推理引擎）。8192 token 以内的提示词使用精确注意力，与[标准文件](https://huggingface.co/infosave/Aion-1.0-Instruct-2.3B-cmf)
相同。

### 快速开始

```bash
cargo install cortiq-cli
hf download infosave/Aion-1.0-Instruct-2.3B-Turbo-cmf Aion-1.0-Instruct-2.3B-turbo.cmf --local-dir .
cortiq run Aion-1.0-Instruct-2.3B-turbo.cmf --prompt-file report.txt --greedy
```

CUDA：安装时加 `--features cuda`，运行时设 `CMF_GPU=cuda`。需要 cortiq 0.9.10 或更新版本；预编译二进制见[发布页面](https://github.com/infosave2007/cmf/releases)。

### 服务器

`cortiq serve` 把文件放在兼容 OpenAI 的 HTTP API 之后，同一端口还有网页面板。

```bash
CMF_GPU=cuda cortiq serve Aion-1.0-Instruct-2.3B-turbo.cmf --port 8080 --host 127.0.0.1   # NVIDIA，CUDA
cortiq serve Aion-1.0-Instruct-2.3B-turbo.cmf --port 8080 --host 127.0.0.1                # Vulkan、DX12 或 Metal
```

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "aion",
  "messages": [{"role": "user", "content": "用五个要点概括下面的报告。\n\n..."}],
  "max_tokens": 512, "temperature": 0, "stream": true
}'
```

接口：`/v1/chat/completions`、`/v1/completions`、`/v1/models`、`/healthz`。工具按 OpenAI 格式放入 `tools`，以
`message.tool_calls` 返回。`--host 127.0.0.1` 使服务器只在本机可用；不检查 API 密钥。

### 原理

**固定上限。** Aion 的 28 层中有 21 层只关注 2048 token 的窗口，显卡上只保留这些行；其余 7 层关注整个提示词。
在这个文件里，提示词达到 8192 token 后，这 7 层精确保留前 4 行和最后 1800 行，其余折叠为每个头 32 个地标点，
即固定大小的远场。显存和每个 token 的开销不再随文本增长。专利申请号 19/738,763。

**精确行。** 单靠远场会丢失密码。每一行按各层对它的处理打分（7 层 × 8 个 KV 头：键离地标张成空间的距离，以及值的
范数；在维基百科埋入事实上拟合的 112 个排名的线性模型，留出提示词中事实所在行排在前 0.02 %），最好的 248 行保持
精确。提示词末尾的问题在回答前读取它们。

### 性能

`cortiq bench --core`，cortiq 0.9.9，在该长度提示词之后生成 64 个 token；显存为运行期间 `nvidia-smi` 的峰值。
标准文件为 `Aion-1.0-Instruct-2.3B-q4tp.cmf`，权重相同。两个后端在同一块显卡上测量，各节请单独阅读。

**CUDA · NVIDIA L4**

测试平台：NVIDIA L4（24 GB），Google Colab，CUDA 后端（`CMF_GPU=cuda`）。

![CUDA 上的标准文件与 Turbo](https://huggingface.co/infosave/Aion-1.0-Instruct-2.3B-Turbo-cmf/resolve/main/assets/turbo_cuda_zh.png)

| 提示词，token 数 | 首 token，秒（标准） | 首 token，秒（**Turbo**） | 生成，tok/s（标准） | 生成，tok/s（**Turbo**） | 显存峰值，GB（标准） | 显存峰值，GB（**Turbo**） |
|---:|---:|---:|---:|---:|---:|---:|
| 2 048 | 0.32 | **0.39** | 116.9 | **117.3** | 2.2 | **2.2** |
| 4 096 | 0.74 | **0.80** | 112.4 | **112.6** | 2.3 | **2.3** |
| 8 192 | 1.72 | **2.80** | 106.3 | **90.7** | 2.5 | **2.6** |
| 12 288 | 2.86 | **4.44** | 100.1 | **90.0** | 2.7 | **2.8** |
| 16 384 | 4.15 | **6.30** | 95.7 | **89.9** | 2.9 | **3.0** |
| 24 576 | 7.30 | **10.2** | 86.3 | **89.7** | 3.3 | **3.4** |
| 30 000 | 9.92 | **13.2** | 79.7 | **89.0** | 3.5 | **3.6** |

在 CUDA 上提示词被整体精确读取，因此显存与标准文件一样增长；从 8 192 token 起生成保持 89.0–90.7 tok/s，
从 24 576 token 起快于标准文件。`CMF_O1_WHOLE=0` 改为分块读取：30 000 token 占 2.8 GB，首 token 11.5 秒，之后
112.4 tok/s。

**Vulkan · NVIDIA L4**

测试平台：同一块 NVIDIA L4（24 GB），通过 Vulkan（默认后端）。

![Vulkan 上的标准文件与 Turbo](https://huggingface.co/infosave/Aion-1.0-Instruct-2.3B-Turbo-cmf/resolve/main/assets/turbo_vulkan_zh.png)

| 提示词，token 数 | 首 token，秒（标准） | 首 token，秒（**Turbo**） | 生成，tok/s（标准） | 生成，tok/s（**Turbo**） | 显存峰值，GB（标准） | 显存峰值，GB（**Turbo**） |
|---:|---:|---:|---:|---:|---:|---:|
| 2 048 | 0.62 | **0.84** | 82.8 | **82.4** | 3.1 | **3.0** |
| 4 096 | 1.54 | **1.73** | 79.8 | **79.8** | 3.2 | **3.2** |
| 8 192 | 3.83 | **5.71** | 73.7 | **80.4** | 3.4 | **3.5** |
| 12 288 | 6.78 | **9.75** | 68.2 | **80.5** | 3.6 | **3.5** |
| 16 384 | 10.3 | **12.2** | 63.5 | **80.5** | 3.9 | **3.5** |
| 24 576 | 19.7 | **17.8** | 56.9 | **80.6** | 4.3 | **3.5** |
| 30 000 | 27.3 | **21.0** | 52.6 | **80.3** | 4.6 | **3.5** |

在 Vulkan 上，从 8 192 token 起显存保持 3.5 GB，生成保持 80.3–80.6 tok/s；30 000 token 时标准文件占用
4.6 GB，生成 52.6 tok/s。`CMF_O1_WHOLE=1` 整体读取提示词：30 000 token 占 4.5 GB，首 token 34.1 秒，之后 64.8 tok/s。

**Metal · MacBook Air M4**

测试平台：MacBook Air M4（24 GB），无风扇，Metal。

| 提示词，token 数 | 文件 | 首 token | 生成，tok/s | 键值 |
|---|---|---:|---:|---:|
| 16 384 | Turbo | 43 s | 44.7 | 0.83 GB |
| 16 384 | 标准 `q4tp` | 44 s | 42.2 | 1.62 GB |

**读取长提示词的两种方式。** 在 Vulkan 上，8192 token 之后的提示词经远场分块读取，读完的行即离开显卡：
显存固定，而这些行算出的结果是近似值。在 CUDA 上，整个提示词被精确读取，远场在最后 512 行之前一次建成：显存像
标准文件一样随提示词增长，但回答基于精确的行。`CMF_O1_WHOLE=1` 让 Vulkan 也整体读取（见上表 30 000 token 一行）；
`CMF_O1_WHOLE=0` 让 CUDA 分块读取：16 000 到 30 000 token 显存 2.6–2.8 GB、生成 112 tok/s，保留的事实较少
（每个长度前四个提示词为 3/4 和 2/4）。

### 质量

8192 token 以内的困惑度与 4 位文件相同（WikiText-2 14.61，Rust 7.83，Python 7.43；bf16 14.24 / 7.66 / 7.26）。之后
回答取决于远场和精确行保留了什么。埋入事实（通行码、访问码、密码、房间号），贪心：

| 提示词 | Transformers bf16 | 标准 `q4tp` | Turbo，CUDA | Turbo，Vulkan | Turbo，Vulkan，`CMF_O1_WHOLE=1` |
|---|---:|---:|---:|---:|---:|
| 约 13 000 token | 12/12 | 12/12 | 12/12 | 9/12 | 12/12 |
| 约 27 000 token | 11/12 | 11/12 | 12/12 | 7/12 | 12/12 |

每个长度 12 个提示词；标准文件在 CUDA 和 Vulkan 上得分相同。整体读取时 Turbo 文件与精确文件一样保留事实；Vulkan 上
的固定显存读取以其中一部分换取恒定的 3.5 GB。

sha256 见英文部分的表格。权重：[MIT](https://huggingface.co/microsoft/Aion-1.0-Instruct-2.3B/blob/main/LICENSE)，
来自 [microsoft/Aion-1.0-Instruct-2.3B](https://huggingface.co/microsoft/Aion-1.0-Instruct-2.3B)。
