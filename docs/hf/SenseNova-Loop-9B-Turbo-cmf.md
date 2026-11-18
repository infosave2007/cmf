---
license: apache-2.0
library_name: cortiq
base_model: sensenova/SenseNova-Loop-9B
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
  - cmf
  - cortiq
  - quantized
  - q4tp
  - looped-transformer
  - mhc
  - long-context
  - low-vram
  - cuda
  - vulkan
language:
  - en
  - ru
  - zh
---

# SenseNova-Loop-9B Turbo — CMF

The 4-bit weights of [SenseNova-Loop-9B](https://huggingface.co/sensenova/SenseNova-Loop-9B) under a header that
holds the card's memory in place on long prompts. SenseNova-Loop runs 24 of its layers twice, and each pass keeps
its own keys and values, so a long prompt costs the standard file twice the memory of a stack without the loop.

- **RTX A5000 (24 GB), CUDA, a 128 000-token prompt:** 8.3 GB of the card against 16.5 GB for the standard file,
  the first token in 140 s (142), then 49.9 tok/s against 30.5.
- **250 000 tokens on the same card:** 8.3 GB, the first token in 265 s, then 50.4 tok/s; the prompt's rows wait in
  RAM or, with `--o1-spill`, in files.
- Facts planted in 60 000, 120 000 and 240 000 characters of Wikipedia (up to 56 000 tokens) come back 4 times of 4
  at each length through CUDA, as from the standard file.

It is one [CMF](https://github.com/infosave2007/cmf) file for `cortiq`, a Rust inference engine with no Python and
no ML framework. Prompts under 8192 tokens run exact attention and the draft tokens, as in the
[standard file](https://huggingface.co/infosave/SenseNova-Loop-9B-cmf).

[English](#quick-start) · [Русский](#документация-на-русском) · [中文](#中文说明)

## Quick start

```bash
cargo install --git https://github.com/infosave2007/cmf --branch sensenova-loop9b cortiq-cli --features cuda
hf download infosave/SenseNova-Loop-9B-Turbo-cmf sensenova-loop-9b-turbo.cmf --local-dir .
CMF_GPU=cuda CMF_MAX_SEQ=262144 cortiq run sensenova-loop-9b-turbo.cmf --prompt-file report.txt --greedy --no-think
```

Without CUDA (Vulkan, DX12, Metal), and with the rows of a long prompt in files instead of RAM:

```bash
cargo install --git https://github.com/infosave2007/cmf --branch sensenova-loop9b cortiq-cli
CMF_MAX_SEQ=262144 cortiq run sensenova-loop-9b-turbo.cmf --prompt-file report.txt --greedy --no-think --o1-spill /mnt/ssd/cortiq
```

The file needs the cortiq release after 0.9.4; until it is out, build from the branch above. `CMF_MAX_SEQ` lifts
the default cap of 32 768 tokens (the model holds 524 288). `--greedy` turns on the draft tokens, `--no-think` asks
for a direct answer.

## Server

`cortiq serve` puts the file behind an OpenAI-compatible HTTP API, with a web dashboard on the same port.

An NVIDIA card through CUDA:

```bash
CMF_GPU=cuda CMF_MAX_SEQ=262144 cortiq serve sensenova-loop-9b-turbo.cmf --port 8080 --host 127.0.0.1
```

Vulkan, DX12 or Metal:

```bash
CMF_MAX_SEQ=262144 cortiq serve sensenova-loop-9b-turbo.cmf --port 8080 --host 127.0.0.1
```

A request:

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "sensenova-loop-9b",
  "messages": [{"role": "user", "content": "Write a Python function that merges two sorted lists."}],
  "max_tokens": 512,
  "temperature": 0,
  "enable_thinking": false,
  "stream": true
}'
```

An image goes in as an `image_url` part (a data URL or an HTTP(S) URL):

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "sensenova-loop-9b",
  "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "https://example.com/chart.png"}},
    {"type": "text", "text": "Read the values of the chart."}]}],
  "max_tokens": 256, "temperature": 0, "enable_thinking": false
}'
```

Any OpenAI client works with `base_url="http://127.0.0.1:8080/v1"` and any API key:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="none")
r = client.chat.completions.create(
    model="sensenova-loop-9b",
    messages=[{"role": "user", "content": open("report.txt").read() + "\n\nSummarize it."}],
    temperature=0,
    extra_body={"enable_thinking": False},
)
print(r.choices[0].message.content)
```

| endpoint | what it does |
|---|---|
| `POST /v1/chat/completions` | a chat turn; `stream: true` sends the answer as it is decoded, `tools` lets the model call functions |
| `POST /v1/completions` | the continuation of a `prompt` |
| `GET /v1/models` | the served model |
| `GET /healthz` | 200 while the server is up |
| `GET /` | the web dashboard |

- `CMF_MAX_SEQ` is the longest conversation a request may hold, 32 768 tokens by default.
- `temperature: 0` turns on the draft tokens, as `--greedy` does for `cortiq run`; `enable_thinking: false` asks
  for a direct answer. A request with an image decodes without drafts.
- `model` may be any name: one file is served.
- `--host 127.0.0.1` keeps the server local; the default is every interface, and no API key is checked.
- A turn that continues the previous one reads only what is new.


## What the file does

**A fixed ceiling.** SenseNova-Loop has 8 full-attention layers, and 6 of them run in both passes of the loop, each
pass with its own keys and values: 14 of them a token. The first 8192 rows of a prompt are read exactly. The keys and
values of later rows leave the card as they are read, for RAM or for files. Before the prompt's last 512 rows,
each of the 14 is rebuilt as a far field: the first 4 rows and the last 1800 stay exact, the rest become 32
landmarks per head. Card memory and the cost of a token stop growing with the text. Patent application
No. 19/738,763.

**Exact rows.** A far field alone loses a password. Each row is scored by what the layers do with it (for each KV
head of the 14: how far the key lies from the span of the landmarks, and the value's norm; a linear fit over the 112
ranks, made on planted facts in Wikipedia), and the 248 best stay exact. The question at the end of the prompt reads
them before it is answered.

**Drafts.** The model's MTP head drafts tokens and the card checks them in one batch, as in the standard file.

## Performance

The two backends were measured on the same card with the same files, but read each section on its own.

### CUDA · RTX A5000

Stand: NVIDIA RTX A5000 (24 GB), the CUDA backend (`CMF_GPU=cuda`), a release build of the branch above.
`cortiq bench --core` without drafts, 128 tokens after a prompt of that length; the standard file is
`sensenova-loop-9b-q4tp.cmf`, the same weights.

![Standard and turbo on CUDA](https://huggingface.co/infosave/SenseNova-Loop-9B-Turbo-cmf/resolve/main/assets/turbo_cuda_en.png)

| prompt, tokens | first token, s (standard) | first token, s (**turbo**) | decode, tok/s (standard) | decode, tok/s (**turbo**) | peak card memory, GB (standard) | peak card memory, GB (**turbo**) |
|---:|---:|---:|---:|---:|---:|---:|
| 16 000 | 10.1 | **16.5** | 47.1 | **50.8** | 8.9 | **8.3** |
| 32 000 | 22.6 | **34.4** | 43.5 | **50.4** | 10.0 | **8.3** |
| 64 000 | 53.0 | **70.3** | 38.1 | **50.5** | 12.2 | **8.3** |
| 128 000 | 141.9 | **139.9** | 30.5 | **49.9** | 16.5 | **8.3** |
| 250 000 | — | **264.6** | — | **50.4** | — | **8.3** |

The standard file was not run past 128 000 tokens. At 250 000 the rows went to files (`--o1-spill`, 27 GB on the
disk until the far field was built). Below 128 000 tokens the turbo file pays for its far field in the first token.

Planted facts (a pass key, an access code, a password, a room number; four prompts each, the answer read from the
reply), CUDA:

| text | tokens | standard | **turbo** |
|---|---:|---:|---:|
| 60 000 characters | 13 300–15 100 | 4 of 4 | **4 of 4** |
| 120 000 characters | 26 900–28 400 | 4 of 4 | **4 of 4** |
| 240 000 characters | 54 400–56 200 | 4 of 4 | **4 of 4** |

### Vulkan · RTX A5000

Stand: the same NVIDIA RTX A5000 (24 GB), over Vulkan, a release build of the branch above. The same bench.

![Standard and turbo over Vulkan](https://huggingface.co/infosave/SenseNova-Loop-9B-Turbo-cmf/resolve/main/assets/turbo_vulkan_en.png)

| prompt, tokens | first token, s (standard) | first token, s (**turbo**) | decode, tok/s (standard) | decode, tok/s (**turbo**) | peak card memory, GB (standard) | peak card memory, GB (**turbo**) |
|---:|---:|---:|---:|---:|---:|---:|
| 16 000 | 11.9 | **38.7** | 31.3 | **33.1** | 6.8 | **5.7** |
| 32 000 | 25.5 | **67.9** | 28.8 | **33.1** | 8.5 | **5.7** |
| 64 000 | 60.2 | **155.7** | 24.3 | **33.1** | 12.1 | **5.7** |
| 128 000 | 163.3 | **318.7** | 19.3 | **33.3** | 19.4 | **5.8** |
| 250 000 | — | **602.4** | — | **33.5** | — | **5.8** |

Over Vulkan the turbo file holds 5.7 GB of the card and 33 tok/s at any length, but it reads a prompt past the seal
at about 410 tokens a second, against 780–1 340 for the standard file: the first token comes 2–3 times later. On
an NVIDIA card take CUDA for long prompts.

Planted facts over Vulkan: the same 12 prompts, 4 of 4 at 60 000, 120 000 and 240 000 characters each (the standard file 4 of 4
at 60 000 and 120 000).


## Options

| option | meaning |
|---|---|
| `--o1-spill DIR` | a long prompt's rows in files in `DIR` until the far field is built, also `CMF_O1_SPILL` |
| `--prompt-file FILE` | a prompt longer than a command line holds |
| `CMF_O1=off` | the plain model, with exact attention at any length |
| `CMF_O1_SPEC=0` | no drafts over the far field |
| `CMF_MAX_SEQ=N` | the longest conversation, 32 768 tokens by default |

## Limits

- The far field runs on CUDA, Vulkan, DX12 and Metal. On the CPU this file runs as the standard one. It was measured
  on an NVIDIA RTX A5000, through CUDA and Vulkan.
- Past the seal the text is approximate: what lies far back fades, apart from the rows kept exact. The fit behind
  those rows saw planted facts in Wikipedia; other kinds of important text were not tested.
- While a long prompt is read, its rows take host memory: 24–29 GB at 128 000 tokens. `--o1-spill` puts them in files
  (27 GB at 250 000 tokens), which go when the far field is built.
- Rows that arrive after a prompt ends stay in the window or go to the far field. A turn that diverges from a sealed
  conversation reads the whole prompt again.
- A request with an image decodes without drafts; an image prompt longer than the seal is refused.

## Building a turbo file

```bash
cortiq far-turbo sensenova-loop-9b-q4tp.cmf sensenova-loop-9b-turbo.cmf --exact-weights keep_sn9_long.bin
```

`keep_sn9_long.bin` is in this repository and in [scripts/snloop](https://github.com/infosave2007/cmf/tree/sensenova-loop9b/scripts/snloop):
112 weights fitted by `scripts/dory/router/q27_feat.py` on 24 prompts of 60 000 characters with a planted fact.

## License

Apache-2.0, as the original release. The weights were converted to CMF and the language model quantized by cortiq.
Patent applications filed with the USPTO: No. 19/738,763 (July 13, 2026); titles and the patent grant in
[PATENTS.md](https://github.com/infosave2007/cmf/blob/master/PATENTS.md).

---

## Документация на русском

4-битные веса [SenseNova-Loop-9B](https://huggingface.co/sensenova/SenseNova-Loop-9B) с заголовком, который держит
память карты неизменной на длинных промптах. SenseNova-Loop выполняет 24 слоя дважды, и каждый проход хранит свои
ключи и значения, поэтому длинный промпт занимает у обычного файла вдвое больше памяти, чем у стека без петли.

- **RTX A5000 (24 ГБ), CUDA, промпт 128 000 токенов:** 8,3 ГБ карты против 16,5 ГБ у обычного файла, первый
  токен через 140 с (142), затем 49,9 ток/с против 30,5.
- **250 000 токенов на той же карте:** 8,3 ГБ, первый токен через 265 с, затем 50,4 ток/с; строки промпта ждут в
  RAM или, с `--o1-spill`, в файлах.
- Факты, спрятанные в 60 000, 120 000 и 240 000 символах Википедии (до 56 000 токенов), находятся 4 раза из 4 на каждой длине через CUDA, как и у обычного файла.

Это один файл [CMF](https://github.com/infosave2007/cmf) для `cortiq`, движка на Rust без Python и
ML-фреймворков. Промпты короче 8192 токенов идут с точным вниманием и черновыми токенами, как в
[обычном файле](https://huggingface.co/infosave/SenseNova-Loop-9B-cmf).

### Быстрый старт

```bash
cargo install --git https://github.com/infosave2007/cmf --branch sensenova-loop9b cortiq-cli --features cuda
hf download infosave/SenseNova-Loop-9B-Turbo-cmf sensenova-loop-9b-turbo.cmf --local-dir .
CMF_GPU=cuda CMF_MAX_SEQ=262144 cortiq run sensenova-loop-9b-turbo.cmf --prompt-file report.txt --greedy --no-think
```

Без CUDA (Vulkan, DX12, Metal) — сборка без `--features cuda`, запуск без `CMF_GPU=cuda`; `--o1-spill DIR` держит
строки длинного промпта в файлах вместо RAM. Нужен релиз cortiq после 0.9.4; до его выхода соберите из ветки выше.
`CMF_MAX_SEQ` поднимает предел в 32 768 токенов (модель вмещает 524 288).

### Сервер

`cortiq serve` отдаёт файл через HTTP API, совместимый с OpenAI, и веб-панель на том же порту.

Карта NVIDIA через CUDA:

```bash
CMF_GPU=cuda CMF_MAX_SEQ=262144 cortiq serve sensenova-loop-9b-turbo.cmf --port 8080 --host 127.0.0.1
```

Vulkan, DX12 или Metal:

```bash
CMF_MAX_SEQ=262144 cortiq serve sensenova-loop-9b-turbo.cmf --port 8080 --host 127.0.0.1
```

Запрос:

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "sensenova-loop-9b",
  "messages": [{"role": "user", "content": "Напиши функцию на Python, которая сливает два отсортированных списка."}],
  "max_tokens": 512,
  "temperature": 0,
  "enable_thinking": false,
  "stream": true
}'
```

Картинка передаётся частью `image_url` (data URL или HTTP(S) URL). Подойдёт любой клиент OpenAI с
`base_url="http://127.0.0.1:8080/v1"` и любым ключом.

| адрес | что делает |
|---|---|
| `POST /v1/chat/completions` | ход диалога; `stream: true` отдаёт ответ по мере декода, `tools` даёт модели вызывать функции |
| `POST /v1/completions` | продолжение `prompt` |
| `GET /v1/models` | обслуживаемая модель |
| `GET /healthz` | 200, пока сервер работает |
| `GET /` | веб-панель |

- `CMF_MAX_SEQ` — самый длинный разговор в запросе, по умолчанию 32 768 токенов.
- `temperature: 0` включает черновые токены, как `--greedy` у `cortiq run`; `enable_thinking: false` просит
  ответить сразу. Запрос с картинкой декодируется без черновиков.
- `--host 127.0.0.1` оставляет сервер локальным; по умолчанию он слушает все интерфейсы и ключи не проверяет.


### Что делает файл

**Неизменный потолок.** У SenseNova-Loop 8 слоёв полного внимания, 6 из них идут в обоих проходах петли, у каждого
прохода свои ключи и значения: 14 на токен. Первые 8192 строки промпта читаются точно. Ключи и значения следующих
строк уходят с карты по мере чтения — в RAM или в файлы. Перед последними 512 строками промпта каждый из 14
собирается заново как дальнее поле: первые 4 строки и последние 1800 остаются точными, остальное сводится к 32
опорным точкам на голову. Память карты и цена токена перестают расти вместе с текстом. Патентная заявка
№ 19/738,763.

**Точные строки.** Одно дальнее поле теряет пароль. Каждая строка оценивается по тому, что с ней делают слои (на
каждую KV-голову из 14: насколько ключ вне опорных точек и норма значения; линейная модель по 112 рангам, подогнана
на фактах в текстах Википедии), и 248 лучших остаются точными. Вопрос в конце промпта читает их до ответа.

**Черновики.** Голова MTP предлагает токены, карта проверяет их одним пакетом, как в обычном файле.

### Скорость

Оба бэкенда измерены на одной карте с одними и теми же файлами, но читайте каждый раздел отдельно.

#### CUDA · RTX A5000

Стенд: NVIDIA RTX A5000 (24 ГБ), бэкенд CUDA (`CMF_GPU=cuda`), релизная сборка ветки выше. `cortiq bench --core`
без черновиков, 128 токенов после промпта; обычный файл — `sensenova-loop-9b-q4tp.cmf`, те же веса.

![Обычный файл и турбо на CUDA](https://huggingface.co/infosave/SenseNova-Loop-9B-Turbo-cmf/resolve/main/assets/turbo_cuda_ru.png)

| промпт, токенов | первый токен, с (обычный) | первый токен, с (**турбо**) | декод, ток/с (обычный) | декод, ток/с (**турбо**) | пик памяти карты, ГБ (обычный) | пик памяти карты, ГБ (**турбо**) |
|---:|---:|---:|---:|---:|---:|---:|
| 16 000 | 10,1 | **16,5** | 47,1 | **50,8** | 8,9 | **8,3** |
| 32 000 | 22,6 | **34,4** | 43,5 | **50,4** | 10,0 | **8,3** |
| 64 000 | 53,0 | **70,3** | 38,1 | **50,5** | 12,2 | **8,3** |
| 128 000 | 141,9 | **139,9** | 30,5 | **49,9** | 16,5 | **8,3** |
| 250 000 | — | **264,6** | — | **50,4** | — | **8,3** |

Обычный файл дальше 128 000 токенов не запускался. При 250 000 строки уходили в файлы (`--o1-spill`, 27 ГБ на
диске до сборки дальнего поля). До 128 000 токенов турбо платит за дальнее поле первым токеном.

Факты в тексте (код доступа, пароль, номер комнаты, ключ; по четыре промпта, ответ берётся из реплики), CUDA:

| текст | токенов | обычный | **турбо** |
|---|---:|---:|---:|
| 60 000 символов | 13 300–15 100 | 4 из 4 | **4 из 4** |
| 120 000 символов | 26 900–28 400 | 4 из 4 | **4 из 4** |
| 240 000 символов | 54 400–56 200 | 4 из 4 | **4 из 4** |

#### Vulkan · RTX A5000

Стенд: та же NVIDIA RTX A5000 (24 ГБ) через Vulkan, релизная сборка ветки выше, тот же бенчмарк.

![Обычный файл и турбо на Vulkan](https://huggingface.co/infosave/SenseNova-Loop-9B-Turbo-cmf/resolve/main/assets/turbo_vulkan_ru.png)

| промпт, токенов | первый токен, с (обычный) | первый токен, с (**турбо**) | декод, ток/с (обычный) | декод, ток/с (**турбо**) | пик памяти карты, ГБ (обычный) | пик памяти карты, ГБ (**турбо**) |
|---:|---:|---:|---:|---:|---:|---:|
| 16 000 | 11,9 | **38,7** | 31,3 | **33,1** | 6,8 | **5,7** |
| 32 000 | 25,5 | **67,9** | 28,8 | **33,1** | 8,5 | **5,7** |
| 64 000 | 60,2 | **155,7** | 24,3 | **33,1** | 12,1 | **5,7** |
| 128 000 | 163,3 | **318,7** | 19,3 | **33,3** | 19,4 | **5,8** |
| 250 000 | — | **602,4** | — | **33,5** | — | **5,8** |

Через Vulkan турбо держит 5,7 ГБ карты и 33 ток/с при любой длине, но промпт после печати читает примерно на 410
токенов в секунду против 780–1 340 у обычного файла: первый токен приходит в 2–3 раза позже. На карте NVIDIA для
длинных промптов берите CUDA.

Факты через Vulkan: те же 12 промптов, по 4 из 4 при 60 000, 120 000 и 240 000 символов (обычный файл 4 из 4 при 60 000 и
120 000).


### Ограничения

- Дальнее поле работает на CUDA, Vulkan, DX12 и Metal; на CPU файл работает как обычный. Измерено на NVIDIA RTX A5000
  через CUDA и Vulkan.
- После печати текст приблизительный: далёкое прошлое тускнеет, кроме точных строк. Модель точных строк видела
  факты в Википедии; другие виды важного текста не проверялись.
- Пока читается длинный промпт, его строки занимают память хоста: 24–29 ГБ при 128 000 токенов. `--o1-spill` кладёт
  их в файлы (27 ГБ при 250 000 токенов), которые удаляются после сборки дальнего поля.
- Запрос с картинкой идёт без черновиков; промпт с картинкой длиннее порога печати отклоняется.

### Как собрать турбо-файл

```bash
cortiq far-turbo sensenova-loop-9b-q4tp.cmf sensenova-loop-9b-turbo.cmf --exact-weights keep_sn9_long.bin
```

`keep_sn9_long.bin` лежит в этом репозитории и в [scripts/snloop](https://github.com/infosave2007/cmf/tree/sensenova-loop9b/scripts/snloop).

### Лицензия

Apache-2.0, как у оригинала. Веса переведены в CMF и квантованы cortiq. Патентная заявка USPTO № 19/738,763.

---

## 中文说明

[SenseNova-Loop-9B](https://huggingface.co/sensenova/SenseNova-Loop-9B) 的 4 bit 权重，加上一个在长提示词下保持显存不变的文件头。
SenseNova-Loop 把 24 层运行两遍，每一遍都有自己的键和值，所以长提示词在普通文件上占用的显存是无循环层堆叠的两倍。

- RTX A5000（24 GB）+ CUDA，128 000 token 的提示词：显存 8.3 GB，普通文件 16.5 GB；首个 token 140 秒（普通文件 142 秒），之后 49.9 tok/s，普通文件 30.5。
- 同一张卡上 250 000 token：8.3 GB，首个 token 265 秒，之后 50.4 tok/s；提示词的行暂存在内存中，或用 `--o1-spill` 存到文件。
- 在 60 000、120 000 和 240 000 字符的维基百科文本（最多 56 000 个 token）中植入的事实，通过 CUDA 每种长度都 4 次找回 4 次，与普通文件相同。

这是一个 [CMF](https://github.com/infosave2007/cmf) 文件，供 `cortiq` 使用（Rust 推理引擎，不依赖 Python 和机器学习框架）。
短于 8192 token 的提示词使用精确注意力和草稿 token，与[普通文件](https://huggingface.co/infosave/SenseNova-Loop-9B-cmf)相同。

### 快速开始

```bash
cargo install --git https://github.com/infosave2007/cmf --branch sensenova-loop9b cortiq-cli --features cuda
hf download infosave/SenseNova-Loop-9B-Turbo-cmf sensenova-loop-9b-turbo.cmf --local-dir .
CMF_GPU=cuda CMF_MAX_SEQ=262144 cortiq run sensenova-loop-9b-turbo.cmf --prompt-file report.txt --greedy --no-think
```

不用 CUDA（Vulkan、DX12、Metal）时，编译不加 `--features cuda`，运行不设 `CMF_GPU=cuda`；`--o1-spill DIR` 把长提示词的行放到文件里而不是内存。
需要 0.9.4 之后的 cortiq 版本；发布之前请从上面的分支编译。`CMF_MAX_SEQ` 提高默认 32 768 token 的上限（模型最多 524 288）。

### 服务器

`cortiq serve` 通过兼容 OpenAI 的 HTTP API 提供该文件，同一端口还有网页面板。

NVIDIA 显卡走 CUDA：

```bash
CMF_GPU=cuda CMF_MAX_SEQ=262144 cortiq serve sensenova-loop-9b-turbo.cmf --port 8080 --host 127.0.0.1
```

Vulkan、DX12 或 Metal：

```bash
CMF_MAX_SEQ=262144 cortiq serve sensenova-loop-9b-turbo.cmf --port 8080 --host 127.0.0.1
```

请求：

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "sensenova-loop-9b",
  "messages": [{"role": "user", "content": "用 Python 写一个合并两个有序列表的函数。"}],
  "max_tokens": 512,
  "temperature": 0,
  "enable_thinking": false,
  "stream": true
}'
```

图片以 `image_url` 部分传入（data URL 或 HTTP(S) URL）。任何 OpenAI 客户端都可以使用，
`base_url="http://127.0.0.1:8080/v1"`，API key 任意。

| 接口 | 作用 |
|---|---|
| `POST /v1/chat/completions` | 一轮对话；`stream: true` 边解码边返回，`tools` 让模型调用函数 |
| `POST /v1/completions` | 续写 `prompt` |
| `GET /v1/models` | 当前提供的模型 |
| `GET /healthz` | 服务运行时返回 200 |
| `GET /` | 网页面板 |

- `CMF_MAX_SEQ` 是一次请求能容纳的最长对话，默认 32 768 个 token。
- `temperature: 0` 打开草稿 token，与 `cortiq run` 的 `--greedy` 相同；`enable_thinking: false` 要求直接回答。带图片的请求不用草稿解码。
- `--host 127.0.0.1` 只在本机监听；默认监听所有网卡，且不检查 API key。


### 这个文件做了什么

固定上限。SenseNova-Loop 有 8 个全注意力层，其中 6 个在循环的两遍中都运行，每一遍有自己的键和值：每个 token 共 14 个。
提示词的前 8192 行精确读取，之后各行的键和值读完就离开显卡，存到内存或文件。在提示词最后 512 行之前，这 14 个层各自重建为远场：
最前 4 行和最后 1800 行保持精确，其余压缩为每个头 32 个地标。显存和每个 token 的代价不再随文本增长。专利申请号 19/738,763。

精确行。只有远场会丢失密码。每一行按各层对它的处理打分（14 个层的每个 KV 头：键离地标张成空间有多远，以及值的范数；
对 112 个排名做线性拟合，在维基百科中植入事实来拟合），得分最高的 248 行保持精确。提示词末尾的问题在回答前先读取它们。

草稿。MTP 头生成草稿 token，显卡一次批量验证，与普通文件相同。

### 速度

两个后端在同一张卡上用同样的文件测量，但请分别阅读各自的小节。

#### CUDA · RTX A5000

测试平台：NVIDIA RTX A5000（24 GB），CUDA 后端（`CMF_GPU=cuda`），上述分支的 release 构建。`cortiq bench --core`，不用草稿，
提示词之后 128 个 token；普通文件是 `sensenova-loop-9b-q4tp.cmf`，权重相同。

![CUDA 上的普通文件与 Turbo](https://huggingface.co/infosave/SenseNova-Loop-9B-Turbo-cmf/resolve/main/assets/turbo_cuda_zh.png)

| 提示词，token | 首个 token，秒（普通） | 首个 token，秒（Turbo） | 解码，tok/s（普通） | 解码，tok/s（Turbo） | 显存峰值，GB（普通） | 显存峰值，GB（Turbo） |
|---:|---:|---:|---:|---:|---:|---:|
| 16 000 | 10.1 | 16.5 | 47.1 | 50.8 | 8.9 | 8.3 |
| 32 000 | 22.6 | 34.4 | 43.5 | 50.4 | 10.0 | 8.3 |
| 64 000 | 53.0 | 70.3 | 38.1 | 50.5 | 12.2 | 8.3 |
| 128 000 | 141.9 | 139.9 | 30.5 | 49.9 | 16.5 | 8.3 |
| 250 000 | — | 264.6 | — | 50.4 | — | 8.3 |

普通文件没有测超过 128 000 token 的情况。250 000 token 时各行写入文件（`--o1-spill`，远场建好前磁盘占用 27 GB）。128 000 token 以下，Turbo 为远场付出的代价体现在首个 token 上。

植入事实（访问码、密码、房间号、通行密钥；每种长度 4 个提示词，从回答中读取），CUDA：

| 文本 | token | 普通 | Turbo |
|---|---:|---:|---:|
| 60 000 字符 | 13 300–15 100 | 4 / 4 | 4 / 4 |
| 120 000 字符 | 26 900–28 400 | 4 / 4 | 4 / 4 |
| 240 000 字符 | 54 400–56 200 | 4 / 4 | 4 / 4 |

#### Vulkan · RTX A5000

测试平台：同一张 NVIDIA RTX A5000（24 GB），走 Vulkan，上述分支的 release 构建，相同的基准测试。

![Vulkan 上的普通文件与 Turbo](https://huggingface.co/infosave/SenseNova-Loop-9B-Turbo-cmf/resolve/main/assets/turbo_vulkan_zh.png)

| 提示词，token | 首个 token，秒（普通） | 首个 token，秒（Turbo） | 解码，tok/s（普通） | 解码，tok/s（Turbo） | 显存峰值，GB（普通） | 显存峰值，GB（Turbo） |
|---:|---:|---:|---:|---:|---:|---:|
| 16 000 | 11.9 | 38.7 | 31.3 | 33.1 | 6.8 | 5.7 |
| 32 000 | 25.5 | 67.9 | 28.8 | 33.1 | 8.5 | 5.7 |
| 64 000 | 60.2 | 155.7 | 24.3 | 33.1 | 12.1 | 5.7 |
| 128 000 | 163.3 | 318.7 | 19.3 | 33.3 | 19.4 | 5.8 |
| 250 000 | — | 602.4 | — | 33.5 | — | 5.8 |

在 Vulkan 上 Turbo 在任意长度下都保持 5.7 GB 显存和 33 tok/s，但封存之后读取提示词约每秒 410 个 token，普通文件为 780–1 340：首个 token 要晚 2–3 倍。
NVIDIA 显卡处理长提示词请用 CUDA。

Vulkan 上的植入事实：同样的 12 个提示词，60 000、120 000 和 240 000 字符时均为 4 / 4（普通文件在 60 000 和 120 000 字符时 4 / 4）。


### 限制

- 远场支持 CUDA、Vulkan、DX12 和 Metal；在 CPU 上该文件按普通文件运行。测量平台：NVIDIA RTX A5000，CUDA 与 Vulkan。
- 封存之后的文本是近似的：远处的内容会淡化，精确行除外。精确行的模型只见过维基百科中植入的事实，其他类型的重要文本未测试。
- 读取长提示词时，各行占用主机内存：128 000 token 时 24–29 GB。`--o1-spill` 把它们放到文件（250 000 token 时 27 GB），远场建好后删除。
- 带图片的请求不用草稿；超过封存长度的图片提示词会被拒绝。

### 构建 turbo 文件

```bash
cortiq far-turbo sensenova-loop-9b-q4tp.cmf sensenova-loop-9b-turbo.cmf --exact-weights keep_sn9_long.bin
```

`keep_sn9_long.bin` 在本仓库和 [scripts/snloop](https://github.com/infosave2007/cmf/tree/sensenova-loop9b/scripts/snloop) 中。

### 许可证

Apache-2.0，与原始发布相同。权重由 cortiq 转换为 CMF 并量化。USPTO 专利申请号 19/738,763。
