---
license: apache-2.0
library_name: cortiq
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: text-generation
tags:
  - cmf
  - cortiq
  - quantized
  - q4tp
  - qwen3.8
  - long-context
  - low-vram
  - vulkan
  - cuda
language:
  - en
  - ru
  - zh
---

# Qwen3.8-27B Turbo — CMF

The 4-bit weights of [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) under a header that holds
device memory in place on long prompts: a 64 000-token prompt takes 15–16 GB, against 22.8 GB for the
standard file. Through CUDA a 16 GB RTX 2000 Ada keeps pace with an RTX 5090 over Vulkan: 64 000 tokens to
the first token in 175 s (177 on the 5090), then 94 tok/s (92). An RTX 3090 through CUDA writes a summary of
100 000 characters at 80 tok/s. Passwords and codes in the text stay exact: facts planted in 100 000
characters come back 8 times of 8. Prompts under 8192 tokens run exact attention, as in the
[standard file](https://huggingface.co/infosave/Qwen3.8-27B-cmf). It is one
[CMF](https://github.com/infosave2007/cmf) file for `cortiq`, a Rust inference engine with no Python
and no ML framework.

[English](#quick-start) · [Русский](#документация-на-русском) · [中文](#中文说明)

## Quick start

```bash
cargo install cortiq-cli
hf download infosave/Qwen3.8-27B-Turbo-cmf qwen38-27b-turbo.cmf --local-dir .
cortiq run qwen38-27b-turbo.cmf --prompt-file report.txt --greedy --no-think
```

Short on RAM as well, with a fast SSD:

```bash
cortiq run qwen38-27b-turbo.cmf --prompt-file report.txt --greedy --no-think --o1-spill /mnt/ssd/cortiq
```

Use cortiq 0.8.59 or later; prebuilt binaries are on the
[releases page](https://github.com/infosave2007/cmf/releases). `--greedy` turns on draft tokens, and
`--no-think` asks for a direct answer.

On an NVIDIA card through CUDA (cortiq 0.8.71 or later; the CUDA toolkit's `libnvrtc` and `libcublas`
12.x at run time):

```bash
cargo install cortiq-cli --features cuda
CMF_GPU=cuda cortiq run qwen38-27b-turbo.cmf --prompt-file report.txt --greedy --no-think
CMF_GPU=cuda cortiq serve qwen38-27b-turbo.cmf --port 8080 --host 127.0.0.1
```

## Server

`cortiq serve` puts the file behind an OpenAI-compatible HTTP API, with a web dashboard on the same port.

Vulkan, DX12 or Metal:

```bash
CMF_MAX_SEQ=131072 cortiq serve qwen38-27b-turbo.cmf --port 8080 --host 127.0.0.1
```

An NVIDIA card through CUDA (a build with `--features cuda`, cortiq 0.8.71 or later):

```bash
CMF_GPU=cuda CMF_MAX_SEQ=131072 cortiq serve qwen38-27b-turbo.cmf --port 8080 --host 127.0.0.1
```

A request:

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "qwen38-27b-turbo",
  "messages": [{"role": "user", "content": "Summarize the report below.\n\n..."}],
  "max_tokens": 512,
  "temperature": 0,
  "enable_thinking": false,
  "stream": true
}'
```

Any OpenAI client works with `base_url="http://127.0.0.1:8080/v1"` and any API key:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="none")
r = client.chat.completions.create(
    model="qwen38-27b-turbo",
    messages=[{"role": "user", "content": open("report.txt").read() + "\n\nSummarize it."}],
    temperature=0,
    extra_body={"enable_thinking": False},
)
print(r.choices[0].message.content)
```

| endpoint | what it does |
|---|---|
| `POST /v1/chat/completions` | a chat turn; `stream: true` sends the answer as it is decoded |
| `POST /v1/completions` | the continuation of a `prompt` |
| `GET /v1/models` | the served model |
| `GET /healthz` | 200 while the server is up |
| `GET /` | the web dashboard |

- `CMF_MAX_SEQ` is the longest conversation a request may hold, 32 768 tokens by default. Set it above your
  longest document: past the seal the card's memory stays where it is.
- `temperature: 0` turns on draft tokens, as `--greedy` does for `cortiq run`. `enable_thinking: false` asks
  for a direct answer.
- `model` may be any name: one file is served.
- `--host 127.0.0.1` keeps the server local; the default is every interface, and no API key is checked.
- A turn that continues the previous one reads only what is new. Past the seal of a long document, a new turn
  reads the conversation again.

## What the file does

![A long prompt on a small card](https://huggingface.co/infosave/Qwen3.8-27B-Turbo-cmf/resolve/main/assets/ceiling_en.png)

**A fixed ceiling.** The first 8192 rows of a prompt are read exactly on the card, 4096 on a 16 GB
card. The keys and values of later rows leave the card as they are read, for RAM or an SSD. Before the
last 512 rows, the far field is built from them one layer at a time, and the rows go away. A far field
keeps the first 4 rows and the last 1800 exact and everything else in 32 landmarks per head, so device
memory and the cost of a token stop growing with the text. Patent application No. 19/738,763.

**Exact rows.** A far field alone loses a password. So each row is scored by what the layers do with
it, and the 248 best stay exact. A question at the end of the prompt reads them before it is answered.

![Passwords and codes stay exact](https://huggingface.co/infosave/Qwen3.8-27B-Turbo-cmf/resolve/main/assets/exact_rows_en.png)

**Short threads and drafts.** A sealed token and a chunk of rows take the attention apart into seven
small stages that fill the whole card, where one long thread per head leaves most of it idle. The
model's MTP head drafts tokens, the card verifies them in one batch, and the far field moves over the
accepted ones only. Greedy text stays the same with and without drafts.

## Performance

![Standard and turbo](https://huggingface.co/infosave/Qwen3.8-27B-Turbo-cmf/resolve/main/assets/compare_en.png)

`cortiq bench --core`, 60 tokens after a prompt of that length. The orange and blue lines are an RTX 5090
over Vulkan (the table below); the green one is the turbo file through CUDA on a 16 GB RTX 2000 Ada,
cortiq 0.8.71: it stays at 15.0 GB and matches the 5090 from 16 000 tokens on.

| prompt, tokens | first token, s (standard) | first token, s (**turbo**) | decode, tok/s (standard) | decode, tok/s (**turbo**) | peak VRAM, GB (standard) | peak VRAM, GB (**turbo**) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 000 | 2.2 | **2.2** | 105 | **105** | 14.5 | **14.4** |
| 4 000 | 8.8 | **8.8** | 104 | **104** | 14.9 | **14.8** |
| 16 000 | 42 | **47** | 87 | **92** | 16.5 | **15.5** |
| 32 000 | 107 | **92** | 71 | **91** | 18.5 | **15.7** |
| 64 000 | 316 | **177** | 51 | **92** | 22.8 | **15.7** |

A real answer, 400 tokens of a summary of Wikipedia articles, with about half of the drafts accepted:

| prompt | first token, s (standard) | first token, s (**turbo**) | decode, tok/s (standard) | decode, tok/s (**turbo**) |
|---|---:|---:|---:|---:|
| 30 000 characters | 19.2 | **18.6** | 76.4 | **76.3** |
| 100 000 characters | 76.7 | **72.1** | 50.9 | **60.4** |

**On a 16 GB card through CUDA.** An RTX 2000 Ada, cortiq 0.8.71, the same bench (its text repeats, so most
drafts hold). The whole run stays within 15.0 GB, the prompt included; the standard file runs out of this
card's memory at 16 000 tokens. Against Vulkan on the same card CUDA reaches the first token 1.4–1.6 times
sooner and decodes 3.1–3.7 times faster.

![Qwen3.8-27B Turbo on CUDA](https://huggingface.co/infosave/Qwen3.8-27B-Turbo-cmf/resolve/main/assets/cuda_en.png)

| prompt, tokens | first token, s (**CUDA**) | first token, s (Vulkan) | decode, tok/s (**CUDA**) | decode, tok/s (Vulkan) | peak VRAM, GB (CUDA / Vulkan) |
|---:|---:|---:|---:|---:|---:|
| 1 000 | **2.55** | 3.58 | **87.3** | 27.9 | 14.6 / 14.1 |
| 4 000 | **9.71** | 14.88 | **86.9** | 26.7 | 14.9 / 14.3 |
| 16 000 | **43.2** | 70.4 | **94.4** | 25.4 | 15.0 / 14.8 |
| 32 000 | **86.2** | 135.6 | **93.5** | 26.7 | 15.0 / 14.8 |
| 64 000 | **174.6** | 277.9 | **93.8** | 26.5 | 15.0 / 14.8 |

A real answer, a summary of 100 000 characters of Wikipedia, up to 400 tokens: first token in 69.8 s, then
31.5 tok/s, the same text without drafts (Vulkan: 114.3 s and 21.3 tok/s). Facts in 60 000 and 100 000
characters: 8 of 8.

**On an RTX 3090 through CUDA** the turbo file decodes at 118–128 tok/s past the seal in the same bench and
writes the summary of 100 000 characters at 80.3 tok/s.

**On a Mac.** The machine is a MacBook Air M4 with 24 GB: no fan, so long runs slow down as it heats, and the
base M4's memory bus, 120 GB/s by Apple's figure and 92 to 95 GB/s as measured. A token reads 13.6 GB of weights,
so the bus sets the speed: 6.4 tok/s without drafts. An M4 Pro's bus is rated at 273 GB/s and an M4 Max's at up to
546 GB/s; those machines were not measured.

At 16 000 tokens the process stays at 6.3 GB beside the mapped weights, first token 545 s, decode 4 tok/s, and a
planted fact in 40 000 characters came back whole: past the seal the file saves memory on a Mac, not time.

Two switches, both off by default, add speed. Drafts, `CMF_GRAPH_SPEC=1`, with `--greedy`, 220 tokens, the same
text with drafts and without:

| answer | without drafts, tok/s | with drafts, tok/s |
|---|---:|---:|
| an essay | 6.4 | **10.1** |
| an explanation in Russian | 6.4 | **10.7** |
| a Rust function with tests | 6.4 | **12.1** |
| quotes from the prompt | 6.4 | **15.0** |
| an edit of the prompt's code | 6.4 | **20.5** |

The Neural Engine, `CMF_ANE=1`, reads a prompt beside the GPU: first token at 3 600 tokens 73.9 → 28.3 s, the same
text. Both switches need cortiq 0.8.59 or later.

**Host memory.** While a long prompt is read, its rows take RAM or a disk. The mapped weights are not
counted here; the system can drop them from RAM.

| prompt, tokens | RAM | with `--o1-spill`: RAM | with `--o1-spill`: disk |
|---:|---:|---:|---:|
| 16 000 | 4.4 GB | 2.2 GB | 4.7 GB |
| 32 000 | 7.0 GB | 2.2 GB | 8.7 GB |
| 64 000 | 13.2 GB | 3.0 GB | 18.7 GB |

On a disk with 4.5 GB/s write and 5.2 GB/s read the speed is the same as in RAM. The rows are needed
only until the far field is built.

**Quality.** A planted fact in 100 000 characters, four kinds of them (a pass key, an access code, a
password, a room number): 12 of 12, 8 of them with a 15 GB budget and an SSD; in 230 000 characters, about 53 000 tokens, 6 of 6 on the
same path. A far field without exact rows finds 0 of 4. Perplexity on the rows past the seal, two windows of 16 384 tokens of Wikipedia: exact 4.76, far field 4.97.

## Options

| option | meaning |
|---|---|
| `--o1-spill DIR` | a long prompt's rows in files in `DIR`, also `CMF_O1_SPILL` |
| `--prompt-file FILE` | a prompt longer than a command line holds |
| `CMF_O1_PREFIX_MAX=N` | seal at `N` rows instead of the choice for the card |
| `CMF_O1=off` | the plain model, with exact attention at any length |
| `CMF_O1_SPEC=0` | no drafts over the far field |

## Limits

- The far field runs on Vulkan, DX12, Metal and CUDA. On the CPU this file runs as the standard one. It was
  measured on NVIDIA cards (Vulkan and CUDA) and an M4.
- Past the seal the text is approximate: what lies far back fades, apart from the rows kept exact. The
  fit behind those rows saw planted facts, and other kinds of important text were not tested.
- Rows that arrive after a prompt ends stay in the window or go to the far field. A turn that diverges
  from a sealed conversation reads the whole prompt again.
- On Vulkan, drafts need greedy decoding without penalties. Without `--greedy` a sealed token runs at
  about 57 tok/s.
- A generation that starts below the seal stays exact, and its memory grows with every token.

## Building a turbo file

The weights of the exact-row fit are in the
[repository](https://github.com/infosave2007/cmf/tree/master/scripts/dory/router), with the scripts that
make them.

```bash
cortiq far-turbo qwen38-27b-q4tp.cmf qwen38-27b-turbo.cmf --exact-weights keep27.bin
```

See [the guide](https://github.com/infosave2007/cmf/blob/master/docs/QWEN38_TURBO.md).

## License

Weights: [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0), from
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). The far field is a subject of U.S. patent
application No. 19/738,763; see
[PATENTS.md](https://github.com/infosave2007/cmf/blob/master/PATENTS.md).

---

## Документация на русском

4-битные веса [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) с заголовком, который держит память
карты на месте при длинных промптах: промпт в 64 000 токенов занимает 15–16 ГБ против 22,8 ГБ у обычного
файла. Через CUDA RTX 2000 Ada с 16 ГБ идёт вровень с RTX 5090 через Vulkan: 64 000 токенов до первого
токена за 175 с (177 на 5090), затем 94 ток/с (92). RTX 3090 через CUDA пишет пересказ 100 000 символов на
80 ток/с. Пароли и коды в тексте остаются точными: факты, вставленные в 100 000 символов, возвращаются 8 раз
из 8. Промпты короче 8192 токенов идут с точным вниманием, как в
[обычном файле](https://huggingface.co/infosave/Qwen3.8-27B-cmf). Это один
[CMF](https://github.com/infosave2007/cmf)-файл для `cortiq`, движка на Rust без Python и ML-фреймворков.

### Быстрый старт

```bash
cargo install cortiq-cli
hf download infosave/Qwen3.8-27B-Turbo-cmf qwen38-27b-turbo.cmf --local-dir .
cortiq run qwen38-27b-turbo.cmf --prompt-file report.txt --greedy --no-think
```

Если не хватает и оперативной памяти, но есть быстрый SSD:

```bash
cortiq run qwen38-27b-turbo.cmf --prompt-file report.txt --greedy --no-think --o1-spill /mnt/ssd/cortiq
```

Нужен cortiq 0.8.59 или новее; готовые сборки лежат на
[странице релизов](https://github.com/infosave2007/cmf/releases). `--greedy` включает черновые токены,
`--no-think` просит ответ без рассуждения.

На карте NVIDIA через CUDA (cortiq 0.8.71 или новее; при запуске нужны `libnvrtc` и `libcublas` 12.x из
CUDA toolkit):

```bash
cargo install cortiq-cli --features cuda
CMF_GPU=cuda cortiq run qwen38-27b-turbo.cmf --prompt-file report.txt --greedy --no-think
CMF_GPU=cuda cortiq serve qwen38-27b-turbo.cmf --port 8080 --host 127.0.0.1
```

### Сервер

`cortiq serve` открывает файл через HTTP API, совместимый с OpenAI, и веб-панель на том же порту.

Vulkan, DX12 или Metal:

```bash
CMF_MAX_SEQ=131072 cortiq serve qwen38-27b-turbo.cmf --port 8080 --host 127.0.0.1
```

Карта NVIDIA через CUDA (сборка с `--features cuda`, cortiq 0.8.71 или новее):

```bash
CMF_GPU=cuda CMF_MAX_SEQ=131072 cortiq serve qwen38-27b-turbo.cmf --port 8080 --host 127.0.0.1
```

Запрос:

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "qwen38-27b-turbo",
  "messages": [{"role": "user", "content": "Перескажи отчёт ниже.\n\n..."}],
  "max_tokens": 512,
  "temperature": 0,
  "enable_thinking": false,
  "stream": true
}'
```

Подойдёт любой клиент OpenAI с `base_url="http://127.0.0.1:8080/v1"` и любым ключом API:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="none")
r = client.chat.completions.create(
    model="qwen38-27b-turbo",
    messages=[{"role": "user", "content": open("report.txt").read() + "\n\nПерескажи это."}],
    temperature=0,
    extra_body={"enable_thinking": False},
)
print(r.choices[0].message.content)
```

| адрес | что делает |
|---|---|
| `POST /v1/chat/completions` | реплика чата; `stream: true` отдаёт ответ по мере декодирования |
| `POST /v1/completions` | продолжение `prompt` |
| `GET /v1/models` | обслуживаемая модель |
| `GET /healthz` | 200, пока сервер работает |
| `GET /` | веб-панель |

- `CMF_MAX_SEQ` — самый длинный диалог в запросе, по умолчанию 32 768 токенов. Ставьте больше самого длинного
  документа: после печати память карты не растёт.
- `temperature: 0` включает черновые токены, как `--greedy` у `cortiq run`. `enable_thinking: false` просит
  прямой ответ.
- `model` может быть любым именем: сервер отдаёт один файл.
- `--host 127.0.0.1` оставляет сервер локальным; по умолчанию он слушает все интерфейсы и ключи API не проверяет.
- Реплика, продолжающая предыдущую, читает только новое. После печати длинного документа новая реплика читает
  диалог заново.

### Что делает файл

![Длинный промпт на маленькой карте](https://huggingface.co/infosave/Qwen3.8-27B-Turbo-cmf/resolve/main/assets/ceiling_ru.png)

**Постоянный потолок.** Первые 8192 строки промпта читаются на карте точно, на карте 16 ГБ — 4096.
Ключи и значения следующих строк уходят с карты по мере чтения, в RAM или на SSD. Перед последними
512 строками из них послойно строится дальнее поле, и сами строки больше не нужны. Дальнее поле держит
точными первые 4 строки и последние 1800, а всё остальное хранит в 32 опорных точках на голову, поэтому
память карты и цена токена перестают расти вместе с текстом. Патентная заявка США № 19/738,763.

**Точные строки.** Одно дальнее поле теряет пароль. Поэтому каждая строка получает оценку по тому, что с
ней делают слои, и 248 лучших остаются точными. Вопрос в конце промпта читает их до ответа.

![Пароли и коды остаются точными](https://huggingface.co/infosave/Qwen3.8-27B-Turbo-cmf/resolve/main/assets/exact_rows_ru.png)

**Короткие потоки и черновики.** Запечатанный токен и пакет строк разбирают внимание на семь небольших
стадий, которые занимают всю карту; один длинный поток на голову оставляет большую её часть без дела.
Голова MTP предлагает токены, карта проверяет их одним пакетом, а дальнее поле сдвигается только на
принятые. Текст при жадном декодировании с черновиками и без них один и тот же.

### Скорость

![Обычный файл и турбо](https://huggingface.co/infosave/Qwen3.8-27B-Turbo-cmf/resolve/main/assets/compare_ru.png)

`cortiq bench --core`, 60 токенов после промпта этой длины. Оранжевая и синяя линии — RTX 5090 через Vulkan
(таблица ниже); зелёная — турбо-файл через CUDA на RTX 2000 Ada с 16 ГБ, cortiq 0.8.71: он держится в 15,0 ГБ
и с 16 000 токенов идёт вровень с 5090.

| промпт, токенов | первый токен, с (обычный) | первый токен, с (**турбо**) | декод, ток/с (обычный) | декод, ток/с (**турбо**) | пик VRAM, ГБ (обычный) | пик VRAM, ГБ (**турбо**) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 000 | 2,2 | **2,2** | 105 | **105** | 14,5 | **14,4** |
| 4 000 | 8,8 | **8,8** | 104 | **104** | 14,9 | **14,8** |
| 16 000 | 42 | **47** | 87 | **92** | 16,5 | **15,5** |
| 32 000 | 107 | **92** | 71 | **91** | 18,5 | **15,7** |
| 64 000 | 316 | **177** | 51 | **92** | 22,8 | **15,7** |

Настоящий ответ: 400 токенов пересказа статей Википедии, принята примерно половина черновиков.

| промпт | первый токен, с (обычный) | первый токен, с (**турбо**) | декод, ток/с (обычный) | декод, ток/с (**турбо**) |
|---|---:|---:|---:|---:|
| 30 000 символов | 19,2 | **18,6** | 76,4 | **76,3** |
| 100 000 символов | 76,7 | **72,1** | 50,9 | **60,4** |

**На карте 16 ГБ через CUDA.** RTX 2000 Ada, cortiq 0.8.71, тот же бенч (его текст повторяется, поэтому
большинство черновиков принимается). Весь прогон вместе с промптом держится в 15,0 ГБ; обычному файлу на
16 000 токенов памяти этой карты не хватает. Против Vulkan на той же карте CUDA доходит до первого токена в
1,4–1,6 раза быстрее и декодирует в 3,1–3,7 раза быстрее.

![Qwen3.8-27B Turbo на CUDA](https://huggingface.co/infosave/Qwen3.8-27B-Turbo-cmf/resolve/main/assets/cuda_ru.png)

| промпт, токенов | первый токен, с (**CUDA**) | первый токен, с (Vulkan) | декод, ток/с (**CUDA**) | декод, ток/с (Vulkan) | пик VRAM, ГБ (CUDA / Vulkan) |
|---:|---:|---:|---:|---:|---:|
| 1 000 | **2,55** | 3,58 | **87,3** | 27,9 | 14,6 / 14,1 |
| 4 000 | **9,71** | 14,88 | **86,9** | 26,7 | 14,9 / 14,3 |
| 16 000 | **43,2** | 70,4 | **94,4** | 25,4 | 15,0 / 14,8 |
| 32 000 | **86,2** | 135,6 | **93,5** | 26,7 | 15,0 / 14,8 |
| 64 000 | **174,6** | 277,9 | **93,8** | 26,5 | 15,0 / 14,8 |

Реальный ответ — пересказ 100 000 символов Википедии, до 400 токенов: первый токен через 69,8 с, затем
31,5 ток/с, тот же текст и без черновиков (Vulkan: 114,3 с и 21,3 ток/с). Факты в 60 000 и 100 000 символов:
8 из 8.

**На RTX 3090 через CUDA** турбо-файл после печати декодирует 118–128 ток/с в том же бенче и пишет пересказ
100 000 символов на 80,3 ток/с.

**На Маке.** Машина — MacBook Air M4, 24 ГБ: без вентилятора, поэтому на долгих прогонах она греется и
замедляется, и с шиной памяти базового M4 — 120 ГБ/с по данным Apple, 92–95 ГБ/с по замеру. Токен читает 13,6 ГБ
весов, так что скорость задаёт шина: 6,4 ток/с без черновиков. У M4 Pro шина заявлена на 273 ГБ/с, у M4 Max — до
546 ГБ/с; на этих машинах замеров не было.

На 16 000 токенов процесс держится на 6,3 ГБ помимо отображённых весов, первый токен 545 с, декод 4 ток/с, а факт,
вставленный в 40 000 символов, вернулся целым: за печатью файл на Маке экономит память, а не время.

Скорость добавляют два переключателя, оба по умолчанию выключены. Черновики, `CMF_GRAPH_SPEC=1`, с `--greedy`,
220 токенов, текст с черновиками и без них один и тот же:

| ответ | без черновиков, ток/с | с черновиками, ток/с |
|---|---:|---:|
| эссе | 6,4 | **10,1** |
| объяснение на русском | 6,4 | **10,7** |
| функция на Rust с тестами | 6,4 | **12,1** |
| цитаты из промпта | 6,4 | **15,0** |
| правка кода из промпта | 6,4 | **20,5** |

Neural Engine, `CMF_ANE=1`, читает промпт рядом с GPU: первый токен на 3 600 токенов 73,9 → 28,3 с, текст тот же.
Обоим переключателям нужен cortiq 0.8.59 или новее.

**Память хоста.** Пока читается длинный промпт, его строки занимают RAM или диск. Отображённые в память
веса здесь не считаются: система может выгрузить их из RAM.

| промпт, токенов | RAM | с `--o1-spill`: RAM | с `--o1-spill`: диск |
|---:|---:|---:|---:|
| 16 000 | 4,4 ГБ | 2,2 ГБ | 4,7 ГБ |
| 32 000 | 7,0 ГБ | 2,2 ГБ | 8,7 ГБ |
| 64 000 | 13,2 ГБ | 3,0 ГБ | 18,7 ГБ |

На диске с записью 4,5 ГБ/с и чтением 5,2 ГБ/с скорость та же, что и в RAM. Строки нужны только до сборки
дальнего поля.

**Качество.** Факт, вставленный в 100 000 символов, четырёх видов (ключ, код доступа, пароль, номер
комнаты): 12 из 12, из них 8 с бюджетом 15 ГБ и SSD; в 230 000 символов, это около 53 000 токенов, 6 из 6 тем же путём. Дальнее поле без точных строк находит 0 из 4.
Перплексия на строках после места запечатывания, два окна по 16 384 токена Википедии: точное внимание 4,76, дальнее поле 4,97.

### Параметры

| параметр | значение |
|---|---|
| `--o1-spill DIR` | строки длинного промпта в файлах в `DIR`, также `CMF_O1_SPILL` |
| `--prompt-file FILE` | промпт длиннее, чем вмещает командная строка |
| `CMF_O1_PREFIX_MAX=N` | запечатать на `N` строках, а не по размеру карты |
| `CMF_O1=off` | обычная модель, точное внимание при любой длине |
| `CMF_O1_SPEC=0` | без черновиков над дальним полем |

### Ограничения

- Дальнее поле работает на Vulkan, DX12, Metal и CUDA. На CPU этот файл идёт как обычный. Замеры сделаны на
  картах NVIDIA (Vulkan и CUDA) и на M4.
- За местом запечатывания текст приближённый: далёкое блекнет, кроме строк, оставленных точными. Модель,
  которая их выбирает, видела вставленные факты; другие виды важного текста не проверялись.
- Строки, пришедшие после конца промпта, остаются в окне или уходят в дальнее поле. Ход, который расходится
  с запечатанным разговором, читает весь промпт заново.
- На Vulkan черновикам нужно жадное декодирование без штрафов. Без `--greedy` запечатанный токен идёт примерно
  на 57 ток/с.
- Генерация, начатая ниже места запечатывания, остаётся точной, и её память растёт с каждым токеном.

### Как собрать турбо-файл

Веса модели точных строк лежат в
[репозитории](https://github.com/infosave2007/cmf/tree/master/scripts/dory/router) вместе со скриптами, которые
их строят.

```bash
cortiq far-turbo qwen38-27b-q4tp.cmf qwen38-27b-turbo.cmf --exact-weights keep27.bin
```

Подробнее в [руководстве](https://github.com/infosave2007/cmf/blob/master/docs/QWEN38_TURBO.ru.md).

### Лицензия

Веса: [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0), из
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). Дальнее поле защищается патентной заявкой США
№ 19/738,763; см. [PATENTS.md](https://github.com/infosave2007/cmf/blob/master/PATENTS.md).

---

## 中文说明

[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) 的 4 位权重,文件头让长提示下的显存保持不变:64 000 个 token 的提示占用 15–16 GB,普通文件为 22.8 GB。通过 CUDA,16 GB 的 RTX 2000 Ada 与走 Vulkan 的 RTX 5090 不相上下:64 000 个 token 的首个 token 用时 175 秒(5090 为 177 秒),之后 94 tok/s(5090 为 92)。RTX 3090 通过 CUDA 以 80 tok/s 写出 100 000 个字符的概括。文本中的密码和编码依然精确:植入 100 000 个字符中的事实 8 次全部答对。短于 8192 个 token 的提示使用精确注意力,与[普通文件](https://huggingface.co/infosave/Qwen3.8-27B-cmf)相同。这是一个供 `cortiq` 使用的 [CMF](https://github.com/infosave2007/cmf) 文件。`cortiq` 是 Rust 推理引擎,不需要 Python,也不需要机器学习框架。

### 快速开始

```bash
cargo install cortiq-cli
hf download infosave/Qwen3.8-27B-Turbo-cmf qwen38-27b-turbo.cmf --local-dir .
cortiq run qwen38-27b-turbo.cmf --prompt-file report.txt --greedy --no-think
```

内存也不够,但有高速 SSD 时:

```bash
cortiq run qwen38-27b-turbo.cmf --prompt-file report.txt --greedy --no-think --o1-spill /mnt/ssd/cortiq
```

需要 cortiq 0.8.59 或更新版本;预编译版本在[发布页](https://github.com/infosave2007/cmf/releases)。`--greedy` 开启草稿 token,`--no-think` 要求直接回答。

在 NVIDIA 显卡上通过 CUDA（cortiq 0.8.71 或更新版本；运行时需要 CUDA toolkit 的 `libnvrtc` 和 `libcublas` 12.x）：

```bash
cargo install cortiq-cli --features cuda
CMF_GPU=cuda cortiq run qwen38-27b-turbo.cmf --prompt-file report.txt --greedy --no-think
CMF_GPU=cuda cortiq serve qwen38-27b-turbo.cmf --port 8080 --host 127.0.0.1
```

### 服务器

`cortiq serve` 通过与 OpenAI 兼容的 HTTP API 提供这个文件,同一端口上还有网页面板。

Vulkan、DX12 或 Metal:

```bash
CMF_MAX_SEQ=131072 cortiq serve qwen38-27b-turbo.cmf --port 8080 --host 127.0.0.1
```

通过 CUDA 使用 NVIDIA 显卡(用 `--features cuda` 构建,cortiq 0.8.71 或更新版本):

```bash
CMF_GPU=cuda CMF_MAX_SEQ=131072 cortiq serve qwen38-27b-turbo.cmf --port 8080 --host 127.0.0.1
```

请求:

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "qwen38-27b-turbo",
  "messages": [{"role": "user", "content": "概括下面的报告。\n\n..."}],
  "max_tokens": 512,
  "temperature": 0,
  "enable_thinking": false,
  "stream": true
}'
```

任何 OpenAI 客户端都可以使用,`base_url="http://127.0.0.1:8080/v1"`,API 密钥随意:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="none")
r = client.chat.completions.create(
    model="qwen38-27b-turbo",
    messages=[{"role": "user", "content": open("report.txt").read() + "\n\n概括一下。"}],
    temperature=0,
    extra_body={"enable_thinking": False},
)
print(r.choices[0].message.content)
```

| 接口 | 作用 |
|---|---|
| `POST /v1/chat/completions` | 一轮对话;`stream: true` 边解码边返回 |
| `POST /v1/completions` | 续写 `prompt` |
| `GET /v1/models` | 提供的模型 |
| `GET /healthz` | 服务器运行时返回 200 |
| `GET /` | 网页面板 |

- `CMF_MAX_SEQ` 是一个请求能容纳的最长对话,默认 32 768 个 token。请设得比最长的文档更大:封存之后显存不再增长。
- `temperature: 0` 开启草稿 token,与 `cortiq run` 的 `--greedy` 相同。`enable_thinking: false` 要求直接回答。
- `model` 可以是任意名字:服务器只提供一个文件。
- `--host 127.0.0.1` 让服务器只在本机可用;默认监听所有网卡,也不检查 API 密钥。
- 接着上一轮的请求只读取新的内容。长文档封存之后,新的一轮会重新读取整段对话。

### 这个文件做了什么

![小显卡上的长提示](https://huggingface.co/infosave/Qwen3.8-27B-Turbo-cmf/resolve/main/assets/ceiling_zh.png)

**固定的上限。** 提示的前 8192 行在显卡上精确读取,16 GB 显卡上为 4096 行。之后各行的键和值一读完就离开显卡,进入内存或 SSD。在最后 512 行之前,由这些行逐层构建远场,此后不再需要它们。远场让前 4 行和后 1800 行保持精确,其余内容存放在每个头 32 个地标中,所以显存和单个 token 的成本不再随文本增长。美国专利申请号 19/738,763。

**精确行。** 只用远场会丢失密码。因此根据各层对每一行的处理为其打分,最好的 248 行保持精确。提示末尾的问题在回答之前会读到它们。

![密码和编码保持精确](https://huggingface.co/infosave/Qwen3.8-27B-Turbo-cmf/resolve/main/assets/exact_rows_zh.png)

**短线程和草稿。** 封存后的 token 和成批的行把注意力拆成七个小阶段,占满整张显卡;而每个头一条长线程会让显卡大部分闲置。MTP 头提出草稿 token,显卡一批验证,远场只前移被接受的部分。贪婪解码时,有无草稿的文本相同。

### 速度

![普通文件与 turbo](https://huggingface.co/infosave/Qwen3.8-27B-Turbo-cmf/resolve/main/assets/compare_zh.png)

`cortiq bench --core`,在该长度的提示之后生成 60 个 token。橙线和蓝线是走 Vulkan 的 RTX 5090(见下表);绿线是 turbo 文件在 16 GB RTX 2000 Ada 上走 CUDA,cortiq 0.8.71:显存保持在 15.0 GB,从 16 000 个 token 起与 5090 不相上下。

| 提示,token 数 | 首个 token,秒(普通) | 首个 token,秒(**turbo**) | 解码,tok/s(普通) | 解码,tok/s(**turbo**) | 显存峰值,GB(普通) | 显存峰值,GB(**turbo**) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 000 | 2.2 | **2.2** | 105 | **105** | 14.5 | **14.4** |
| 4 000 | 8.8 | **8.8** | 104 | **104** | 14.9 | **14.8** |
| 16 000 | 42 | **47** | 87 | **92** | 16.5 | **15.5** |
| 32 000 | 107 | **92** | 71 | **91** | 18.5 | **15.7** |
| 64 000 | 316 | **177** | 51 | **92** | 22.8 | **15.7** |

真实回答:400 个 token 的维基百科文章摘要,约一半草稿被接受。

| 提示 | 首个 token,秒(普通) | 首个 token,秒(**turbo**) | 解码,tok/s(普通) | 解码,tok/s(**turbo**) |
|---|---:|---:|---:|---:|
| 30 000 字符 | 19.2 | **18.6** | 76.4 | **76.3** |
| 100 000 字符 | 76.7 | **72.1** | 50.9 | **60.4** |

**16 GB 显卡上走 CUDA。** RTX 2000 Ada,cortiq 0.8.71,同样的 bench(文本是重复的,所以大多数草稿被接受)。整个运行连同提示都不超过 15.0 GB;普通文件在 16 000 个 token 时超出这块显卡的显存。与同一显卡上的 Vulkan 相比,CUDA 首个 token 快 1.4–1.6 倍,解码快 3.1–3.7 倍。

![CUDA 上的 Qwen3.8-27B Turbo](https://huggingface.co/infosave/Qwen3.8-27B-Turbo-cmf/resolve/main/assets/cuda_zh.png)

| 提示,token | 首个 token,秒(**CUDA**) | 首个 token,秒(Vulkan) | 解码,tok/s(**CUDA**) | 解码,tok/s(Vulkan) | 显存峰值,GB(CUDA / Vulkan) |
|---:|---:|---:|---:|---:|---:|
| 1 000 | **2.55** | 3.58 | **87.3** | 27.9 | 14.6 / 14.1 |
| 4 000 | **9.71** | 14.88 | **86.9** | 26.7 | 14.9 / 14.3 |
| 16 000 | **43.2** | 70.4 | **94.4** | 25.4 | 15.0 / 14.8 |
| 32 000 | **86.2** | 135.6 | **93.5** | 26.7 | 15.0 / 14.8 |
| 64 000 | **174.6** | 277.9 | **93.8** | 26.5 | 15.0 / 14.8 |

真实回答——概括 100 000 个字符的维基百科,最多 400 个 token:首个 token 69.8 秒,之后 31.5 tok/s,不用草稿时文本相同(Vulkan:114.3 秒、21.3 tok/s)。60 000 和 100 000 字符中的事实:8/8。

**RTX 3090 走 CUDA** 时,turbo 文件封存之后在同样的 bench 中解码 118–128 tok/s,以 80.3 tok/s 写出 100 000 个字符的概括。

**在 Mac 上。** 测量机器是 MacBook Air M4,24 GB:没有风扇,长时间运行会发热降速;内存总线是基础版 M4 的,Apple 标称 120 GB/s,实测 92 到 95 GB/s。每个 token 要读取 13.6 GB 权重,所以速度由总线决定:不带草稿时 6.4 tok/s。M4 Pro 的总线标称 273 GB/s,M4 Max 最高 546 GB/s;这些机器没有测量。

16 000 个 token 时,进程除内存映射的权重外保持在 6.3 GB,首个 token 545 秒,解码 4 tok/s,插入 40 000 字符中的事实完整返回:封存点之后,这个文件在 Mac 上节省的是内存,不是时间。

两个开关可以提速,默认都关闭。草稿,`CMF_GRAPH_SPEC=1`,配合 `--greedy`,220 个 token,带草稿和不带草稿的文本相同:

| 回答 | 不带草稿,tok/s | 带草稿,tok/s |
|---|---:|---:|
| 一篇文章 | 6.4 | **10.1** |
| 俄语讲解 | 6.4 | **10.7** |
| 带测试的 Rust 函数 | 6.4 | **12.1** |
| 引用提示中的原文 | 6.4 | **15.0** |
| 修改提示中的代码 | 6.4 | **20.5** |

Neural Engine,`CMF_ANE=1`,与 GPU 一起读取提示:3 600 个 token 时首个 token 73.9 → 28.3 秒,文本相同。两个开关都需要 cortiq 0.8.59 或更新版本。

**主机内存。** 读取长提示时,它的行占用内存或磁盘。这里不计内存映射的权重,系统可以把它们移出内存。

| 提示,token 数 | 内存 | 使用 `--o1-spill`:内存 | 使用 `--o1-spill`:磁盘 |
|---:|---:|---:|---:|
| 16 000 | 4.4 GB | 2.2 GB | 4.7 GB |
| 32 000 | 7.0 GB | 2.2 GB | 8.7 GB |
| 64 000 | 13.2 GB | 3.0 GB | 18.7 GB |

在写入 4.5 GB/s、读取 5.2 GB/s 的磁盘上,速度与使用内存相同。这些行只在构建远场之前需要。

**质量。** 在 100 000 字符中插入事实,共四类(密钥、访问码、密码、房间号):12/12,其中 8 次使用 15 GB 预算和 SSD;在 230 000 字符(约 53 000 token)中同样路径 6/6。没有精确行的远场为 0/4。封存点之后各行的困惑度,两个 16 384 token 的维基百科窗口:精确注意力 4.76,远场 4.97。

### 参数

| 参数 | 含义 |
|---|---|
| `--o1-spill DIR` | 把长提示的行放在 `DIR` 的文件中,也可用 `CMF_O1_SPILL` |
| `--prompt-file FILE` | 比命令行能容纳的更长的提示 |
| `CMF_O1_PREFIX_MAX=N` | 在 `N` 行封存,不按显卡选择 |
| `CMF_O1=off` | 普通模型,任何长度都用精确注意力 |
| `CMF_O1_SPEC=0` | 远场上不用草稿 |

### 限制

- 远场运行在 Vulkan、DX12、Metal 和 CUDA 上。在 CPU 上,这个文件按普通文件运行。测量在 NVIDIA 显卡(Vulkan 和 CUDA)和 M4 上进行。
- 封存点之后的文本是近似的:远处的内容会淡化,保持精确的行除外。选择这些行的拟合只见过插入的事实,其他类型的重要文本没有测试。
- 提示结束后到达的行留在窗口中或进入远场。与已封存对话不一致的回合会重新读取整个提示。
- 在 Vulkan 上,草稿需要不带惩罚的贪婪解码。不加 `--greedy` 时,封存后的 token 约为 57 tok/s。
- 从封存点以下开始的生成保持精确,其内存随每个 token 增长。

### 构建 turbo 文件

精确行拟合的权重及其生成脚本在[仓库](https://github.com/infosave2007/cmf/tree/master/scripts/dory/router)中。

```bash
cortiq far-turbo qwen38-27b-q4tp.cmf qwen38-27b-turbo.cmf --exact-weights keep27.bin
```

详见[指南](https://github.com/infosave2007/cmf/blob/master/docs/QWEN38_TURBO.zh.md)。

### 许可证

权重:[Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0),来自 [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)。远场受美国专利申请号 19/738,763 保护;见 [PATENTS.md](https://github.com/infosave2007/cmf/blob/master/PATENTS.md)。
