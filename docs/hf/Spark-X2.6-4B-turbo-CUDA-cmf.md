---
license: apache-2.0
library_name: cortiq
base_model:
  - XHToken/Spark-X2.6-4B
base_model_relation: quantized
pipeline_tag: text-generation
tags:
  - cmf
  - cortiq
  - quantized
  - q4tp
  - cuda
  - nvidia
  - long-context
  - sliding-window-attention
  - tool-calling
language:
  - multilingual
  - en
  - zh
  - ru
---

# Spark-X2.6-4B Turbo on CUDA — CMF

[Spark-X2.6-4B](https://huggingface.co/XHToken/Spark-X2.6-4B) in 4 and 8 bits on an NVIDIA card, run by the native CUDA backend of `cortiq`, a
Rust inference engine with no Python and no ML framework. On an A40 the 4-bit file reads a 64 000-token
prompt in 42.3 s and then decodes at 100.7 tok/s in 3.9 GB of the card: past 8 192 tokens the
file's far field holds decode speed and memory in place. The same file over Vulkan on the same card needs
60.3 s and decodes at 85.9 tok/s. A 4 000-token prompt is read at 6096 tok/s and decoded at
111.5 tok/s, against 3789 and 80.2 over Vulkan. Passwords and codes planted in texts of up to 240 000
characters are found 8 times of 8. Each file is one [CMF](https://github.com/infosave2007/cmf) file, the same one that
runs on Metal, Vulkan, DX12 and the CPU.

[English](#quick-start) · [Русский](#документация-на-русском) · [中文](#中文说明)

## Quick start

```bash
cargo install --git https://github.com/infosave2007/cmf --branch cuda-backend cortiq-cli --features cuda
hf download infosave/Spark-X2.6-4B-turbo-CUDA-cmf spark-x2.6-4b-q4tp-turbo.cmf --local-dir .
CMF_GPU=cuda CMF_MAX_SEQ=131072 cortiq run spark-x2.6-4b-q4tp-turbo.cmf --prompt-file report.txt --greedy --no-think
```

A short prompt, and a benchmark of your own card:

```bash
CMF_GPU=cuda cortiq run spark-x2.6-4b-q4tp-turbo.cmf --prompt "Explain how a hash map works." --greedy --no-think
CMF_GPU=cuda CMF_MAX_SEQ=131072 cortiq bench spark-x2.6-4b-q4tp-turbo.cmf --ctx 16000 --tokens 60 --core
```

| what it needs | |
|---|---|
| card | NVIDIA, with its driver (`libcuda.so.1`) |
| libraries | CUDA 12.x: `libnvrtc.so.12` and `libcublas.so.12` on the loader's path; the CUDA Toolkit installs them |
| build | stable Rust; no CUDA toolkit at build time, nothing is linked |
| engine | `cortiq` from the `cuda-backend` branch with the `cuda` feature; the prebuilt release binaries do not include the backend |

`CMF_MAX_SEQ` lifts the default cap of 32 768 tokens. `cortiq gpu` says whether the card and the libraries
are there:

```
CUDA: NVIDIA A40 (compute_86, 47.7 GB) — `CMF_GPU=cuda` runs dense attention stacks on it
CUDA: kernels compiled by NVRTC, prompt GEMMs on cuBLAS
```

If a library is missing the line reads `CUDA: unavailable` with the reason, and `CMF_GPU=cuda` runs on
the CPU.

## Server

```bash
CMF_GPU=cuda CMF_MAX_SEQ=131072 cortiq serve spark-x2.6-4b-q4tp-turbo.cmf --port 8080 --host 127.0.0.1
```

An OpenAI-compatible HTTP API with a web dashboard on the same port; any OpenAI client works with the
base URL `http://127.0.0.1:8080/v1`.

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "spark",
  "messages": [{"role": "user", "content": "Explain how a hash map works."}],
  "max_tokens": 256,
  "temperature": 0,
  "enable_thinking": false,
  "stream": true
}'
```

| endpoint | what it does |
|---|---|
| `POST /v1/chat/completions` | a chat turn; `stream: true` sends the answer as it is decoded and ends with `data: [DONE]` |
| `POST /v1/completions` | the continuation of a `prompt` |
| `GET /v1/models` | the served model |
| `GET /healthz` | 200 while the server is up |
| `GET /` | the web dashboard |

- `model` is required in a request and may be any name: one file is served.
- The default `--host` is `0.0.0.0`, every interface, and the server checks no keys; `--host 127.0.0.1`
  keeps it local.
- `CMF_MAX_SEQ` is the context cap of every request, 32 768 by default.
- A request that continues the previous one reads only what is new: the rows of the shared beginning stay
  on the card. Past 8 192 tokens a new turn reads the conversation again.
- `CMF_SERVE_SLOTS=N` keeps N conversations at once (one for every four CPU cores by default, four at
  most); their requests take turns on the card and give the same texts as one after the other.
- After each request the server prints what it took, as llama-server does. A document of 20 283 tokens on
  an A40, read, sealed and answered:

```
prompt eval time =    8334.96 ms / 20283 tokens (    0.41 ms per token,  2433.49 tokens per second)
       eval time =     710.31 ms /    63 tokens (   11.27 ms per token,    88.69 tokens per second)
      total time =    9045.26 ms / 20347 tokens
```

## Long prompts: the far field

![Long prompts: the far field on CUDA](https://huggingface.co/infosave/Spark-X2.6-4B-turbo-CUDA-cmf/resolve/main/assets/turbo_en.png)

Past 8 192 tokens the turbo header takes over. The prompt is read whole with exact attention on the card
and sealed once after it: 248 rows are chosen to stay exact, the first 4 and the last 1800 stay exact
too, and 32 landmarks a head hold everything else. From there the cost of a token and the memory of the
card do not grow with the text.

NVIDIA A40 (46 GB), driver 595.91, CUDA 12.8 libraries, Ubuntu 24.04. The 4-bit file, `cortiq bench --core`,
60 tokens after a prompt of that length, medians of three runs, the three paths taking turns; then one run
of 500 tokens a path for the long decode and the memory while decoding.

| prompt, tokens | path | prompt, tok/s | first token, s | decode, tok/s | decode over 500 tokens, tok/s | peak VRAM, GB | VRAM while decoding, GB |
|---:|---|---:|---:|---:|---:|---:|---:|
| 16 000 | Vulkan, far field | 1001 | 16.0 | 83.6 | 85.2 | 4.6 | 4.2 |
| 16 000 | **CUDA, far field** | **2622** | **6.1** | 95.1 | **103.0** | 4.9 | **3.9** |
| 16 000 | CUDA, exact attention | 4828 | 3.3 | 92.6 | 92.1 | 4.9 | 4.9 |
| 32 000 | Vulkan, far field | 1030 | 31.1 | 83.8 | 87.0 | 4.6 | 4.2 |
| 32 000 | **CUDA, far field** | **2161** | **14.8** | 91.1 | **102.5** | 6.0 | **3.9** |
| 32 000 | CUDA, exact attention | 3533 | 9.1 | 74.7 | 74.8 | 6.0 | 6.0 |
| 64 000 | Vulkan, far field | 1061 | 60.3 | 86.0 | 85.9 | 4.5 | 4.2 |
| 64 000 | **CUDA, far field** | **1511** | **42.3** | 84.9 | **100.7** | 8.2 | **3.9** |
| 64 000 | CUDA, exact attention | 2149 | 29.8 | 53.5 | 53.4 | 8.2 | 8.2 |

The 8-bit file on CUDA: 2535 tok/s, first token in 6.3 s and 68.5 tok/s at 16 000 tokens; 2113 tok/s, 15.1 s and 68.4 tok/s at 32 000.

The first token after the seal moves the far-field state onto the card and frees the prompt's rows, so a
60-token decode reads lower than a long one. Until the seal the K/V rows of the prompt are on the card, so the peak grows with the prompt
(8.2 GB at 64 000 tokens); then they leave, and decoding holds 3.9 GB. Over Vulkan the far field
seals at 8 192 rows and reads the rest of the prompt through it: 4.5–4.6 GB at any length and a slower
prompt. `CMF_O1=off` runs the file with exact attention at any length, the third row.

## Exact attention

![CUDA and Vulkan on the same card](https://huggingface.co/infosave/Spark-X2.6-4B-turbo-CUDA-cmf/resolve/main/assets/speed_en.png)

Below the seal, and at any length with `CMF_O1=off`, both paths run exact attention. The same card and
method.

| file | prompt, tokens | prompt, tok/s (Vulkan) | prompt, tok/s (**CUDA**) | first token, s (Vulkan) | first token, s (**CUDA**) | decode, tok/s (Vulkan) | decode, tok/s (**CUDA**) | peak VRAM, GB (Vulkan) | peak VRAM, GB (**CUDA**) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| q4tp | 1 000 | 3177 | **6132** | 0.31 | **0.16** | 82.9 | **116.9** | 3.6 | **3.5** |
| q4tp | 4 000 | 3789 | **6096** | 1.06 | **0.66** | 80.2 | **111.5** | 4.5 | **3.9** |
| q4tp | 16 000 | 2949 | **4808** | 5.43 | **3.33** | 54.3 | **92.7** | 5.5 | **4.9** |
| q8_2f | 1 000 | 513 | **5926** | 1.95 | **0.17** | 63.5 | **82.7** | 5.2 | **5.2** |
| q8_2f | 4 000 | 528 | **5927** | 7.58 | **0.67** | 60.1 | **80.0** | 6.1 | **5.6** |
| q8_2f | 16 000 | 507 | **4732** | 31.53 | **3.38** | 45.3 | **69.4** | 7.1 | **6.6** |

## Quality

A fact (a pass key, an access code, a password, a room number) at a random place of Wikipedia text, the
question at the end, eight trials a length, the 4-bit file on CUDA:

| text | far field | exact attention |
|---|---:|---:|
| 60 000 characters (about 14 000 tokens) | **8 of 8** | 8 of 8 |
| 120 000 characters (about 28 000 tokens) | **8 of 8** | 8 of 8 |
| 240 000 characters (about 56 000 tokens) | **8 of 8** | 8 of 8 |

Against the host with float activations (`CMF_GPU=0 CMF_SDOT=0`), prompts below the seal, 96 decoded tokens
each, every path forced along the same ids:

| file | path | top-1 as the reference | largest logit difference |
|---|---|---:|---:|
| q4tp | **CUDA** | 384/384 | 0.020 |
| q4tp | CUDA, `CMF_CUDA_PREC=exact` | 384/384 | 0.0002 |
| q4tp | Vulkan | 384/384 | 0.020 |
| q8_2f | **CUDA** | 384/384 | 0.023 |
| q8_2f | CUDA, `CMF_CUDA_PREC=exact` | 384/384 | 0.0002 |
| q8_2f | Vulkan | 384/384 | 0.015 |

With exact attention CUDA and Vulkan give the same greedy text: 576 of 576 tokens on six prompts of 20 to
20 255 tokens for each file.
The far-field kernels were checked against the host's from one and the same seal of a 20 255-token prompt:
96 of 96 tokens the same, logits within 0.0004. Past the seal the text depends on the prompt's arithmetic
more than exact attention does, so two paths, or two precisions of one, can part after some tokens.

## How it runs

![A token and a prompt on the card](https://huggingface.co/infosave/Spark-X2.6-4B-turbo-CUDA-cmf/resolve/main/assets/paths_en.png)

**Decode** keeps the weights in their codec on the card and reads each matrix once a token. One kernel
multiplies several matrices by the same input row in a launch. Attention is three small kernels over the
K/V rows the prompt left on the card, the head runs there too, and the logits come back once a token.

**The prompt** goes through the 36 layers in chunks of 1024 rows. Each weight is expanded to an f16 plane
once a chunk and multiplied by cuBLAS on the tensor cores with f32 accumulation. `CMF_CUDA_PREC` sets its
arithmetic: `fast` (the default: f16 operands, f32 accumulation; 6096 tok/s at 4 000 tokens), `tf32` (4152)
or `exact` (f32; 1788).

**The far field** adds two launches a layer to a token: the row leaving the window goes into the far sums,
and the far part of the token's attention joins the near rows under one shift. Its state is 18 MB a layer.

The backend is described in
[docs/CUDA.md](https://github.com/infosave2007/cmf/blob/cuda-backend/docs/CUDA.md), the far field on the
[Turbo page](https://huggingface.co/infosave/Spark-X2.6-4B-turbo-cmf).

## Options

| option | meaning |
|---|---|
| `CMF_GPU=cuda` | select the backend (a build with `--features cuda`) |
| `CMF_MAX_SEQ=N` | the context cap (32 768 by default) |
| `CMF_O1=off` | exact attention at any length |
| `CMF_CUDA_PREC=fast\|tf32\|exact` | the prompt's precision |
| `CMF_CUDA_DEVICE=N` | the card's ordinal, 0 by default |
| `CMF_CUDA_BATCH=N` | prompt rows per chunk, 1024 by default |
| `CMF_GPU_VRAM_MB=N` | the weight budget instead of the card's memory less 2 GB |
| `--prompt-file FILE` | a prompt longer than a command line holds |

## Limits

- One card was measured, an A40 under Linux.
- The prompt of the far field is read whole: its K/V rows stay on the card until the seal (8.2 GB at
  64 000 tokens), and its time grows faster than the text.
- A new turn of a sealed conversation reads the conversation again.
- No speculative decoding on this backend.
- The bindings are for CUDA 12.x, and the two libraries must be on the loader's path.
- The backend covers dense attention stacks of the Spark-X2.x class. Under `CMF_GPU=cuda` other
  architectures run on the CPU; use Vulkan for them.

## Files

| file | size | what it is | sha256 |
|---|---:|---|---|
| `spark-x2.6-4b-q4tp-turbo.cmf` | 2.15 GB | 4-bit weights, the default | `ed22ff69b15f5f29c09b45704b146efd87e0a5a68b80bd2511efc3bce0ed100b` |
| `spark-x2.6-4b-q8_2f-turbo.cmf` | 4.13 GB | 8-bit weights | `b9eb6ec30f220dff3f1097105cd91603dc1d376eec0ef5408eda79135171da44` |

They are the files of the [Turbo page](https://huggingface.co/infosave/Spark-X2.6-4B-turbo-cmf), byte for byte.

## License

Weights: [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0), from
[XHToken/Spark-X2.6-4B](https://huggingface.co/XHToken/Spark-X2.6-4B). The far field is a subject of U.S. patent application No. 19/738,763; see
[PATENTS.md](https://github.com/infosave2007/cmf/blob/master/PATENTS.md).

---

## Документация на русском

[Spark-X2.6-4B](https://huggingface.co/XHToken/Spark-X2.6-4B) в 4 и 8 битах на карте NVIDIA через нативный бэкенд CUDA движка `cortiq` — движка
вывода на Rust без Python и без ML-фреймворка. На A40 4-битный файл читает промпт в 64 000 токенов за
42,3 с и затем генерирует 100,7 tok/s в 3,9 ГБ карты: после 8 192 токенов дальнее поле файла
держит скорость генерации и память на месте. Тому же файлу через Vulkan на той же карте нужно 60,3 с,
генерация 85,9 tok/s. Промпт в 4 000 токенов читается на 6096 tok/s и генерируется на 111,5 tok/s —
против 3789 и 80,2 через Vulkan. Пароли и коды, подложенные в тексты до 240 000 символов, найдены 8 раз
из 8. Каждый файл — один файл [CMF](https://github.com/infosave2007/cmf), тот же, что работает на Metal,
Vulkan, DX12 и CPU.

### Быстрый старт

```bash
cargo install --git https://github.com/infosave2007/cmf --branch cuda-backend cortiq-cli --features cuda
hf download infosave/Spark-X2.6-4B-turbo-CUDA-cmf spark-x2.6-4b-q4tp-turbo.cmf --local-dir .
CMF_GPU=cuda CMF_MAX_SEQ=131072 cortiq run spark-x2.6-4b-q4tp-turbo.cmf --prompt-file report.txt --greedy --no-think
```

Короткий промпт и замер на своей карте:

```bash
CMF_GPU=cuda cortiq run spark-x2.6-4b-q4tp-turbo.cmf --prompt "Explain how a hash map works." --greedy --no-think
CMF_GPU=cuda CMF_MAX_SEQ=131072 cortiq bench spark-x2.6-4b-q4tp-turbo.cmf --ctx 16000 --tokens 60 --core
```

| что нужно | |
|---|---|
| карта | NVIDIA с драйвером (`libcuda.so.1`) |
| библиотеки | CUDA 12.x: `libnvrtc.so.12` и `libcublas.so.12` на пути загрузчика; их ставит CUDA Toolkit |
| сборка | стабильный Rust; toolkit CUDA при сборке не нужен, ничего не линкуется |
| движок | `cortiq` из ветки `cuda-backend` с feature `cuda`; в готовых бинарниках релизов бэкенда нет |

`CMF_MAX_SEQ` снимает потолок по умолчанию в 32 768 токенов. `cortiq gpu` говорит, на месте ли карта и
библиотеки:

```
CUDA: NVIDIA A40 (compute_86, 47.7 GB) — `CMF_GPU=cuda` runs dense attention stacks on it
CUDA: kernels compiled by NVRTC, prompt GEMMs on cuBLAS
```

Если библиотеки нет, строка сообщает `CUDA: unavailable` и причину, а `CMF_GPU=cuda` работает на CPU.

### Сервер

```bash
CMF_GPU=cuda CMF_MAX_SEQ=131072 cortiq serve spark-x2.6-4b-q4tp-turbo.cmf --port 8080 --host 127.0.0.1
```

OpenAI-совместимый HTTP API и веб-панель на том же порту; любой клиент OpenAI работает с базовым адресом
`http://127.0.0.1:8080/v1`.

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "spark",
  "messages": [{"role": "user", "content": "Explain how a hash map works."}],
  "max_tokens": 256,
  "temperature": 0,
  "enable_thinking": false,
  "stream": true
}'
```

| адрес | что делает |
|---|---|
| `POST /v1/chat/completions` | реплика чата; `stream: true` отдаёт ответ по мере генерации и заканчивает его строкой `data: [DONE]` |
| `POST /v1/completions` | продолжение текста `prompt` |
| `GET /v1/models` | обслуживаемая модель |
| `GET /healthz` | 200, пока сервер работает |
| `GET /` | веб-панель |

- Поле `model` в запросе обязательно, имя может быть любым: сервер обслуживает один файл.
- По умолчанию `--host` равен `0.0.0.0`, то есть все интерфейсы, и сервер не проверяет ключи;
  `--host 127.0.0.1` оставляет его локальным.
- `CMF_MAX_SEQ` — потолок контекста каждого запроса, по умолчанию 32 768.
- Запрос, продолжающий предыдущий, читает только новое: строки общего начала остаются на карте. После
  8 192 токенов новая реплика читает диалог заново.
- `CMF_SERVE_SLOTS=N` держит N диалогов одновременно (по умолчанию один на каждые четыре ядра CPU, не
  больше четырёх); их запросы идут по карте по очереди и дают те же тексты, что и по одному.
- После каждого запроса сервер печатает, сколько он занял, как llama-server. Документ в 20 283 токена на
  A40 — чтение, печать и ответ:

```
prompt eval time =    8334.96 ms / 20283 tokens (    0.41 ms per token,  2433.49 tokens per second)
       eval time =     710.31 ms /    63 tokens (   11.27 ms per token,    88.69 tokens per second)
      total time =    9045.26 ms / 20347 tokens
```

### Длинные промпты: дальнее поле

![Длинные промпты: дальнее поле на CUDA](https://huggingface.co/infosave/Spark-X2.6-4B-turbo-CUDA-cmf/resolve/main/assets/turbo_ru.png)

После 8 192 токенов работает турбо-заголовок. Промпт читается целиком с точным вниманием на карте и
запечатывается один раз после него: 248 строк выбираются точными, первые 4 и последние 1800 тоже остаются
точными, а всё остальное держат 32 опорные точки на голову. Дальше цена токена и память карты не растут
вместе с текстом.

NVIDIA A40 (46 ГБ), драйвер 595.91, библиотеки CUDA 12.8, Ubuntu 24.04. 4-битный файл, `cortiq bench --core`,
60 токенов после промпта такой длины, медианы трёх прогонов, три пути по очереди; затем по одному прогону
на 500 токенов на путь — для длинной генерации и памяти при декоде.

| промпт, токенов | путь | промпт, tok/s | первый токен, с | генерация, tok/s | генерация на 500 токенах, tok/s | пик VRAM, ГБ | VRAM при декоде, ГБ |
|---:|---|---:|---:|---:|---:|---:|---:|
| 16 000 | Vulkan, дальнее поле | 1001 | 16,0 | 83,6 | 85,2 | 4,6 | 4,2 |
| 16 000 | **CUDA, дальнее поле** | **2622** | **6,1** | 95,1 | **103,0** | 4,9 | **3,9** |
| 16 000 | CUDA, точное внимание | 4828 | 3,3 | 92,6 | 92,1 | 4,9 | 4,9 |
| 32 000 | Vulkan, дальнее поле | 1030 | 31,1 | 83,8 | 87,0 | 4,6 | 4,2 |
| 32 000 | **CUDA, дальнее поле** | **2161** | **14,8** | 91,1 | **102,5** | 6,0 | **3,9** |
| 32 000 | CUDA, точное внимание | 3533 | 9,1 | 74,7 | 74,8 | 6,0 | 6,0 |
| 64 000 | Vulkan, дальнее поле | 1061 | 60,3 | 86,0 | 85,9 | 4,5 | 4,2 |
| 64 000 | **CUDA, дальнее поле** | **1511** | **42,3** | 84,9 | **100,7** | 8,2 | **3,9** |
| 64 000 | CUDA, точное внимание | 2149 | 29,8 | 53,5 | 53,4 | 8,2 | 8,2 |

8-битный файл на CUDA: 2535 tok/s, первый токен через 6,3 с и 68,5 tok/s на 16 000 токенов; 2113 tok/s, 15,1 с и 68,4 tok/s на 32 000.

Первый токен после печати переносит состояние дальнего поля на карту и освобождает строки промпта,
поэтому генерация на 60 токенах ниже, чем на длинном ответе. До печати строки K/V промпта лежат на карте, поэтому пик растёт вместе с промптом (8,2 ГБ на
64 000 токенов); затем они уходят, и декод держит 3,9 ГБ. Через Vulkan дальнее поле запечатывается на
8 192 строках и читает остальной промпт через себя: 4,5–4,6 ГБ на любой длине и более медленный промпт.
`CMF_O1=off` запускает файл с точным вниманием на любой длине — третья строка.

### Точное внимание

![CUDA и Vulkan на одной карте](https://huggingface.co/infosave/Spark-X2.6-4B-turbo-CUDA-cmf/resolve/main/assets/speed_ru.png)

Ниже печати, а с `CMF_O1=off` на любой длине, оба пути считают точное внимание. Та же карта и тот же
способ замера.

| файл | промпт, токенов | промпт, tok/s (Vulkan) | промпт, tok/s (**CUDA**) | первый токен, с (Vulkan) | первый токен, с (**CUDA**) | генерация, tok/s (Vulkan) | генерация, tok/s (**CUDA**) | пик VRAM, ГБ (Vulkan) | пик VRAM, ГБ (**CUDA**) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| q4tp | 1 000 | 3177 | **6132** | 0,31 | **0,16** | 82,9 | **116,9** | 3,6 | **3,5** |
| q4tp | 4 000 | 3789 | **6096** | 1,06 | **0,66** | 80,2 | **111,5** | 4,5 | **3,9** |
| q4tp | 16 000 | 2949 | **4808** | 5,43 | **3,33** | 54,3 | **92,7** | 5,5 | **4,9** |
| q8_2f | 1 000 | 513 | **5926** | 1,95 | **0,17** | 63,5 | **82,7** | 5,2 | **5,2** |
| q8_2f | 4 000 | 528 | **5927** | 7,58 | **0,67** | 60,1 | **80,0** | 6,1 | **5,6** |
| q8_2f | 16 000 | 507 | **4732** | 31,53 | **3,38** | 45,3 | **69,4** | 7,1 | **6,6** |

### Качество

Факт (ключ, код доступа, пароль, номер комнаты) в случайном месте текста Википедии, вопрос в конце, по
восемь попыток на длину, 4-битный файл на CUDA:

| текст | дальнее поле | точное внимание |
|---|---:|---:|
| 60 000 символов (около 14 000 токенов) | **8 из 8** | 8 из 8 |
| 120 000 символов (около 28 000 токенов) | **8 из 8** | 8 из 8 |
| 240 000 символов (около 56 000 токенов) | **8 из 8** | 8 из 8 |

Против хоста с float-активациями (`CMF_GPU=0 CMF_SDOT=0`), промпты ниже печати, по 96 токенов, все пути идут
по одним и тем же id:

| файл | путь | top-1 как у эталона | наибольшее расхождение логитов |
|---|---|---:|---:|
| q4tp | **CUDA** | 384/384 | 0,020 |
| q4tp | CUDA, `CMF_CUDA_PREC=exact` | 384/384 | 0,0002 |
| q4tp | Vulkan | 384/384 | 0,020 |
| q8_2f | **CUDA** | 384/384 | 0,023 |
| q8_2f | CUDA, `CMF_CUDA_PREC=exact` | 384/384 | 0,0002 |
| q8_2f | Vulkan | 384/384 | 0,015 |

С точным вниманием CUDA и Vulkan дают один и тот же жадный текст: 576 из 576 токенов на шести промптах от
20 до 20 255 токенов для каждого файла. Ядра дальнего поля сверены с реализацией хоста от одной и той же печати промпта в 20 255 токенов:
96 из 96 токенов те же, логиты в пределах 0,0004. После печати текст зависит от арифметики промпта сильнее,
чем при точном внимании, поэтому два пути или две точности одного пути могут разойтись через несколько
токенов.

### Как это работает

![Токен и промпт на карте](https://huggingface.co/infosave/Spark-X2.6-4B-turbo-CUDA-cmf/resolve/main/assets/paths_ru.png)

**Декод** держит веса на карте в их кодеке и читает каждую матрицу один раз на токен. Одно ядро за запуск
умножает несколько матриц на одну входную строку. Внимание — три небольших ядра над строками K/V, которые
промпт оставил на карте; голова считается там же, логиты возвращаются один раз на токен.

**Промпт** проходит 36 слоёв кусками по 1024 строки. Каждый вес раз в кусок разворачивается в плоскость
f16 и умножается через cuBLAS на тензорных ядрах с накоплением в f32. Арифметику задаёт `CMF_CUDA_PREC`:
`fast` (по умолчанию: операнды f16, накопление f32; 6096 tok/s на 4 000 токенов), `tf32` (4152) или
`exact` (f32; 1788).

**Дальнее поле** добавляет токену два запуска на слой: строка, выходящая из окна, уходит в дальние суммы,
а дальняя часть внимания токена складывается с ближними строками под одним сдвигом. Его состояние — 18 МБ
на слой.

Бэкенд описан в
[docs/CUDA.ru.md](https://github.com/infosave2007/cmf/blob/cuda-backend/docs/CUDA.ru.md), дальнее поле — на
[странице Turbo](https://huggingface.co/infosave/Spark-X2.6-4B-turbo-cmf).

### Параметры

| параметр | смысл |
|---|---|
| `CMF_GPU=cuda` | выбрать бэкенд (сборка с `--features cuda`) |
| `CMF_MAX_SEQ=N` | потолок контекста (32 768 по умолчанию) |
| `CMF_O1=off` | точное внимание на любой длине |
| `CMF_CUDA_PREC=fast\|tf32\|exact` | точность промпта |
| `CMF_CUDA_DEVICE=N` | номер карты, по умолчанию 0 |
| `CMF_CUDA_BATCH=N` | строк промпта в куске, по умолчанию 1024 |
| `CMF_GPU_VRAM_MB=N` | бюджет весов вместо памяти карты минус 2 ГБ |
| `--prompt-file FILE` | промпт длиннее командной строки |

### Ограничения

- Измерена одна карта, A40 под Linux.
- Промпт под дальнее поле читается целиком: его строки K/V лежат на карте до печати (8,2 ГБ на
  64 000 токенов), и его время растёт быстрее текста.
- Новая реплика запечатанного диалога читает диалог заново.
- Спекулятивного декода на этом бэкенде нет.
- Привязки рассчитаны на CUDA 12.x, обе библиотеки должны быть на пути загрузчика.
- Бэкенд покрывает плотные стеки с вниманием класса Spark-X2.x. Другие архитектуры под `CMF_GPU=cuda`
  идут на CPU; для них используйте Vulkan.

### Файлы

| файл | размер | что это | sha256 |
|---|---:|---|---|
| `spark-x2.6-4b-q4tp-turbo.cmf` | 2,15 ГБ | 4-битные веса, по умолчанию | `ed22ff69b15f5f29c09b45704b146efd87e0a5a68b80bd2511efc3bce0ed100b` |
| `spark-x2.6-4b-q8_2f-turbo.cmf` | 4,13 ГБ | 8-битные веса | `b9eb6ec30f220dff3f1097105cd91603dc1d376eec0ef5408eda79135171da44` |

Это файлы [страницы Turbo](https://huggingface.co/infosave/Spark-X2.6-4B-turbo-cmf), байт в байт.

### Лицензия

Веса: [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0), от
[XHToken/Spark-X2.6-4B](https://huggingface.co/XHToken/Spark-X2.6-4B). Дальнее поле — предмет патентной заявки США № 19/738,763; см.
[PATENTS.md](https://github.com/infosave2007/cmf/blob/master/PATENTS.md).

---

## 中文说明

[Spark-X2.6-4B](https://huggingface.co/XHToken/Spark-X2.6-4B) 的 4 位和 8 位文件，在 NVIDIA 显卡上由 `cortiq` 的原生 CUDA 后端运行；cortiq 是一个 Rust
推理引擎，不依赖 Python 和任何 ML 框架。在 A40 上，4 位文件用 42.3 秒读完 64 000 词元的提示，随后以
100.7 tok/s 生成，占显卡 3.9 GB：超过 8 192 词元后，文件的远场把生成速度和内存固定住。同一个文件在同一块
显卡上走 Vulkan 需要 60.3 秒，生成 85.9 tok/s。4 000 词元的提示以 6096 tok/s 读入、111.5 tok/s 生成，
Vulkan 为 3789 和 80.2。埋在最长 240 000 字符文本中的密码和编码，8 次全部找到。每个文件都是一个
[CMF](https://github.com/infosave2007/cmf) 文件，与在 Metal、Vulkan、DX12 和 CPU 上运行的是同一个文件。

### 快速开始

```bash
cargo install --git https://github.com/infosave2007/cmf --branch cuda-backend cortiq-cli --features cuda
hf download infosave/Spark-X2.6-4B-turbo-CUDA-cmf spark-x2.6-4b-q4tp-turbo.cmf --local-dir .
CMF_GPU=cuda CMF_MAX_SEQ=131072 cortiq run spark-x2.6-4b-q4tp-turbo.cmf --prompt-file report.txt --greedy --no-think
```

短提示，以及在自己的显卡上测速：

```bash
CMF_GPU=cuda cortiq run spark-x2.6-4b-q4tp-turbo.cmf --prompt "Explain how a hash map works." --greedy --no-think
CMF_GPU=cuda CMF_MAX_SEQ=131072 cortiq bench spark-x2.6-4b-q4tp-turbo.cmf --ctx 16000 --tokens 60 --core
```

| 需要什么 | |
|---|---|
| 显卡 | NVIDIA 显卡及其驱动（`libcuda.so.1`） |
| 库 | CUDA 12.x：`libnvrtc.so.12` 和 `libcublas.so.12` 在加载器路径上，由 CUDA Toolkit 安装 |
| 构建 | 稳定版 Rust；构建时不需要 CUDA toolkit，不链接任何库 |
| 引擎 | `cuda-backend` 分支的 `cortiq`，打开 `cuda` feature；发布页的预编译二进制不含该后端 |

`CMF_MAX_SEQ` 解除默认 32 768 词元的上限。`cortiq gpu` 会说明显卡和库是否就位：

```
CUDA: NVIDIA A40 (compute_86, 47.7 GB) — `CMF_GPU=cuda` runs dense attention stacks on it
CUDA: kernels compiled by NVRTC, prompt GEMMs on cuBLAS
```

缺少某个库时，这一行显示 `CUDA: unavailable` 和原因，`CMF_GPU=cuda` 改在 CPU 上运行。

### 服务器

```bash
CMF_GPU=cuda CMF_MAX_SEQ=131072 cortiq serve spark-x2.6-4b-q4tp-turbo.cmf --port 8080 --host 127.0.0.1
```

OpenAI 兼容的 HTTP API，同一端口上还有网页面板；任何 OpenAI 客户端把基础地址设为
`http://127.0.0.1:8080/v1` 即可使用。

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "spark",
  "messages": [{"role": "user", "content": "Explain how a hash map works."}],
  "max_tokens": 256,
  "temperature": 0,
  "enable_thinking": false,
  "stream": true
}'
```

| 地址 | 作用 |
|---|---|
| `POST /v1/chat/completions` | 一轮对话；`stream: true` 边生成边返回，以 `data: [DONE]` 结束 |
| `POST /v1/completions` | 续写 `prompt` |
| `GET /v1/models` | 所服务的模型 |
| `GET /healthz` | 服务器运行时返回 200 |
| `GET /` | 网页面板 |

- 请求里必须有 `model` 字段，名字任意：服务器只服务一个文件。
- `--host` 默认是 `0.0.0.0`，即所有网络接口，而且服务器不检查密钥；`--host 127.0.0.1` 让它只在本机可用。
- `CMF_MAX_SEQ` 是每个请求的上下文上限，默认 32 768。
- 接着上一个请求的新请求只读新增部分：共同开头的行留在显卡上。超过 8 192 词元后，新一轮会把对话重新读一遍。
- `CMF_SERVE_SLOTS=N` 同时保留 N 个对话（默认每四个 CPU 核一个，最多四个）；它们的请求在显卡上轮流执行，
  得到的文本与逐个发出时相同。
- 每个请求结束后，服务器像 llama-server 一样打印耗时。A40 上一份 20 283 词元的文档，读入、封口并回答：

```
prompt eval time =    8334.96 ms / 20283 tokens (    0.41 ms per token,  2433.49 tokens per second)
       eval time =     710.31 ms /    63 tokens (   11.27 ms per token,    88.69 tokens per second)
      total time =    9045.26 ms / 20347 tokens
```

### 长提示：远场

![长提示：CUDA 上的远场](https://huggingface.co/infosave/Spark-X2.6-4B-turbo-CUDA-cmf/resolve/main/assets/turbo_zh.png)

超过 8 192 词元后由 turbo 文件头接管。提示先在显卡上以精确注意力整段读完，之后封口一次：选出 248 行保持
精确，最前 4 行和最后 1800 行也保持精确，其余内容由每个头的 32 个地标承载。此后每个词元的代价和显卡内存
不再随文本增长。

NVIDIA A40（46 GB），驱动 595.91，CUDA 12.8 库，Ubuntu 24.04。4 位文件，`cortiq bench --core`，在该长度的提示
之后生成 60 个词元，三次运行的中位数，三条路径交替进行；然后每条路径各跑一次 500 词元，得到长生成的速度和
解码时的显存。

| 提示，词元 | 路径 | 提示，tok/s | 首词元，秒 | 生成，tok/s | 500 词元的生成，tok/s | 显存峰值，GB | 解码时显存，GB |
|---:|---|---:|---:|---:|---:|---:|---:|
| 16 000 | Vulkan，远场 | 1001 | 16.0 | 83.6 | 85.2 | 4.6 | 4.2 |
| 16 000 | **CUDA，远场** | **2622** | **6.1** | 95.1 | **103.0** | 4.9 | **3.9** |
| 16 000 | CUDA，精确注意力 | 4828 | 3.3 | 92.6 | 92.1 | 4.9 | 4.9 |
| 32 000 | Vulkan，远场 | 1030 | 31.1 | 83.8 | 87.0 | 4.6 | 4.2 |
| 32 000 | **CUDA，远场** | **2161** | **14.8** | 91.1 | **102.5** | 6.0 | **3.9** |
| 32 000 | CUDA，精确注意力 | 3533 | 9.1 | 74.7 | 74.8 | 6.0 | 6.0 |
| 64 000 | Vulkan，远场 | 1061 | 60.3 | 86.0 | 85.9 | 4.5 | 4.2 |
| 64 000 | **CUDA，远场** | **1511** | **42.3** | 84.9 | **100.7** | 8.2 | **3.9** |
| 64 000 | CUDA，精确注意力 | 2149 | 29.8 | 53.5 | 53.4 | 8.2 | 8.2 |

8 位文件在 CUDA 上：16 000 词元时 2535 tok/s、首词元 6.3 秒、生成 68.5 tok/s；32 000 词元时 2113 tok/s、15.1 秒、68.4 tok/s。

封口后的第一个词元要把远场状态传到显卡并释放提示的行，所以 60 词元的生成速度低于长生成。封口之前，提示的 K/V 行都在显卡上，所以峰值随提示增长（64 000 词元时 8.2 GB）；之后这些行离开显卡，
解码保持 3.9 GB。走 Vulkan 时远场在 8 192 行处封口，其余提示经远场读取：任何长度都是 4.5–4.6 GB，
但提示更慢。`CMF_O1=off` 让文件在任何长度都用精确注意力，即第三行。

### 精确注意力

![同一块显卡上的 CUDA 与 Vulkan](https://huggingface.co/infosave/Spark-X2.6-4B-turbo-CUDA-cmf/resolve/main/assets/speed_zh.png)

封口以下，以及在任何长度加上 `CMF_O1=off` 时，两条路径都是精确注意力。同一块显卡，同样的测法。

| 文件 | 提示，词元 | 提示，tok/s（Vulkan） | 提示，tok/s（**CUDA**） | 首词元，秒（Vulkan） | 首词元，秒（**CUDA**） | 生成，tok/s（Vulkan） | 生成，tok/s（**CUDA**） | 显存峰值，GB（Vulkan） | 显存峰值，GB（**CUDA**） |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| q4tp | 1 000 | 3177 | **6132** | 0.31 | **0.16** | 82.9 | **116.9** | 3.6 | **3.5** |
| q4tp | 4 000 | 3789 | **6096** | 1.06 | **0.66** | 80.2 | **111.5** | 4.5 | **3.9** |
| q4tp | 16 000 | 2949 | **4808** | 5.43 | **3.33** | 54.3 | **92.7** | 5.5 | **4.9** |
| q8_2f | 1 000 | 513 | **5926** | 1.95 | **0.17** | 63.5 | **82.7** | 5.2 | **5.2** |
| q8_2f | 4 000 | 528 | **5927** | 7.58 | **0.67** | 60.1 | **80.0** | 6.1 | **5.6** |
| q8_2f | 16 000 | 507 | **4732** | 31.53 | **3.38** | 45.3 | **69.4** | 7.1 | **6.6** |

### 质量

在维基百科文本的随机位置埋入一个事实（密钥、访问码、密码、房间号），问题放在最后，每个长度八次，CUDA 上的
4 位文件：

| 文本 | 远场 | 精确注意力 |
|---|---:|---:|
| 60 000 字符（约 14 000 词元） | **8 中 8** | 8 中 8 |
| 120 000 字符（约 28 000 词元） | **8 中 8** | 8 中 8 |
| 240 000 字符（约 56 000 词元） | **8 中 8** | 8 中 8 |

与使用浮点激活的主机（`CMF_GPU=0 CMF_SDOT=0`）对比，封口以下的提示，各解码 96 个词元，所有路径沿同一组 id 前进：

| 文件 | 路径 | top-1 与参照相同 | 最大 logit 差 |
|---|---|---:|---:|
| q4tp | **CUDA** | 384/384 | 0.020 |
| q4tp | CUDA，`CMF_CUDA_PREC=exact` | 384/384 | 0.0002 |
| q4tp | Vulkan | 384/384 | 0.020 |
| q8_2f | **CUDA** | 384/384 | 0.023 |
| q8_2f | CUDA，`CMF_CUDA_PREC=exact` | 384/384 | 0.0002 |
| q8_2f | Vulkan | 384/384 | 0.015 |

使用精确注意力时，CUDA 和 Vulkan 的贪心文本相同：每个文件在六个 20 到 20 255 词元的提示上 576 个词元全部一致。远场内核与主机的实现从
同一次封口（20 255 词元的提示）出发对比：96 个词元全部相同，logits 相差不超过 0.0004。封口之后，文本对提示
算术的依赖比精确注意力更强，因此两条路径，或同一路径的两种精度，可能在若干词元后分开。

### 工作方式

![显卡上的一个 token 和一段提示](https://huggingface.co/infosave/Spark-X2.6-4B-turbo-CUDA-cmf/resolve/main/assets/paths_zh.png)

**解码**把权重以其编码留在显卡上，每个词元把每个矩阵读一次。一个内核在一次启动中用同一输入行乘多个矩阵。
注意力是三个小内核，读取提示留在显卡上的 K/V 行；输出头也在显卡上计算，logits 每个词元读回一次。

**提示**以每块 1024 行通过 36 层。每个权重每块展开成一个 f16 平面，由 cuBLAS 在张量核上相乘，累加为 f32。
`CMF_CUDA_PREC` 决定其算术：`fast`（默认：f16 操作数、f32 累加；4 000 词元时 6096 tok/s）、`tf32`（4152）或
`exact`（f32；1788）。

**远场**让每个词元在每层多两次启动：离开窗口的行并入远场累加量，词元注意力的远场部分与近处各行在同一个
偏移下相加。它的状态是每层 18 MB。

后端的说明见
[docs/CUDA.zh.md](https://github.com/infosave2007/cmf/blob/cuda-backend/docs/CUDA.zh.md)，远场见
[Turbo 页面](https://huggingface.co/infosave/Spark-X2.6-4B-turbo-cmf)。

### 选项

| 选项 | 含义 |
|---|---|
| `CMF_GPU=cuda` | 选择该后端（需以 `--features cuda` 构建） |
| `CMF_MAX_SEQ=N` | 上下文上限（默认 32 768） |
| `CMF_O1=off` | 任何长度都用精确注意力 |
| `CMF_CUDA_PREC=fast\|tf32\|exact` | 提示的精度 |
| `CMF_CUDA_DEVICE=N` | 显卡序号，默认 0 |
| `CMF_CUDA_BATCH=N` | 每块的提示行数，默认 1024 |
| `CMF_GPU_VRAM_MB=N` | 权重预算，代替显卡内存减 2 GB |
| `--prompt-file FILE` | 命令行放不下的提示 |

### 限制

- 只测了一块显卡：Linux 下的 A40。
- 远场的提示是整段读取的：封口之前它的 K/V 行都在显卡上（64 000 词元时 8.2 GB），所需时间比文本增长得更快。
- 已封口对话的新一轮会把对话重新读一遍。
- 该后端没有推测解码。
- 绑定针对 CUDA 12.x，两个库必须在加载器的路径上。
- 后端覆盖 Spark-X2.x 一类的稠密注意力堆栈。其他架构在 `CMF_GPU=cuda` 下走 CPU，请对它们使用 Vulkan。

### 文件

| 文件 | 大小 | 说明 | sha256 |
|---|---:|---|---|
| `spark-x2.6-4b-q4tp-turbo.cmf` | 2.15 GB | 4 位权重，默认 | `ed22ff69b15f5f29c09b45704b146efd87e0a5a68b80bd2511efc3bce0ed100b` |
| `spark-x2.6-4b-q8_2f-turbo.cmf` | 4.13 GB | 8 位权重 | `b9eb6ec30f220dff3f1097105cd91603dc1d376eec0ef5408eda79135171da44` |

它们与 [Turbo 页面](https://huggingface.co/infosave/Spark-X2.6-4B-turbo-cmf)上的文件逐字节相同。

### 许可

权重：[Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)，来自
[XHToken/Spark-X2.6-4B](https://huggingface.co/XHToken/Spark-X2.6-4B)。远场是美国专利申请 No. 19/738,763 的主题；见
[PATENTS.md](https://github.com/infosave2007/cmf/blob/master/PATENTS.md)。
