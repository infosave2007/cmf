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
  - q8_2f
  - looped-transformer
  - mhc
  - mtp
  - vision
  - cuda
  - vulkan
  - metal
language:
  - en
  - ru
  - zh
---

# SenseNova-Loop-9B — CMF

[SenseNova-Loop-9B](https://huggingface.co/sensenova/SenseNova-Loop-9B) (released 2026-11-17) as single
[CMF](https://github.com/infosave2007/cmf) files for `cortiq`, a Rust inference engine with no Python and no ML
framework. The model runs 24 of its 32 decoder layers twice (56 layer passes), carries two residual streams
(mHC) inside the loop, mixes linear and full attention, reads images, and holds 512K tokens of context. The files
keep its MTP head, which drafts tokens that the card checks in one batch.

- **RTX A5000 (24 GB), CUDA:** `q4tp` writes code at 88 tok/s with drafts and 51 without, and reads a
  32 000-token prompt in 22.6 s.
- **The same card, Vulkan:** code at 57.5 tok/s with drafts, 49.3 without.
- **Apple M4 (MacBook Air), Metal:** code at 14.4 tok/s with drafts, 9.5 without.
- Greedy text of `q8_2f` = Transformers float32 on the reference text prompts, on CUDA and Vulkan.
- Long documents in fixed card memory: [SenseNova-Loop-9B-Turbo-cmf](https://huggingface.co/infosave/SenseNova-Loop-9B-Turbo-cmf).

[English](#quick-start) · [Русский](#документация-на-русском) · [中文](#中文说明)

## Quick start

An NVIDIA card through CUDA, the fastest path:

```bash
cargo install cortiq-cli --features cuda
hf download infosave/SenseNova-Loop-9B-cmf sensenova-loop-9b-q4tp.cmf --local-dir .
CMF_GPU=cuda cortiq run sensenova-loop-9b-q4tp.cmf --prompt "Write a Python function that merges two sorted lists." --greedy --no-think
```

Vulkan or DX12 (NVIDIA, AMD, Intel), Metal, or the CPU:

```bash
cargo install cortiq-cli
cortiq run sensenova-loop-9b-q4tp.cmf --image photo.jpg --prompt "Describe this image." --no-think
```

These files need cortiq 0.9.5 or newer; 0.9.4 reads them and writes the same text, but on a GPU it runs them
op by op. `--greedy` turns on the draft tokens, and
`--no-think` asks for a direct answer: the model thinks before it answers otherwise, and the thinking counts
toward `--max-tokens`. `--image` takes a path, a data URL or an HTTP(S) URL and may be repeated.

| | CUDA | Vulkan / DX12 | Metal |
|---|---|---|---|
| card | NVIDIA with its driver | any with Vulkan 1.3 or DX12 | Apple silicon |
| libraries | CUDA 12.x `libnvrtc` and `libcublas` at run time | on headless Linux: `libvulkan1 libglvnd0 libegl1 libgl1 libglx0`, `XDG_RUNTIME_DIR=/tmp` | none |
| build | `--features cuda`, stable Rust, no CUDA toolkit at build time | default features | default features |
| card memory, `q4tp` / `q8_2f`, short prompts | 7.8 / 9.1 GB (RTX A5000) | 4.8 / 8.2 GB (RTX A5000) | not measured for this release |

## Files

| file | language model | size | use |
|---|---|---:|---|
| `sensenova-loop-9b-q4tp.cmf` | 4-bit, GPTQ | 5.74 GB | **default**, with draft tokens |
| `sensenova-loop-9b-q8_2f.cmf` | 8-bit | 10.16 GB | closest to the original |

Both carry the vision encoder (16-bit), the MTP head, the loop plan (layers 4–27 twice, each pass with its own
cache), the mHC routers in float32 as the release ships them, and the image processor. The release applies YaRN
(factor 2) at every length; the files do too.

| | wikitext-2, first 2048 tokens | chat text, 4096 tokens |
|---|---:|---:|
| Transformers, float32 | 11.90 | — |
| `q8_2f` | 11.78 | 1.848 |
| `q4tp` | 12.95 | 1.942 |

Perplexity. The chat text is the `q8_2f` file's own greedy answers to 30 held-out prompts in the model's template;
`q4tp` was calibrated on answers of the same kind to other prompts.

## CUDA · RTX A5000

Stand: NVIDIA RTX A5000 (24 GB), the CUDA backend (`CMF_GPU=cuda`), cortiq 0.9.5, a release build.

**Answers**, `cortiq run --greedy --no-think`, 256 tokens, four prompts of each kind:

| prompts | with drafts, tok/s | without, tok/s | drafts accepted |
|---|---:|---:|---:|
| code | **88.1** (74.4–98.9) | 51.0 | 30–52 % |
| general questions | **57.9** (52.7–62.4) | 49.8 | 19–40 % |

The text is the same with drafts and without. Answers of a sentence or two gain nothing.

![Draft tokens on CUDA](https://huggingface.co/infosave/SenseNova-Loop-9B-cmf/resolve/main/assets/drafts_cuda_en.png)

**Prompt length**, `cortiq bench --core`, without drafts, 128 tokens after the prompt:

| prompt, tokens | first token, s | decode, tok/s | peak card memory, GB |
|---:|---:|---:|---:|
| 1 000 | 1.5 | 51.9 | 7.8 |
| 4 000 | 2.3 | 50.4 | 8.0 |
| 16 000 | 10.1 | 47.1 | 8.9 |
| 32 000 | 22.6 | 43.5 | 10.0 |
| 64 000 | 53.0 | 38.1 | 12.2 |
| 128 000 | 141.9 | 30.5 | 16.5 |

`q8_2f`: 34.0 / 33.7 / 32.1 tok/s and 9.1 / 9.3 / 10.1 GB at 1 000 / 4 000 / 16 000 tokens. Past 64 000 tokens
the [Turbo file](https://huggingface.co/infosave/SenseNova-Loop-9B-Turbo-cmf) holds memory and speed in place.

## Vulkan · RTX A5000

Stand: the same NVIDIA RTX A5000 (24 GB), over Vulkan (the default backend of a build without `--features cuda`), cortiq
0.9.5, a release build.

**Answers**, `cortiq run --greedy --no-think`, 256 tokens, the same prompts:

| prompts | with drafts, tok/s | without, tok/s | drafts accepted |
|---|---:|---:|---:|
| code | **57.5** (47.9–63.7) | 49.3 | 38–61 % |
| general questions | 42.4 (35.8–45.4) | **47.4** | 23–36 % |

Over Vulkan the drafts pay on code only; on general questions a token is faster without them (`CMF_MTP=0` turns
them off). Four of the eight answers parted from the plain greedy text at a near tie: the batch that checks the
drafts rounds differently from the one-token graph. The draft head also reads the prompt: a 13 185-token prompt
reached the first token in 16.4–17.1 s with drafts and in 12.8–14.6 s without.

![Draft tokens over Vulkan](https://huggingface.co/infosave/SenseNova-Loop-9B-cmf/resolve/main/assets/drafts_vulkan_en.png)

**Prompt length**, `cortiq bench --core`, without drafts, 128 tokens after the prompt:

| prompt, tokens | first token, s | decode, tok/s | peak card memory, GB |
|---:|---:|---:|---:|
| 1 000 | 0.9 | 47.2 | 4.8 |
| 4 000 | 2.8 | 42.7 | 5.3 |
| 16 000 | 10.8 | 41.0 | 6.8 |
| 32 000 | 24.4 | 36.5 | 8.5 |
| 64 000 | 60.2 | 31.6 | 12.1 |
| 128 000 | 161.7 | 23.5 | 19.4 |

`q8_2f`: 31.1 / 29.0 / 30.0 tok/s and 8.2 / 8.7 / 10.0 GB at 1 000 / 4 000 / 16 000 tokens; it reads a prompt at
about 480 tok/s over Vulkan against about 1 300 for `q4tp`.


## Metal · Apple M4

Stand: MacBook Air M4 (24 GB), Metal (the default backend on a Mac), cortiq 0.9.5, a release build. The Air has
no fan and the base M4's memory bus; it slows down under a long load, and the runs went one after another.

**Answers**, `cortiq run --greedy --no-think`, 256 tokens, two prompts of each kind:

| prompts | with drafts, tok/s | without, tok/s | drafts accepted |
|---|---:|---:|---:|
| code | **14.4** | 9.5 | 62–70 % |
| general questions | **11.6** (10.4–12.7) | 9.7 | 37–47 % |

The text is the same with drafts and without.

![Draft tokens on Metal](https://huggingface.co/infosave/SenseNova-Loop-9B-cmf/resolve/main/assets/drafts_metal_en.png)

**Prompt length**, `cortiq bench --core`, without drafts, 128 tokens after the prompt:

| prompt, tokens | first token, s | decode, tok/s |
|---:|---:|---:|
| 1 000 | 10.4 | 11.3 |
| 4 000 | 49.6 | 9.4 |
| 16 000 | 220.3 | 7.6 |

The M4's GPU reads a prompt at 75–95 tokens a second, close to its arithmetic: for long documents take the
[Turbo file](https://huggingface.co/infosave/SenseNova-Loop-9B-Turbo-cmf) or an NVIDIA card.

## Server

`cortiq serve` puts the file behind an OpenAI-compatible HTTP API, with a web dashboard on the same port.

An NVIDIA card through CUDA:

```bash
CMF_GPU=cuda CMF_MAX_SEQ=131072 cortiq serve sensenova-loop-9b-q4tp.cmf --port 8080 --host 127.0.0.1
```

Vulkan, DX12 or Metal:

```bash
CMF_MAX_SEQ=131072 cortiq serve sensenova-loop-9b-q4tp.cmf --port 8080 --host 127.0.0.1
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

## Accuracy

- `q8_2f`, greedy, 64 tokens, against Transformers 5.18 in float32 on the same prompts: a one-sentence answer, a
  haiku and the start of a Python answer match word for word on CUDA and on Vulkan. An image description (a
  1024×523 diagram, 533 prompt tokens) matches word for word from the f16 export on the CPU (`CMF_GPU=0`); `q8_2f`
  on CUDA and Vulkan parts from it at a near tie after 7 and 18 words.
- YaRN: the frequencies match Transformers' `inv_freq` exactly, and its 1.0693 scale on cos and sin is applied
  through the q/k norms on every backend.
- Through CUDA the greedy text was the same with drafts and without in all 12 answers of the comparison (four each
  of code, general questions and short instructions).

## Verify the download

```bash
sha256sum -c sensenova-loop-9b-q4tp.cmf.sha256
```

```
1ad6e559d1f973c8944ef18eee1e6a0297c2c889d96d6a30478abd6ec45bc2d3  sensenova-loop-9b-q4tp.cmf
8cebe012a9a046802d1869d285221e7ed7e29b379c9cc201b85ba154520cb23b  sensenova-loop-9b-q8_2f.cmf
```

## Reproduce

```bash
hf download sensenova/SenseNova-Loop-9B --local-dir sn9
cortiq convert --model sn9 --quant q8_2f --output sensenova-loop-9b-q8_2f.cmf --also-output f16=sn9-f16.cmf
cortiq quantize-gptq sn9-f16.cmf --codec q4tp --calib-model sensenova-loop-9b-q8_2f.cmf \
  --calib wiki.train.raw=0.5 --calib chat_calib.txt=0.5 --tokens 16384 --window 2048 --lambda 0.01 \
  --calib-chunk-layers 8 --output sensenova-loop-9b-q4tp.cmf
```

`chat_calib.txt` is the `q8_2f` file's own greedy answers to 52 prompts (code, general, instructions, Chinese,
agent tasks), rendered through the model's chat template. Scripts: [scripts/snloop](https://github.com/infosave2007/cmf/tree/master/scripts/snloop).

## License

Apache-2.0, as the original release. The weights were converted to CMF and the language model quantized
(`q4tp`, `q8_2f`) by cortiq; the vision encoder stays 16-bit.

---

## Документация на русском

[SenseNova-Loop-9B](https://huggingface.co/sensenova/SenseNova-Loop-9B) (релиз 17.11.2026) одиночными файлами
[CMF](https://github.com/infosave2007/cmf) для `cortiq` — движка на Rust без Python и ML-фреймворков. Модель
выполняет 24 из 32 слоёв декодера дважды (56 проходов слоёв), внутри петли держит два остаточных потока (mHC),
сочетает линейное и полное внимание, читает изображения и вмещает 512K токенов контекста. В файлах сохранена голова
MTP: она предлагает черновые токены, а карта проверяет их одним пакетом.

- **RTX A5000 (24 ГБ), CUDA:** `q4tp` пишет код со скоростью 88 ток/с с черновиками и 51 без них; промпт в
  32 000 токенов читается за 23 с.
- **Та же карта, Vulkan:** 57,5 ток/с на коде с черновиками, 49,3 без них.
- **Apple M4 (MacBook Air), Metal:** 14,4 ток/с на коде с черновиками, 9,5 без них.
- Жадный текст `q8_2f` совпадает с Transformers float32 на текстовых эталонных промптах, на CUDA и Vulkan.
- Длинные документы при неизменной памяти карты: [SenseNova-Loop-9B-Turbo-cmf](https://huggingface.co/infosave/SenseNova-Loop-9B-Turbo-cmf).

### Быстрый старт

```bash
cargo install cortiq-cli --features cuda
hf download infosave/SenseNova-Loop-9B-cmf sensenova-loop-9b-q4tp.cmf --local-dir .
CMF_GPU=cuda cortiq run sensenova-loop-9b-q4tp.cmf --prompt "Напиши функцию на Python, которая сливает два отсортированных списка." --greedy --no-think
```

Без CUDA (Vulkan, DX12, Metal, CPU) — сборка без `--features cuda` и запуск без `CMF_GPU=cuda`.
Файлам нужен cortiq 0.9.5 или новее; 0.9.4 читает файлы и пишет тот же текст, но на GPU выполняет их по одной
операции. `--greedy` включает черновые токены, `--no-think` просит ответить
сразу; `--image` принимает путь, data URL или HTTP(S) URL.

| файл | языковая модель | размер | назначение |
|---|---|---:|---|
| `sensenova-loop-9b-q4tp.cmf` | 4 бита, GPTQ | 5,74 ГБ | **по умолчанию**, с черновыми токенами |
| `sensenova-loop-9b-q8_2f.cmf` | 8 бит | 10,16 ГБ | ближе всего к оригиналу |

Перплексия wikitext-2 (первые 2048 токенов): Transformers float32 11,90, `q8_2f` 11,78, `q4tp` 12,95. На тексте
чата (собственные ответы `q8_2f` на 30 отложенных вопросов, 4096 токенов): `q8_2f` 1,848, `q4tp` 1,942.

### CUDA · RTX A5000

Стенд: NVIDIA RTX A5000 (24 ГБ), бэкенд CUDA (`CMF_GPU=cuda`), cortiq 0.9.5, релизная сборка.

| запросы | с черновиками, ток/с | без, ток/с | принято черновиков |
|---|---:|---:|---:|
| код | **88,1** (74,4–98,9) | 51,0 | 30–52 % |
| общие вопросы | **57,9** (52,7–62,4) | 49,8 | 19–40 % |

`cortiq run --greedy --no-think`, 256 токенов, по четыре запроса; текст с черновиками и без одинаковый. Ответы в
одно-два предложения не ускоряются.

![Черновые токены на CUDA](https://huggingface.co/infosave/SenseNova-Loop-9B-cmf/resolve/main/assets/drafts_cuda_ru.png)

| промпт, токенов | первый токен, с | декод, ток/с | пик памяти карты, ГБ |
|---:|---:|---:|---:|
| 1 000 | 1,5 | 51,9 | 7,8 |
| 4 000 | 2,3 | 50,4 | 8,0 |
| 16 000 | 10,1 | 47,1 | 8,9 |
| 32 000 | 22,6 | 43,5 | 10,0 |
| 64 000 | 53,0 | 38,1 | 12,2 |
| 128 000 | 141,9 | 30,5 | 16,5 |

`cortiq bench --core` без черновиков, 128 токенов после промпта. `q8_2f`: 34,0 / 33,7 / 32,1 ток/с и
9,1 / 9,3 / 10,1 ГБ при 1 000 / 4 000 / 16 000 токенов.

### Vulkan · RTX A5000

Стенд: та же NVIDIA RTX A5000 (24 ГБ) через Vulkan (бэкенд по умолчанию у сборки без `--features cuda`), cortiq
0.9.5, релизная сборка.

| запросы | с черновиками, ток/с | без, ток/с | принято черновиков |
|---|---:|---:|---:|
| код | **57,5** (47,9–63,7) | 49,3 | 38–61 % |
| общие вопросы | 42,4 (35,8–45,4) | **47,4** | 23–36 % |

Через Vulkan черновики окупаются только на коде; на общих вопросах токен без них быстрее (`CMF_MTP=0` их выключает).
Четыре ответа из восьми разошлись с обычным жадным текстом на почти равном выборе: пакет, проверяющий черновики,
округляет иначе, чем граф одного токена. Голова черновиков тоже читает промпт: до первого токена промпта в 13 185
токенов 16,4–17,1 с с черновиками и 12,8–14,6 с без них.

![Черновые токены на Vulkan](https://huggingface.co/infosave/SenseNova-Loop-9B-cmf/resolve/main/assets/drafts_vulkan_ru.png)

| промпт, токенов | первый токен, с | декод, ток/с | пик памяти карты, ГБ |
|---:|---:|---:|---:|
| 1 000 | 0,9 | 47,2 | 4,8 |
| 4 000 | 2,8 | 42,7 | 5,3 |
| 16 000 | 10,8 | 41,0 | 6,8 |
| 32 000 | 24,4 | 36,5 | 8,5 |
| 64 000 | 60,2 | 31,6 | 12,1 |
| 128 000 | 161,7 | 23,5 | 19,4 |

`q8_2f`: 31,1 / 29,0 / 30,0 ток/с и 8,2 / 8,7 / 10,0 ГБ при 1 000 / 4 000 / 16 000 токенов; промпт он читает через
Vulkan примерно на 480 ток/с против 1 300 у `q4tp`.

### Metal · Apple M4

Стенд: MacBook Air M4 (24 ГБ), Metal (бэкенд по умолчанию на Mac), cortiq 0.9.5, релизная сборка. У Air нет
вентилятора и базовая шина памяти M4; под долгой нагрузкой он замедляется, прогоны шли один за другим.

**Ответы**, `cortiq run --greedy --no-think`, 256 токенов, по два запроса каждого вида:

| запросы | с черновиками, ток/с | без, ток/с | принято черновиков |
|---|---:|---:|---:|
| код | **14,4** | 9,5 | 62–70 % |
| общие вопросы | **11,6** (10,4–12,7) | 9,7 | 37–47 % |

Текст с черновиками и без одинаковый.

![Черновые токены на Metal](https://huggingface.co/infosave/SenseNova-Loop-9B-cmf/resolve/main/assets/drafts_metal_ru.png)

| промпт, токенов | первый токен, с | декод, ток/с |
|---:|---:|---:|
| 1 000 | 10,4 | 11,3 |
| 4 000 | 49,6 | 9,4 |
| 16 000 | 220,3 | 7,6 |

GPU M4 читает промпт со скоростью 75–95 токенов в секунду, близко к пределу своей арифметики: для длинных документов
берите [Turbo-файл](https://huggingface.co/infosave/SenseNova-Loop-9B-Turbo-cmf) или карту NVIDIA.

### Сервер

`cortiq serve` отдаёт файл через HTTP API, совместимый с OpenAI, и веб-панель на том же порту.

Карта NVIDIA через CUDA:

```bash
CMF_GPU=cuda CMF_MAX_SEQ=131072 cortiq serve sensenova-loop-9b-q4tp.cmf --port 8080 --host 127.0.0.1
```

Vulkan, DX12 или Metal:

```bash
CMF_MAX_SEQ=131072 cortiq serve sensenova-loop-9b-q4tp.cmf --port 8080 --host 127.0.0.1
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

### Точность

- `q8_2f`, жадно, 64 токена, против Transformers 5.18 в float32 на тех же промптах: ответ в одно предложение, хайку
  и начало ответа на Python совпадают слово в слово на CUDA и на Vulkan. Описание изображения (схема 1024×523,
  533 токена промпта) совпадает слово в слово у f16 на CPU (`CMF_GPU=0`); `q8_2f` на CUDA и Vulkan расходится с ним
  на почти равном выборе после 7 и 18 слов.
- YaRN: частоты совпадают с `inv_freq` Transformers точно, а его множитель 1,0693 для cos и sin вносится через
  q/k-нормы на всех бэкендах.

### Проверка загрузки

```bash
sha256sum -c sensenova-loop-9b-q4tp.cmf.sha256
```

### Воспроизведение

```bash
hf download sensenova/SenseNova-Loop-9B --local-dir sn9
cortiq convert --model sn9 --quant q8_2f --output sensenova-loop-9b-q8_2f.cmf --also-output f16=sn9-f16.cmf
cortiq quantize-gptq sn9-f16.cmf --codec q4tp --calib-model sensenova-loop-9b-q8_2f.cmf \
  --calib wiki.train.raw=0.5 --calib chat_calib.txt=0.5 --tokens 16384 --window 2048 --lambda 0.01 \
  --calib-chunk-layers 8 --output sensenova-loop-9b-q4tp.cmf
```

`chat_calib.txt` — собственные жадные ответы `q8_2f` на 52 вопроса в шаблоне чата модели. Скрипты:
[scripts/snloop](https://github.com/infosave2007/cmf/tree/master/scripts/snloop).

### Лицензия

Apache-2.0, как у оригинала. Веса переведены в CMF и квантованы cortiq; энкодер изображений остаётся 16-битным.


---

## 中文说明

[SenseNova-Loop-9B](https://huggingface.co/sensenova/SenseNova-Loop-9B)（2026-11-17 发布）的单文件
[CMF](https://github.com/infosave2007/cmf) 版本，供 `cortiq` 使用。cortiq 是 Rust 推理引擎，不依赖 Python 和机器学习框架。
该模型把 32 层解码器中的 24 层运行两遍（共 56 次层计算），循环内保持两条残差流（mHC），混合线性注意力与全注意力，
可以读图，上下文长度 512K。文件保留了 MTP 头：它生成草稿 token，显卡一次性批量验证。

- RTX A5000（24 GB）+ CUDA：`q4tp` 写代码时用草稿 88 tok/s，不用草稿 51 tok/s；32 000 token 的提示词 23 秒读完。
- 同一张卡 + Vulkan：代码用草稿 57.5 tok/s，不用 49.3。
- Apple M4（MacBook Air）+ Metal：代码用草稿 14.4 tok/s，不用 9.5。
- `q8_2f` 的贪心输出与 Transformers float32 在文本参考提示词上一致（CUDA 与 Vulkan）。
- 长文档且显存不增长：[SenseNova-Loop-9B-Turbo-cmf](https://huggingface.co/infosave/SenseNova-Loop-9B-Turbo-cmf)。

### 快速开始

```bash
cargo install cortiq-cli --features cuda
hf download infosave/SenseNova-Loop-9B-cmf sensenova-loop-9b-q4tp.cmf --local-dir .
CMF_GPU=cuda cortiq run sensenova-loop-9b-q4tp.cmf --prompt "用 Python 写一个合并两个有序列表的函数。" --greedy --no-think
```

不用 CUDA（Vulkan、DX12、Metal、CPU）时，编译不加 `--features cuda`，运行不设 `CMF_GPU=cuda`。
这些文件需要 cortiq 0.9.5 或更新版本；0.9.4 能读取这些文件、输出相同文本，但在 GPU 上逐个算子执行。
`--greedy` 打开草稿 token，`--no-think` 要求直接回答；`--image` 接受路径、data URL 或 HTTP(S) URL。

| 文件 | 语言模型 | 大小 | 用途 |
|---|---|---:|---|
| `sensenova-loop-9b-q4tp.cmf` | 4 bit，GPTQ | 5.74 GB | 默认，支持草稿 token |
| `sensenova-loop-9b-q8_2f.cmf` | 8 bit | 10.16 GB | 最接近原始模型 |

wikitext-2 困惑度（前 2048 个 token）：Transformers float32 11.90，`q8_2f` 11.78，`q4tp` 12.95。
对话文本（`q8_2f` 对 30 个留出问题的回答，4096 个 token）：`q8_2f` 1.848，`q4tp` 1.942。

### CUDA · RTX A5000

测试平台：NVIDIA RTX A5000（24 GB），CUDA 后端（`CMF_GPU=cuda`），cortiq 0.9.5 release 构建。

| 请求 | 用草稿，tok/s | 不用，tok/s | 草稿接受率 |
|---|---:|---:|---:|
| 代码 | 88.1（74.4–98.9） | 51.0 | 30–52 % |
| 通用问题 | 57.9（52.7–62.4） | 49.8 | 19–40 % |

`cortiq run --greedy --no-think`，256 个 token，每类 4 个请求；有无草稿文本相同。一两句话的回答不会加速。

![CUDA 上的草稿 token](https://huggingface.co/infosave/SenseNova-Loop-9B-cmf/resolve/main/assets/drafts_cuda_zh.png)

| 提示词，token | 首个 token，秒 | 解码，tok/s | 显存峰值，GB |
|---:|---:|---:|---:|
| 1 000 | 1.5 | 51.9 | 7.8 |
| 4 000 | 2.3 | 50.4 | 8.0 |
| 16 000 | 10.1 | 47.1 | 8.9 |
| 32 000 | 22.6 | 43.5 | 10.0 |
| 64 000 | 53.0 | 38.1 | 12.2 |
| 128 000 | 141.9 | 30.5 | 16.5 |

`cortiq bench --core`，不用草稿，提示词之后 128 个 token。`q8_2f`：1 000 / 4 000 / 16 000 token 时 34.0 / 33.7 / 32.1 tok/s，显存 9.1 / 9.3 / 10.1 GB。

### Vulkan · RTX A5000

测试平台：同一张 NVIDIA RTX A5000（24 GB），走 Vulkan（不加 `--features cuda` 编译时的默认后端），cortiq 0.9.5 release 构建。

| 请求 | 用草稿，tok/s | 不用，tok/s | 草稿接受率 |
|---|---:|---:|---:|
| 代码 | 57.5（47.9–63.7） | 49.3 | 38–61 % |
| 通用问题 | 42.4（35.8–45.4） | 47.4 | 23–36 % |

在 Vulkan 上，草稿只在代码上有收益；通用问题上不用草稿每个 token 更快（`CMF_MTP=0` 关闭草稿）。8 个回答中有 4 个在近似并列处与普通贪心文本分开：验证草稿的批处理与单 token 图的舍入不同。草稿头也要读取提示词：13 185 token 的提示词，用草稿时首个 token 需要 16.4–17.1 秒，不用时 12.8–14.6 秒。

![Vulkan 上的草稿 token](https://huggingface.co/infosave/SenseNova-Loop-9B-cmf/resolve/main/assets/drafts_vulkan_zh.png)

| 提示词，token | 首个 token，秒 | 解码，tok/s | 显存峰值，GB |
|---:|---:|---:|---:|
| 1 000 | 0.9 | 47.2 | 4.8 |
| 4 000 | 2.8 | 42.7 | 5.3 |
| 16 000 | 10.8 | 41.0 | 6.8 |
| 32 000 | 24.4 | 36.5 | 8.5 |
| 64 000 | 60.2 | 31.6 | 12.1 |
| 128 000 | 161.7 | 23.5 | 19.4 |

`q8_2f`：1 000 / 4 000 / 16 000 token 时 31.1 / 29.0 / 30.0 tok/s，显存 8.2 / 8.7 / 10.0 GB；它在 Vulkan 上读提示词约 480 tok/s，`q4tp` 约 1 300。

### Metal · Apple M4

测试平台：MacBook Air M4（24 GB），Metal（Mac 上的默认后端），cortiq 0.9.5 release 构建。Air 没有风扇，内存总线为 M4 基础款；长时间负载下会降速，各项测试依次进行。

**回答**，`cortiq run --greedy --no-think`，256 个 token，每类两个提示词：

| 请求 | 用草稿，tok/s | 不用，tok/s | 草稿接受率 |
|---|---:|---:|---:|
| 代码 | 14.4 | 9.5 | 62–70 % |
| 通用问题 | 11.6（10.4–12.7） | 9.7 | 37–47 % |

有无草稿文本相同。

![Metal 上的草稿 token](https://huggingface.co/infosave/SenseNova-Loop-9B-cmf/resolve/main/assets/drafts_metal_zh.png)

| 提示词，token | 首个 token，秒 | 解码，tok/s |
|---:|---:|---:|
| 1 000 | 10.4 | 11.3 |
| 4 000 | 49.6 | 9.4 |
| 16 000 | 220.3 | 7.6 |

M4 的 GPU 读提示词约每秒 75–95 个 token，接近其算力上限：长文档请用 [Turbo 文件](https://huggingface.co/infosave/SenseNova-Loop-9B-Turbo-cmf) 或 NVIDIA 显卡。

### 服务器

`cortiq serve` 通过兼容 OpenAI 的 HTTP API 提供该文件，同一端口还有网页面板。

NVIDIA 显卡走 CUDA：

```bash
CMF_GPU=cuda CMF_MAX_SEQ=131072 cortiq serve sensenova-loop-9b-q4tp.cmf --port 8080 --host 127.0.0.1
```

Vulkan、DX12 或 Metal：

```bash
CMF_MAX_SEQ=131072 cortiq serve sensenova-loop-9b-q4tp.cmf --port 8080 --host 127.0.0.1
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

### 精度

- `q8_2f` 贪心生成 64 个 token，与 Transformers 5.18 float32 在相同提示词上对比：一句话的回答、俳句和 Python 回答的开头在 CUDA 和 Vulkan 上逐字一致。
  图片描述（1024×523 的示意图，提示词 533 个 token）由 f16 在 CPU（`CMF_GPU=0`）上逐字一致；`q8_2f` 在 CUDA 和 Vulkan 上分别在第 7 和第 18 个词处因近似并列而分开。
- YaRN：频率与 Transformers 的 `inv_freq` 完全一致，其 cos 和 sin 上的 1.0693 系数在所有后端都通过 q/k 归一化施加。

### 校验下载

```bash
sha256sum -c sensenova-loop-9b-q4tp.cmf.sha256
```

### 复现

```bash
hf download sensenova/SenseNova-Loop-9B --local-dir sn9
cortiq convert --model sn9 --quant q8_2f --output sensenova-loop-9b-q8_2f.cmf --also-output f16=sn9-f16.cmf
cortiq quantize-gptq sn9-f16.cmf --codec q4tp --calib-model sensenova-loop-9b-q8_2f.cmf \
  --calib wiki.train.raw=0.5 --calib chat_calib.txt=0.5 --tokens 16384 --window 2048 --lambda 0.01 \
  --calib-chunk-layers 8 --output sensenova-loop-9b-q4tp.cmf
```

`chat_calib.txt` 是 `q8_2f` 用模型对话模板对 52 个问题的贪心回答。脚本见 [scripts/snloop](https://github.com/infosave2007/cmf/tree/master/scripts/snloop)。

### 许可证

Apache-2.0，与原始发布相同。权重由 cortiq 转换为 CMF 并量化；图像编码器保持 16 bit。

