---
license: apache-2.0
library_name: cortiq
base_model:
  - taobao-mnn/MNN-Pocket-0.3B
  - taobao-mnn/MNN-Pocket-0.3B-Base
  - taobao-mnn/MNN-Pocket-0.3B-SFT
  - taobao-mnn/MNN-Pocket-0.3B-DFlash2
  - taobao-mnn/MNN-Pocket-0.3B-Dec
base_model_relation: quantized
pipeline_tag: text-generation
tags:
  - cmf
  - cortiq
  - quantized
  - on-device
  - speculative-decoding
  - decision-model
  - vulkan
  - metal
language:
  - en
  - zh
---

# MNN-Pocket-0.3B — CMF

The [MNN-Pocket-0.3B](https://huggingface.co/collections/taobao-mnn/mnn-pocket-03b) family by the MNN team
(Alibaba) — a 0.3B Chinese/English assistant, its base and SFT checkpoints, a DFlash2 draft for
speculative decoding and a decision model — as single [CMF](https://github.com/infosave2007/cmf) files.
They run with `cortiq`, a Rust inference engine with no Python and no ML framework: NVIDIA, AMD and
Intel GPUs through Vulkan/DX12, Apple silicon through Metal, and the CPU.

[English](#quick-start) · [Русский](#документация-на-русском) · [中文](#中文文档)

## Quick start

```bash
cargo install cortiq-cli
hf download infosave/MNN-Pocket-0.3B-cmf MNN-Pocket-0.3B-q8_2f.cmf --local-dir .
cortiq run MNN-Pocket-0.3B-q8_2f.cmf --prompt "用三句话介绍一下杭州。"
```

Use cortiq 0.8.66 or later; prebuilt binaries for Linux, macOS and Windows are on the
[releases page](https://github.com/infosave2007/cmf/releases).

## Files

| file | model | quantization | size | use |
|---|---|---|---:|---|
| `MNN-Pocket-0.3B-q8_2f.cmf` | assistant | 8-bit | 316 MB | **default** |
| `MNN-Pocket-0.3B-q4tp.cmf` | assistant, the authors' INT4 (QAT) weights | 4-bit GPTQ; attention, embeddings, head 8-bit | 217 MB | smaller and faster |
| `MNN-Pocket-0.3B-q8_2f.dflash.cmf` | DFlash2 draft for the `q8_2f` file | 4-bit | 90 MB | speculative decoding, see Performance |
| `MNN-Pocket-0.3B-q4tp.dflash.cmf` | DFlash2 draft for the `q4tp` file | 4-bit | 90 MB | the same for `q4tp` |
| `MNN-Pocket-0.3B-Base-q8_2f.cmf` | pre-trained base | 8-bit | 316 MB | text completion, fine-tuning |
| `MNN-Pocket-0.3B-SFT-q8_2f.cmf` | SFT checkpoint | 8-bit | 316 MB | research |
| `MNN-Pocket-0.3B-Dec-q8_2f.cmf` | decision model | 8-bit | 318 MB | choice / yes-no / score decisions |
| `MNN-Pocket-0.3B-Dec-f16.cmf` | decision model | 16-bit | 620 MB | the reference's probabilities |

The GGUF and MNN builds of the authors are formats of the assistant; its INT4 weights are the source of
`q4tp`. A draft sidecar is bound to its file: keep the pair in one directory under these names.

### Quality

Perplexity over 16 windows of 512 tokens of chat text (9 925 tokens: answers of the bf16 assistant to 48
English and Chinese requests, in its chat template) and 12 windows of 512 tokens of WikiText-2 test. The
source column is `transformers` in float32 on the same tokens; cortiq's 16-bit conversion of the
assistant gives the same 16.111 and 4741.09.

| file | chat text: source → file | WikiText-2: source → file |
|---|---:|---:|
| `q8_2f` | 16.11 → 16.53 | 4741 → 4736 |
| `q4tp` | 13.57 → 13.47 (INT4 weights) | 2733 → 2185 (INT4 weights) |
| `Base-q8_2f` | 5.37 → 5.40 | 18.25 → 18.31 |
| `SFT-q8_2f` | 10.65 → 10.21 | 783 → 767 |

The assistant is very sure of itself: on raw WikiText its rare misses cost 30+ nats a token, hence
the thousands; the base model is the one to read there. `q4tp` is rounded from the authors' INT4 (QAT)
weights, a separately trained model that answers differently from bf16 (the authors report 49.59
against 50.00 on their benchmarks); rounding the bf16 weights to 4 bits instead gave 18.6–23.7 on the
chat text. Its GPTQ calibration: 4 096 tokens, half WikiText-2, half the assistant's chats.

The decision model's probabilities, against the authors' `inference.py` (float32, unmerged LoRA) on 49
questions of 17 requests: `Dec-f16` picks the same answer 49/49, probabilities within 0.001; `Dec-q8_2f`
47/49 on Metal (the two were near ties) and 36/36 on Vulkan, within 0.05.

## Performance

One stream, `cortiq bench --core`; the prompt columns give the time to the first token and the decode
speed after a prompt of that length. Memory: VRAM above idle, or peak RAM, for the 2k and 8k prompts.

| hardware | backend | file | decode, tok/s | 2k prompt, first token | decode at 2k | 8k prompt, first token | decode at 8k | memory |
|---|---|---|---:|---:|---:|---:|---:|---|
| RTX 2000 Ada (16 GB) | Vulkan | `q8_2f` | 333 | 1.65 s | 292 | 10.5 s | 224 | 0.4–0.7 GB VRAM |
| RTX 2000 Ada (16 GB) | Vulkan | `q4tp` | 401 | 0.76 s | 342 | 7.0 s | 256 | 0.4–0.6 GB VRAM |
| MacBook Air M4 (24 GB) | Metal | `q8_2f` | 202 | 0.69 s | 144 | 5.9 s | 101 | 0.7–1.3 GB RAM |
| MacBook Air M4 (24 GB) | Metal | `q4tp` | 252 | 0.67 s | 183 | 5.7 s | 122 | 0.5–1.2 GB RAM |
| MacBook Air M4 (24 GB) | CPU, 10 cores | `q8_2f` | 178 | 2.1 s | 104 | — | — | 0.6 GB RAM |
| MacBook Air M4 (24 GB) | CPU, 10 cores | `q4tp` | 191 | 2.0 s | 110 | — | — | 0.5 GB RAM |

`Base-q8_2f`, `SFT-q8_2f` and `Dec-q8_2f` have the layout of `q8_2f` and its speed. The MacBook Air has
no fan and the base M4's memory bus; an M4 Pro or Max was not measured.

DFlash2 drafts on the RTX 2000 Ada with `q8_2f`, greedy, plain → `CMF_DFLASH=1`: Python 320 → 415
tok/s (88 % of the drafted tokens kept), Chinese code 325 → 318–347, arithmetic 324 → 306–315, chat
307 → 235–256 (a third kept); the text is the plain decode's in every run. With `q4tp` the drafts do
not pay yet (its verify runs the four-bit kernels): use them with `q8_2f` for code.

## Usage

```bash
cortiq run MNN-Pocket-0.3B-q8_2f.cmf                          # interactive chat
cortiq run MNN-Pocket-0.3B-q8_2f.cmf --prompt "..."           # one answer
cortiq run MNN-Pocket-0.3B-q8_2f.cmf --prompt "..." --greedy  # deterministic output
cortiq run MNN-Pocket-0.3B-Base-q8_2f.cmf --raw --prompt "The capital of France is"
```

The files carry the authors' sampling — temperature 0.8, top-p 0.8, top-k 20, no repetition
penalty — and a loop stop: after a code block the model sometimes keeps printing backticks (the bf16
weights do the same in `transformers`), and the answer then ends with finish reason `loop` instead of
running to the token limit. Flags and request fields override the sampling.

### OpenAI-compatible server

```bash
cortiq serve MNN-Pocket-0.3B-q8_2f.cmf --port 8080

curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "MNN-Pocket-0.3B-q8_2f", "messages": [{"role": "user", "content": "What is MNN?"}]}'
```

Endpoints: `/v1/chat/completions`, `/v1/completions`, `/v1/models`, plus a web dashboard on the same port.

### Speculative decoding with DFlash2

```bash
CMF_DFLASH=1 cortiq run MNN-Pocket-0.3B-q8_2f.cmf --greedy --prompt "Write a Python function ..."
```

The draft proposes a block of 8 tokens in one pass and the assistant checks them in one pass; rounds
that do not pay sit out on their own. It runs with greedy decoding on Vulkan/DX12; on Metal and the
CPU, `CMF_DFLASH=1` has no effect.

### Decisions

```bash
cat > request.json <<'EOF'
{"state": "The shoes arrived, but they are the wrong size. I want to send them back for an exchange.",
 "questions": {
   "department": {"type": "choice", "instructions": "Which team should handle this request?",
                  "criteria": {"returns": "Returns, exchanges and refunds", "billing": "Charges and payment problems"}},
   "needs_human": {"type": "noul", "instructions": "Does this request require a human agent?"},
   "urgency": {"type": "score", "instructions": "How urgent is this request?",
               "criteria": ["No time pressure", "Routine", "Handle today"]}}}
EOF
cortiq decide-pointer MNN-Pocket-0.3B-Dec-q8_2f.cmf --input request.json
```

The answer is the authors' JSON: per question its `type` and `probabilities`, plus `choice`,
`decision` and `noul` (the probability of yes), or `score` and `legend`; `input_tokens` and the
`temperature` applied (1.111, calibrated by the authors; `--temperature` overrides it). A list of
requests gets a list of answers. Every question is one pass over the state and its own branch, no token
is generated: a request of three questions takes about 0.2 s on the M4 and 0.3 s on the RTX 2000 Ada
with `Dec-q8_2f`, 1.2 and 1.9 s with `Dec-f16`.

## Long context

The model was trained for 32 768 tokens, the default size of the KV cache (28 KB a token, 0.9 GB).

## Verify the download

```bash
cortiq verify MNN-Pocket-0.3B-q8_2f.cmf
cortiq info MNN-Pocket-0.3B-q8_2f.cmf
```

Weights: [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0), from the
[MNN-Pocket-0.3B collection](https://huggingface.co/collections/taobao-mnn/mnn-pocket-03b).

---

## Документация на русском

Семейство [MNN-Pocket-0.3B](https://huggingface.co/collections/taobao-mnn/mnn-pocket-03b) от команды MNN
(Alibaba) — ассистент на 0,3 млрд параметров для китайского и английского, его базовый и SFT-чекпойнты,
драфтер DFlash2 для спекулятивного декодирования и модель решений — в виде отдельных файлов
[CMF](https://github.com/infosave2007/cmf). Запускается движком `cortiq` на Rust без Python и
ML-фреймворков: GPU NVIDIA, AMD и Intel через Vulkan/DX12, Apple silicon через Metal, а также CPU.

### Быстрый старт

```bash
cargo install cortiq-cli
hf download infosave/MNN-Pocket-0.3B-cmf MNN-Pocket-0.3B-q8_2f.cmf --local-dir .
cortiq run MNN-Pocket-0.3B-q8_2f.cmf --prompt "用三句话介绍一下杭州。"
```

Нужен cortiq 0.8.66 или новее; готовые сборки — на [странице релизов](https://github.com/infosave2007/cmf/releases).

### Файлы

| файл | модель | квантизация | размер | назначение |
|---|---|---|---:|---|
| `MNN-Pocket-0.3B-q8_2f.cmf` | ассистент | 8 бит | 316 МБ | **по умолчанию** |
| `MNN-Pocket-0.3B-q4tp.cmf` | ассистент, INT4-веса авторов (QAT) | 4 бита GPTQ; внимание, эмбеддинги и голова 8 бит | 217 МБ | меньше и быстрее |
| `MNN-Pocket-0.3B-q8_2f.dflash.cmf` | драфтер DFlash2 для файла `q8_2f` | 4 бита | 90 МБ | спекулятивное декодирование, см. «Скорость» |
| `MNN-Pocket-0.3B-q4tp.dflash.cmf` | драфтер DFlash2 для файла `q4tp` | 4 бита | 90 МБ | то же для `q4tp` |
| `MNN-Pocket-0.3B-Base-q8_2f.cmf` | базовая предобученная | 8 бит | 316 МБ | продолжение текста, дообучение |
| `MNN-Pocket-0.3B-SFT-q8_2f.cmf` | SFT-чекпойнт | 8 бит | 316 МБ | исследования |
| `MNN-Pocket-0.3B-Dec-q8_2f.cmf` | модель решений | 8 бит | 318 МБ | выбор / да-нет / оценка |
| `MNN-Pocket-0.3B-Dec-f16.cmf` | модель решений | 16 бит | 620 МБ | вероятности как у эталона |

Сборки авторов GGUF и MNN — форматы того же ассистента; его INT4-веса — источник `q4tp`. Драфтер
привязан к своему файлу: держите пару в одной папке под этими именами.

#### Качество

Перплексия на 16 окнах по 512 токенов чат-текста (9 925 токенов: ответы bf16-ассистента на 48 запросов
на английском и китайском в его шаблоне) и на 12 окнах по 512 токенов WikiText-2 test. Столбец
«исходник» — `transformers` во float32 на тех же токенах; 16-битная конвертация ассистента в cortiq даёт
те же 16,111 и 4741,09.

| файл | чат-текст: исходник → файл | WikiText-2: исходник → файл |
|---|---:|---:|
| `q8_2f` | 16,11 → 16,53 | 4741 → 4736 |
| `q4tp` | 13,57 → 13,47 (INT4-веса) | 2733 → 2185 (INT4-веса) |
| `Base-q8_2f` | 5,37 → 5,40 | 18,25 → 18,31 |
| `SFT-q8_2f` | 10,65 → 10,21 | 783 → 767 |

Ассистент очень уверен в себе: на сыром WikiText его редкие промахи стоят 30+ нат на токен, отсюда
тысячи; там смотреть стоит на базовую модель. `q4tp` округлён из INT4-весов авторов (QAT) — это отдельно
обученная модель, она отвечает иначе, чем bf16 (авторы дают 49,59 против 50,00 на своих бенчмарках);
округление bf16-весов до 4 бит давало 18,6–23,7 на чат-тексте. Калибровка GPTQ: 4 096 токенов, половина
WikiText-2, половина диалогов ассистента.

Вероятности модели решений против `inference.py` авторов (float32, LoRA не слита) на 49 вопросах из 17
запросов: `Dec-f16` выбирает тот же ответ в 49 из 49, вероятности в пределах 0,001; `Dec-q8_2f` — 47 из
49 на Metal (оба расхождения — почти равные варианты) и 36 из 36 на Vulkan, в пределах 0,05.

### Скорость

Один поток, `cortiq bench --core`; в столбцах промпта — время до первого токена и скорость генерации
после промпта такой длины. Память — VRAM сверх простоя или пик RAM для промптов 2k и 8k.

| железо | бэкенд | файл | генерация, ток/с | промпт 2k, первый токен | генерация на 2k | промпт 8k, первый токен | генерация на 8k | память |
|---|---|---|---:|---:|---:|---:|---:|---|
| RTX 2000 Ada (16 ГБ) | Vulkan | `q8_2f` | 333 | 1,65 с | 292 | 10,5 с | 224 | 0,4–0,7 ГБ VRAM |
| RTX 2000 Ada (16 ГБ) | Vulkan | `q4tp` | 401 | 0,76 с | 342 | 7,0 с | 256 | 0,4–0,6 ГБ VRAM |
| MacBook Air M4 (24 ГБ) | Metal | `q8_2f` | 202 | 0,69 с | 144 | 5,9 с | 101 | 0,7–1,3 ГБ RAM |
| MacBook Air M4 (24 ГБ) | Metal | `q4tp` | 252 | 0,67 с | 183 | 5,7 с | 122 | 0,5–1,2 ГБ RAM |
| MacBook Air M4 (24 ГБ) | CPU, 10 ядер | `q8_2f` | 178 | 2,1 с | 104 | — | — | 0,6 ГБ RAM |
| MacBook Air M4 (24 ГБ) | CPU, 10 ядер | `q4tp` | 191 | 2,0 с | 110 | — | — | 0,5 ГБ RAM |

`Base-q8_2f`, `SFT-q8_2f` и `Dec-q8_2f` устроены как `q8_2f` и работают с той же скоростью. У MacBook Air
нет вентилятора и шина памяти базового M4; M4 Pro и Max не измерялись.

Драфтер DFlash2 на RTX 2000 Ada с `q8_2f`, жадная генерация, обычная → `CMF_DFLASH=1`: Python 320 → 415
ток/с (принято 88 % предложенных токенов), код по-китайски 325 → 318–347, арифметика 324 → 306–315,
беседа 307 → 235–256 (принята треть); текст в каждом прогоне совпадает с обычной генерацией. С `q4tp`
драфтер пока не окупается (его проверка идёт на 4-битных ядрах): включайте его с `q8_2f` для кода.

### Использование

```bash
cortiq run MNN-Pocket-0.3B-q8_2f.cmf                          # интерактивный чат
cortiq run MNN-Pocket-0.3B-q8_2f.cmf --prompt "..."           # один ответ
cortiq run MNN-Pocket-0.3B-q8_2f.cmf --prompt "..." --greedy  # детерминированный вывод
cortiq run MNN-Pocket-0.3B-Base-q8_2f.cmf --raw --prompt "The capital of France is"
```

В файлах записан сэмплинг авторов — temperature 0,8, top-p 0,8, top-k 20, без штрафа за повторы — и
стоп по зацикливанию: после блока кода модель иногда продолжает печатать обратные кавычки (bf16-веса в
`transformers` делают то же), и ответ тогда заканчивается с причиной `loop`, а не на лимите токенов. Флаги
и поля запроса переопределяют сэмплинг.

#### Сервер с API OpenAI

```bash
cortiq serve MNN-Pocket-0.3B-q8_2f.cmf --port 8080

curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "MNN-Pocket-0.3B-q8_2f", "messages": [{"role": "user", "content": "What is MNN?"}]}'
```

Эндпоинты: `/v1/chat/completions`, `/v1/completions`, `/v1/models` и веб-панель на том же порту.

#### Спекулятивное декодирование с DFlash2

```bash
CMF_DFLASH=1 cortiq run MNN-Pocket-0.3B-q8_2f.cmf --greedy --prompt "Write a Python function ..."
```

Драфтер предлагает блок из 8 токенов за один проход, ассистент проверяет их тоже за один; раунды, которые
не окупаются, сами уходят в паузу. Работает при жадной генерации на Vulkan/DX12; на Metal и CPU
`CMF_DFLASH=1` ничего не меняет.

#### Решения

```bash
cortiq decide-pointer MNN-Pocket-0.3B-Dec-q8_2f.cmf --input request.json
```

Формат запроса и ответа — как у авторов (пример `request.json` — в английском разделе): по каждому
вопросу `type` и `probabilities`, плюс `choice`, либо `decision` и `noul` (вероятность «да»), либо `score`
и `legend`; `input_tokens` и применённая `temperature` (1,111, откалибрована авторами; `--temperature`
её заменяет). На список запросов — список ответов. Каждый вопрос — один проход по состоянию и своей ветке,
токены не генерируются: запрос из трёх вопросов — около 0,2 с на M4 и 0,3 с на RTX 2000 Ada с
`Dec-q8_2f`, 1,2 и 1,9 с с `Dec-f16`.

### Длинный контекст

Модель обучена на 32 768 токенах — это и размер KV-кэша по умолчанию (28 КБ на токен, 0,9 ГБ).

### Проверка загрузки

```bash
cortiq verify MNN-Pocket-0.3B-q8_2f.cmf
cortiq info MNN-Pocket-0.3B-q8_2f.cmf
```

Веса: [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0), из
[коллекции MNN-Pocket-0.3B](https://huggingface.co/collections/taobao-mnn/mnn-pocket-03b).

---

## 中文文档

MNN 团队（阿里巴巴）的 [MNN-Pocket-0.3B](https://huggingface.co/collections/taobao-mnn/mnn-pocket-03b) 系列——0.3B 参数的中英文助手、它的基座与 SFT 检查点、用于投机解码的 DFlash2 草稿模型以及决策模型——打包为单个 [CMF](https://github.com/infosave2007/cmf) 文件。由 `cortiq` 运行：这是一个不依赖 Python 和机器学习框架的 Rust 推理引擎，通过 Vulkan/DX12 支持 NVIDIA、AMD、Intel 显卡，通过 Metal 支持 Apple 芯片，也可在 CPU 上运行。

### 快速开始

```bash
cargo install cortiq-cli
hf download infosave/MNN-Pocket-0.3B-cmf MNN-Pocket-0.3B-q8_2f.cmf --local-dir .
cortiq run MNN-Pocket-0.3B-q8_2f.cmf --prompt "用三句话介绍一下杭州。"
```

需要 cortiq 0.8.66 或更高版本；Linux、macOS、Windows 的预编译版本见[发布页面](https://github.com/infosave2007/cmf/releases)。

### 文件

| 文件 | 模型 | 量化 | 大小 | 用途 |
|---|---|---|---:|---|
| `MNN-Pocket-0.3B-q8_2f.cmf` | 助手 | 8 位 | 316 MB | 默认 |
| `MNN-Pocket-0.3B-q4tp.cmf` | 助手，作者的 INT4（QAT）权重 | 4 位 GPTQ；注意力、嵌入和输出头为 8 位 | 217 MB | 更小更快 |
| `MNN-Pocket-0.3B-q8_2f.dflash.cmf` | `q8_2f` 文件的 DFlash2 草稿模型 | 4 位 | 90 MB | 投机解码，见“性能” |
| `MNN-Pocket-0.3B-q4tp.dflash.cmf` | `q4tp` 文件的 DFlash2 草稿模型 | 4 位 | 90 MB | 同上，用于 `q4tp` |
| `MNN-Pocket-0.3B-Base-q8_2f.cmf` | 预训练基座 | 8 位 | 316 MB | 文本续写、微调 |
| `MNN-Pocket-0.3B-SFT-q8_2f.cmf` | SFT 检查点 | 8 位 | 316 MB | 研究 |
| `MNN-Pocket-0.3B-Dec-q8_2f.cmf` | 决策模型 | 8 位 | 318 MB | 选择 / 是否 / 评分决策 |
| `MNN-Pocket-0.3B-Dec-f16.cmf` | 决策模型 | 16 位 | 620 MB | 与参考实现一致的概率 |

作者的 GGUF 和 MNN 版本是同一助手的不同格式；其 INT4 权重是 `q4tp` 的来源。草稿模型与其对应文件绑定：请以上述文件名把两者放在同一目录。

#### 质量

困惑度：对话文本 16 个 512 token 窗口（9 925 个 token：bf16 助手对 48 条中英文请求的回答，使用其对话模板），以及 WikiText-2 test 的 12 个 512 token 窗口。“源模型”列为 `transformers` float32 在相同 token 上的结果；cortiq 对助手的 16 位转换给出相同的 16.111 与 4741.09。

| 文件 | 对话文本：源模型 → 文件 | WikiText-2：源模型 → 文件 |
|---|---:|---:|
| `q8_2f` | 16.11 → 16.53 | 4741 → 4736 |
| `q4tp` | 13.57 → 13.47（INT4 权重） | 2733 → 2185（INT4 权重） |
| `Base-q8_2f` | 5.37 → 5.40 | 18.25 → 18.31 |
| `SFT-q8_2f` | 10.65 → 10.21 | 783 → 767 |

助手模型非常自信：在原始 WikiText 上，它偶尔的失误每个 token 要付出 30 nat 以上，因此困惑度达到数千；WikiText 应看基座模型。`q4tp` 由作者的 INT4（QAT）权重量化而来，这是单独训练的模型，回答与 bf16 不同（作者在其基准上报告 49.59 对 50.00）；若改为把 bf16 权重量化到 4 位，对话文本困惑度为 18.6–23.7。GPTQ 校准：4 096 个 token，一半 WikiText-2，一半助手的对话。

决策模型的概率与作者的 `inference.py`（float32，LoRA 未合并）在 17 个请求的 49 个问题上对比：`Dec-f16` 49/49 选择相同答案，概率差在 0.001 以内；`Dec-q8_2f` 在 Metal 上 47/49（两处均为几乎持平的选项）、在 Vulkan 上 36/36，差在 0.05 以内。

### 性能

单路，`cortiq bench --core`；提示列给出首个 token 的时间以及该长度提示之后的解码速度。内存：2k 与 8k 提示下超出空闲的显存或峰值内存。

| 硬件 | 后端 | 文件 | 解码 tok/s | 2k 提示首 token | 2k 处解码 | 8k 提示首 token | 8k 处解码 | 内存 |
|---|---|---|---:|---:|---:|---:|---:|---|
| RTX 2000 Ada (16 GB) | Vulkan | `q8_2f` | 333 | 1.65 s | 292 | 10.5 s | 224 | 0.4–0.7 GB 显存 |
| RTX 2000 Ada (16 GB) | Vulkan | `q4tp` | 401 | 0.76 s | 342 | 7.0 s | 256 | 0.4–0.6 GB 显存 |
| MacBook Air M4 (24 GB) | Metal | `q8_2f` | 202 | 0.69 s | 144 | 5.9 s | 101 | 0.7–1.3 GB 内存 |
| MacBook Air M4 (24 GB) | Metal | `q4tp` | 252 | 0.67 s | 183 | 5.7 s | 122 | 0.5–1.2 GB 内存 |
| MacBook Air M4 (24 GB) | CPU，10 核 | `q8_2f` | 178 | 2.1 s | 104 | — | — | 0.6 GB 内存 |
| MacBook Air M4 (24 GB) | CPU，10 核 | `q4tp` | 191 | 2.0 s | 110 | — | — | 0.5 GB 内存 |

`Base-q8_2f`、`SFT-q8_2f` 和 `Dec-q8_2f` 与 `q8_2f` 结构相同，速度一致。MacBook Air 无风扇，且为基础款 M4 的内存总线；未测试 M4 Pro 或 Max。

DFlash2 草稿模型在 RTX 2000 Ada 上配合 `q8_2f`，贪心解码，普通 → `CMF_DFLASH=1`：Python 320 → 415 tok/s（保留 88 % 的草稿 token），中文代码 325 → 318–347，算术 324 → 306–315，闲聊 307 → 235–256（保留约三分之一）；每次运行的文本都与普通解码一致。配合 `q4tp` 时草稿暂不划算（其验证走 4 位内核）：写代码时请与 `q8_2f` 一起使用。

### 使用

```bash
cortiq run MNN-Pocket-0.3B-q8_2f.cmf                          # 交互式对话
cortiq run MNN-Pocket-0.3B-q8_2f.cmf --prompt "..."           # 单次回答
cortiq run MNN-Pocket-0.3B-q8_2f.cmf --prompt "..." --greedy  # 确定性输出
cortiq run MNN-Pocket-0.3B-Base-q8_2f.cmf --raw --prompt "The capital of France is"
```

文件中记录了作者的采样参数——temperature 0.8、top-p 0.8、top-k 20、无重复惩罚——以及循环停止：在代码块之后模型有时会不断输出反引号（bf16 权重在 `transformers` 中也是如此），此时回答以结束原因 `loop` 终止，而不是一直生成到 token 上限。命令行参数和请求字段可覆盖采样设置。

#### OpenAI 兼容服务

```bash
cortiq serve MNN-Pocket-0.3B-q8_2f.cmf --port 8080

curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "MNN-Pocket-0.3B-q8_2f", "messages": [{"role": "user", "content": "What is MNN?"}]}'
```

接口：`/v1/chat/completions`、`/v1/completions`、`/v1/models`，同一端口上还有网页控制台。

#### DFlash2 投机解码

```bash
CMF_DFLASH=1 cortiq run MNN-Pocket-0.3B-q8_2f.cmf --greedy --prompt "Write a Python function ..."
```

草稿模型一次前向提出 8 个 token 的块，助手一次前向完成验证；不划算的轮次会自动暂停。仅在 Vulkan/DX12 的贪心解码下生效；在 Metal 和 CPU 上 `CMF_DFLASH=1` 不起作用。

#### 决策

```bash
cortiq decide-pointer MNN-Pocket-0.3B-Dec-q8_2f.cmf --input request.json
```

请求与回答格式与作者一致（`request.json` 示例见英文部分）：每个问题给出 `type` 和 `probabilities`，另有 `choice`，或 `decision` 与 `noul`（“是”的概率），或 `score` 与 `legend`；以及 `input_tokens` 和所用的 `temperature`（1.111，由作者校准；`--temperature` 可覆盖）。请求列表返回回答列表。每个问题对状态和其自身分支做一次前向，不生成 token：三个问题的请求在 M4 上约 0.2 秒、在 RTX 2000 Ada 上约 0.3 秒（`Dec-q8_2f`），`Dec-f16` 分别约 1.2 和 1.9 秒。

### 长上下文

模型按 32 768 个 token 训练，这也是 KV 缓存的默认大小（每个 token 28 KB，共 0.9 GB）。

### 校验下载

```bash
cortiq verify MNN-Pocket-0.3B-q8_2f.cmf
cortiq info MNN-Pocket-0.3B-q8_2f.cmf
```

权重：[Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)，来自 [MNN-Pocket-0.3B 系列](https://huggingface.co/collections/taobao-mnn/mnn-pocket-03b)。
