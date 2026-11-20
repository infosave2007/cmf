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
  - summarization
  - rewriting
  - chat
  - cuda
  - vulkan
  - metal
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

# Aion-1.0-Instruct-2.3B — CMF

[Aion-1.0-Instruct-2.3B](https://huggingface.co/microsoft/Aion-1.0-Instruct-2.3B), Microsoft's 2.3B on-device model
for summaries, rewriting and chat in eight languages with a 32 768-token context, packaged as single
[CMF](https://github.com/infosave2007/cmf) files. They run with `cortiq`, a Rust inference engine with no Python and
no ML framework: NVIDIA GPUs through CUDA or Vulkan, AMD and Intel GPUs through Vulkan/DX12, Apple silicon through
Metal, and the CPU. Measured:

- **NVIDIA L4 (24 GB), CUDA:** 129 tok/s for the 4-bit file and 86 for the 8-bit one; a 2 048-token prompt is
  read in 0.32 s and a 30 000-token one in 9.92 s, in 3.5 GB of the card.
- **The same card over Vulkan:** 110 and 79 tok/s; a 16 384-token prompt in 10.3 s.
- **MacBook Air M4 (24 GB):** 59.6 and 38.6 tok/s through Metal, 40.2 and 28.2 on the CPU alone.
- **Accuracy:** WikiText-2 perplexity 14.25 (8-bit) and 14.61 (4-bit) against 14.24 for Transformers bf16; greedy
  answers of the 8-bit file match Transformers word for word.

[English](#quick-start) · [Русский](#документация-на-русском) · [中文](#中文文档)

## Quick start

```bash
cargo install cortiq-cli
hf download infosave/Aion-1.0-Instruct-2.3B-cmf Aion-1.0-Instruct-2.3B-q8_2f.cmf --local-dir .
cortiq run Aion-1.0-Instruct-2.3B-q8_2f.cmf --prompt "Summarize in three bullet points: ..."
```

Use cortiq 0.9.9 or later (it renders the model's chat template and parses its tool calls); prebuilt binaries for
Linux, macOS and Windows are on the [releases page](https://github.com/infosave2007/cmf/releases). Add `--features cuda` to the install for the CUDA
backend (`CMF_GPU=cuda`).

## Files

| file | quantization | size | WikiText-2 ppl | Rust ppl | Python ppl |
|---|---|---:|---:|---:|---:|
| `Aion-1.0-Instruct-2.3B-q8_2f.cmf` | 8-bit, **default** | 2.31 GB | 14.25 | 7.66 | 7.26 |
| `Aion-1.0-Instruct-2.3B-q4tp.cmf` | 4-bit GPTQ, embeddings 8-bit | 1.41 GB | 14.61 | 7.83 | 7.43 |
| reference: Transformers bf16 | — | 4.58 GB | 14.24 | 7.66 | 7.26 |

Perplexity over 12 windows of 512 tokens on WikiText-2 test, 1.8 MB of Rust and 477 KB of Python, the same windows
for every row. Over 4 windows of 4096 tokens (past the 2048-token sliding window): q8_2f 9.47, bf16 9.46, q4tp 9.70.
The 4-bit weights are rounded with GPTQ on 4096 tokens of WikiText-2, code and the model's own answers in its chat
template, with channel scales and a code refinement. Greedy answers of the 8-bit file match Transformers word for
word on CUDA, Vulkan and the CPU.

For long prompts with the card's memory held in place:
[Aion-1.0-Instruct-2.3B Turbo](https://huggingface.co/infosave/Aion-1.0-Instruct-2.3B-Turbo-cmf).

## Performance

One stream, `cortiq bench --core`, cortiq 0.9.9: the time to the first token of a prompt of that length and the
decode speed after it (64 tokens). Card memory is the peak `nvidia-smi` shows during the run. Read each backend's
section on its own.

### CUDA · NVIDIA L4

Stand: NVIDIA L4 (24 GB), Google Colab, the CUDA backend (`CMF_GPU=cuda`). Without a prompt the 4-bit file decodes
at 129.1 tok/s, the 8-bit file at 86.3.

![The two files on CUDA](https://huggingface.co/infosave/Aion-1.0-Instruct-2.3B-cmf/resolve/main/assets/files_cuda_en.png)

| prompt, tokens | first token, s (`q4tp`) | first token, s (`q8_2f`) | decode, tok/s (`q4tp`) | decode, tok/s (`q8_2f`) | peak card memory, GB (`q4tp`) | peak card memory, GB (`q8_2f`) |
|---:|---:|---:|---:|---:|---:|---:|
| 2 048 | 0.32 | 0.34 | 116.9 | 80.4 | 2.2 | 3.3 |
| 8 192 | 1.72 | 1.79 | 106.3 | 75.2 | 2.5 | 3.6 |
| 16 384 | 4.15 | 4.31 | 95.7 | 69.8 | 2.9 | 4.0 |
| 30 000 | 9.92 | 10.1 | 79.7 | 61.0 | 3.5 | 4.7 |

### Vulkan · NVIDIA L4

Stand: the same NVIDIA L4 (24 GB), over Vulkan (the default backend). Without a prompt: 110.0 tok/s for the 4-bit
file, 78.6 for the 8-bit one.

![The two files on Vulkan](https://huggingface.co/infosave/Aion-1.0-Instruct-2.3B-cmf/resolve/main/assets/files_vulkan_en.png)

| prompt, tokens | first token, s (`q4tp`) | first token, s (`q8_2f`) | decode, tok/s (`q4tp`) | decode, tok/s (`q8_2f`) | peak card memory, GB (`q4tp`) | peak card memory, GB (`q8_2f`) |
|---:|---:|---:|---:|---:|---:|---:|
| 2 048 | 0.62 | 1.21 | 82.8 | 64.0 | 3.1 | 3.7 |
| 8 192 | 3.83 | 6.20 | 73.7 | 57.8 | 3.4 | 4.2 |
| 16 384 | 10.3 | 15.0 | 63.5 | 52.0 | 3.9 | 4.6 |
| 30 000 | 27.3 | 35.8 | 52.6 | 44.5 | 4.6 | 5.3 |

### Metal · MacBook Air M4

Stand: MacBook Air M4 (24 GB), fanless, Metal. Without a prompt: 59.6 tok/s for the 4-bit file, 38.6 for the 8-bit
one.

![The two files on Metal](https://huggingface.co/infosave/Aion-1.0-Instruct-2.3B-cmf/resolve/main/assets/files_metal_en.png)

| prompt, tokens | first token, s (`q4tp`) | first token, s (`q8_2f`) | decode, tok/s (`q4tp`) | decode, tok/s (`q8_2f`) |
|---:|---:|---:|---:|---:|
| 2 048 | 3.48 | 4.17 | 53.4 | 33.8 |
| 8 192 | 18.4 | 22.7 | 47.4 | 33.0 |
| 16 384 | 44.0 | — | 42.2 | — |

The fanless Air has the base M4's memory bus (about 100 GB/s): the 8-bit file reads its 2.3 GB a token at about
90 GB/s. At 16 384 tokens the 4-bit file holds 1.6 GB of keys and values.

### MacBook Air M4 (24 GB), CPU (10 cores)

| file | decode, tok/s | 2k prompt | decode at 2k |
|---|---:|---:|---:|
| `q4tp` | 40.2 | 11.3 s | 26.6 |
| `q8_2f` | 28.2 | 12.5 s | 20.3 |

### Colab L4 host, CPU (Xeon 2.2 GHz, 6 cores / 12 threads)

| file | decode, tok/s | 2k prompt | decode at 2k |
|---|---:|---:|---:|
| `q8_2f` | 15.5 | 50.8 s | 14.0 |
| `q4tp` | 15.1 | 45.0 s | 11.1 |

## Usage

```bash
cortiq run Aion-1.0-Instruct-2.3B-q8_2f.cmf                          # interactive chat
cortiq run Aion-1.0-Instruct-2.3B-q8_2f.cmf --prompt "..."           # one answer
cortiq run Aion-1.0-Instruct-2.3B-q8_2f.cmf --prompt "..." --greedy  # deterministic output
cortiq run Aion-1.0-Instruct-2.3B-q8_2f.cmf --prompt-file email.txt --max-tokens 512
```

The embedded template is the model's own: its system prompt names the model, the knowledge cutoff and today's date.
Without flags the CLI samples at temperature 0.7, top-p 0.9 and writes at most 256 new tokens.

### OpenAI-compatible server

```bash
cortiq serve Aion-1.0-Instruct-2.3B-q8_2f.cmf --port 8080

curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Aion-1.0-Instruct-2.3B-q8_2f",
       "messages": [{"role": "user", "content": "Rewrite politely: send me the report by Friday."}]}'
```

Endpoints: `/v1/chat/completions`, `/v1/completions`, `/v1/models`, and a web dashboard on the same port. Tools go in
`tools` in the OpenAI format; the model answers `<|tool_call|>{"name": …, "arguments": …}<|/tool_call|>`, which the
server returns as `message.tool_calls`.

## Long context

The model holds 32 768 tokens. 21 of its 28 layers attend over a 2048-token window and keep only those rows; the 7
full-attention layers keep every row, about 28 KB a token on the card: a 30 000-token prompt takes the 4-bit file
to 3.5 GB on an L4 with CUDA, decoding at 80 tok/s after it. `CMF_MAX_SEQ` sets the cap (32 768 by default). Planted
facts read back from a long Wikipedia text, greedy, 12 prompts a length: 12/12 at about 13 000 tokens and 11/12 at
about 27 000 for the 4-bit file on CUDA and Vulkan, the same as Transformers bf16.

## Verify the download

```bash
cortiq verify Aion-1.0-Instruct-2.3B-q8_2f.cmf
cortiq info Aion-1.0-Instruct-2.3B-q8_2f.cmf
```

| file | sha256 |
|---|---|
| `Aion-1.0-Instruct-2.3B-q8_2f.cmf` | `f194db0dbde7eded280146124507a5a031d664f8876aa79594d7e3916ceff7e4` |
| `Aion-1.0-Instruct-2.3B-q4tp.cmf` | `5737b006bd9ac6fa77bb814db92885129e944fc89cdeaca0e4fd053516a7c6f1` |

Weights: [MIT](https://huggingface.co/microsoft/Aion-1.0-Instruct-2.3B/blob/main/LICENSE), from
[microsoft/Aion-1.0-Instruct-2.3B](https://huggingface.co/microsoft/Aion-1.0-Instruct-2.3B). The model card's notes
on intended use, limitations and safety apply to these files as well.

---

## Документация на русском

[Aion-1.0-Instruct-2.3B](https://huggingface.co/microsoft/Aion-1.0-Instruct-2.3B) — модель Microsoft на 2,3 млрд
параметров для работы на устройстве: сводки, переписывание текста и чат на восьми языках, контекст 32 768 токенов.
Здесь она упакована в отдельные файлы [CMF](https://github.com/infosave2007/cmf) для `cortiq` — движка на Rust без
Python и ML-фреймворков: GPU NVIDIA через CUDA или Vulkan, AMD и Intel через Vulkan/DX12, Apple silicon через Metal,
а также CPU. Измерено:

- **NVIDIA L4 (24 ГБ), CUDA:** 129 ток/с у 4-битного файла и 86 у 8-битного; промпт из 2 048 токенов
  читается за 0,32 с, из 30 000 — за 9,92 с, в 3,5 ГБ карты.
- **Та же карта через Vulkan:** 110 и 79 ток/с; промпт из 16 384 токенов — за 10,3 с.
- **MacBook Air M4 (24 ГБ):** 59,6 и 38,6 ток/с через Metal, 40,2 и 28,2 на одном CPU.
- **Точность:** перплексия WikiText-2 14,25 (8 бит) и 14,61 (4 бита) против 14,24 у Transformers bf16; жадные
  ответы 8-битного файла совпадают с Transformers слово в слово.

Для длинных промптов при неизменной памяти карты:
[Aion-1.0-Instruct-2.3B Turbo](https://huggingface.co/infosave/Aion-1.0-Instruct-2.3B-Turbo-cmf).

### Скорость

Один поток, `cortiq bench --core`, cortiq 0.9.9: время до первого токена при промпте такой длины и генерация
после него (64 токена). Память карты — пик по `nvidia-smi` за прогон. Каждый раздел читается сам по себе.

**CUDA · NVIDIA L4**

Стенд: NVIDIA L4 (24 ГБ), Google Colab, бэкенд CUDA (`CMF_GPU=cuda`). Без промпта 4-битный файл пишет
129,1 ток/с, 8-битный — 86,3.

![Два файла на CUDA](https://huggingface.co/infosave/Aion-1.0-Instruct-2.3B-cmf/resolve/main/assets/files_cuda_ru.png)

| промпт, токенов | первый токен, с (`q4tp`) | первый токен, с (`q8_2f`) | генерация, ток/с (`q4tp`) | генерация, ток/с (`q8_2f`) | пик памяти карты, ГБ (`q4tp`) | пик памяти карты, ГБ (`q8_2f`) |
|---:|---:|---:|---:|---:|---:|---:|
| 2 048 | 0,32 | 0,34 | 116,9 | 80,4 | 2,2 | 3,3 |
| 8 192 | 1,72 | 1,79 | 106,3 | 75,2 | 2,5 | 3,6 |
| 16 384 | 4,15 | 4,31 | 95,7 | 69,8 | 2,9 | 4,0 |
| 30 000 | 9,92 | 10,1 | 79,7 | 61,0 | 3,5 | 4,7 |

**Vulkan · NVIDIA L4**

Стенд: та же NVIDIA L4 (24 ГБ), через Vulkan (бэкенд по умолчанию). Без промпта: 110,0 ток/с у 4-битного
файла, 78,6 у 8-битного.

![Два файла на Vulkan](https://huggingface.co/infosave/Aion-1.0-Instruct-2.3B-cmf/resolve/main/assets/files_vulkan_ru.png)

| промпт, токенов | первый токен, с (`q4tp`) | первый токен, с (`q8_2f`) | генерация, ток/с (`q4tp`) | генерация, ток/с (`q8_2f`) | пик памяти карты, ГБ (`q4tp`) | пик памяти карты, ГБ (`q8_2f`) |
|---:|---:|---:|---:|---:|---:|---:|
| 2 048 | 0,62 | 1,21 | 82,8 | 64,0 | 3,1 | 3,7 |
| 8 192 | 3,83 | 6,20 | 73,7 | 57,8 | 3,4 | 4,2 |
| 16 384 | 10,3 | 15,0 | 63,5 | 52,0 | 3,9 | 4,6 |
| 30 000 | 27,3 | 35,8 | 52,6 | 44,5 | 4,6 | 5,3 |

**Metal · MacBook Air M4**

Стенд: MacBook Air M4 (24 ГБ), без вентилятора, Metal. Без промпта: 59,6 ток/с у 4-битного файла, 38,6 у
8-битного.

![Два файла на Metal](https://huggingface.co/infosave/Aion-1.0-Instruct-2.3B-cmf/resolve/main/assets/files_metal_ru.png)

| промпт, токенов | первый токен, с (`q4tp`) | первый токен, с (`q8_2f`) | генерация, ток/с (`q4tp`) | генерация, ток/с (`q8_2f`) |
|---:|---:|---:|---:|---:|
| 2 048 | 3,48 | 4,17 | 53,4 | 33,8 |
| 8 192 | 18,4 | 22,7 | 47,4 | 33,0 |
| 16 384 | 44,0 | — | 42,2 | — |

У безвентиляторного Air шина памяти базового M4 (около 100 ГБ/с): 8-битный файл читает свои 2,3 ГБ на токен
примерно на 90 ГБ/с. На 16 384 токенах 4-битный файл держит 1,6 ГБ ключей и значений.

**MacBook Air M4 (24 ГБ), CPU (10 ядер)**

| файл | генерация, ток/с | промпт 2k | генерация на 2k |
|---|---:|---:|---:|
| `q4tp` | 40,2 | 11,3 с | 26,6 |
| `q8_2f` | 28,2 | 12,5 с | 20,3 |

**Хост Colab L4, CPU (Xeon 2,2 ГГц, 6 ядер / 12 потоков)**

| файл | генерация, ток/с | промпт 2k | генерация на 2k |
|---|---:|---:|---:|
| `q8_2f` | 15,5 | 50,8 с | 14,0 |
| `q4tp` | 15,1 | 45,0 с | 11,1 |

### Использование

```bash
cortiq run Aion-1.0-Instruct-2.3B-q8_2f.cmf                          # интерактивный чат
cortiq run Aion-1.0-Instruct-2.3B-q8_2f.cmf --prompt "..."           # один ответ
cortiq run Aion-1.0-Instruct-2.3B-q8_2f.cmf --prompt "..." --greedy  # детерминированный вывод
cortiq run Aion-1.0-Instruct-2.3B-q8_2f.cmf --prompt-file email.txt --max-tokens 512
```

Встроенный шаблон — собственный шаблон модели: системный промпт называет модель, дату среза знаний и сегодняшнюю
дату. Без флагов CLI сэмплирует с temperature 0,7, top-p 0,9 и пишет не больше 256 новых токенов.

Сервер, совместимый с OpenAI:

```bash
cortiq serve Aion-1.0-Instruct-2.3B-q8_2f.cmf --port 8080

curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Aion-1.0-Instruct-2.3B-q8_2f",
       "messages": [{"role": "user", "content": "Перепиши вежливо: пришли отчёт к пятнице."}]}'
```

Эндпоинты: `/v1/chat/completions`, `/v1/completions`, `/v1/models` и веб-панель на том же порту. Инструменты
передаются в `tools` в формате OpenAI; модель отвечает `<|tool_call|>{"name": …, "arguments": …}<|/tool_call|>`, и
сервер возвращает это как `message.tool_calls`.

### Длинный контекст

Модель держит 32 768 токенов. 21 из 28 слоёв смотрит в окно из 2048 токенов и хранит только эти строки; 7 слоёв
полного внимания хранят все строки, около 28 КБ на токен на карте: промпт из 30 000 токенов занимает у 4-битного
файла 3,5 ГБ на L4 с CUDA, генерация после него — 79 ток/с. `CMF_MAX_SEQ` задаёт предел (по умолчанию 32 768).
Факты, спрятанные в длинном тексте Википедии (жадно, по 12 промптов на длину): 12/12 примерно на 13 000 токенов и
11/12 примерно на 27 000 у 4-битного файла на CUDA и Vulkan — как у Transformers bf16.

### Проверка загрузки

```bash
cortiq verify Aion-1.0-Instruct-2.3B-q8_2f.cmf
```

Суммы sha256 — в таблице английского раздела. Веса:
[MIT](https://huggingface.co/microsoft/Aion-1.0-Instruct-2.3B/blob/main/LICENSE), от
[microsoft/Aion-1.0-Instruct-2.3B](https://huggingface.co/microsoft/Aion-1.0-Instruct-2.3B); замечания карточки
модели о назначении, ограничениях и безопасности относятся и к этим файлам.

---

## 中文文档

[Aion-1.0-Instruct-2.3B](https://huggingface.co/microsoft/Aion-1.0-Instruct-2.3B) 是微软面向端侧的 23 亿参数模型，
用于摘要、改写和八种语言的对话，上下文 32 768 个 token。这里打包为独立的 [CMF](https://github.com/infosave2007/cmf)
文件，由 `cortiq` 运行——一个不依赖 Python 和机器学习框架的 Rust 推理引擎：NVIDIA 显卡走 CUDA 或 Vulkan，AMD、
Intel 显卡走 Vulkan/DX12，Apple silicon 走 Metal，也可在 CPU 上运行。实测：

- **NVIDIA L4（24 GB），CUDA：** 4 位文件 129 tok/s，8 位文件 86；2 048 token 的提示词读取 0.32 秒，30 000 token
  读取 9.92 秒，占用 3.5 GB 显存。
- **同一块显卡，Vulkan：** 110 和 79 tok/s；16 384 token 的提示词 10.3 秒。
- **MacBook Air M4（24 GB）：** Metal 59.6 和 38.6 tok/s，仅 CPU 为 40.2 和 28.2。
- **精度：** WikiText-2 困惑度 14.25（8 位）和 14.61（4 位），Transformers bf16 为 14.24；8 位文件的贪心回答与
  Transformers 逐词一致。


长提示词且显存保持不变：[Aion-1.0-Instruct-2.3B Turbo](https://huggingface.co/infosave/Aion-1.0-Instruct-2.3B-Turbo-cmf)。

### 性能

单流，`cortiq bench --core`，cortiq 0.9.9：该长度提示词的首 token 时间和之后的生成速度（64 个 token）。显存为
运行期间 `nvidia-smi` 的峰值。各节请单独阅读。

**CUDA · NVIDIA L4**

测试平台：NVIDIA L4（24 GB），Google Colab，CUDA 后端（`CMF_GPU=cuda`）。无提示词时 4 位文件生成 129.1 tok/s，
8 位文件 86.3。

![CUDA 上的两个文件](https://huggingface.co/infosave/Aion-1.0-Instruct-2.3B-cmf/resolve/main/assets/files_cuda_zh.png)

| 提示词，token 数 | 首 token，秒（`q4tp`） | 首 token，秒（`q8_2f`） | 生成，tok/s（`q4tp`） | 生成，tok/s（`q8_2f`） | 显存峰值，GB（`q4tp`） | 显存峰值，GB（`q8_2f`） |
|---:|---:|---:|---:|---:|---:|---:|
| 2 048 | 0.32 | 0.34 | 116.9 | 80.4 | 2.2 | 3.3 |
| 8 192 | 1.72 | 1.79 | 106.3 | 75.2 | 2.5 | 3.6 |
| 16 384 | 4.15 | 4.31 | 95.7 | 69.8 | 2.9 | 4.0 |
| 30 000 | 9.92 | 10.1 | 79.7 | 61.0 | 3.5 | 4.7 |

**Vulkan · NVIDIA L4**

测试平台：同一块 NVIDIA L4（24 GB），通过 Vulkan（默认后端）。无提示词时：4 位文件 110.0 tok/s，8 位文件 78.6。

![Vulkan 上的两个文件](https://huggingface.co/infosave/Aion-1.0-Instruct-2.3B-cmf/resolve/main/assets/files_vulkan_zh.png)

| 提示词，token 数 | 首 token，秒（`q4tp`） | 首 token，秒（`q8_2f`） | 生成，tok/s（`q4tp`） | 生成，tok/s（`q8_2f`） | 显存峰值，GB（`q4tp`） | 显存峰值，GB（`q8_2f`） |
|---:|---:|---:|---:|---:|---:|---:|
| 2 048 | 0.62 | 1.21 | 82.8 | 64.0 | 3.1 | 3.7 |
| 8 192 | 3.83 | 6.20 | 73.7 | 57.8 | 3.4 | 4.2 |
| 16 384 | 10.3 | 15.0 | 63.5 | 52.0 | 3.9 | 4.6 |
| 30 000 | 27.3 | 35.8 | 52.6 | 44.5 | 4.6 | 5.3 |

**Metal · MacBook Air M4**

测试平台：MacBook Air M4（24 GB），无风扇，Metal。无提示词时：4 位文件 59.6 tok/s，8 位文件 38.6。

![Metal 上的两个文件](https://huggingface.co/infosave/Aion-1.0-Instruct-2.3B-cmf/resolve/main/assets/files_metal_zh.png)

| 提示词，token 数 | 首 token，秒（`q4tp`） | 首 token，秒（`q8_2f`） | 生成，tok/s（`q4tp`） | 生成，tok/s（`q8_2f`） |
|---:|---:|---:|---:|---:|
| 2 048 | 3.48 | 4.17 | 53.4 | 33.8 |
| 8 192 | 18.4 | 22.7 | 47.4 | 33.0 |
| 16 384 | 44.0 | — | 42.2 | — |

无风扇的 Air 采用基础款 M4 的内存总线（约 100 GB/s）：8 位文件每个 token 以约 90 GB/s 读取 2.3 GB。16 384 token
时 4 位文件的键值占 1.6 GB。

**MacBook Air M4（24 GB），CPU（10 核）**

| 文件 | 生成，tok/s | 2k 提示词 | 2k 处生成 |
|---|---:|---:|---:|
| `q4tp` | 40.2 | 11.3 s | 26.6 |
| `q8_2f` | 28.2 | 12.5 s | 20.3 |

**Colab L4 主机，CPU（Xeon 2.2 GHz，6 核 / 12 线程）**

| 文件 | 生成，tok/s | 2k 提示词 | 2k 处生成 |
|---|---:|---:|---:|
| `q8_2f` | 15.5 | 50.8 s | 14.0 |
| `q4tp` | 15.1 | 45.0 s | 11.1 |

### 使用

```bash
cortiq run Aion-1.0-Instruct-2.3B-q8_2f.cmf                          # 交互式对话
cortiq run Aion-1.0-Instruct-2.3B-q8_2f.cmf --prompt "..."           # 单次回答
cortiq run Aion-1.0-Instruct-2.3B-q8_2f.cmf --prompt "..." --greedy  # 确定性输出
```

内嵌模板即模型自带模板：系统提示包含模型名称、知识截止日期和当天日期。不加参数时 CLI 以 temperature 0.7、
top-p 0.9 采样，最多生成 256 个新 token。

OpenAI 兼容服务器：

```bash
cortiq serve Aion-1.0-Instruct-2.3B-q8_2f.cmf --port 8080
```

接口：`/v1/chat/completions`、`/v1/completions`、`/v1/models`，同端口还有网页面板。工具以 OpenAI 格式放在
`tools` 中；模型回答 `<|tool_call|>{"name": …, "arguments": …}<|/tool_call|>`，服务器将其返回为
`message.tool_calls`。

### 长上下文

模型支持 32 768 个 token。28 层中有 21 层只关注 2048 token 的窗口并只保留这些行；7 个全注意力层保留所有行，
每个 token 在显卡上约 28 KB：30 000 token 的提示词让 4 位文件在 L4（CUDA）上占用 3.5 GB，之后生成 79 tok/s。
`CMF_MAX_SEQ` 设定上限（默认 32 768）。在长篇维基百科文本中埋入的事实（贪心，每个长度 12 个提示词）：4 位文件约
13 000 token 时 12/12，约 27 000 token 时 11/12（CUDA 与 Vulkan 相同），与 Transformers bf16 一致。

### 校验下载

```bash
cortiq verify Aion-1.0-Instruct-2.3B-q8_2f.cmf
```

sha256 见英文部分的表格。权重：[MIT](https://huggingface.co/microsoft/Aion-1.0-Instruct-2.3B/blob/main/LICENSE)，
来自 [microsoft/Aion-1.0-Instruct-2.3B](https://huggingface.co/microsoft/Aion-1.0-Instruct-2.3B)；原模型卡关于用途、
局限和安全的说明同样适用于这些文件。
