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
Metal, and the CPU. On a 24 GB NVIDIA L4 with CUDA the 4-bit file writes 129 tokens a second and reads a 2048-token
prompt in 0.33 s.

[English](#quick-start) · [Русский](#документация-на-русском) · [中文](#中文文档)

## Quick start

```bash
cargo install --git https://github.com/infosave2007/cmf --branch aion cortiq-cli
hf download infosave/Aion-1.0-Instruct-2.3B-cmf Aion-1.0-Instruct-2.3B-q8_2f.cmf --local-dir .
cortiq run Aion-1.0-Instruct-2.3B-q8_2f.cmf --prompt "Summarize in three bullet points: ..."
```

The files need the cortiq release after 0.9.8 (it renders the model's chat template and parses its tool calls);
until it is out, build from the branch above. Add `--features cuda` to the install for the CUDA backend
(`CMF_GPU=cuda`).

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

One stream, `cortiq bench --core`: decode speed, then the time to the first token of a prompt of that length and
the decode speed after it. Card memory is the peak `nvidia-smi` shows during the run.

### NVIDIA L4 (24 GB, Google Colab), CUDA

| file | decode, tok/s | 2k prompt | decode at 2k | 8k prompt | decode at 8k | 16k prompt | decode at 16k | card memory |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `q4tp` | 129.1 | 0.33 s | 115.6 | 1.78 s | 105.0 | 4.29 s | 94.8 | 2.0–3.0 GB |
| `q8_2f` | 86.3 | 0.34 s | 80.0 | 1.83 s | 75.0 | 4.38 s | 68.8 | 3.2–4.2 GB |

### NVIDIA L4 (24 GB, Google Colab), Vulkan

| file | decode, tok/s | 2k prompt | decode at 2k | 8k prompt | decode at 8k | 16k prompt | decode at 16k | card memory |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `q4tp` | 107.1 | 0.62 s | 80.1 | 3.82 s | 66.5 | 10.37 s | 53.4 | 2.6–4.0 GB |
| `q8_2f` | 76.4 | 1.24 s | 61.4 | 6.34 s | 52.9 | 15.5 s | 44.3 | 3.2–4.7 GB |

### MacBook Air M4 (24 GB), Metal

| file | decode, tok/s | 2k prompt | decode at 2k | 8k prompt | decode at 8k |
|---|---:|---:|---:|---:|---:|
| `q4tp` | 59.7 | 4.6 s | 45.4 | 25.4 s | 39.4 |
| `q8_2f` | 39.5 | 3.8 s | 31.6 | 22.5 s | 29.5 |

The fanless Air has the base M4's memory bus (about 100 GB/s): the 8-bit file reads its 2.3 GB a token at 92 GB/s.

### MacBook Air M4 (24 GB), CPU (10 cores)

| file | decode, tok/s | 2k prompt | decode at 2k |
|---|---:|---:|---:|
| `q4tp` | 29.9 | 14.8 s | 16.6 |
| `q8_2f` | 30.2 | 15.5 s | 12.4 |

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
to 3.6 GB on an L4 with CUDA, decoding at 79 tok/s after it. `CMF_MAX_SEQ` sets the cap (32 768 by default). Planted
facts read back from a long Wikipedia text, CUDA, greedy: 4/4 at about 13 000 tokens and 3/4 at about 27 000 for the
4-bit file, the same as Transformers bf16 (the same question missed).

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
а также CPU. На NVIDIA L4 (24 ГБ) с CUDA 4-битный файл пишет 129 токенов в секунду и читает промпт из 2048 токенов
за 0,33 с.

### Быстрый старт

```bash
cargo install --git https://github.com/infosave2007/cmf --branch aion cortiq-cli
hf download infosave/Aion-1.0-Instruct-2.3B-cmf Aion-1.0-Instruct-2.3B-q8_2f.cmf --local-dir .
cortiq run Aion-1.0-Instruct-2.3B-q8_2f.cmf --prompt "Сделай краткую сводку в трёх пунктах: ..."
```

Файлам нужен релиз cortiq после 0.9.8 (в нём шаблон чата модели и разбор её вызовов инструментов); до выхода —
сборка из ветки выше. Для бэкенда CUDA добавьте `--features cuda` и запускайте с `CMF_GPU=cuda`.

### Файлы

| файл | квантизация | размер | ppl WikiText-2 | ppl Rust | ppl Python |
|---|---|---:|---:|---:|---:|
| `Aion-1.0-Instruct-2.3B-q8_2f.cmf` | 8 бит, **по умолчанию** | 2,31 ГБ | 14,25 | 7,66 | 7,26 |
| `Aion-1.0-Instruct-2.3B-q4tp.cmf` | 4 бита GPTQ, эмбеддинги 8 бит | 1,41 ГБ | 14,61 | 7,83 | 7,43 |
| эталон: Transformers bf16 | — | 4,58 ГБ | 14,24 | 7,66 | 7,26 |

Перплексия на 12 окнах по 512 токенов (WikiText-2 test, 1,8 МБ кода на Rust, 477 КБ на Python), окна одни и те же
для всех строк. На 4 окнах по 4096 токенов (дальше скользящего окна в 2048): q8_2f 9,47, bf16 9,46, q4tp 9,70.
4-битные веса округлены GPTQ на 4096 токенах WikiText-2, кода и собственных ответов модели в её шаблоне, с
масштабами каналов и уточнением кодов. Жадные ответы 8-битного файла слово в слово совпадают с Transformers на
CUDA, Vulkan и CPU.

Для длинных промптов при неизменной памяти карты:
[Aion-1.0-Instruct-2.3B Turbo](https://huggingface.co/infosave/Aion-1.0-Instruct-2.3B-Turbo-cmf).

### Скорость

Один поток, `cortiq bench --core`: скорость генерации, затем время до первого токена при промпте такой длины и
генерация после него. Память карты — пик по `nvidia-smi` за прогон.

**NVIDIA L4 (24 ГБ, Google Colab), CUDA**

| файл | генерация, ток/с | промпт 2k | генерация на 2k | промпт 8k | генерация на 8k | промпт 16k | генерация на 16k | память карты |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `q4tp` | 129,1 | 0,33 с | 115,6 | 1,78 с | 105,0 | 4,29 с | 94,8 | 2,0–3,0 ГБ |
| `q8_2f` | 86,3 | 0,34 с | 80,0 | 1,83 с | 75,0 | 4,38 с | 68,8 | 3,2–4,2 ГБ |

**NVIDIA L4 (24 ГБ, Google Colab), Vulkan**

| файл | генерация, ток/с | промпт 2k | генерация на 2k | промпт 8k | генерация на 8k | промпт 16k | генерация на 16k | память карты |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `q4tp` | 107,1 | 0,62 с | 80,1 | 3,82 с | 66,5 | 10,37 с | 53,4 | 2,6–4,0 ГБ |
| `q8_2f` | 76,4 | 1,24 с | 61,4 | 6,34 с | 52,9 | 15,5 с | 44,3 | 3,2–4,7 ГБ |

**MacBook Air M4 (24 ГБ), Metal**

| файл | генерация, ток/с | промпт 2k | генерация на 2k | промпт 8k | генерация на 8k |
|---|---:|---:|---:|---:|---:|
| `q4tp` | 59,7 | 4,6 с | 45,4 | 25,4 с | 39,4 |
| `q8_2f` | 39,5 | 3,8 с | 31,6 | 22,5 с | 29,5 |

У безвентиляторного Air шина памяти базового M4 (около 100 ГБ/с): 8-битный файл читает свои 2,3 ГБ на токен со
скоростью 92 ГБ/с.

**MacBook Air M4 (24 ГБ), CPU (10 ядер)**

| файл | генерация, ток/с | промпт 2k | генерация на 2k |
|---|---:|---:|---:|
| `q4tp` | 29,9 | 14,8 с | 16,6 |
| `q8_2f` | 30,2 | 15,5 с | 12,4 |

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
файла 3,6 ГБ на L4 с CUDA, генерация после него — 79 ток/с. `CMF_MAX_SEQ` задаёт предел (по умолчанию 32 768).
Факты, спрятанные в длинном тексте Википедии (CUDA, жадно): 4/4 примерно на 13 000 токенов и 3/4 примерно на
27 000 у 4-битного файла — как у Transformers bf16 (промах на том же вопросе).

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
Intel 显卡走 Vulkan/DX12，Apple silicon 走 Metal，也可在 CPU 上运行。在 24 GB 的 NVIDIA L4 上用 CUDA，4 位文件每秒
生成 129 个 token，读取 2048 个 token 的提示词只需 0.33 秒。

### 快速开始

```bash
cargo install --git https://github.com/infosave2007/cmf --branch aion cortiq-cli
hf download infosave/Aion-1.0-Instruct-2.3B-cmf Aion-1.0-Instruct-2.3B-q8_2f.cmf --local-dir .
cortiq run Aion-1.0-Instruct-2.3B-q8_2f.cmf --prompt "用三个要点总结：..."
```

需要 0.9.8 之后的 cortiq 版本（它能渲染模型的对话模板并解析其工具调用）；发布前请从上面的分支构建。CUDA 后端
需在安装时加 `--features cuda`，并以 `CMF_GPU=cuda` 运行。

### 文件

| 文件 | 量化 | 大小 | WikiText-2 困惑度 | Rust 困惑度 | Python 困惑度 |
|---|---|---:|---:|---:|---:|
| `Aion-1.0-Instruct-2.3B-q8_2f.cmf` | 8 位，**默认** | 2.31 GB | 14.25 | 7.66 | 7.26 |
| `Aion-1.0-Instruct-2.3B-q4tp.cmf` | 4 位 GPTQ，嵌入层 8 位 | 1.41 GB | 14.61 | 7.83 | 7.43 |
| 参考：Transformers bf16 | — | 4.58 GB | 14.24 | 7.66 | 7.26 |

困惑度基于 12 个 512 token 的窗口（WikiText-2 test、1.8 MB Rust 代码、477 KB Python 代码），各行使用相同窗口。
4 个 4096 token 的窗口（超过 2048 的滑动窗口）：q8_2f 9.47，bf16 9.46，q4tp 9.70。4 位权重用 GPTQ 在 4096 个
token 上取整（WikiText-2、代码以及模型自身模板中的回答），并使用通道缩放与码值细化。8 位文件的贪心输出在
CUDA、Vulkan 和 CPU 上与 Transformers 逐字一致。

长提示词且显存保持不变：[Aion-1.0-Instruct-2.3B Turbo](https://huggingface.co/infosave/Aion-1.0-Instruct-2.3B-Turbo-cmf)。

### 性能

单流，`cortiq bench --core`：生成速度，以及该长度提示词的首 token 时间和之后的生成速度。显存为运行期间
`nvidia-smi` 的峰值。

**NVIDIA L4（24 GB，Google Colab），CUDA**

| 文件 | 生成，tok/s | 2k 提示词 | 2k 处生成 | 8k 提示词 | 8k 处生成 | 16k 提示词 | 16k 处生成 | 显存 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `q4tp` | 129.1 | 0.33 s | 115.6 | 1.78 s | 105.0 | 4.29 s | 94.8 | 2.0–3.0 GB |
| `q8_2f` | 86.3 | 0.34 s | 80.0 | 1.83 s | 75.0 | 4.38 s | 68.8 | 3.2–4.2 GB |

**NVIDIA L4（24 GB，Google Colab），Vulkan**

| 文件 | 生成，tok/s | 2k 提示词 | 2k 处生成 | 8k 提示词 | 8k 处生成 | 16k 提示词 | 16k 处生成 | 显存 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `q4tp` | 107.1 | 0.62 s | 80.1 | 3.82 s | 66.5 | 10.37 s | 53.4 | 2.6–4.0 GB |
| `q8_2f` | 76.4 | 1.24 s | 61.4 | 6.34 s | 52.9 | 15.5 s | 44.3 | 3.2–4.7 GB |

**MacBook Air M4（24 GB），Metal**

| 文件 | 生成，tok/s | 2k 提示词 | 2k 处生成 | 8k 提示词 | 8k 处生成 |
|---|---:|---:|---:|---:|---:|
| `q4tp` | 59.7 | 4.6 s | 45.4 | 25.4 s | 39.4 |
| `q8_2f` | 39.5 | 3.8 s | 31.6 | 22.5 s | 29.5 |

无风扇的 Air 采用基础款 M4 的内存总线（约 100 GB/s）：8 位文件每个 token 以 92 GB/s 读取 2.3 GB。

**MacBook Air M4（24 GB），CPU（10 核）**

| 文件 | 生成，tok/s | 2k 提示词 | 2k 处生成 |
|---|---:|---:|---:|
| `q4tp` | 29.9 | 14.8 s | 16.6 |
| `q8_2f` | 30.2 | 15.5 s | 12.4 |

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
每个 token 在显卡上约 28 KB：30 000 token 的提示词让 4 位文件在 L4（CUDA）上占用 3.6 GB，之后生成 79 tok/s。
`CMF_MAX_SEQ` 设定上限（默认 32 768）。在长篇维基百科文本中埋入的事实（CUDA，贪心）：4 位文件约 13 000 token
时 4/4，约 27 000 token 时 3/4，与 Transformers bf16 相同（错在同一个问题上）。

### 校验下载

```bash
cortiq verify Aion-1.0-Instruct-2.3B-q8_2f.cmf
```

sha256 见英文部分的表格。权重：[MIT](https://huggingface.co/microsoft/Aion-1.0-Instruct-2.3B/blob/main/LICENSE)，
来自 [microsoft/Aion-1.0-Instruct-2.3B](https://huggingface.co/microsoft/Aion-1.0-Instruct-2.3B)；原模型卡关于用途、
局限和安全的说明同样适用于这些文件。
