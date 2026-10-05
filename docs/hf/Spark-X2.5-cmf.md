---
license: apache-2.0
library_name: cortiq
base_model:
  - XHToken/Spark-X2.5-4B
  - XHToken/Spark-X2.5-1.7B
base_model_relation: quantized
pipeline_tag: text-generation
tags:
  - cmf
  - cortiq
  - quantized
  - sliding-window-attention
  - tool-calling
language:
  - multilingual
  - en
  - zh
  - ru
---

# Spark-X2.5 — CMF (4B · 1.7B)

[Spark-X2.5-4B](https://huggingface.co/XHToken/Spark-X2.5-4B) and
[Spark-X2.5-1.7B](https://huggingface.co/XHToken/Spark-X2.5-1.7B) (hybrid
attention: three 512-token sliding-window layers per full layer, 1M-token
context, thinking mode, tool calling, 200+ languages) packaged as single
[CMF](https://github.com/infosave2007/cmf) files. They run with `cortiq`, a
Rust inference engine with no Python and no ML framework: NVIDIA, AMD and
Intel GPUs through Vulkan/DX12, Apple silicon through Metal, and the CPU.

[English](#quick-start) · [Русский](#документация-на-русском) · [中文](#中文文档)

## Quick start

```bash
cargo install cortiq-cli
hf download infosave/Spark-X2.5-cmf Spark-X2.5-4B-q8_2f.cmf --local-dir .
cortiq run Spark-X2.5-4B-q8_2f.cmf --prompt "Explain quicksort in three sentences." --no-think
```

Use cortiq 0.8.12 or later; prebuilt binaries for Linux, macOS and Windows are
on the [releases page](https://github.com/infosave2007/cmf/releases). The
engine picks the GPU and its settings itself. `--no-think` asks for a direct
answer; without it the model reasons in a `<think>` block first, so raise
`--max-tokens` (default 256).

## Files

| file | quantization | size | wikitext-2 ppl | recommended for |
|---|---|---:|---:|---|
| `Spark-X2.5-4B-q8_2f.cmf` | 8-bit | 4.14 GB | 10.68 | **default**: practically lossless |
| `Spark-X2.5-4B-q4mix.cmf` | gate/up 4-bit (q4tp), everything else 8-bit | 3.23 GB | 10.81 | less memory, faster on GPUs (+1.9 % ppl) |
| `Spark-X2.5-1.7B-q8_2f.cmf` | 8-bit | 1.73 GB | 14.47 | small machines, practically lossless |
| `Spark-X2.5-1.7B-q4mix.cmf` | gate/up 4-bit (q4tp), everything else 8-bit | 1.36 GB | 14.88 | the smallest and fastest (+2.7 % ppl) |

All files are quantized from the bf16 checkpoints. Perplexity: the first 4096
tokens of wikitext-2 test with the BOS token in front; the unquantized models
score 10.61 (4B) and 14.49 (1.7B; transformers in float32 gives the same 14.49).
Spark is sensitive to 4-bit attention and down projections, so only gate/up
are 4-bit in the compact files.

## Performance

`cortiq bench --core --ignore-eos`, decode tok/s, one stream:

| hardware | backend | 4B q8_2f | 4B q4mix | 1.7B q8_2f | 1.7B q4mix |
|---|---|---:|---:|---:|---:|
| RTX 2000 Ada (16 GB) | Vulkan | 40.3 | 48.2 | 85.6 | 101.1 |
| Mac mini M4 (24 GB) | Metal | 20.5 | 25.3 | 41.0 | 52.7 |
| Mac mini M4 (24 GB) | CPU, 10 cores | 21.2 | 23.2 | 47.8 | 51.7 |
| EPYC 9354, 7 cores | CPU | 14.0 | 13.7 | 30.8 | 30.0 |

Prompt processing, tok/s:

| hardware | backend | prompt | 4B q8_2f | 4B q4mix | 1.7B q8_2f | 1.7B q4mix |
|---|---|---:|---:|---:|---:|---:|
| RTX 2000 Ada (16 GB) | Vulkan | 1000 | 377 | 282 | 906 | 772 |
| RTX 2000 Ada (16 GB) | Vulkan | 16000 | 361 | 254 | 833 | 718 |
| Mac mini M4 (24 GB) | Metal | 1000 | 155 | 157 | 350 | 353 |
| EPYC 9354, 7 cores | CPU | 1000 | 39 | 40 | 90 | 84 |

A 16000-token prompt takes 41 s on the 4B q8_2f and 18.5 s on the 1.7B q8_2f
(RTX 2000 Ada).

On the RTX 2000 Ada the 4B q8_2f file takes 4.5 GB of VRAM at a 3k-token
context.

## Usage

```bash
cortiq run Spark-X2.5-4B-q8_2f.cmf                             # interactive chat
cortiq run Spark-X2.5-4B-q8_2f.cmf --prompt "..."              # one answer, with reasoning
cortiq run Spark-X2.5-4B-q8_2f.cmf --prompt "..." --no-think   # answer without the thinking block
cortiq run Spark-X2.5-4B-q8_2f.cmf --prompt "..." --greedy     # deterministic output
```

### Sampling

XHToken recommends temperature 1.0 and top-p 0.95:

```bash
cortiq run Spark-X2.5-4B-q8_2f.cmf --temperature 1.0 --top-p 0.95 --top-k 0 --min-p 0 --rep-penalty 1.0 --max-tokens 4096 --prompt "..."
```

Without flags the CLI uses temperature 0.7, top-p 0.9, top-k 40, min-p 0.05,
repetition penalty 1.1 and at most 256 new tokens. `--top-k 0` and
`--rep-penalty 1.0` switch those filters off.

### OpenAI-compatible server

```bash
cortiq serve Spark-X2.5-4B-q8_2f.cmf --port 8080

curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Spark-X2.5-4B-q8_2f", "temperature": 1.0, "top_p": 0.95,
       "messages": [{"role": "user", "content": "Hi!"}]}'
```

Endpoints: `/v1/chat/completions`, `/v1/completions`, `/v1/models`, plus a web
dashboard on the same port. `--host 127.0.0.1` keeps the server local. The
reply reads `<think>…</think>answer`; `"enable_thinking": false` in the request
gives a direct answer, and `temperature: 0` means greedy decoding.

### Tool calling

Send `tools` in the usual OpenAI format; the embedded template lists them in
the system prompt the way the model was trained. The model calls a tool as

```
<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>
```

and the server returns it as `message.tool_calls` with
`finish_reason: "tool_calls"`, streamed or not; `arguments` is a JSON string
whose values follow the types in the tool's schema. Run the tool, then send the
assistant turn back with `tool_calls` and the result as a
`{"role": "tool", ...}` message.

Tool calls were correct, and so were the answers from the results, in 6 of 6
test conversations for both models with thinking on (the default). With
`enable_thinking: false` the 4B called the tool in 5 of 6 and the 1.7B in 4 of 6;
the rest were answered without a call. For agent loops keep thinking on.

## Hardware

**Apple silicon (Metal).** Runs on the GPU out of the box.

**Choosing a GPU (Vulkan).** `cortiq gpu` lists the adapters;
`CMF_GPU_ADAPTER=1` (index or name substring) picks one. If the list is empty
on Linux, install `libvulkan1 libglvnd0 libegl1 libgl1 libglx0`; on headless
machines set `XDG_RUNTIME_DIR=/tmp`.

**CPU only.** `CMF_GPU=0` keeps everything on the CPU.

## Long context

The model supports 1 048 576 tokens; the KV cache holds 32 768 by default.
Raise it with `CMF_MAX_SEQ`:

```bash
CMF_MAX_SEQ=131072 cortiq run Spark-X2.5-4B-q8_2f.cmf --prompt "..." --max-tokens 4096
```

Only the full-attention layers keep every token (74 KB per token for the 4B,
29 KB for the 1.7B); the sliding layers keep only the rows their
512-token window reads (at most 1024). The
4B's cache takes 1.3 GB at 16k tokens and 2.6 GB at 32k, the 1.7B's 0.5 GB at
16k.

## Verify the download

```bash
sha256sum -c Spark-X2.5-4B-q8_2f.cmf.sha256
cortiq verify Spark-X2.5-4B-q8_2f.cmf
cortiq info Spark-X2.5-4B-q8_2f.cmf
```

Weights derive from XHToken's release and remain under its Apache-2.0 terms.

---

## Документация на русском

[Spark-X2.5-4B](https://huggingface.co/XHToken/Spark-X2.5-4B) и
[Spark-X2.5-1.7B](https://huggingface.co/XHToken/Spark-X2.5-1.7B) (гибридное
внимание: три слоя со скользящим окном 512 токенов на один полный, контекст до
1M токенов, режим рассуждений, вызов инструментов, 200+ языков) в виде
отдельных файлов [CMF](https://github.com/infosave2007/cmf). Запускается
движком `cortiq` на Rust без Python и ML-фреймворков: GPU NVIDIA, AMD и Intel
через Vulkan/DX12, Apple silicon через Metal, а также CPU.

### Быстрый старт

```bash
cargo install cortiq-cli
hf download infosave/Spark-X2.5-cmf Spark-X2.5-4B-q8_2f.cmf --local-dir .
cortiq run Spark-X2.5-4B-q8_2f.cmf --prompt "Объясни квиксорт в трёх предложениях." --no-think
```

Используйте cortiq 0.8.12 или новее; готовые сборки для Linux, macOS и Windows —
на [странице релизов](https://github.com/infosave2007/cmf/releases). Движок сам
выбирает GPU и настройки. `--no-think` даёт прямой ответ; без него модель
сначала рассуждает в блоке `<think>`, поэтому увеличьте `--max-tokens` (по
умолчанию 256).

### Файлы

| файл | квантизация | размер | ppl wikitext-2 | назначение |
|---|---|---:|---:|---|
| `Spark-X2.5-4B-q8_2f.cmf` | 8 бит | 4.14 ГБ | 10.68 | **по умолчанию**: практически без потерь |
| `Spark-X2.5-4B-q4mix.cmf` | gate/up 4 бита (q4tp), остальное 8 бит | 3.23 ГБ | 10.81 | меньше памяти, быстрее на GPU (+1.9 % ppl) |
| `Spark-X2.5-1.7B-q8_2f.cmf` | 8 бит | 1.73 ГБ | 14.47 | слабые машины, практически без потерь |
| `Spark-X2.5-1.7B-q4mix.cmf` | gate/up 4 бита (q4tp), остальное 8 бит | 1.36 ГБ | 14.88 | самый компактный и быстрый (+2.7 % ppl) |

Все файлы квантованы из bf16-чекпойнтов. Перплексия — первые 4096 токенов
wikitext-2 test с BOS-токеном в начале; неквантованные модели дают 10.61 (4B)
и 14.49 (1.7B; transformers в float32 — те же 14.49). Spark чувствителен к
4-битным attention и down-проекциям, поэтому в компактных файлах 4 бита только
у gate/up.

### Скорость

`cortiq bench --core --ignore-eos`, декодирование, ток/с, один поток:

| железо | бэкенд | 4B q8_2f | 4B q4mix | 1.7B q8_2f | 1.7B q4mix |
|---|---|---:|---:|---:|---:|
| RTX 2000 Ada (16 ГБ) | Vulkan | 40.3 | 48.2 | 85.6 | 101.1 |
| Mac mini M4 (24 ГБ) | Metal | 20.5 | 25.3 | 41.0 | 52.7 |
| Mac mini M4 (24 ГБ) | CPU, 10 ядер | 21.2 | 23.2 | 47.8 | 51.7 |
| EPYC 9354, 7 ядер | CPU | 14.0 | 13.7 | 30.8 | 30.0 |

Обработка промпта, ток/с:

| железо | бэкенд | промпт | 4B q8_2f | 4B q4mix | 1.7B q8_2f | 1.7B q4mix |
|---|---|---:|---:|---:|---:|---:|
| RTX 2000 Ada (16 ГБ) | Vulkan | 1000 | 377 | 282 | 906 | 772 |
| RTX 2000 Ada (16 ГБ) | Vulkan | 16000 | 361 | 254 | 833 | 718 |
| Mac mini M4 (24 ГБ) | Metal | 1000 | 155 | 157 | 350 | 353 |
| EPYC 9354, 7 ядер | CPU | 1000 | 39 | 40 | 90 | 84 |

Промпт из 16000 токенов обрабатывается за 41 с на 4B q8_2f и за 18.5 с на
1.7B q8_2f (RTX 2000 Ada).

На RTX 2000 Ada файл 4B q8_2f занимает 4.5 ГБ видеопамяти при контексте 3k токенов.

### Использование

```bash
cortiq run Spark-X2.5-4B-q8_2f.cmf                             # интерактивный чат
cortiq run Spark-X2.5-4B-q8_2f.cmf --prompt "..."              # один ответ с рассуждением
cortiq run Spark-X2.5-4B-q8_2f.cmf --prompt "..." --no-think   # ответ без блока рассуждений
cortiq run Spark-X2.5-4B-q8_2f.cmf --prompt "..." --greedy     # детерминированный вывод
```

XHToken рекомендует temperature 1.0 и top-p 0.95:

```bash
cortiq run Spark-X2.5-4B-q8_2f.cmf --temperature 1.0 --top-p 0.95 --top-k 0 --min-p 0 --rep-penalty 1.0 --max-tokens 4096 --prompt "..."
```

Без флагов CLI использует temperature 0.7, top-p 0.9, top-k 40, min-p 0.05,
repetition penalty 1.1 и не более 256 новых токенов. `--top-k 0` и
`--rep-penalty 1.0` отключают эти фильтры.

**Сервер с OpenAI-совместимым API:**

```bash
cortiq serve Spark-X2.5-4B-q8_2f.cmf --port 8080

curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Spark-X2.5-4B-q8_2f", "temperature": 1.0, "top_p": 0.95,
       "messages": [{"role": "user", "content": "Привет!"}]}'
```

Эндпоинты `/v1/chat/completions`, `/v1/completions`, `/v1/models` и веб-панель
на том же порту. `--host 127.0.0.1` оставляет сервер локальным. Ответ имеет вид
`<think>…</think>ответ`; `"enable_thinking": false` в запросе даёт прямой
ответ, `temperature: 0` — greedy-декодирование.

**Вызов инструментов.** Передавайте `tools` в обычном формате OpenAI;
встроенный шаблон помещает их в системный промпт так, как модель обучали.
Модель вызывает инструмент так:

```
<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>
```

Сервер возвращает это как `message.tool_calls` с `finish_reason: "tool_calls"`,
в потоковом режиме тоже; `arguments` — строка JSON, значения в которой приведены
к типам из схемы инструмента. Выполните инструмент и отправьте обратно ход
ассистента с `tool_calls` и результат сообщением `{"role": "tool", ...}`.

С включёнными рассуждениями (по умолчанию) вызов и ответ по результату были
верны в 6 из 6 тестовых диалогов у обеих моделей. С `enable_thinking: false`
4B вызвала инструмент в 5 из 6, а 1.7B в 4 из 6; остальные ответила без вызова.
Для агентных сценариев оставляйте рассуждения включёнными.

### Железо

**Apple silicon (Metal).** Работает на GPU без настройки.

**Выбор GPU (Vulkan).** `cortiq gpu` показывает адаптеры; `CMF_GPU_ADAPTER=1`
(индекс или часть имени) закрепляет нужный. Если на Linux список пуст,
установите `libvulkan1 libglvnd0 libegl1 libgl1 libglx0`; на машинах без дисплея
задайте `XDG_RUNTIME_DIR=/tmp`.

**Только CPU.** `CMF_GPU=0` оставляет всё на процессоре.

### Длинный контекст

Модель поддерживает 1 048 576 токенов; KV-кэш по умолчанию рассчитан на 32 768.
Увеличьте его через `CMF_MAX_SEQ`:
`CMF_MAX_SEQ=131072 cortiq run Spark-X2.5-4B-q8_2f.cmf --prompt "..." --max-tokens 4096`.

Каждый токен хранят только слои с полным вниманием (74 КБ на токен у 4B,
29 КБ у 1.7B); скользящие слои держат только строки,
которые читает их окно в 512 токенов (не больше 1024). Кэш 4B
занимает 1.3 ГБ на 16k токенов и 2.6 ГБ на 32k, кэш 1.7B — 0.5 ГБ на 16k.

### Проверка загрузки

```bash
sha256sum -c Spark-X2.5-4B-q8_2f.cmf.sha256
cortiq verify Spark-X2.5-4B-q8_2f.cmf
cortiq info Spark-X2.5-4B-q8_2f.cmf
```

Веса получены из релиза XHToken и распространяются на условиях Apache-2.0.

---

## 中文文档

[Spark-X2.5-4B](https://huggingface.co/XHToken/Spark-X2.5-4B) 与
[Spark-X2.5-1.7B](https://huggingface.co/XHToken/Spark-X2.5-1.7B)（混合注意力：
每三层 512 token 滑动窗口注意力配一层全注意力，最长 1M token 上下文，支持思考模式、
工具调用和 200 多种语言）打包为独立的 [CMF](https://github.com/infosave2007/cmf)
文件，由 `cortiq` 运行——一个不依赖 Python 和机器学习框架的 Rust 推理引擎：NVIDIA、
AMD、Intel 显卡走 Vulkan/DX12，Apple silicon 走 Metal，也可在 CPU 上运行。

### 快速开始

```bash
cargo install cortiq-cli
hf download infosave/Spark-X2.5-cmf Spark-X2.5-4B-q8_2f.cmf --local-dir .
cortiq run Spark-X2.5-4B-q8_2f.cmf --prompt "用三句话解释快速排序。" --no-think
```

建议使用 cortiq 0.8.12 或更新版本；Linux、macOS、Windows 预编译二进制见
[发布页面](https://github.com/infosave2007/cmf/releases)。引擎会自动选择 GPU 和设置。
`--no-think` 直接给出回答；不加该参数时模型会先在 `<think>` 块中思考，请调高
`--max-tokens`（默认 256）。

### 文件

| 文件 | 量化 | 大小 | wikitext-2 困惑度 | 适用场景 |
|---|---|---:|---:|---|
| `Spark-X2.5-4B-q8_2f.cmf` | 8 位 | 4.14 GB | 10.68 | **默认**：几乎无损 |
| `Spark-X2.5-4B-q4mix.cmf` | gate/up 为 4 位（q4tp），其余 8 位 | 3.23 GB | 10.81 | 内存更省，GPU 上更快（困惑度 +1.9%） |
| `Spark-X2.5-1.7B-q8_2f.cmf` | 8 位 | 1.73 GB | 14.47 | 配置较低的机器，几乎无损 |
| `Spark-X2.5-1.7B-q4mix.cmf` | gate/up 为 4 位（q4tp），其余 8 位 | 1.36 GB | 14.88 | 体积最小、速度最快（困惑度 +2.7%） |

所有文件均由 bf16 检查点量化。困惑度基于 wikitext-2 test 的前 4096 个 token，开头加
BOS token；未量化模型分别为 10.61（4B）和 14.49（1.7B；transformers float32 同为
14.49）。Spark 对 4 位的注意力和 down 投影较敏感，因此紧凑文件中只有 gate/up 为 4 位。

### 性能

`cortiq bench --core --ignore-eos`，解码速度，tok/s，单流：

| 硬件 | 后端 | 4B q8_2f | 4B q4mix | 1.7B q8_2f | 1.7B q4mix |
|---|---|---:|---:|---:|---:|
| RTX 2000 Ada（16 GB） | Vulkan | 40.3 | 48.2 | 85.6 | 101.1 |
| Mac mini M4（24 GB） | Metal | 20.5 | 25.3 | 41.0 | 52.7 |
| Mac mini M4（24 GB） | CPU，10 核 | 21.2 | 23.2 | 47.8 | 51.7 |
| EPYC 9354，7 核 | CPU | 14.0 | 13.7 | 30.8 | 30.0 |

提示词处理速度，tok/s：

| 硬件 | 后端 | 提示词 | 4B q8_2f | 4B q4mix | 1.7B q8_2f | 1.7B q4mix |
|---|---|---:|---:|---:|---:|---:|
| RTX 2000 Ada（16 GB） | Vulkan | 1000 | 377 | 282 | 906 | 772 |
| RTX 2000 Ada（16 GB） | Vulkan | 16000 | 361 | 254 | 833 | 718 |
| Mac mini M4（24 GB） | Metal | 1000 | 155 | 157 | 350 | 353 |
| EPYC 9354，7 核 | CPU | 1000 | 39 | 40 | 90 | 84 |

在 RTX 2000 Ada 上，16000 token 的提示词在 4B q8_2f 上需要 41 秒，在 1.7B q8_2f 上需要 18.5 秒。

在 RTX 2000 Ada 上，4B q8_2f 文件在 3k token 上下文时占用 4.5 GB 显存。

### 使用

```bash
cortiq run Spark-X2.5-4B-q8_2f.cmf                             # 交互式对话
cortiq run Spark-X2.5-4B-q8_2f.cmf --prompt "..."              # 单次回答（含思考）
cortiq run Spark-X2.5-4B-q8_2f.cmf --prompt "..." --no-think   # 不输出思考块
cortiq run Spark-X2.5-4B-q8_2f.cmf --prompt "..." --greedy     # 确定性输出
```

XHToken 推荐 temperature 1.0、top-p 0.95：

```bash
cortiq run Spark-X2.5-4B-q8_2f.cmf --temperature 1.0 --top-p 0.95 --top-k 0 --min-p 0 --rep-penalty 1.0 --max-tokens 4096 --prompt "..."
```

不加参数时 CLI 使用 temperature 0.7、top-p 0.9、top-k 40、min-p 0.05、repetition
penalty 1.1，最多生成 256 个新 token。`--top-k 0` 和 `--rep-penalty 1.0` 关闭这两项过滤。

**OpenAI 兼容 API 服务器：**

```bash
cortiq serve Spark-X2.5-4B-q8_2f.cmf --port 8080

curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Spark-X2.5-4B-q8_2f", "temperature": 1.0, "top_p": 0.95,
       "messages": [{"role": "user", "content": "你好！"}]}'
```

提供 `/v1/chat/completions`、`/v1/completions`、`/v1/models` 端点，同一端口还有网页
控制台。`--host 127.0.0.1` 仅限本机访问。回答形式为 `<think>…</think>回答`；请求中
`"enable_thinking": false` 直接给出回答，`temperature: 0` 表示贪心解码。

**工具调用。** 按常规 OpenAI 格式传入 `tools`；内置模板会按模型的训练方式把它们放进
系统提示词。模型以如下形式调用工具：

```
<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>
```

服务器会把它转换为 `message.tool_calls`，并给出 `finish_reason: "tool_calls"`，流式输出
同样如此；`arguments` 是 JSON 字符串，其中的值按工具 schema 声明的类型转换。执行工具后，
把带 `tool_calls` 的助手回合和 `{"role": "tool", ...}` 结果消息一并发回。

开启思考（默认）时，两个模型在 6 个测试对话中的调用和基于结果的回答全部正确。使用
`enable_thinking: false` 时，4B 在 6 次中调用了 5 次，1.7B 调用了 4 次，其余未调用工具
直接作答。智能体场景请保持思考开启。

### 硬件

**Apple silicon（Metal）。** 开箱即用 GPU 加速。

**选择显卡（Vulkan）。** `cortiq gpu` 列出所有适配器；`CMF_GPU_ADAPTER=1`（索引或名称
片段）指定使用哪一块。若 Linux 上列表为空，请安装 `libvulkan1 libglvnd0 libegl1 libgl1
libglx0`；无显示器的机器需设置 `XDG_RUNTIME_DIR=/tmp`。

**仅用 CPU。** `CMF_GPU=0` 让所有计算留在 CPU 上。

### 长上下文

模型支持 1 048 576 个 token；KV 缓存默认容纳 32 768 个，可用 `CMF_MAX_SEQ` 调高：
`CMF_MAX_SEQ=131072 cortiq run Spark-X2.5-4B-q8_2f.cmf --prompt "..." --max-tokens 4096`。

只有全注意力层保存每个 token（4B 每个 token 74 KB，1.7B 为 29 KB），滑动窗口层只保留
其 512 token 窗口读取的行（最多 1024 行）。4B 的缓存在 16k token 时占 1.3 GB，32k 时占 2.6 GB；1.7B 在 16k
时占 0.5 GB。

### 校验下载

```bash
sha256sum -c Spark-X2.5-4B-q8_2f.cmf.sha256
cortiq verify Spark-X2.5-4B-q8_2f.cmf
cortiq info Spark-X2.5-4B-q8_2f.cmf
```

权重来自 XHToken 的发布，遵循其 Apache-2.0 许可。
