---
license: apache-2.0
library_name: cortiq
base_model: openbmb/MiniCPM5-2B
base_model_relation: quantized
pipeline_tag: text-generation
tags:
  - cmf
  - cortiq
  - quantized
language:
  - en
  - zh
---

# MiniCPM5-2B — CMF

[MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B) (2.5B parameters,
42 layers, 128k context, thinking mode, tool calling; English and Chinese)
packaged as single [CMF](https://github.com/infosave2007/cmf) files. They run
with `cortiq`, a Rust inference engine with no Python and no ML framework:
NVIDIA, AMD and Intel GPUs through Vulkan/DX12, Apple silicon through Metal,
and the CPU.

[English](#quick-start) · [Русский](#документация-на-русском) · [中文](#中文文档)

## Quick start

```bash
cargo install cortiq-cli
hf download infosave/MiniCPM5-2B-cmf MiniCPM5-2B-q8_2f.cmf --local-dir .
cortiq run MiniCPM5-2B-q8_2f.cmf --prompt "Explain quicksort in three sentences." --no-think
```

Use cortiq 0.7.4 or later; prebuilt binaries for Linux, macOS and Windows are
on the [releases page](https://github.com/infosave2007/cmf/releases). The
engine picks the GPU and its settings itself. `--no-think` asks for a direct
answer; without it the model reasons in a `<think>` block first, so raise
`--max-tokens` (default 256).

## Files

| file | quantization | size | wikitext-2 ppl | recommended for |
|---|---|---:|---:|---|
| `MiniCPM5-2B-q4tp.cmf` | 4-bit, ladder scales | 1.32 GB | 21.21 | smallest and fastest (+19 % ppl) |
| `MiniCPM5-2B-q8_2f.cmf` | 8-bit | 2.53 GB | 17.83 | **default**: practically lossless |

Both files are quantized directly from the bf16 checkpoint. Perplexity: twelve
512-token windows of wikitext-2; the unquantized model scores 17.81 on the
same windows.

## Performance

One stream, medians of three runs:

| hardware | backend | file | `cortiq bench --core`, tok/s | chat, 300 tokens, tok/s | 2k-token prompt, time to first token |
|---|---|---|---:|---:|---:|
| RTX 3090 (24 GB) | Vulkan | `q4tp` | 104.5 | 87.7 | 18.6 s |
| RTX 3090 (24 GB) | Vulkan | `q8_2f` | 90.1 | 74.0 | 37.4 s |
| Mac mini M4 (24 GB) | Metal | `q4tp` | 49.1 | 42.2 | 6.6 s |

The chat and prompt columns are measured through `cortiq serve` after one
warm-up request, with the recommended sampling below. On Vulkan/DX12, long
prompts are read much faster with `CMF_PREFILL_CHUNK=512`: 8.2 s instead of
18.6 s (`q4tp`) and 25.0 s instead of 37.4 s (`q8_2f`) for the same 2k-token
prompt, with the same output. Apple silicon already uses that setting.

## Usage

```bash
cortiq run MiniCPM5-2B-q8_2f.cmf                             # interactive chat
cortiq run MiniCPM5-2B-q8_2f.cmf --prompt "..."              # one answer, with reasoning
cortiq run MiniCPM5-2B-q8_2f.cmf --prompt "..." --no-think   # answer without the thinking block
cortiq run MiniCPM5-2B-q8_2f.cmf --prompt "..." --greedy     # deterministic output
```

### Sampling

OpenBMB recommends temperature 1.0, top-p 0.95, min-p 0; if the output starts
repeating, add a repetition penalty of 1.05:

```bash
cortiq run MiniCPM5-2B-q8_2f.cmf --temperature 1.0 --top-p 0.95 --top-k 0 --min-p 0 --rep-penalty 1.0 --max-tokens 4096 --prompt "..."
```

Without flags the CLI uses temperature 0.7, top-p 0.9, top-k 40, min-p 0.05,
repetition penalty 1.1 and at most 256 new tokens. `--top-k 0` and
`--rep-penalty 1.0` switch those filters off. Answers stop at `<|im_end|>` or
`</s>`.

### OpenAI-compatible server

```bash
cortiq serve MiniCPM5-2B-q8_2f.cmf --port 8080

curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "MiniCPM5-2B-q8_2f", "temperature": 1.0, "top_p": 0.95,
       "messages": [{"role": "user", "content": "Hi!"}]}'
```

Endpoints: `/v1/chat/completions`, `/v1/completions`, `/v1/models`, plus a web
dashboard on the same port. `--host 127.0.0.1` keeps the server local.
`temperature: 0` means greedy decoding; `"enable_thinking": false` in the
request gives a direct answer.

### Tool calling

Send `tools` in the usual OpenAI format; the embedded template puts them in the
system prompt the way MiniCPM5 was trained. The model calls a tool in XML:

```
<function name="get_weather"><param name="city">Paris</param></function>
```

From cortiq 0.7.6 the server turns this into `message.tool_calls` with
`finish_reason: "tool_calls"`, streamed or not; `arguments` is a JSON string
whose values follow the types in the tool's schema. Older versions leave the
XML in `message.content`. Run the tool, then send the assistant turn back with
`tool_calls` and the result as a `{"role": "tool", ...}` message. On
Vulkan/DX12 start the server with `CMF_KV_REUSE=0` for tool conversations:
with the default cross-turn cache the model often repeats the call instead of
answering from the result. With it, and on Apple silicon by default, the call
and the answer from the result were correct in 6 of 6 test conversations for
both files. The embedded template differs from the upstream file only in
writing `tojson` without `ensure_ascii=False`, which gives the same output, and
the four tool-markup tokens are marked as ordinary text. cortiq 0.7.6 and later
also accept the upstream template and tokenizer unchanged.

## Hardware

**Apple silicon (Metal).** Runs on the GPU out of the box.

**Choosing a GPU (Vulkan).** `cortiq gpu` lists the adapters;
`CMF_GPU_ADAPTER=1` (index or name substring) picks one. If the list is empty
on Linux, install `libvulkan1 libglvnd0 libegl1 libgl1 libglx0`; on headless
machines set `XDG_RUNTIME_DIR=/tmp`.

## Long context

The model supports 131 072 tokens; the KV cache holds 32 768 by default. Raise
it with `CMF_MAX_SEQ`:

```bash
CMF_MAX_SEQ=131072 cortiq run MiniCPM5-2B-q8_2f.cmf --prompt "..." --max-tokens 4096
```

The cache takes about 86 KB per token: 2.8 GB at 32k tokens, 11.3 GB at the
full 131k. Decoding slows down as the context grows (about 18 tok/s at 16k
tokens for `q4tp` on the RTX 3090).

## Verify the download

```bash
sha256sum -c MiniCPM5-2B-q8_2f.cmf.sha256
cortiq verify MiniCPM5-2B-q8_2f.cmf
cortiq info MiniCPM5-2B-q8_2f.cmf
```

Weights derive from OpenBMB's release and remain under its Apache-2.0 terms.

---

## Документация на русском

[MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B) (2.5 млрд
параметров, 42 слоя, контекст 128k, режим рассуждений, вызов инструментов;
английский и китайский) в виде отдельных файлов
[CMF](https://github.com/infosave2007/cmf). Запускается движком `cortiq` на
Rust без Python и ML-фреймворков: GPU NVIDIA, AMD и Intel через Vulkan/DX12,
Apple silicon через Metal, а также CPU.

### Быстрый старт

```bash
cargo install cortiq-cli
hf download infosave/MiniCPM5-2B-cmf MiniCPM5-2B-q8_2f.cmf --local-dir .
cortiq run MiniCPM5-2B-q8_2f.cmf --prompt "Объясни квиксорт в трёх предложениях." --no-think
```

Используйте cortiq 0.7.4 или новее; готовые сборки для Linux, macOS и
Windows — на [странице релизов](https://github.com/infosave2007/cmf/releases).
Движок сам выбирает GPU и настройки. `--no-think` даёт прямой ответ; без него
модель сначала рассуждает в блоке `<think>`, поэтому увеличьте `--max-tokens`
(по умолчанию 256).

### Файлы

| файл | квантизация | размер | ppl wikitext-2 | назначение |
|---|---|---:|---:|---|
| `MiniCPM5-2B-q4tp.cmf` | 4 бита, лестница масштабов | 1.32 ГБ | 21.21 | самый компактный и быстрый (+19 % ppl) |
| `MiniCPM5-2B-q8_2f.cmf` | 8 бит | 2.53 ГБ | 17.83 | **по умолчанию**: практически без потерь |

Оба файла квантованы напрямую из bf16-чекпойнта. Перплексия измерена на 12
окнах wikitext-2 по 512 токенов; неквантованная модель на тех же окнах даёт
17.81.

### Скорость

Один поток, медианы трёх запусков:

| железо | бэкенд | файл | `cortiq bench --core`, ток/с | чат, 300 токенов, ток/с | промпт 2k токенов, время до первого токена |
|---|---|---|---:|---:|---:|
| RTX 3090 (24 ГБ) | Vulkan | `q4tp` | 104.5 | 87.7 | 18.6 с |
| RTX 3090 (24 ГБ) | Vulkan | `q8_2f` | 90.1 | 74.0 | 37.4 с |
| Mac mini M4 (24 ГБ) | Metal | `q4tp` | 49.1 | 42.2 | 6.6 с |

Столбцы «чат» и «промпт» измерены через `cortiq serve` после одного
прогревочного запроса, с рекомендуемым сэмплированием (ниже). На Vulkan/DX12
длинные промпты читаются заметно быстрее с `CMF_PREFILL_CHUNK=512`: 8.2 с
вместо 18.6 (`q4tp`) и 25.0 с вместо 37.4 (`q8_2f`) на том же промпте в 2k
токенов, вывод тот же. На Apple silicon эта настройка уже действует.

### Использование

```bash
cortiq run MiniCPM5-2B-q8_2f.cmf                             # интерактивный чат
cortiq run MiniCPM5-2B-q8_2f.cmf --prompt "..."              # один ответ с рассуждением
cortiq run MiniCPM5-2B-q8_2f.cmf --prompt "..." --no-think   # ответ без блока рассуждений
cortiq run MiniCPM5-2B-q8_2f.cmf --prompt "..." --greedy     # детерминированный вывод
```

OpenBMB рекомендует temperature 1.0, top-p 0.95, min-p 0; если вывод начинает
повторяться, добавьте repetition penalty 1.05:

```bash
cortiq run MiniCPM5-2B-q8_2f.cmf --temperature 1.0 --top-p 0.95 --top-k 0 --min-p 0 --rep-penalty 1.0 --max-tokens 4096 --prompt "..."
```

Без флагов CLI использует temperature 0.7, top-p 0.9, top-k 40, min-p 0.05,
repetition penalty 1.1 и не более 256 новых токенов. `--top-k 0` и
`--rep-penalty 1.0` отключают эти фильтры. Ответ заканчивается на `<|im_end|>`
или `</s>`.

**Сервер с OpenAI-совместимым API:**

```bash
cortiq serve MiniCPM5-2B-q8_2f.cmf --port 8080

curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "MiniCPM5-2B-q8_2f", "temperature": 1.0, "top_p": 0.95,
       "messages": [{"role": "user", "content": "Привет!"}]}'
```

Эндпоинты `/v1/chat/completions`, `/v1/completions`, `/v1/models` и
веб-панель на том же порту. `--host 127.0.0.1` оставляет сервер локальным.
`temperature: 0` означает greedy-декодирование; `"enable_thinking": false` в
запросе даёт прямой ответ.

**Вызов инструментов.** Передавайте `tools` в обычном формате OpenAI;
встроенный шаблон помещает их в системный промпт так, как модель обучали.
Модель вызывает инструмент в XML:

```
<function name="get_weather"><param name="city">Paris</param></function>
```

Начиная с cortiq 0.7.6 сервер превращает такой вызов в `message.tool_calls` с
`finish_reason: "tool_calls"`, и в потоковом режиме тоже; `arguments` — строка
JSON, значения в которой приведены к типам из схемы инструмента. Более старые
версии оставляют XML в `message.content`. Выполните инструмент и отправьте
обратно ход ассистента с `tool_calls` и результат сообщением
`{"role": "tool", ...}`. На
Vulkan/DX12 для диалогов с инструментами запускайте сервер с
`CMF_KV_REUSE=0`: с межходовым кэшем по умолчанию модель часто повторяет
вызов вместо ответа по результату. С этой настройкой, а на Apple silicon и
без неё, вызов и ответ по результату были верны в 6 из 6 тестовых диалогов
для обоих файлов. Встроенный шаблон отличается от исходного только тем, что
`tojson` записан без `ensure_ascii=False` (результат тот же), а четыре токена
разметки вызова помечены как обычный текст. cortiq 0.7.6 и новее принимает и
исходные шаблон и токенизатор без изменений.

### Железо

**Apple silicon (Metal).** Работает на GPU без настройки.

**Выбор GPU (Vulkan).** `cortiq gpu` показывает адаптеры; `CMF_GPU_ADAPTER=1`
(индекс или часть имени) закрепляет нужный. Если на Linux список пуст,
установите `libvulkan1 libglvnd0 libegl1 libgl1 libglx0`; на машинах без
дисплея задайте `XDG_RUNTIME_DIR=/tmp`.

### Длинный контекст

Модель поддерживает 131 072 токена; KV-кэш по умолчанию рассчитан на 32 768.
Увеличьте его через `CMF_MAX_SEQ`:
`CMF_MAX_SEQ=131072 cortiq run MiniCPM5-2B-q8_2f.cmf --prompt "..." --max-tokens 4096`.

Кэш занимает около 86 КБ на токен: 2.8 ГБ на 32k токенов, 11.3 ГБ на полных
131k. С ростом контекста декодирование замедляется (около 18 ток/с на 16k
токенов для `q4tp` на RTX 3090).

### Проверка загрузки

```bash
sha256sum -c MiniCPM5-2B-q8_2f.cmf.sha256
cortiq verify MiniCPM5-2B-q8_2f.cmf
cortiq info MiniCPM5-2B-q8_2f.cmf
```

Веса получены из релиза OpenBMB и распространяются на условиях Apache-2.0.

---

## 中文文档

[MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B)（25 亿参数，42 层，
128k 上下文，支持思考模式和工具调用，中英双语）打包为独立的
[CMF](https://github.com/infosave2007/cmf) 文件，由 `cortiq` 运行——一个不依赖
Python 和机器学习框架的 Rust 推理引擎：NVIDIA、AMD、Intel 显卡走 Vulkan/DX12，
Apple silicon 走 Metal，也可在 CPU 上运行。

### 快速开始

```bash
cargo install cortiq-cli
hf download infosave/MiniCPM5-2B-cmf MiniCPM5-2B-q8_2f.cmf --local-dir .
cortiq run MiniCPM5-2B-q8_2f.cmf --prompt "用三句话解释快速排序。" --no-think
```

建议使用 cortiq 0.7.4 或更新版本；Linux、macOS、Windows 预编译二进制见
[发布页面](https://github.com/infosave2007/cmf/releases)。引擎会自动选择 GPU 和
设置。`--no-think` 直接给出回答；不加该参数时模型会先在 `<think>` 块中思考，
请调高 `--max-tokens`（默认 256）。

### 文件

| 文件 | 量化 | 大小 | wikitext-2 困惑度 | 适用场景 |
|---|---|---:|---:|---|
| `MiniCPM5-2B-q4tp.cmf` | 4 位，阶梯缩放 | 1.32 GB | 21.21 | 体积最小、速度最快（困惑度 +19%） |
| `MiniCPM5-2B-q8_2f.cmf` | 8 位 | 2.53 GB | 17.83 | **默认**：几乎无损 |

两个文件均直接由 bf16 检查点量化。困惑度基于 wikitext-2 的 12 个 512 token
窗口；未量化模型在相同窗口上为 17.81。

### 性能

单流，三次运行取中位数：

| 硬件 | 后端 | 文件 | `cortiq bench --core`，tok/s | 对话，300 token，tok/s | 2k token 提示词，首 token 时间 |
|---|---|---|---:|---:|---:|
| RTX 3090（24 GB） | Vulkan | `q4tp` | 104.5 | 87.7 | 18.6 s |
| RTX 3090（24 GB） | Vulkan | `q8_2f` | 90.1 | 74.0 | 37.4 s |
| Mac mini M4（24 GB） | Metal | `q4tp` | 49.1 | 42.2 | 6.6 s |

「对话」和「提示词」两列通过 `cortiq serve` 测得（先发一次预热请求），使用下文
推荐的采样参数。在 Vulkan/DX12 上，设置 `CMF_PREFILL_CHUNK=512` 可显著加快长
提示词的读取：同一 2k token 提示词，`q4tp` 由 18.6 s 降至 8.2 s，`q8_2f` 由
37.4 s 降至 25.0 s，输出不变。Apple silicon 默认已使用该设置。

### 使用

```bash
cortiq run MiniCPM5-2B-q8_2f.cmf                             # 交互式对话
cortiq run MiniCPM5-2B-q8_2f.cmf --prompt "..."              # 单次回答（含思考）
cortiq run MiniCPM5-2B-q8_2f.cmf --prompt "..." --no-think   # 不输出思考块
cortiq run MiniCPM5-2B-q8_2f.cmf --prompt "..." --greedy     # 确定性输出
```

OpenBMB 推荐 temperature 1.0、top-p 0.95、min-p 0；如出现重复输出，可加
repetition penalty 1.05：

```bash
cortiq run MiniCPM5-2B-q8_2f.cmf --temperature 1.0 --top-p 0.95 --top-k 0 --min-p 0 --rep-penalty 1.0 --max-tokens 4096 --prompt "..."
```

不加参数时 CLI 使用 temperature 0.7、top-p 0.9、top-k 40、min-p 0.05、
repetition penalty 1.1，最多生成 256 个新 token。`--top-k 0` 和
`--rep-penalty 1.0` 关闭这两项过滤。回答在 `<|im_end|>` 或 `</s>` 处结束。

**OpenAI 兼容 API 服务器：**

```bash
cortiq serve MiniCPM5-2B-q8_2f.cmf --port 8080

curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "MiniCPM5-2B-q8_2f", "temperature": 1.0, "top_p": 0.95,
       "messages": [{"role": "user", "content": "你好！"}]}'
```

提供 `/v1/chat/completions`、`/v1/completions`、`/v1/models` 端点，同一端口还有
网页控制台。`--host 127.0.0.1` 仅限本机访问。请求中 `temperature: 0` 表示贪心
解码；`"enable_thinking": false` 直接给出回答。

**工具调用。** 按常规 OpenAI 格式传入 `tools`；内置模板会按 MiniCPM5 的训练方式
把它们放进系统提示词。模型以 XML 形式调用工具：

```
<function name="get_weather"><param name="city">Paris</param></function>
```

从 cortiq 0.7.6 起，服务器会把它转换为 `message.tool_calls`，并给出
`finish_reason: "tool_calls"`，流式输出同样如此；`arguments` 是 JSON 字符串，其中的值
按工具 schema 声明的类型转换。更早的版本会把 XML 留在 `message.content` 中。
执行工具后，把带 `tool_calls` 的助手回合和
`{"role": "tool", ...}` 结果消息一并发回。在 Vulkan/DX12 上进行工具对话时，请用
`CMF_KV_REUSE=0` 启动服务器：默认的跨回合缓存下，模型常会重复调用而不是根据结果
作答。设置后（Apple silicon 默认即可），两个文件在 6 个测试对话中调用和基于结果的
回答全部正确。内置模板与上游文件的唯一区别是 `tojson` 去掉了
`ensure_ascii=False`（输出相同），并且四个工具标记 token 被标为普通文本。
cortiq 0.7.6 及以后版本也可以直接使用未经修改的上游模板和分词器。

### 硬件

**Apple silicon（Metal）。** 开箱即用 GPU 加速。

**选择显卡（Vulkan）。** `cortiq gpu` 列出所有适配器；`CMF_GPU_ADAPTER=1`（索引或
名称片段）指定使用哪一块。若 Linux 上列表为空，请安装
`libvulkan1 libglvnd0 libegl1 libgl1 libglx0`；无显示器的机器需设置
`XDG_RUNTIME_DIR=/tmp`。

### 长上下文

模型支持 131 072 个 token；KV 缓存默认容纳 32 768 个，可用 `CMF_MAX_SEQ` 调高：
`CMF_MAX_SEQ=131072 cortiq run MiniCPM5-2B-q8_2f.cmf --prompt "..." --max-tokens 4096`。

缓存每个 token 约占 86 KB：32k token 为 2.8 GB，完整 131k 为 11.3 GB。上下文越长
解码越慢（RTX 3090 上 `q4tp` 在 16k token 时约 18 tok/s）。

### 校验下载

```bash
sha256sum -c MiniCPM5-2B-q8_2f.cmf.sha256
cortiq verify MiniCPM5-2B-q8_2f.cmf
cortiq info MiniCPM5-2B-q8_2f.cmf
```

权重来自 OpenBMB 的发布，遵循其 Apache-2.0 许可。
