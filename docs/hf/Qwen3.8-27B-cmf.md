---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
  - cmf
  - cortiq
  - quantized
language:
  - en
  - ru
  - zh
library_name: cortiq
pipeline_tag: text-generation
base_model_relation: quantized
---

# Qwen3.8-27B — CMF

[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (48 GatedDeltaNet
linear-attention layers + 16 full-attention layers, 262k context, thinking
mode) packaged as a single [CMF](https://github.com/infosave2007/cmf) file.
It runs with `cortiq`, a Rust inference engine with no Python and no ML
framework: NVIDIA GPUs through CUDA or Vulkan/DX12, AMD and Intel GPUs
through Vulkan/DX12, Apple silicon through Metal, and the CPU. On an RTX 3090
it decodes chat at 120–128 tok/s through CUDA.

[English](#quick-start) · [Русский](#документация-на-русском) · [中文](#中文文档)

## Quick start

```bash
cargo install cortiq-cli
hf download infosave/Qwen3.8-27B-cmf qwen38-27b-q4tp.cmf --local-dir .
cortiq run qwen38-27b-q4tp.cmf --prompt "Explain quicksort in three sentences." --no-think
```

Use cortiq 0.8.15 or later for fast prompt processing; prebuilt binaries for
Linux, macOS and Windows are on the [releases page](https://github.com/infosave2007/cmf/releases).
The engine picks the GPU and its settings itself. `--no-think` asks for a
direct answer; without it the model reasons first, so raise `--max-tokens`
(default 256).

On an NVIDIA card the CUDA backend is the fastest path (cortiq 0.8.69 or later):

```bash
cargo install cortiq-cli --features cuda
hf download infosave/Qwen3.8-27B-cmf qwen38-27b-q4tp.cmf qwen38-27b-q4tp.dflash.cmf --local-dir .
CMF_GPU=cuda CMF_DFLASH=1 cortiq run qwen38-27b-q4tp.cmf --prompt "Explain quicksort in three sentences." --greedy --no-think
```

See [NVIDIA via CUDA](#nvidia-via-cuda) for the settings.

## Files

| file | quantization | size | wikitext-2 ppl | recommended for |
|---|---|---:|---:|---|
| `qwen38-27b-q4tp.cmf` | 4-bit, ladder scales | 14.3 GB | 8.79 | default choice: Macs and 16–24 GB GPUs; speculative decoding |
| `qwen38-27b-q4t.cmf` | 4-bit | 15.4 GB | 8.86 | alternative 4-bit layout, without speculative decoding |
| `qwen38-27b-q8_2f.cmf` | 8-bit | 27.4 GB | — | highest precision, 32 GB of memory or more |
| `qwen38-27b-q4tp.dflash.cmf` | DFlash 2 drafter, 4-bit | 1.3 GB | — | beside `q4tp` on CUDA: faster chat with `CMF_DFLASH=1` |

The 4-bit files are quantized directly from the bf16 checkpoint.
Perplexity: twelve 512-token windows of wikitext-2. The drafter is
[z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2)
converted for `q4tp`, with a router fitted at an RTX 3090's prices.

## Performance

Decode speed of `qwen38-27b-q4tp.cmf`, one stream, tok/s:

| hardware | backend | plain decode | code prompt | `cortiq bench --core` |
|---|---|---:|---:|---:|
| RTX 3090 (24 GB) | **CUDA** | **49.1** | **125.8** | **232** |
| RTX 5090 (32 GB) | Vulkan | 48.7 | 56.5 | 76 |
| RTX 3090 (24 GB) | Vulkan | 37.1 | 43.6 | 62.4 |
| RTX PRO 4000 Blackwell (24 GB) | Vulkan | 27.9 | 43.9 | 54.8 |
| Mac mini M4 (24 GB) | Metal | 6.8 | 17.3 | 20.0 |

The code prompt and bench columns use speculative decoding with `--greedy`;
on the RTX 5090 the code prompt carries a 2.3k-token context. The CUDA row is
cortiq 0.8.69 with the DFlash drafter on an LRU cache module, 512 tokens, and
the bench after a 4 000-token prompt. RTX 5090
figures were measured on an earlier release. RTX 3090 figures were measured
with cortiq 0.8.15 on Ubuntu 24.04: plain decode with `CMF_GRAPH_SPEC=0`, a
400-token code answer, 91% of drafted tokens accepted in the bench.

Prompt processing on the RTX 3090 (`cortiq bench --ctx N`):

| prompt | prompt processing, tok/s | time to first token | decode at this depth, tok/s |
|---|---:|---:|---:|
| 1000 tokens | 199 | 5.0 s | 57.8 |
| 4000 tokens | 199 | 20.1 s | 33.9 |

Decode in this table is speculative: 87% of drafted tokens are accepted after
a 1000-token prompt, 50% after a 4000-token prompt. Through CUDA the same card
reads prompts four to five times faster:

| prompt | prompt processing, tok/s | time to first token |
|---|---:|---:|
| 4 000 tokens | 983 | 4.1 s |
| 16 000 tokens | 929 | 17.2 s |
| 32 000 tokens | 819 | 39.1 s |

**Speculative decoding** is on by default for `qwen38-27b-q4tp.cmf`: the
model's own MTP head drafts several tokens and the GPU verifies them in one
batch. It pays most on code and structured output; on free prose it runs
near the plain speed, and the engine switches it off wherever it does not
pay. On Apple silicon it covers every mode, including the default sampling.
On Vulkan and CUDA it covers greedy decoding without penalties, so add
`--greedy` there. `CMF_GRAPH_SPEC=0` disables it. On Vulkan the int8 verify
can pick a different token when two are almost equally likely;
`CMF_VERIFY_I8=0` makes greedy output identical to plain decoding. On CUDA
the text with drafts is the text without them.

## Usage

```bash
cortiq run qwen38-27b-q4tp.cmf                           # interactive chat
cortiq run qwen38-27b-q4tp.cmf --prompt "..."            # one answer
cortiq run qwen38-27b-q4tp.cmf --prompt "..." --no-think # answer without the thinking block
cortiq run qwen38-27b-q4tp.cmf --prompt "..." --greedy   # deterministic output
```

### Sampling

Qwen's recommended settings:

| mode | temperature | top_p | top_k | presence_penalty | repetition_penalty |
|---|---|---|---|---|---|
| thinking | 1.0 | 0.95 | 20 | 0.0 | 1.0 |
| instruct (`--no-think`) | 0.7 | 0.80 | 20 | 1.5 | 1.0 |

Flags: `--temperature`, `--top-p`, `--top-k`, `--min-p`, `--presence-penalty`,
`--rep-penalty`, `--seed`, `--max-tokens`. Without flags the CLI uses
temperature 0.7, top-p 0.9, top-k 40, min-p 0.05, repetition penalty 1.1
and at most 256 new tokens.

### OpenAI-compatible server

```bash
cortiq serve qwen38-27b-q4tp.cmf --port 8080

curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen38-27b-q4tp", "messages": [{"role": "user", "content": "Hi!"}]}'
```

Endpoints: `/v1/chat/completions`, `/v1/completions`, `/v1/models`, plus a
web dashboard on the same port. `--host 127.0.0.1` keeps the server local.
`temperature: 0` means greedy decoding; `repetition_penalty` and
`presence_penalty` are accepted as request fields.

## Hardware

### NVIDIA via CUDA

```bash
cargo install cortiq-cli --features cuda
cortiq gpu      # says whether the card and the CUDA libraries are found
CMF_GPU=cuda cortiq run qwen38-27b-q4tp.cmf --prompt "..." --greedy --no-think
CMF_GPU=cuda CMF_DFLASH=1 cortiq serve qwen38-27b-q4tp.cmf --host 127.0.0.1 --port 8080
```

The build links nothing. At run time the backend opens the driver's
`libcuda` and two CUDA 12.x toolkit libraries, `libnvrtc` and `libcublas`;
without them `CMF_GPU=cuda` falls back to the CPU and says why. The first run
compiles the kernels and keeps them in `~/.cache/cortiq/cuda`. `q4tp` and its
drafter fit a 24 GB card.

RTX 3090 (24 GB), cortiq 0.8.69, `--greedy --no-think`, 512 tokens, tok/s:

| prompt | Vulkan | **CUDA** | **CUDA + DFlash** |
|---|---:|---:|---:|
| plain decode (`CMF_GRAPH_SPEC=0`) | 40.8 | **49.1** | — |
| eight chat prompts (code, prose, Danish, JSON), mean | 57.7 | **120.2** | **128.4** |
| an LRU cache module | 53.6 | **121.2** | **125.8** |
| 60 lines of a document copied | 74.8 | **198.9** | 183.5 |
| a rename in a 10 KB file | 71.6 | **220.3** | 197.6 |

- The text with drafts and DFlash is the text without them, through `cortiq run` and `cortiq serve`.
- `CMF_DFLASH=1` adds 7 % to chat on average and takes 8–10 % off copies and edits; the first token comes
  0.5 s later. Without it the model's MTP head drafts alone.
- The server keeps a chat's GDN state at its last turn, so the next turn reads only the new message:
  0.5 s instead of 4.1 s in a 4 000-token chat. `CMF_KV_REUSE=0` reads every turn from the start.
- Through `cortiq serve`, greedy requests with and without thinking: 86–173 tok/s.

### Apple silicon (Metal)

Runs on the GPU out of the box. A 24 GB Mac runs `q4tp` or `q4t`; `q8_2f`
does not fit. With `RUST_LOG=info` the first generation prints one line that
names the active Metal path.

### Choosing a GPU (Vulkan)

```bash
cortiq gpu                                                        # list adapters
CMF_GPU_ADAPTER=1 cortiq run qwen38-27b-q4tp.cmf --prompt "..."   # index or name substring
```

If the list is empty on Linux, install the loader libraries:

```bash
sudo apt install libvulkan1 libglvnd0 libegl1 libgl1 libglx0
XDG_RUNTIME_DIR=/tmp cortiq gpu     # headless machines
```

### Two GPUs

```bash
cortiq run qwen38-27b-q4tp.cmf --prompt "..." --gpus 2   # one stream, layers split across both cards
cortiq serve qwen38-27b-q4tp.cmf --gpus 2                # many requests: a replica per card when it fits
```

### Across the network

```bash
# machine B — runs the tail layers
cortiq worker qwen38-27b-q4tp.cmf --listen 0.0.0.0:9911 --token SECRET
# machine A — the coordinator, same .cmf file
cortiq run qwen38-27b-q4tp.cmf --prompt "..." --peer 192.168.1.42:9911 --net-token SECRET --net-dtype f16
```

`--peer-split N` sets the first layer the worker runs (default: half).
`--net-dtype f16` halves the traffic; `f32` is exact. `cortiq peers` lists
workers on the local network.

## Long context

The KV cache holds 32768 tokens by default. For longer generations raise it:

```bash
CMF_MAX_SEQ=65536 cortiq run qwen38-27b-q4tp.cmf --prompt "..." --max-tokens 50000
```

For very long contexts on memory-limited machines, the O(1) mode replaces
full attention with a bounded approximation (an exact window of recent
tokens plus landmarks), so memory stops growing with the context. It pays
past ~8k tokens; its output is close to full attention but not identical,
and it runs without speculative decoding.

```bash
CMF_O1_GPU=1   cortiq run qwen38-27b-q4tp.cmf --o1 all --prompt "..."   # Vulkan
CMF_O1_METAL=1 cortiq run qwen38-27b-q4tp.cmf --o1 all --prompt "..."   # Apple silicon
```

| flag | default | meaning |
|---|---|---|
| `--o1 all\|deepN\|i,j,k\|off` | file hint | which full-attention layers switch to O(1) |
| `--o1-m` | 32 | landmark budget (GPU kernels accept at most 32) |
| `--o1-window` | 128 | recent tokens attended exactly |
| `--o1-sink` | 4 | exact tokens kept from the sequence start |

## Examples

A Three.js aquarium generated in one pass from the same 7 KB spec with
Qwen's instruct settings — open in a browser:
[q4tp](https://huggingface.co/infosave/Qwen3.8-27B-cmf/blob/main/examples/aquarium-q4tp.html) ·
[q4t](https://huggingface.co/infosave/Qwen3.8-27B-cmf/blob/main/examples/aquarium-q4t.html) ·
[q8_2f](https://huggingface.co/infosave/Qwen3.8-27B-cmf/blob/main/examples/aquarium-q8_2f.html).

## Verify the download

```bash
sha256sum -c qwen38-27b-q4tp.cmf.sha256
sha256sum -c qwen38-27b-q4tp.dflash.cmf.sha256   # the drafter, if downloaded
cortiq info qwen38-27b-q4tp.cmf
```

---

## Документация на русском

[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (48 слоёв линейного
внимания GatedDeltaNet + 16 слоёв полного внимания, контекст 262k, режим
рассуждений) в виде одного файла [CMF](https://github.com/infosave2007/cmf).
Запускается движком `cortiq` на Rust без Python и ML-фреймворков: GPU NVIDIA
через CUDA или Vulkan/DX12, AMD и Intel через Vulkan/DX12, Apple silicon через
Metal, а также CPU. На RTX 3090 через CUDA чат декодируется со скоростью
120–128 токенов/с.

### Быстрый старт

```bash
cargo install cortiq-cli
hf download infosave/Qwen3.8-27B-cmf qwen38-27b-q4tp.cmf --local-dir .
cortiq run qwen38-27b-q4tp.cmf --prompt "Объясни квиксорт в трёх предложениях." --no-think
```

Для быстрой обработки промпта используйте cortiq 0.8.15 или новее; готовые
сборки для Linux, macOS и Windows — на [странице релизов](https://github.com/infosave2007/cmf/releases).
Движок сам выбирает GPU и настройки. `--no-think` даёт прямой ответ; без него
модель сначала рассуждает, поэтому увеличьте `--max-tokens` (по умолчанию 256).

На карте NVIDIA быстрее всего бэкенд CUDA (cortiq 0.8.69 или новее):

```bash
cargo install cortiq-cli --features cuda
hf download infosave/Qwen3.8-27B-cmf qwen38-27b-q4tp.cmf qwen38-27b-q4tp.dflash.cmf --local-dir .
CMF_GPU=cuda CMF_DFLASH=1 cortiq run qwen38-27b-q4tp.cmf --prompt "Объясни квиксорт в трёх предложениях." --greedy --no-think
```

Настройки — в разделе «NVIDIA через CUDA» ниже.

### Файлы

| файл | квантизация | размер | ppl wikitext-2 | назначение |
|---|---|---:|---:|---|
| `qwen38-27b-q4tp.cmf` | 4 бита, лестница масштабов | 14.3 ГБ | 8.79 | выбор по умолчанию: Mac и GPU на 16–24 ГБ; спекулятивное декодирование |
| `qwen38-27b-q4t.cmf` | 4 бита | 15.4 ГБ | 8.86 | альтернативная 4-битная раскладка, без спекулятивного декодирования |
| `qwen38-27b-q8_2f.cmf` | 8 бит | 27.4 ГБ | — | максимальная точность, от 32 ГБ памяти |
| `qwen38-27b-q4tp.dflash.cmf` | драфтер DFlash 2, 4 бита | 1.3 ГБ | — | рядом с `q4tp` на CUDA: быстрее чат с `CMF_DFLASH=1` |

4-битные файлы квантованы напрямую из bf16-чекпойнта. Перплексия измерена на
12 окнах wikitext-2 по 512 токенов. Драфтер —
[z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2),
сконвертированный под `q4tp`, с роутером, подобранным по ценам RTX 3090.

### Скорость

Скорость декодирования `qwen38-27b-q4tp.cmf`, один поток, токенов в секунду:

| железо | бэкенд | обычное декодирование | промпт с кодом | `cortiq bench --core` |
|---|---|---:|---:|---:|
| RTX 3090 (24 ГБ) | **CUDA** | **49.1** | **125.8** | **232** |
| RTX 5090 (32 ГБ) | Vulkan | 48.7 | 56.5 | 76 |
| RTX 3090 (24 ГБ) | Vulkan | 37.1 | 43.6 | 62.4 |
| RTX PRO 4000 Blackwell (24 ГБ) | Vulkan | 27.9 | 43.9 | 54.8 |
| Mac mini M4 (24 ГБ) | Metal | 6.8 | 17.3 | 20.0 |

Столбцы «промпт с кодом» и bench измерены со спекулятивным декодированием и
`--greedy`; на RTX 5090 промпт с кодом идёт с контекстом 2.3k токенов. Строка
CUDA — cortiq 0.8.69 с драфтером DFlash на модуле LRU-кэша, 512 токенов, а
bench — после промпта в 4 000 токенов. Данные для RTX 5090 получены на одном из прошлых релизов. Данные для RTX 3090 измерены
на cortiq 0.8.15 под Ubuntu 24.04: обычное декодирование — с `CMF_GRAPH_SPEC=0`,
ответ на промпт с кодом — 400 токенов, в bench принимается 91 % черновых токенов.

Обработка промпта на RTX 3090 (`cortiq bench --ctx N`):

| промпт | обработка промпта, токенов/с | время до первого токена | декодирование на этой глубине, токенов/с |
|---|---:|---:|---:|
| 1000 токенов | 199 | 5.0 с | 57.8 |
| 4000 токенов | 199 | 20.1 с | 33.9 |

Декодирование в этой таблице спекулятивное: после промпта в 1000 токенов
принимается 87 % черновых токенов, после промпта в 4000 токенов — 50 %. Через
CUDA та же карта читает промпт в четыре-пять раз быстрее:

| промпт | обработка промпта, токенов/с | время до первого токена |
|---|---:|---:|
| 4 000 токенов | 983 | 4.1 с |
| 16 000 токенов | 929 | 17.2 с |
| 32 000 токенов | 819 | 39.1 с |

**Спекулятивное декодирование** включено по умолчанию для
`qwen38-27b-q4tp.cmf`: собственная MTP-голова модели предлагает несколько
токенов, а GPU проверяет их одним пакетом. Больше всего оно ускоряет код и
структурированный вывод; на свободном тексте скорость близка к обычной, и
движок сам отключает спекуляцию там, где она не окупается. На Apple silicon
оно работает во всех режимах, включая сэмплирование по умолчанию; на Vulkan и
CUDA — при greedy без штрафов, поэтому там добавьте `--greedy`.
`CMF_GRAPH_SPEC=0` отключает его. На Vulkan int8-проверка может выбрать другой
токен при почти равных вероятностях; `CMF_VERIFY_I8=0` делает greedy-вывод
идентичным обычному декодированию. На CUDA текст с черновиками совпадает с
текстом без них.

### Использование

```bash
cortiq run qwen38-27b-q4tp.cmf                           # интерактивный чат
cortiq run qwen38-27b-q4tp.cmf --prompt "..."            # один ответ
cortiq run qwen38-27b-q4tp.cmf --prompt "..." --no-think # ответ без блока рассуждений
cortiq run qwen38-27b-q4tp.cmf --prompt "..." --greedy   # детерминированный вывод
```

Рекомендуемые настройки Qwen:

| режим | temperature | top_p | top_k | presence_penalty | repetition_penalty |
|---|---|---|---|---|---|
| с рассуждениями | 1.0 | 0.95 | 20 | 0.0 | 1.0 |
| instruct (`--no-think`) | 0.7 | 0.80 | 20 | 1.5 | 1.0 |

Флаги: `--temperature`, `--top-p`, `--top-k`, `--min-p`, `--presence-penalty`,
`--rep-penalty`, `--seed`, `--max-tokens`. Без флагов CLI использует
temperature 0.7, top-p 0.9, top-k 40, min-p 0.05, repetition penalty 1.1 и не
более 256 новых токенов.

**Сервер с OpenAI-совместимым API:**

```bash
cortiq serve qwen38-27b-q4tp.cmf --port 8080

curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen38-27b-q4tp", "messages": [{"role": "user", "content": "Привет!"}]}'
```

Эндпоинты `/v1/chat/completions`, `/v1/completions`, `/v1/models` и
веб-панель на том же порту. `--host 127.0.0.1` оставляет сервер локальным.
`temperature: 0` в запросе означает greedy-декодирование; поля
`repetition_penalty` и `presence_penalty` поддерживаются.

### Железо

**NVIDIA через CUDA.**

```bash
cargo install cortiq-cli --features cuda
cortiq gpu      # покажет, найдены ли карта и библиотеки CUDA
CMF_GPU=cuda cortiq run qwen38-27b-q4tp.cmf --prompt "..." --greedy --no-think
CMF_GPU=cuda CMF_DFLASH=1 cortiq serve qwen38-27b-q4tp.cmf --host 127.0.0.1 --port 8080
```

Сборка ничего не линкует. При запуске бэкенд открывает `libcuda` из драйвера
и две библиотеки CUDA 12.x — `libnvrtc` и `libcublas`; без них `CMF_GPU=cuda`
переходит на CPU и пишет почему. Первый запуск компилирует ядра и хранит их в
`~/.cache/cortiq/cuda`. `q4tp` вместе с драфтером помещается на карту 24 ГБ.

RTX 3090 (24 ГБ), cortiq 0.8.69, `--greedy --no-think`, 512 токенов, токенов/с:

| промпт | Vulkan | **CUDA** | **CUDA + DFlash** |
|---|---:|---:|---:|
| обычное декодирование (`CMF_GRAPH_SPEC=0`) | 40.8 | **49.1** | — |
| восемь чат-промптов (код, проза, датский, JSON), среднее | 57.7 | **120.2** | **128.4** |
| модуль LRU-кэша | 53.6 | **121.2** | **125.8** |
| копирование 60 строк документа | 74.8 | **198.9** | 183.5 |
| переименование в файле на 10 КБ | 71.6 | **220.3** | 197.6 |

- Текст с черновиками и с DFlash совпадает с текстом без них — и в `cortiq run`, и в `cortiq serve`.
- `CMF_DFLASH=1` в среднем прибавляет чату 7 % и отнимает 8–10 % у копирования и правок; первый токен
  приходит на 0.5 с позже. Без него черновики пишет только MTP-голова модели.
- Сервер хранит состояние GDN чата на последнем ходе, поэтому следующий ход читает только новое
  сообщение: 0.5 с вместо 4.1 с в чате на 4 000 токенов. `CMF_KV_REUSE=0` читает каждый ход с начала.
- Через `cortiq serve` greedy-запросы с рассуждениями и без: 86–173 токенов/с.

**Apple silicon (Metal).** Работает на GPU без настройки. На Mac с 24 ГБ
запускаются `q4tp` и `q4t`; `q8_2f` не помещается. С `RUST_LOG=info` первая
генерация печатает строку с активным путём Metal.

**Выбор GPU (Vulkan).** `cortiq gpu` показывает адаптеры; `CMF_GPU_ADAPTER=1`
(индекс или часть имени) закрепляет нужный. Если на Linux список пуст,
установите `libvulkan1 libglvnd0 libegl1 libgl1 libglx0`; на машинах без
дисплея задайте `XDG_RUNTIME_DIR=/tmp`.

**Две карты.** `cortiq run … --gpus 2` делит слои одного потока между двумя
картами; `cortiq serve … --gpus 2` для многих запросов держит по реплике на
каждой карте, если модель помещается.

**По сети.**

```bash
# машина B — хвостовые слои
cortiq worker qwen38-27b-q4tp.cmf --listen 0.0.0.0:9911 --token SECRET
# машина A — координатор, тот же файл .cmf
cortiq run qwen38-27b-q4tp.cmf --prompt "..." --peer 192.168.1.42:9911 --net-token SECRET --net-dtype f16
```

`--peer-split N` задаёт первый слой воркера (по умолчанию половина стека).
`f16` вдвое сокращает трафик, `f32` передаёт данные точно. `cortiq peers`
показывает воркеры в локальной сети.

### Длинный контекст

По умолчанию KV-кэш рассчитан на 32768 токенов. Для длинной генерации
увеличьте его: `CMF_MAX_SEQ=65536 cortiq run … --max-tokens 50000`.

Режим O(1) заменяет полное внимание ограниченным приближением (точное окно
последних токенов плюс опорные точки), и память перестаёт расти с контекстом.
Он полезен после ~8k токенов на машинах с ограниченной памятью. Вывод близок к
полному вниманию, но не совпадает с ним; спекулятивное декодирование в этом
режиме не используется.

```bash
CMF_O1_GPU=1   cortiq run qwen38-27b-q4tp.cmf --o1 all --prompt "..."   # Vulkan
CMF_O1_METAL=1 cortiq run qwen38-27b-q4tp.cmf --o1 all --prompt "..."   # Apple silicon
```

Параметры: `--o1 all|deepN|i,j,k|off` (какие слои полного внимания переходят
в O(1), по умолчанию — из файла), `--o1-m` (32, максимум для GPU),
`--o1-window` (128), `--o1-sink` (4).

### Примеры

Аквариум на Three.js, сгенерированный за один проход по одной спецификации
на 7 КБ с instruct-настройками Qwen — откройте в браузере:
[q4tp](https://huggingface.co/infosave/Qwen3.8-27B-cmf/blob/main/examples/aquarium-q4tp.html) ·
[q4t](https://huggingface.co/infosave/Qwen3.8-27B-cmf/blob/main/examples/aquarium-q4t.html) ·
[q8_2f](https://huggingface.co/infosave/Qwen3.8-27B-cmf/blob/main/examples/aquarium-q8_2f.html).

### Проверка загрузки

```bash
sha256sum -c qwen38-27b-q4tp.cmf.sha256
sha256sum -c qwen38-27b-q4tp.dflash.cmf.sha256   # драфтер, если скачан
cortiq info qwen38-27b-q4tp.cmf
```

---

## 中文文档

[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)（48 层 GatedDeltaNet
线性注意力 + 16 层全注意力，262k 上下文，支持思考模式）打包为单个
[CMF](https://github.com/infosave2007/cmf) 文件，由 `cortiq` 运行——一个不依赖
Python 和机器学习框架的 Rust 推理引擎：NVIDIA 显卡走 CUDA 或 Vulkan/DX12，AMD、
Intel 显卡走 Vulkan/DX12，Apple silicon 走 Metal，也可在 CPU 上运行。在 RTX 3090 上
通过 CUDA，聊天解码速度为 120–128 tok/s。

### 快速开始

```bash
cargo install cortiq-cli
hf download infosave/Qwen3.8-27B-cmf qwen38-27b-q4tp.cmf --local-dir .
cortiq run qwen38-27b-q4tp.cmf --prompt "用三句话解释快速排序。" --no-think
```

建议使用 cortiq 0.8.15 或更新版本，以获得快速的提示词处理；Linux、macOS、Windows
预编译二进制见[发布页面](https://github.com/infosave2007/cmf/releases)。引擎会自动选择 GPU 和
设置。`--no-think` 直接给出回答；不加该参数时模型会先思考，请调高 `--max-tokens`
（默认 256）。

在 NVIDIA 显卡上，最快的路径是 CUDA 后端（cortiq 0.8.69 或更新版本）：

```bash
cargo install cortiq-cli --features cuda
hf download infosave/Qwen3.8-27B-cmf qwen38-27b-q4tp.cmf qwen38-27b-q4tp.dflash.cmf --local-dir .
CMF_GPU=cuda CMF_DFLASH=1 cortiq run qwen38-27b-q4tp.cmf --prompt "用三句话解释快速排序。" --greedy --no-think
```

设置见下文「NVIDIA（CUDA）」一节。

### 文件

| 文件 | 量化 | 大小 | wikitext-2 困惑度 | 适用场景 |
|---|---|---:|---:|---|
| `qwen38-27b-q4tp.cmf` | 4 位，阶梯缩放 | 14.3 GB | 8.79 | 默认选择：Mac 和 16–24 GB 显卡；支持推测解码 |
| `qwen38-27b-q4t.cmf` | 4 位 | 15.4 GB | 8.86 | 另一种 4 位布局，不使用推测解码 |
| `qwen38-27b-q8_2f.cmf` | 8 位 | 27.4 GB | — | 最高精度，需要 32 GB 以上内存 |
| `qwen38-27b-q4tp.dflash.cmf` | DFlash 2 草稿模型，4 位 | 1.3 GB | — | 在 CUDA 上与 `q4tp` 搭配：`CMF_DFLASH=1` 让聊天更快 |

4 位文件直接由 bf16 检查点量化。困惑度基于 wikitext-2 的 12 个 512 token 窗口。草稿模型
是为 `q4tp` 转换的 [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2)，
其路由器按 RTX 3090 的代价拟合。

### 性能

`qwen38-27b-q4tp.cmf` 解码速度，单流，tok/s：

| 硬件 | 后端 | 普通解码 | 代码类提示词 | `cortiq bench --core` |
|---|---|---:|---:|---:|
| RTX 3090（24 GB） | **CUDA** | **49.1** | **125.8** | **232** |
| RTX 5090（32 GB） | Vulkan | 48.7 | 56.5 | 76 |
| RTX 3090（24 GB） | Vulkan | 37.1 | 43.6 | 62.4 |
| RTX PRO 4000 Blackwell（24 GB） | Vulkan | 27.9 | 43.9 | 54.8 |
| Mac mini M4（24 GB） | Metal | 6.8 | 17.3 | 20.0 |

「代码类提示词」和 bench 两列使用推测解码和 `--greedy`；RTX 5090 的代码类提示词
带 2.3k token 上下文。CUDA 一行为 cortiq 0.8.69 配合 DFlash 草稿模型，提示词是一个 LRU
缓存模块，512 个 token；bench 在 4 000 token 提示词之后测得。RTX 5090 数据测于较早的版本。RTX 3090 数据使用 cortiq 0.8.15
在 Ubuntu 24.04 上测得：普通解码设置 `CMF_GRAPH_SPEC=0`，代码类提示词生成 400 个
token，bench 中 91% 的草稿 token 被接受。

RTX 3090 上的提示词处理（`cortiq bench --ctx N`）：

| 提示词 | 提示词处理，tok/s | 首 token 延迟 | 该深度下的解码，tok/s |
|---|---:|---:|---:|
| 1000 token | 199 | 5.0 秒 | 57.8 |
| 4000 token | 199 | 20.1 秒 | 33.9 |

该表中的解码为推测解码：1000 token 提示词之后草稿 token 接受率为 87%，4000 token
提示词之后为 50%。通过 CUDA，同一块显卡读取提示词快四到五倍：

| 提示词 | 提示词处理，tok/s | 首 token 延迟 |
|---|---:|---:|
| 4 000 token | 983 | 4.1 秒 |
| 16 000 token | 929 | 17.2 秒 |
| 32 000 token | 819 | 39.1 秒 |

`qwen38-27b-q4tp.cmf` **默认开启推测解码**：模型自带的 MTP 头一次起草多个
token，由 GPU 批量验证。它在代码和结构化输出上收益最大；自由文本接近普通速度，
引擎会在不划算时自动关闭。在 Apple silicon 上它覆盖所有模式，包括默认采样；在
Vulkan 和 CUDA 上仅用于无惩罚的贪心解码，因此请加 `--greedy`。`CMF_GRAPH_SPEC=0`
关闭该功能。Vulkan 上的 int8 验证在两个 token 概率几乎相同时可能选择不同的 token；
`CMF_VERIFY_I8=0` 可使贪心输出与普通解码完全一致。在 CUDA 上，有草稿与无草稿的文本相同。

### 使用

```bash
cortiq run qwen38-27b-q4tp.cmf                           # 交互式对话
cortiq run qwen38-27b-q4tp.cmf --prompt "..."            # 单次回答
cortiq run qwen38-27b-q4tp.cmf --prompt "..." --no-think # 不输出思考块
cortiq run qwen38-27b-q4tp.cmf --prompt "..." --greedy   # 确定性输出
```

Qwen 推荐的采样参数：

| 模式 | temperature | top_p | top_k | presence_penalty | repetition_penalty |
|---|---|---|---|---|---|
| 思考模式 | 1.0 | 0.95 | 20 | 0.0 | 1.0 |
| 指令模式（`--no-think`） | 0.7 | 0.80 | 20 | 1.5 | 1.0 |

参数：`--temperature`、`--top-p`、`--top-k`、`--min-p`、`--presence-penalty`、
`--rep-penalty`、`--seed`、`--max-tokens`。不加参数时 CLI 使用 temperature 0.7、
top-p 0.9、top-k 40、min-p 0.05、repetition penalty 1.1，最多生成 256 个新 token。

**OpenAI 兼容 API 服务器：**

```bash
cortiq serve qwen38-27b-q4tp.cmf --port 8080

curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen38-27b-q4tp", "messages": [{"role": "user", "content": "你好！"}]}'
```

提供 `/v1/chat/completions`、`/v1/completions`、`/v1/models` 端点，同一端口还有
网页控制台。`--host 127.0.0.1` 仅限本机访问。请求中 `temperature: 0` 表示贪心
解码；支持 `repetition_penalty` 和 `presence_penalty` 字段。

### 硬件

**NVIDIA（CUDA）。**

```bash
cargo install cortiq-cli --features cuda
cortiq gpu      # 显示是否找到显卡和 CUDA 库
CMF_GPU=cuda cortiq run qwen38-27b-q4tp.cmf --prompt "..." --greedy --no-think
CMF_GPU=cuda CMF_DFLASH=1 cortiq serve qwen38-27b-q4tp.cmf --host 127.0.0.1 --port 8080
```

构建时不链接任何库。运行时后端打开驱动的 `libcuda` 以及两个 CUDA 12.x 工具包库
`libnvrtc` 和 `libcublas`；缺少它们时 `CMF_GPU=cuda` 会退回 CPU 并说明原因。首次运行
编译内核并缓存到 `~/.cache/cortiq/cuda`。`q4tp` 加上草稿模型可装入 24 GB 显卡。

RTX 3090（24 GB），cortiq 0.8.69，`--greedy --no-think`，512 个 token，tok/s：

| 提示词 | Vulkan | **CUDA** | **CUDA + DFlash** |
|---|---:|---:|---:|
| 普通解码（`CMF_GRAPH_SPEC=0`） | 40.8 | **49.1** | — |
| 八个聊天提示词（代码、散文、丹麦语、JSON）平均 | 57.7 | **120.2** | **128.4** |
| LRU 缓存模块 | 53.6 | **121.2** | **125.8** |
| 复制文档的 60 行 | 74.8 | **198.9** | 183.5 |
| 在 10 KB 文件中重命名 | 71.6 | **220.3** | 197.6 |

- 有草稿和 DFlash 时的文本与没有时相同，`cortiq run` 和 `cortiq serve` 都是如此。
- `CMF_DFLASH=1` 让聊天平均快 7%，复制和编辑慢 8–10%；首个 token 晚 0.5 秒。不开启时只由模型的
  MTP 头起草。
- 服务器在聊天的最后一轮保存 GDN 状态，下一轮只读取新消息：4 000 token 的聊天中为 0.5 秒而不是
  4.1 秒。`CMF_KV_REUSE=0` 让每一轮都从头读取。
- 通过 `cortiq serve`，带思考和不带思考的贪心请求：86–173 tok/s。

**Apple silicon（Metal）。** 开箱即用 GPU 加速。24 GB 内存的 Mac 可运行 `q4tp`
和 `q4t`；`q8_2f` 无法装入。设置 `RUST_LOG=info` 后，首次生成时会打印一行当前
使用的 Metal 路径。

**选择显卡（Vulkan）。** `cortiq gpu` 列出所有适配器；`CMF_GPU_ADAPTER=1`（索引或
名称片段）指定使用哪一块。若 Linux 上列表为空，请安装
`libvulkan1 libglvnd0 libegl1 libgl1 libglx0`；无显示器的机器需设置
`XDG_RUNTIME_DIR=/tmp`。

**双显卡。** `cortiq run … --gpus 2` 把单个流的层拆分到两块卡上；
`cortiq serve … --gpus 2` 面向大量请求，模型装得下时每块卡各运行一个副本。

**跨机网络。**

```bash
# 机器 B —— 运行尾部层
cortiq worker qwen38-27b-q4tp.cmf --listen 0.0.0.0:9911 --token SECRET
# 机器 A —— 协调端，使用同一个 .cmf 文件
cortiq run qwen38-27b-q4tp.cmf --prompt "..." --peer 192.168.1.42:9911 --net-token SECRET --net-dtype f16
```

`--peer-split N` 指定 worker 的起始层（默认一半）；`f16` 使流量减半，`f32` 为精确
传输；`cortiq peers` 列出局域网中的 worker。

### 长上下文

KV 缓存默认容纳 32768 个 token；生成长文本时请调高：
`CMF_MAX_SEQ=65536 cortiq run … --max-tokens 50000`。

O(1) 模式用有界近似（最近 token 的精确窗口加地标点）替代全注意力，内存不再随上下文
增长。它适合在内存有限的机器上处理约 8k token 以上的上下文；输出接近但不完全等同于
全注意力，且该模式下不使用推测解码。

```bash
CMF_O1_GPU=1   cortiq run qwen38-27b-q4tp.cmf --o1 all --prompt "..."   # Vulkan
CMF_O1_METAL=1 cortiq run qwen38-27b-q4tp.cmf --o1 all --prompt "..."   # Apple silicon
```

参数：`--o1 all|deepN|i,j,k|off`（哪些全注意力层切换到 O(1)，默认取自文件）、
`--o1-m`（32，GPU 内核上限）、`--o1-window`（128）、`--o1-sink`（4）。

### 示例

使用 Qwen 指令模式参数、根据同一份 7 KB 规格一次生成的 Three.js 水族馆（在浏览器
中打开）：
[q4tp](https://huggingface.co/infosave/Qwen3.8-27B-cmf/blob/main/examples/aquarium-q4tp.html) ·
[q4t](https://huggingface.co/infosave/Qwen3.8-27B-cmf/blob/main/examples/aquarium-q4t.html) ·
[q8_2f](https://huggingface.co/infosave/Qwen3.8-27B-cmf/blob/main/examples/aquarium-q8_2f.html)。

### 校验下载

```bash
sha256sum -c qwen38-27b-q4tp.cmf.sha256
sha256sum -c qwen38-27b-q4tp.dflash.cmf.sha256   # 草稿模型（如已下载）
cortiq info qwen38-27b-q4tp.cmf
```
