---
license: apache-2.0
library_name: cortiq
base_model: maxwell-model/Maxwell-V1-Flash
base_model_relation: quantized
pipeline_tag: text-generation
tags:
  - cmf
  - cortiq
  - quantized
  - moe
  - q4tp
  - reasoning
  - tool-calling
  - long-context
  - cuda
  - vulkan
language:
  - en
  - zh
  - ru
  - th
  - vi
  - id
---

# Maxwell-V1-Flash — CMF

[Maxwell-V1-Flash](https://huggingface.co/maxwell-model/Maxwell-V1-Flash), the Maxwell team's 35B mixture-of-experts model for reasoning and agentic work (3B
parameters active a token, a 256K-token context, Apache-2.0), packaged as single
[CMF](https://github.com/infosave2007/cmf) files. They run with `cortiq`, a Rust inference engine with no Python and
no ML framework: NVIDIA GPUs through CUDA or Vulkan, AMD and Intel GPUs through Vulkan/DX12, and the CPU. Measured:

- **The whole 35B model on one 24 GB card.** NVIDIA L4, CUDA, the 4-bit file: 66.3 tok/s with every weight on
  the card; a 2 048-token prompt is read in 2.01 s and a 30 000-token one in 68.2 s, after which it still
  writes 33.6 tok/s in 20.3 GB of the card.
- **The 8-bit file is 37.82 GB and runs on the same 24 GB card** at 38.0 tok/s after a 2 048-token prompt
  (21.8 from a cold start): its experts stay in host memory and pass through the card as the router picks them.
- **Vulkan on the same card:** 38.8 tok/s for the 4-bit file, the block-sparse attention on the card as well: a
  16 384-token prompt is read in 109.7 s, after which it writes 24.6 tok/s.
- **Accuracy:** WikiText-2 perplexity 33.07 (8-bit) and 31.78 (4-bit) against 32.80 for Transformers bf16; over
  windows of 6 144 tokens, where the model's block-sparse attention picks its own blocks, 20.68 and 20.61 against
  20.75. Greedy answers of the 8-bit file were checked against Transformers bf16 token by token on four prompts of
  40 tokens: three are identical, the fourth differs in one token.

[English](#quick-start) · [Русский](#документация-на-русском) · [中文](#中文文档)

## Quick start

```bash
cargo install --git https://github.com/infosave2007/cmf --branch maxwell cortiq-cli --features cuda
hf download infosave/Maxwell-V1-Flash-cmf Maxwell-V1-Flash-q4tp.cmf --local-dir .
CMF_GPU=cuda cortiq run Maxwell-V1-Flash-q4tp.cmf --prompt "Why is the sky blue? Answer in three sentences."
```

The model's support is in cortiq after 0.9.10 ([PR #182](https://github.com/infosave2007/cmf/pull/182)): until the
next release, build from that branch as above (once the PR is merged, drop `--branch maxwell`). Leave out
`--features cuda` and `CMF_GPU=cuda` for Vulkan or the CPU.

## Files

| file | quantization | size | WikiText-2 ppl | Rust ppl | Python ppl | WikiText-2, 6 144-token windows |
|---|---|---:|---:|---:|---:|---:|
| `Maxwell-V1-Flash-q4tp.cmf` | 4-bit: routed experts, attention, shared experts, dense FFNs; head 8-bit | 20.50 GB | 31.78 | 7.21 | 7.51 | 20.61 |
| `Maxwell-V1-Flash-q8_2f.cmf` | 8-bit | 37.82 GB | 33.07 | 7.05 | 7.16 | 20.68 |
| reference: Transformers bf16 | — | 73.64 GB | 32.80 | 7.05 | 7.03 | 20.75 |

Perplexity over 12 windows of 512 tokens on WikiText-2 test, 1.8 MB of Rust and 477 KB of Python, and over 2
windows of 6 144 tokens of WikiText-2, the same windows for every row. In the 4-bit file every matrix but the
head is GPTQ-calibrated on prose, Rust and Python that the evaluation texts do not hold: the routed experts on
65 536 tokens, the attention, the shared experts and the dense FFNs on 3 072; its head is 8-bit. In both files the embedding is
f16, the routers, the hyper-connections and the attention's block indexer are f32. Both files carry the
checkpoint's multi-token-prediction layer: with `CMF_MAXWELL_SPEC=1` the CUDA decode drafts the next token with
it (the same text; on the L4 three quarters of the drafts are taken and decode goes from 65.5 to 67.2 tok/s, so
it is off by default).

What each file needs:

| file | card | host memory | where the experts are |
|---|---|---:|---|
| `Maxwell-V1-Flash-q4tp.cmf` | 24 GB | 24 GB | on the card, with about 18 000 positions beside them; a longer context cap (`CMF_MAX_SEQ`) streams them from page-locked host memory |
| `Maxwell-V1-Flash-q8_2f.cmf` | 24 GB | 42 GB | in page-locked host memory, 18 GB of them at a time on the card |

## Performance

One stream, `cortiq bench --core`, cortiq built from PR #182: the time to the first token of a prompt
of that length and the decode speed after it (64 tokens). Card memory is the peak `nvidia-smi` shows during the
run. Read each backend's section on its own.

### CUDA · NVIDIA L4

Stand: NVIDIA L4 (24 GB), Google Colab (12 vCPU, 52 GB), the CUDA backend (`CMF_GPU=cuda`). Without a prompt the
4-bit file decodes at 66.3 tok/s, the 8-bit file at 21.8. By default the card keeps every weight that fits and
the positions that fit beside them (about 18 000 for the 4-bit file); `CMF_MAX_SEQ` names a longer context, and the
routed experts then stream from host memory (55.0 tok/s with `CMF_MAX_SEQ=32768`). Each row below ran with the
cap at its prompt's length.

![The two files on CUDA](https://huggingface.co/infosave/Maxwell-V1-Flash-cmf/resolve/main/assets/files_cuda_en.png)

| prompt, tokens | first token, s (`q4tp`) | first token, s (`q8_2f`) | decode, tok/s (`q4tp`) | decode, tok/s (`q8_2f`) | peak card memory, GB (`q4tp`) | peak card memory, GB (`q8_2f`) |
|---:|---:|---:|---:|---:|---:|---:|
| 2 048 | 2.01 | 4.10 | 61.4 | 38.0 | 19.7 | 21.6 |
| 8 192 | 10.9 | 17.7 | 47.7 | 31.4 | 20.6 | 20.2 |
| 16 384 | 27.9 | — | 42.0 | — | 21.6 | — |
| 30 000 | 68.2 | 94.9 | 33.6 | 26.1 | 20.3 | 20.4 |

### Vulkan · NVIDIA L4

Stand: the same NVIDIA L4 (24 GB), over Vulkan (the default backend), the 4-bit file. Without a prompt:
38.8 tok/s.

![The 4-bit file on Vulkan](https://huggingface.co/infosave/Maxwell-V1-Flash-cmf/resolve/main/assets/files_vulkan_en.png)

| prompt, tokens | first token, s | decode, tok/s | peak card memory, GB |
|---:|---:|---:|---:|
| 512 | 0.91 | 38.5 | 19.6 |
| 2 048 | 4.93 | 35.4 | 19.8 |
| 4 096 | 13.5 | 28.9 | 20.2 |
| 8 192 | 39.8 | 27.6 | 21.0 |
| 16 384 | 109.7 | 24.6 | 21.4 |

Over Vulkan the block-sparse attention runs on the card too (the indexer, the choice of blocks and the attention over
them): WikiText-2 over windows of 6 144 tokens gives 20.66, against 20.61 on CUDA. The weights stay on the card whole
(17.5 GB of the 4-bit file) and the rest of the card holds the context: about 17 000 positions on a 24 GB card;
past that the sequence continues on the CPU, with a warning. The 8-bit file needs a larger card over Vulkan.

### CPU

| stand | file | decode, tok/s |
|---|---|---:|
| Colab L4 host (Xeon 2.2 GHz, 6 cores / 12 threads, 52 GB) | `q4tp` | 4.1 |
| MacBook Air M4 (10 cores, 24 GB) | `q4tp` | 7.9 |

On Apple Silicon the model runs on the CPU (its stack has no Metal graph yet). The file is memory-mapped: that
run peaked at 12 GB of resident memory.

## Usage

```bash
cortiq run Maxwell-V1-Flash-q4tp.cmf                          # interactive chat
cortiq run Maxwell-V1-Flash-q4tp.cmf --prompt "..."           # one answer, with its reasoning
cortiq run Maxwell-V1-Flash-q4tp.cmf --prompt "..." --no-think            # the answer alone
cortiq run Maxwell-V1-Flash-q4tp.cmf --prompt-file task.txt --max-tokens 2048
```

The embedded template is the model's own; it thinks before it answers unless `--no-think` is given. The files carry
the release's sampling (temperature 0.6, top-p 0.95, top-k 20), which `run` and `serve` use by default; `--greedy`
gives deterministic output.

### OpenAI-compatible server

```bash
CMF_GPU=cuda cortiq serve Maxwell-V1-Flash-q4tp.cmf --port 8080 --host 127.0.0.1

curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "maxwell", "max_tokens": 1024,
       "messages": [{"role": "user", "content": "Plan a three-step migration from MySQL to Postgres."}]}'
```

Endpoints: `/v1/chat/completions`, `/v1/completions`, `/v1/models`, and a web dashboard on the same port. The
reasoning comes back as `message.reasoning_content`; `"enable_thinking": false` turns it off. Tools go in `tools` in
the OpenAI format; the model answers `<tool_call><function=name>…</function></tool_call>`, which the server returns
as `message.tool_calls`. `--host 127.0.0.1` keeps the server local; no API key is checked.

## Long context

The model's attention is block-sparse: a query attends to the first 1 024 positions, to its own block of 128 and the
seven before it, and to 16 more blocks that each layer's indexer picks for each of its four key groups by the best
token score of the block. Below 4 096 positions that is every position. On CUDA and over Vulkan the indexer and the
block choice run on the card, as in the reference (the 6 144-token perplexity above). Every position keeps its keys, values and
index keys on the card, 101 KB a token: a 30 000-token prompt takes the 4-bit file to 20.3 GB. `CMF_MAX_SEQ`
sets the context cap (262 144 at most; without it, the positions that fit beside the weights, 32 768 at most);
the card holds that many positions' room, and experts that no longer fit beside it stream from host memory.

## Verify the download

```bash
cortiq verify Maxwell-V1-Flash-q4tp.cmf
cortiq info Maxwell-V1-Flash-q4tp.cmf
```

| file | sha256 |
|---|---|
| `Maxwell-V1-Flash-q4tp.cmf` | `350017b5bc2942f03fabb5611f3d9f7729ff9631607716e8961a4c0ae3a5a974` |
| `Maxwell-V1-Flash-q8_2f.cmf` | `8b96d4e0b65ae258eed226032c5af646adabd25d62fc5a912a7ad4f291972221` |

Weights: [Apache-2.0](https://huggingface.co/maxwell-model/Maxwell-V1-Flash/blob/main/LICENSE), from [maxwell-model/Maxwell-V1-Flash](https://huggingface.co/maxwell-model/Maxwell-V1-Flash). The model card's
notes on intended use and limitations apply to these files as well.

---

## Документация на русском

[Maxwell-V1-Flash](https://huggingface.co/maxwell-model/Maxwell-V1-Flash) — модель команды Maxwell со смесью экспертов на 35 млрд параметров для рассуждений и
агентной работы (3 млрд параметров активны на токен, контекст 256K токенов, Apache-2.0). Здесь она упакована в
отдельные файлы [CMF](https://github.com/infosave2007/cmf) для `cortiq` — движка на Rust без Python и
ML-фреймворков: GPU NVIDIA через CUDA или Vulkan, AMD и Intel через Vulkan/DX12, а также CPU. Измерено:

- **Вся модель на 35 млрд параметров на одной карте 24 ГБ.** NVIDIA L4, CUDA, 4-битный файл: 66,3 ток/с, все
  веса лежат на карте; промпт из 2 048 токенов читается за 2,01 с, из 30 000 — за 68,2 с, после чего
  генерация идёт на 33,6 ток/с в 20,3 ГБ карты.
- **8-битный файл занимает 37,82 ГБ и работает на той же карте 24 ГБ** на 38,0 ток/с после промпта из 2 048
  токенов (21,8 с холодного старта): эксперты лежат в памяти хоста и проходят через карту по выбору роутера.
- **Vulkan на той же карте:** 38,8 ток/с у 4-битного файла, блочно-разреженное внимание тоже на карте: промпт
  из 16 384 токенов читается за 109,7 с, после чего генерация идёт на 24,6 ток/с.
- **Точность:** перплексия WikiText-2 33,07 (8 бит) и 31,78 (4 бита) против 32,80 у Transformers bf16; на окнах
  по 6 144 токена, где блочно-разреженное внимание модели само выбирает блоки, — 20,68 и 20,61 против 20,75.
  Жадные ответы 8-битного файла сверены с Transformers bf16 токен за токеном на четырёх
  промптах по 40 токенов: три совпали полностью, четвёртый отличается одним токеном.

### Быстрый старт

```bash
cargo install --git https://github.com/infosave2007/cmf --branch maxwell cortiq-cli --features cuda
hf download infosave/Maxwell-V1-Flash-cmf Maxwell-V1-Flash-q4tp.cmf --local-dir .
CMF_GPU=cuda cortiq run Maxwell-V1-Flash-q4tp.cmf --prompt "Почему небо голубое? Ответь тремя предложениями."
```

Поддержка модели входит в cortiq после 0.9.10 ([PR #182](https://github.com/infosave2007/cmf/pull/182)): до
следующего релиза соберите движок из этой ветки, как выше (после слияния PR уберите `--branch maxwell`). Для Vulkan
или CPU уберите `--features cuda` и `CMF_GPU=cuda`.

### Файлы

| файл | квантование | размер | перплексия WikiText-2 | Rust | Python | WikiText-2, окна по 6 144 токена |
|---|---|---:|---:|---:|---:|---:|
| `Maxwell-V1-Flash-q4tp.cmf` | 4 бита: маршрутизируемые эксперты, внимание, общие эксперты, плотные FFN; голова 8 бит | 20,50 ГБ | 31,78 | 7,21 | 7,51 | 20,61 |
| `Maxwell-V1-Flash-q8_2f.cmf` | 8 бит | 37,82 ГБ | 33,07 | 7,05 | 7,16 | 20,68 |
| эталон: Transformers bf16 | — | 73,64 ГБ | 32,80 | 7,05 | 7,03 | 20,75 |

Перплексия на 12 окнах по 512 токенов: тест WikiText-2, 1,8 МБ кода на Rust и 477 КБ на Python, а также на 2 окнах
по 6 144 токена WikiText-2; окна одни и те же для всех строк. В 4-битном файле все матрицы, кроме
головы, откалиброваны методом GPTQ на прозе, Rust и Python, которых нет в оценочных текстах: маршрутизируемые
эксперты — на 65 536 токенах, внимание, общие эксперты и плотные FFN — на 3 072; голова 8-битная. В обоих файлах эмбеддинг хранится в f16,
роутеры, гиперсвязи и блочный индексатор внимания — в f32. В обоих файлах есть слой предсказания
нескольких токенов (MTP) из чекпойнта: с `CMF_MAXWELL_SPEC=1` генерация на CUDA строит по нему черновик следующего
токена (текст тот же; на L4 принимается три четверти черновиков, скорость растёт с 65,5 до 67,2 ток/с, поэтому
по умолчанию он выключен).

Что нужно каждому файлу:

| файл | карта | память хоста | где лежат эксперты |
|---|---|---:|---|
| `Maxwell-V1-Flash-q4tp.cmf` | 24 ГБ | 24 ГБ | на карте, рядом помещается около 18 000 позиций; при большем пределе контекста (`CMF_MAX_SEQ`) подаются из закреплённой памяти хоста |
| `Maxwell-V1-Flash-q8_2f.cmf` | 24 ГБ | 42 ГБ | в закреплённой памяти хоста, на карте одновременно 18 ГБ из них |

### Скорость

Один поток, `cortiq bench --core`, cortiq, собранный из PR #182: время до первого токена при промпте такой
длины и генерация после него (64 токена). Память карты — пик по `nvidia-smi` за прогон. Каждый раздел читается сам
по себе.

**CUDA · NVIDIA L4**

Стенд: NVIDIA L4 (24 ГБ), Google Colab (12 vCPU, 52 ГБ), бэкенд CUDA (`CMF_GPU=cuda`). Без промпта 4-битный файл
пишет 66,3 ток/с, 8-битный — 21,8. По умолчанию карта держит все веса, которые на неё помещаются, и столько
позиций, сколько помещается рядом (около 18 000 у 4-битного файла); `CMF_MAX_SEQ` задаёт более длинный контекст, и
тогда маршрутизируемые эксперты подаются из памяти хоста (55,0 ток/с при `CMF_MAX_SEQ=32768`). Каждая строка
ниже измерена с пределом, равным длине её промпта.

![Два файла на CUDA](https://huggingface.co/infosave/Maxwell-V1-Flash-cmf/resolve/main/assets/files_cuda_ru.png)

| промпт, токенов | первый токен, с (`q4tp`) | первый токен, с (`q8_2f`) | генерация, ток/с (`q4tp`) | генерация, ток/с (`q8_2f`) | пик памяти карты, ГБ (`q4tp`) | пик памяти карты, ГБ (`q8_2f`) |
|---:|---:|---:|---:|---:|---:|---:|
| 2 048 | 2,01 | 4,10 | 61,4 | 38,0 | 19,7 | 21,6 |
| 8 192 | 10,9 | 17,7 | 47,7 | 31,4 | 20,6 | 20,2 |
| 16 384 | 27,9 | — | 42,0 | — | 21,6 | — |
| 30 000 | 68,2 | 94,9 | 33,6 | 26,1 | 20,3 | 20,4 |

**Vulkan · NVIDIA L4**

Стенд: та же NVIDIA L4 (24 ГБ), через Vulkan (бэкенд по умолчанию), 4-битный файл. Без промпта: 38,8 ток/с.

![4-битный файл на Vulkan](https://huggingface.co/infosave/Maxwell-V1-Flash-cmf/resolve/main/assets/files_vulkan_ru.png)

| промпт, токенов | первый токен, с | генерация, ток/с | пик памяти карты, ГБ |
|---:|---:|---:|---:|
| 512 | 0,91 | 38,5 | 19,6 |
| 2 048 | 4,93 | 35,4 | 19,8 |
| 4 096 | 13,5 | 28,9 | 20,2 |
| 8 192 | 39,8 | 27,6 | 21,0 |
| 16 384 | 109,7 | 24,6 | 21,4 |

Через Vulkan блочно-разреженное внимание тоже работает на карте (индексатор, выбор блоков и внимание по ним):
перплексия WikiText-2 на окнах по 6 144 токена — 20,66 против 20,61 на CUDA. Веса лежат на карте целиком (17,5 ГБ
у 4-битного файла), а остаток карты занимает контекст: около 17 000 позиций на карте 24 ГБ; дальше
последовательность продолжается на CPU с предупреждением. 8-битному файлу через Vulkan нужна карта большего объёма.

**CPU**

| стенд | файл | генерация, ток/с |
|---|---|---:|
| хост Colab L4 (Xeon 2,2 ГГц, 6 ядер / 12 потоков, 52 ГБ) | `q4tp` | 4,1 |
| MacBook Air M4 (10 ядер, 24 ГБ) | `q4tp` | 7,9 |

На Apple Silicon модель работает на CPU (графа Metal для её стека пока нет). Файл отображается в память: этот
запуск занял до 12 ГБ резидентной памяти.

### Использование

```bash
cortiq run Maxwell-V1-Flash-q4tp.cmf                          # интерактивный чат
cortiq run Maxwell-V1-Flash-q4tp.cmf --prompt "..."           # один ответ с рассуждением
cortiq run Maxwell-V1-Flash-q4tp.cmf --prompt "..." --no-think            # только ответ
cortiq run Maxwell-V1-Flash-q4tp.cmf --prompt-file task.txt --max-tokens 2048
```

Встроенный шаблон — собственный шаблон модели; перед ответом она рассуждает, если не задан `--no-think`. В файлы
записаны настройки сэмплирования авторов (temperature 0,6, top-p 0,95, top-k 20) — `run` и `serve` берут их по
умолчанию; `--greedy` даёт детерминированный вывод.

Сервер, совместимый с OpenAI:

```bash
CMF_GPU=cuda cortiq serve Maxwell-V1-Flash-q4tp.cmf --port 8080 --host 127.0.0.1

curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "maxwell", "max_tokens": 1024,
       "messages": [{"role": "user", "content": "Составь план миграции с MySQL на Postgres из трёх шагов."}]}'
```

Эндпоинты: `/v1/chat/completions`, `/v1/completions`, `/v1/models` и веб-панель на том же порту. Рассуждение
возвращается в `message.reasoning_content`; `"enable_thinking": false` его отключает. Инструменты передаются в
`tools` в формате OpenAI; модель отвечает `<tool_call><function=имя>…</function></tool_call>`, и сервер возвращает
это как `message.tool_calls`. `--host 127.0.0.1` оставляет сервер локальным; ключ API не проверяется.

### Длинный контекст

Внимание модели блочно-разреженное: запрос видит первые 1 024 позиции, свой блок из 128 позиций и семь блоков
перед ним, а также ещё 16 блоков, которые индексатор каждого слоя выбирает для каждой из четырёх групп ключей по
лучшей оценке токена в блоке. До 4 096 позиций это все позиции. На CUDA и через Vulkan индексатор и выбор блоков
работают на карте, как в эталоне (перплексия на окнах по 6 144 токена выше). Каждая позиция хранит на карте ключи, значения и
ключи индексатора, 101 КБ на токен: промпт из 30 000 токенов доводит 4-битный файл до 20,3 ГБ. Предел
контекста задаёт `CMF_MAX_SEQ` (максимум 262 144; без него — столько позиций, сколько помещается рядом с весами,
но не больше 32 768): карта держит место под столько позиций, а эксперты, которым рядом не хватило места,
подаются из памяти хоста.

### Проверка загрузки

```bash
cortiq verify Maxwell-V1-Flash-q4tp.cmf
cortiq info Maxwell-V1-Flash-q4tp.cmf
```

| файл | sha256 |
|---|---|
| `Maxwell-V1-Flash-q4tp.cmf` | `350017b5bc2942f03fabb5611f3d9f7729ff9631607716e8961a4c0ae3a5a974` |
| `Maxwell-V1-Flash-q8_2f.cmf` | `8b96d4e0b65ae258eed226032c5af646adabd25d62fc5a912a7ad4f291972221` |

Веса: [Apache-2.0](https://huggingface.co/maxwell-model/Maxwell-V1-Flash/blob/main/LICENSE), источник —
[maxwell-model/Maxwell-V1-Flash](https://huggingface.co/maxwell-model/Maxwell-V1-Flash). Замечания карточки модели о назначении и ограничениях относятся и к этим
файлам.

---

## 中文文档

[Maxwell-V1-Flash](https://huggingface.co/maxwell-model/Maxwell-V1-Flash) 是 Maxwell 团队的 350 亿参数混合专家模型，面向推理与智能体任务（每个 token 激活
30 亿参数，上下文 256K token，Apache-2.0）。这里打包为独立的 [CMF](https://github.com/infosave2007/cmf)
文件，由 `cortiq` 运行——一个不依赖 Python 和机器学习框架的 Rust 推理引擎：NVIDIA 显卡走 CUDA 或 Vulkan，AMD、
Intel 显卡走 Vulkan/DX12，也可在 CPU 上运行。实测：

- **整个 350 亿参数模型放进一块 24 GB 显卡。** NVIDIA L4，CUDA，4 位文件：66.3 tok/s，全部权重都在显卡上；
  2 048 token 的提示词读取 2.01 秒，30 000 token 读取 68.2 秒，之后仍以 33.6 tok/s 生成，占用
  20.3 GB 显存。
- **8 位文件 37.82 GB，在同一块 24 GB 显卡上运行，** 2 048 token 提示词之后 38.0 tok/s（冷启动时
  21.8）：专家权重留在主机内存中，由路由器选中后经过显卡。
- **同一块显卡，Vulkan：** 4 位文件 38.8 tok/s，块稀疏注意力同样在显卡上运行：16 384 token 的提示词
  读取 109.7 秒，之后以 24.6 tok/s 生成。
- **精度：** WikiText-2 困惑度 33.07（8 位）和 31.78（4 位），Transformers bf16 为 32.80；在 6 144 token
  的窗口上（此时模型的块稀疏注意力自行选择块）为 20.68 和 20.61，参考值 20.75。8 位文件的贪心回答在四个提示词上（各 40 个 token）与 Transformers bf16
  逐 token 比对：三个完全一致，第四个有一个 token 不同。

### 快速开始

```bash
cargo install --git https://github.com/infosave2007/cmf --branch maxwell cortiq-cli --features cuda
hf download infosave/Maxwell-V1-Flash-cmf Maxwell-V1-Flash-q4tp.cmf --local-dir .
CMF_GPU=cuda cortiq run Maxwell-V1-Flash-q4tp.cmf --prompt "天空为什么是蓝色的？用三句话回答。"
```

对该模型的支持在 cortiq 0.9.10 之后（[PR #182](https://github.com/infosave2007/cmf/pull/182)）：下一个版本发布
之前，请按上面的命令从该分支构建（PR 合并后去掉 `--branch maxwell`）。使用 Vulkan 或 CPU 时去掉
`--features cuda` 和 `CMF_GPU=cuda`。

### 文件

| 文件 | 量化 | 大小 | WikiText-2 困惑度 | Rust | Python | WikiText-2，6 144 token 窗口 |
|---|---|---:|---:|---:|---:|---:|
| `Maxwell-V1-Flash-q4tp.cmf` | 4 位：路由专家、注意力、共享专家、稠密 FFN；输出头 8 位 | 20.50 GB | 31.78 | 7.21 | 7.51 | 20.61 |
| `Maxwell-V1-Flash-q8_2f.cmf` | 8 位 | 37.82 GB | 33.07 | 7.05 | 7.16 | 20.68 |
| 参考：Transformers bf16 | — | 73.64 GB | 32.80 | 7.05 | 7.03 | 20.75 |

困惑度在 12 个 512 token 的窗口上计算：WikiText-2 测试集、1.8 MB Rust 代码和 477 KB Python 代码；另有
WikiText-2 的 2 个 6 144 token 窗口；每一行使用相同的窗口。4 位文件中，除输出头外的所有矩阵都用 GPTQ 校准，
校准文本为散文、Rust 和 Python，且不在评测文本之内：路由专家用 65 536 个 token，注意力、共享专家和稠密 FFN 用
3 072 个；输出头为 8 位。两个文件中词嵌入为 f16，路由器、超连接和
注意力的块索引器为 f32。两个文件都带有检查点的多 token 预测层（MTP）：
设置 `CMF_MAXWELL_SPEC=1` 后，CUDA 解码用它为下一个 token 打草稿（文本不变；在 L4 上约四分之三的草稿被采纳，
速度从 65.5 提高到 67.2 tok/s，因此默认关闭）。

每个文件的要求：

| 文件 | 显卡 | 主机内存 | 专家权重的位置 |
|---|---|---:|---|
| `Maxwell-V1-Flash-q4tp.cmf` | 24 GB | 24 GB | 在显卡上，旁边可容纳约 18 000 个位置；更大的上下文上限（`CMF_MAX_SEQ`）时从锁页的主机内存流式调入 |
| `Maxwell-V1-Flash-q8_2f.cmf` | 24 GB | 42 GB | 在锁页的主机内存中，显卡上同时保留其中 18 GB |

### 性能

单路推理，`cortiq bench --core`，由 PR #182 构建的 cortiq：给定长度提示词的首个 token 时间，以及之后的生成
速度（64 个 token）。显存为运行期间 `nvidia-smi` 的峰值。各后端的小节请单独阅读。

**CUDA · NVIDIA L4**

测试平台：NVIDIA L4（24 GB），Google Colab（12 vCPU，52 GB），CUDA 后端（`CMF_GPU=cuda`）。无提示词时 4 位文件
66.3 tok/s，8 位文件 21.8。默认情况下，显卡保留能放下的全部权重，以及旁边能容纳的位置数（4 位文件约
18 000 个）；`CMF_MAX_SEQ` 指定更长的上下文，此时路由专家从主机内存流式调入（`CMF_MAX_SEQ=32768` 时
55.0 tok/s）。下表每一行的上限等于该行提示词的长度。

![CUDA 上的两个文件](https://huggingface.co/infosave/Maxwell-V1-Flash-cmf/resolve/main/assets/files_cuda_zh.png)

| 提示词，token 数 | 首个 token，秒（`q4tp`） | 首个 token，秒（`q8_2f`） | 生成，tok/s（`q4tp`） | 生成，tok/s（`q8_2f`） | 显存峰值，GB（`q4tp`） | 显存峰值，GB（`q8_2f`） |
|---:|---:|---:|---:|---:|---:|---:|
| 2 048 | 2.01 | 4.10 | 61.4 | 38.0 | 19.7 | 21.6 |
| 8 192 | 10.9 | 17.7 | 47.7 | 31.4 | 20.6 | 20.2 |
| 16 384 | 27.9 | — | 42.0 | — | 21.6 | — |
| 30 000 | 68.2 | 94.9 | 33.6 | 26.1 | 20.3 | 20.4 |

**Vulkan · NVIDIA L4**

测试平台：同一块 NVIDIA L4（24 GB），通过 Vulkan（默认后端），4 位文件。无提示词时 38.8 tok/s。

![Vulkan 上的 4 位文件](https://huggingface.co/infosave/Maxwell-V1-Flash-cmf/resolve/main/assets/files_vulkan_zh.png)

| 提示词，token 数 | 首个 token，秒 | 生成，tok/s | 显存峰值，GB |
|---:|---:|---:|---:|
| 512 | 0.91 | 38.5 | 19.6 |
| 2 048 | 4.93 | 35.4 | 19.8 |
| 4 096 | 13.5 | 28.9 | 20.2 |
| 8 192 | 39.8 | 27.6 | 21.0 |
| 16 384 | 109.7 | 24.6 | 21.4 |

通过 Vulkan 时块稀疏注意力同样在显卡上运行（索引器、块的选择以及在所选块上的注意力）：WikiText-2 在 6 144 token
窗口上的困惑度为 20.66，CUDA 为 20.61。权重完整地放在显卡上（4 位文件为 17.5 GB），显卡其余空间用于上下文：
24 GB 显卡约 17 000 个位置；超过之后序列转到 CPU 上继续，并给出一次警告。8 位文件通过 Vulkan 需要更大显存的显卡。

**CPU**

| 测试平台 | 文件 | 生成，tok/s |
|---|---|---:|
| Colab L4 主机（Xeon 2.2 GHz，6 核 / 12 线程，52 GB） | `q4tp` | 4.1 |
| MacBook Air M4（10 核，24 GB） | `q4tp` | 7.9 |

在 Apple Silicon 上模型运行于 CPU（其结构暂无 Metal 图）。文件通过内存映射读取：该次运行的常驻内存峰值为
12 GB。

### 用法

```bash
cortiq run Maxwell-V1-Flash-q4tp.cmf                          # 交互式对话
cortiq run Maxwell-V1-Flash-q4tp.cmf --prompt "..."           # 单次回答，附带推理过程
cortiq run Maxwell-V1-Flash-q4tp.cmf --prompt "..." --no-think            # 只输出回答
cortiq run Maxwell-V1-Flash-q4tp.cmf --prompt-file task.txt --max-tokens 2048
```

内置模板即模型自带的模板；未指定 `--no-think` 时，模型先推理再回答。文件中带有发布方的采样设置（temperature
0.6、top-p 0.95、top-k 20），`run` 和 `serve` 默认使用；`--greedy` 给出确定性输出。

兼容 OpenAI 的服务器：

```bash
CMF_GPU=cuda cortiq serve Maxwell-V1-Flash-q4tp.cmf --port 8080 --host 127.0.0.1

curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "maxwell", "max_tokens": 1024,
       "messages": [{"role": "user", "content": "制定从 MySQL 迁移到 Postgres 的三步计划。"}]}'
```

接口：`/v1/chat/completions`、`/v1/completions`、`/v1/models`，同一端口上还有网页面板。推理过程以
`message.reasoning_content` 返回；`"enable_thinking": false` 可将其关闭。工具以 OpenAI 格式放在 `tools`
中；模型以 `<tool_call><function=名称>…</function></tool_call>` 作答，服务器将其作为 `message.tool_calls`
返回。`--host 127.0.0.1` 使服务器只在本机可用；不校验 API 密钥。

### 长上下文

模型的注意力是块稀疏的：每个查询关注最前面的 1 024 个位置、自己所在的 128 位置块及其之前的七个块，另外还有
16 个块，由每一层的索引器按块内最高的 token 得分为四个键组分别选出。在 4 096 个位置以内，这就是全部位置。
在 CUDA 和 Vulkan 上，索引器和块的选择都在显卡上执行，与参考实现一致（见上面 6 144 token 窗口的困惑度）。每个位置
在显卡上保留键、值和索引键，每个 token 101 KB：30 000 token 的提示词使 4 位文件占用 20.3 GB。上下文上限由
`CMF_MAX_SEQ` 设置（最大 262 144；未设置时为权重旁边能容纳的位置数，最多 32 768）：显卡为这么多位置预留空间，
放不下的专家权重从主机内存流式送入。

### 校验下载

```bash
cortiq verify Maxwell-V1-Flash-q4tp.cmf
cortiq info Maxwell-V1-Flash-q4tp.cmf
```

| 文件 | sha256 |
|---|---|
| `Maxwell-V1-Flash-q4tp.cmf` | `350017b5bc2942f03fabb5611f3d9f7729ff9631607716e8961a4c0ae3a5a974` |
| `Maxwell-V1-Flash-q8_2f.cmf` | `8b96d4e0b65ae258eed226032c5af646adabd25d62fc5a912a7ad4f291972221` |

权重：[Apache-2.0](https://huggingface.co/maxwell-model/Maxwell-V1-Flash/blob/main/LICENSE)，来源
[maxwell-model/Maxwell-V1-Flash](https://huggingface.co/maxwell-model/Maxwell-V1-Flash)。原模型卡中关于用途和局限的说明同样适用于这些文件。
