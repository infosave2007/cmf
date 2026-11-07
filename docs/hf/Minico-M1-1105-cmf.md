---
license: other
license_name: lfm1.0
license_link: LICENSE
library_name: cortiq
base_model: Mincofficial/Minico-M1-1105
base_model_relation: quantized
pipeline_tag: text-generation
tags:
  - cmf
  - cortiq
  - quantized
  - q8_2f
  - q4tp
  - lfm2
  - liquid
  - reasoning
  - reasoning-effort
  - tool-calling
  - long-context
  - edge
  - metal
  - vulkan
language:
  - en
  - ru
  - zh
---

# Minico M1 1105 — CMF

[Minico M1 1105](https://huggingface.co/Mincofficial/Minico-M1-1105) is a 350M reasoning model from Mincofficial,
fine-tuned from LiquidAI's LFM2.5-350M (10 short-convolution and 6 attention layers, a 128,000-token context). It
thinks in `<think>`, has five reasoning-effort levels from `off` to `max`, and calls tools. Here it comes as single
[CMF](https://github.com/infosave2007/cmf) files for `cortiq`, a Rust inference engine with no Python and no ML
framework: Apple silicon through Metal, NVIDIA, AMD and Intel GPUs through Vulkan/DX12, and the CPU.

[English](#quick-start) · [Русский](#документация-на-русском) · [中文](#中文说明)

## Quick start

```bash
cargo install cortiq-cli
hf download infosave/Minico-M1-1105-cmf minico-m1-1105-q8_2f.cmf --local-dir .
cortiq run minico-m1-1105-q8_2f.cmf --prompt "What is 5 + 5 x 5?" --reasoning-effort low --max-tokens 640
```

```
<think>
The user is asking for the result of 5 + 5 x 5. Let's break it down step by step. First, we evaluate the
parentheses: 5 x 5 is 25. Then, we add 5 to the result: 5 + 25 = 30.
</think>

The answer is $\boxed{30}$.
```

`q8_2f` and `q4tp` run on cortiq 0.8.61 and later, at the template's default effort (`medium`) or `--no-think`
(`off`). The other effort levels, tool calls in `cortiq serve` and the turbo file's exact rows on Apple silicon come
with the release after 0.8.62 ([pull request #93](https://github.com/infosave2007/cmf/pull/93)). Prebuilt binaries
are on the [releases page](https://github.com/infosave2007/cmf/releases).

## Files

| file | weights | size | wikitext-2 ppl | top-1 vs original | use |
|---|---|---:|---:|---:|---|
| `minico-m1-1105-q8_2f.cmf` | 8-bit | 367 MB | 62.14 | 94.1 % | **default** |
| `minico-m1-1105-turbo.cmf` | 8-bit | 360 MB | 62.14 | 94.1 % | long prompts: the same weights with a far field |
| `minico-m1-1105-q4tp.cmf` | 4-bit | 197 MB | 71.49 | 58.6 % | the smallest file |
| bf16 original (Transformers) | — | 709 MB | 62.44 | — | — |

Perplexity over the first 4096 tokens of wikitext-2 test; *top-1 vs original*: positions where the file's top
prediction matches the bf16 checkpoint run in float32 by Transformers over the same tokens. A 350M model loses a lot
at 4 bits, so prefer `q8_2f` unless the size matters. The conversion is exact: an f16 conversion of the checkpoint
matches Transformers position for position (perplexity 62.441 against 62.441, top-1 100 %), and greedy answers are
text-identical at every effort level.

## Performance

MacBook Air M4, 24 GB: no fan, and the base M4's memory bus. `cortiq bench --core`, 128 tokens after a short
prompt:

| file | Metal, tok/s | CPU, tok/s |
|---|---:|---:|
| `q8_2f` | 158 | 164 |
| `q4tp` | 276 | 199 |

## The turbo file

The turbo file holds the `q8_2f` weights under a header that changes how a long prompt is kept. Prompts under 8192
tokens run exact attention, as in the standard file, with the same text. On a longer prompt, before its last 512
rows, the 6 attention layers move to a far field: the first 4 rows and the last 1800 stay exact, everything else
lives in 32 landmarks per head, so the cost of a token stops growing with the text. The far field is a subject of
U.S. patent application No. 19/738,763.

![What a long prompt keeps](https://huggingface.co/infosave/Minico-M1-1105-cmf/resolve/main/assets/far_field_en.png)

**Exact rows.** A far field alone loses a password. So each row is scored by what the layers do with it, with no
word lists, and the 248 best stay exact; the question at the end of the prompt reads them before it is answered. On
prompts the scoring did not see, the rows of a planted fact land in the top 0.01 % of the prompt on average and in
the top 0.02 % at worst.

![How passwords stay exact](https://huggingface.co/infosave/Minico-M1-1105-cmf/resolve/main/assets/exact_rows_en.png)

![Standard and turbo](https://huggingface.co/infosave/Minico-M1-1105-cmf/resolve/main/assets/compare_en.png)

MacBook Air M4, Metal, `cortiq run --greedy --no-think`, 128 tokens of a summary after a prompt of that length:

| prompt, tokens | first token, s (standard) | first token, s (**turbo**) | decode, tok/s (standard) | decode, tok/s (**turbo**) | peak memory, GB (standard) | peak memory, GB (**turbo**) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 000 | 0.9 | **0.9** | 155 | **153** | 0.32 | **0.32** |
| 4 000 | 3.6 | **3.6** | 137 | **138** | 0.59 | **0.59** |
| 16 000 | 19.8 | **24.5** | 92.2 | **122** | 1.30 | **1.05** |
| 32 000 | 56.5 | **61.2** | 65.1 | **104** | 1.95 | **1.75** |
| 64 000 | 183 | **193** | 38.6 | **106** | 3.69 | **3.38** |

Peak memory is the process's footprint (`/usr/bin/time -l`) beside the mapped weights. The standard file at 64 000
tokens needs `CMF_MAX_SEQ=70000`; the turbo file needs no such setting up to the model's 128,000 tokens.

**Quality.** Facts of four kinds (a pass key, an access code, a password, a room number) planted in 100 000
characters of Wikipedia, about 23 000 tokens, and asked at the end: the standard file finds 3 of 8, the turbo file
4 of 8, a far field without exact rows 0 of 8. A 350M model misses some of them with exact attention too.
Perplexity on the rows past the seal, two windows of 16 384 tokens: exact 37.6, far field 42.4.

## Usage

```bash
cortiq run minico-m1-1105-q8_2f.cmf                                         # interactive chat
cortiq run minico-m1-1105-q8_2f.cmf --prompt "..." --reasoning-effort high  # off | low | medium | high | max
cortiq run minico-m1-1105-q8_2f.cmf --prompt "..." --no-think               # the same as off
cortiq run minico-m1-1105-turbo.cmf --prompt-file report.txt --no-think     # a long document
```

| effort | what it does | `--max-tokens` the model's authors use |
|---|---|---:|
| `off` | no thinking, an empty think block is prefilled | 1024 |
| `low` | a short thought | 640 |
| `medium` (default) | a normal thought | 1152 |
| `high` | longer reasoning | 2368 |
| `max` | the longest reasoning | 4608 |

The thinking counts toward `--max-tokens` (default 256). For "What is 5 + 5 x 5?" the reply took 30 tokens at
`off`, 76 at `low`, 195 at `high` and 280 at `max`, each with the answer 30. The files carry the authors'
sampling — temperature 0.3, top-k 50, repetition penalty 1.05 — as the `run` and `serve` defaults. The embedded
chat template is the checkpoint's own. Pick one effort per conversation: the model was trained on chats that keep
one setting.

### OpenAI-compatible server

```bash
cortiq serve minico-m1-1105-q8_2f.cmf --port 8080

curl http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "minico", "max_tokens": 1152, "reasoning_effort": "high",
  "messages": [{"role": "user", "content": "A shirt costs $25 and is 20% off. How much do I pay?"}]}'
```

`reasoning_effort` (or `chat_template_kwargs.reasoning_effort`) picks the level; `"enable_thinking": false` is
`off`. Send `tools` in the OpenAI format: the model answers with `<|tool_call_start|>[set_volume(level=30)]<|tool_call_end|>`,
and the server returns it as `message.tool_calls` with `finish_reason: "tool_calls"`. In the test conversations it
called `set_volume(level=30)` and `get_weather(city="Paris")` with the right arguments. Given the tool's result,
it repeated the result's JSON instead of a sentence, as the checkpoint does in Transformers.

## Limits

- It is a 350M model: it gets facts and harder math wrong. English was the only language tested by its authors.
- The turbo file's far field runs on Metal, Vulkan and DX12; on the CPU it runs as the standard file. It was
  measured on the M4 only. On Apple silicon both files read a long prompt through the same batched host path (the
  GPU's batched prompt graph does not take short convolutions), so the turbo file reads it in about the time the
  standard one takes, plus its last 512 rows one at a time.
- Past the seal the text is approximate: what lies far back fades, apart from the rows kept exact. The scoring
  behind those rows saw planted facts; other kinds of important text were not tested.

## Verify the download

```bash
sha256sum -c minico-m1-1105-q8_2f.cmf.sha256
cortiq verify minico-m1-1105-q8_2f.cmf
```

| file | size, bytes | sha256 |
|---|---:|---|
| `minico-m1-1105-q8_2f.cmf` | 367,302,093 | `2c37c0d61ca0df55de83f9d7666c3e5fa6f6bf798d3c085d5e582ce49e26090c` |
| `minico-m1-1105-turbo.cmf` | 360,294,221 | `93fec8a30eeb6608cf741ac52be6340f103d427dcfaca7eb97962f69057e344a` |
| `minico-m1-1105-q4tp.cmf` | 197,282,253 | `f8485a4778bcce778eae60d65d881a6aabcc8b5360d15cabe25829e403f7ff90` |

## Reproduce

```bash
cortiq convert --model Mincofficial/Minico-M1-1105 --quant q8_2f --output minico-m1-1105-q8_2f.cmf \
  --also-output q4tp=minico-m1-1105-q4tp.cmf \
  --sampling temperature=0.3,top_k=50,top_p=1.0,min_p=0,repetition_penalty=1.05
cortiq far-turbo minico-m1-1105-q8_2f.cmf minico-m1-1105-turbo.cmf --exact-weights keep_minico.bin
```

`keep_minico.bin` and the Transformers reference scripts are in
[scripts/minico](https://github.com/infosave2007/cmf/tree/master/scripts/minico); the fit comes from
[q27_feat.py](https://github.com/infosave2007/cmf/blob/master/scripts/dory/router/q27_feat.py) with 96 features
(2 per KV head of the 6 attention layers).

## License

These files are a modified form of Minico M1 1105, itself a modified form of Liquid AI's LFM2.5-350M: the weights
were converted to CMF and quantized by cortiq. They remain under the LFM Open License v1.0, a copy of which, with
the attribution notices of the base model, is in this repository ([LICENSE](LICENSE)). Among its terms, commercial
use by an entity with annual revenue of US$10 million or more is not licensed. See the
[upstream card](https://huggingface.co/Mincofficial/Minico-M1-1105).

---

## Документация на русском

[Minico M1 1105](https://huggingface.co/Mincofficial/Minico-M1-1105) — рассуждающая модель на 350 млн параметров
от Mincofficial, дообученная из LFM2.5-350M от Liquid AI (10 слоёв короткой свёртки и 6 слоёв внимания, контекст
128 000 токенов). Она рассуждает в `<think>`, умеет пять уровней усилия от `off` до `max` и вызывает инструменты.
Здесь она в виде отдельных файлов [CMF](https://github.com/infosave2007/cmf) для движка `cortiq` на Rust без Python
и ML-фреймворков.

### Быстрый старт

```bash
cargo install cortiq-cli
hf download infosave/Minico-M1-1105-cmf minico-m1-1105-q8_2f.cmf --local-dir .
cortiq run minico-m1-1105-q8_2f.cmf --prompt "What is 5 + 5 x 5?" --reasoning-effort low --max-tokens 640
```

`q8_2f` и `q4tp` работают с cortiq 0.8.61 и новее — с уровнем шаблона по умолчанию (`medium`) или с `--no-think`
(`off`). Остальные уровни, вызовы инструментов в `cortiq serve` и точные строки турбо-файла на Apple silicon
появятся в релизе после 0.8.62 ([pull request #93](https://github.com/infosave2007/cmf/pull/93)). Авторы модели
проверяли её только на английском.

### Файлы

| файл | веса | размер | ppl wikitext-2 | top-1 к оригиналу | назначение |
|---|---|---:|---:|---:|---|
| `minico-m1-1105-q8_2f.cmf` | 8 бит | 367 МБ | 62,14 | 94,1 % | **по умолчанию** |
| `minico-m1-1105-turbo.cmf` | 8 бит | 360 МБ | 62,14 | 94,1 % | длинные промпты: те же веса с дальним полем |
| `minico-m1-1105-q4tp.cmf` | 4 бита | 197 МБ | 71,49 | 58,6 % | самый маленький файл |
| bf16-оригинал (Transformers) | — | 709 МБ | 62,44 | — | — |

Конвертация точная: f16-конвертация совпадает с Transformers по каждой позиции (ppl 62,441 против 62,441, top-1
100 %), жадные ответы совпадают по тексту на всех уровнях усилия. Модель на 350 млн параметров заметно теряет в 4
битах, поэтому берите `q8_2f`, если размер не важен.

### Скорость

MacBook Air M4, 24 ГБ (без вентилятора, базовая шина памяти M4), `cortiq bench --core`, 128 токенов: `q8_2f` —
158 ток/с на Metal и 164 на CPU, `q4tp` — 276 и 199.

### Турбо-файл

Те же веса `q8_2f` с заголовком, который меняет хранение длинного промпта. Промпты короче 8192 токенов идут с точным
вниманием, как в обычном файле, и дают тот же текст. На длинном промпте перед последними 512 строками 6 слоёв
внимания переходят на дальнее поле: первые 4 строки и последние 1800 остаются точными, остальное хранится в 32
опорных точках на голову, и цена токена перестаёт расти с текстом. Патентная заявка США № 19/738,763.

![Что хранится от длинного промпта](https://huggingface.co/infosave/Minico-M1-1105-cmf/resolve/main/assets/far_field_ru.png)

Одно дальнее поле теряет пароль. Поэтому каждая строка получает оценку по тому, что с ней делают слои, без списков
слов, и 248 лучших остаются точными; вопрос в конце промпта читает их до ответа. На промптах, которых оценка не
видела, строки вставленного факта попадают в верхние 0,01 % промпта в среднем и в 0,02 % в худшем случае.

![Как пароли остаются точными](https://huggingface.co/infosave/Minico-M1-1105-cmf/resolve/main/assets/exact_rows_ru.png)

![Обычный и турбо](https://huggingface.co/infosave/Minico-M1-1105-cmf/resolve/main/assets/compare_ru.png)

MacBook Air M4, Metal, `cortiq run --greedy --no-think`, 128 токенов пересказа после промпта такой длины:

| промпт, токенов | первый токен, с (обычный) | первый токен, с (**турбо**) | генерация, ток/с (обычный) | генерация, ток/с (**турбо**) | пик памяти, ГБ (обычный) | пик памяти, ГБ (**турбо**) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 000 | 0,9 | **0,9** | 155 | **153** | 0,32 | **0,32** |
| 4 000 | 3,6 | **3,6** | 137 | **138** | 0,59 | **0,59** |
| 16 000 | 19,8 | **24,5** | 92,2 | **122** | 1,30 | **1,05** |
| 32 000 | 56,5 | **61,2** | 65,1 | **104** | 1,95 | **1,75** |
| 64 000 | 183 | **193** | 38,6 | **106** | 3,69 | **3,38** |

Обычному файлу на 64 000 токенов нужен `CMF_MAX_SEQ=70000`; турбо-файлу эта настройка не нужна вплоть до 128 000 токенов модели.
Факты в 100 000 символов Википедии (около 23 000 токенов): обычный файл находит 3 из 8, турбо — 4 из 8, дальнее поле
без точных строк — 0 из 8. Perplexity на строках после печати (два окна по 16 384 токена): точное внимание 37,6,
дальнее поле 42,4. На Apple silicon оба файла читают длинный промпт одним и тем же пакетным путём на хосте, поэтому
турбо-файл читает его примерно столько же, сколько обычный, плюс последние 512 строк по одной.

### Уровни усилия и сервер

```bash
cortiq run minico-m1-1105-q8_2f.cmf --prompt "..." --reasoning-effort high  # off | low | medium | high | max
cortiq serve minico-m1-1105-q8_2f.cmf --port 8080
```

Рассуждение входит в `--max-tokens` (по умолчанию 256); авторы модели дают `off` 1024 токена, `low` 640, `medium`
1152, `high` 2368, `max` 4608. В API уровень задаёт `reasoning_effort` (или `chat_template_kwargs.reasoning_effort`),
`"enable_thinking": false` — это `off`. Инструменты передаются в формате OpenAI; сервер возвращает вызовы модели как
`message.tool_calls`. Сэмплирование по умолчанию — авторское: temperature 0,3, top-k 50, repetition penalty 1,05.

Файлы — изменённая форма Minico M1 1105, которая сама изменённая форма LFM2.5-350M от Liquid AI (конвертация в CMF и
квантование cortiq), и распространяются по LFM Open License v1.0 (копия с атрибуцией базовой модели — в
[LICENSE](LICENSE)); в частности, коммерческое использование организацией с годовой выручкой от 10 млн долларов
США лицензией не покрывается.

## 中文说明

[Minico M1 1105](https://huggingface.co/Mincofficial/Minico-M1-1105) 是 Mincofficial 的 350M 推理模型，由 Liquid AI
的 LFM2.5-350M 微调而来（10 层短卷积、6 层注意力，128,000 token 上下文）。它在 `<think>` 中思考，提供从 `off` 到
`max` 的五档推理强度，并支持工具调用。本仓库提供可由 Rust 推理引擎 `cortiq` 加载的 CMF 文件。模型作者只在英文上做过测试。

```bash
cargo install cortiq-cli
hf download infosave/Minico-M1-1105-cmf minico-m1-1105-q8_2f.cmf --local-dir .
cortiq run minico-m1-1105-q8_2f.cmf --prompt "What is 5 + 5 x 5?" --reasoning-effort low --max-tokens 640
```

`q8_2f` 与 `q4tp` 可在 cortiq 0.8.61 及以上版本运行，使用模板默认强度（`medium`）或 `--no-think`（`off`）。其余强度档位、
`cortiq serve` 中的工具调用，以及 Apple 芯片上 turbo 文件的精确行，将随 0.8.62 之后的版本发布
（[pull request #93](https://github.com/infosave2007/cmf/pull/93)）。

| 文件 | 权重 | 大小 | wikitext-2 ppl | 与原模型 top-1 一致率 | 适用 |
|---|---|---:|---:|---:|---|
| `minico-m1-1105-q8_2f.cmf` | 8 位 | 367 MB | 62.14 | 94.1 % | **默认** |
| `minico-m1-1105-turbo.cmf` | 8 位 | 360 MB | 62.14 | 94.1 % | 长提示词：相同权重加远场 |
| `minico-m1-1105-q4tp.cmf` | 4 位 | 197 MB | 71.49 | 58.6 % | 最小的文件 |
| bf16 原模型（Transformers） | — | 709 MB | 62.44 | — | — |

转换是精确的：f16 转换与 Transformers 逐位置一致（ppl 62.441 对 62.441，top-1 100 %），各强度档位的贪心回答文本相同。
MacBook Air M4（24 GB，无风扇）上 `cortiq bench --core`：`q8_2f` 在 Metal 上 158 token/s、CPU 上 164，`q4tp` 为
276 与 199。

turbo 文件使用相同的 `q8_2f` 权重：8192 token 以下的提示词保持精确注意力，文本与标准文件相同；更长的提示词在最后 512 行之前，
6 个注意力层转入远场（前 4 行和最后 1800 行精确，其余每个头 32 个地标），并按各层自身的作用为每行打分，保留得分最高的
248 行为精确行，使密码和编码不会丢失。美国专利申请号 19/738,763。

![长提示词保存什么](https://huggingface.co/infosave/Minico-M1-1105-cmf/resolve/main/assets/far_field_zh.png)

![密码如何保持精确](https://huggingface.co/infosave/Minico-M1-1105-cmf/resolve/main/assets/exact_rows_zh.png)

![标准与 turbo](https://huggingface.co/infosave/Minico-M1-1105-cmf/resolve/main/assets/compare_zh.png)

M4 上 Metal，在 16 000 / 32 000 / 64 000 token 提示词之后生成速度：标准 92.2 / 65.1 / 38.6 token/s，turbo 122 / 104 / 106；
首 token 时间 19.8 / 56.5 / 183 秒对 24.5 / 61.2 / 193 秒；峰值内存 1.30 / 1.95 / 3.69 GB 对 1.05 / 1.75 / 3.38 GB。
在约 23 000 token 的维基百科文本中植入的事实：标准文件找到 8 个中的 3 个，turbo 找到 4 个，无精确行的远场为 0 个。

`--reasoning-effort off|low|medium|high|max` 选择推理强度；思考计入 `--max-tokens`（默认 256）。`cortiq serve` 提供 OpenAI
兼容接口：`reasoning_effort` 选择强度，工具调用以 `message.tool_calls` 返回。默认采样为作者推荐的 temperature 0.3、
top-k 50、repetition penalty 1.05。

这些文件是 Minico M1 1105 的修改形式（其本身是 Liquid AI LFM2.5-350M 的修改形式；由 cortiq 转换为 CMF 并量化），仍受
LFM Open License v1.0 约束（副本及基础模型的署名声明见 [LICENSE](LICENSE)）；其中，年收入 1000 万美元及以上的实体进行
商业使用不在许可范围内。
