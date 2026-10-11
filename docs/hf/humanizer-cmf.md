---
license: apache-2.0
library_name: cortiq
base_model: jialinyyzz/humanizer
base_model_relation: quantized
pipeline_tag: text-generation
tags:
  - cmf
  - cortiq
  - humanizer
  - text-rewriting
  - gemma4
  - quantized
language:
  - en
  - zh
---

# humanizer 12B — CMF

[humanizer](https://huggingface.co/jialinyyzz/humanizer) by jialinyyzz rewrites
AI-written drafts (emails, essays, reports, forum posts; English and Chinese)
so they read like a person wrote them; it is trained to keep every number,
unit, date, name and quotation. It is a fine-tune of Google's
[Gemma 4 12B](https://huggingface.co/google/gemma-4-12B). This repo packages its
text model as single [CMF](https://github.com/infosave2007/cmf) files for
`cortiq`, a Rust inference engine with no Python and no ML framework: NVIDIA,
AMD and Intel GPUs through Vulkan/DX12, Apple silicon through Metal, and the CPU.

From the authors: [model card and evaluation](https://huggingface.co/jialinyyzz/humanizer#evaluation-details) ·
[desktop app for Mac and Windows](https://github.com/sgaofen/humanizer-local-model/releases/latest) ·
[GGUF files for llama.cpp](https://huggingface.co/jialinyyzz/humanizer-GGUF) ·
[source and docs](https://github.com/sgaofen/humanizer-local-model). They report
that Originality.ai at its strictest setting judged 95 % of 210 English
rewrites human, and that a strict LLM judge found no factual problem in 376 of
420 English rewrites. These are the authors' measurements of their own files,
not of the CMF files here; method and caveats are in
[their evaluation](https://huggingface.co/jialinyyzz/humanizer#evaluation-details).

[English](#quick-start) · [Русский](#документация-на-русском) · [中文](#中文文档)

## Quick start

```bash
cargo install cortiq-cli
hf download infosave/humanizer-cmf humanizer-12b-q8_2f.cmf --local-dir .
cortiq run humanizer-12b-q8_2f.cmf --max-tokens 2048 --prompt "$(cat draft.txt)"
```

`humanizer-12b-q8_2f.cmf` (11.97 GB) is the default: 31.7 tokens/s on an RTX
PRO 4000 and 6.6 on an M4 Mac mini (Metal). `humanizer-12b-q4tp-ffn.cmf`
(7.89 GB) fits GPUs with 12 GB and decodes 41.6 tokens/s on the RTX PRO 4000,
at slightly lower quality ([Files](#files)).

Use cortiq 0.8.16 or later; prebuilt binaries for Linux, macOS and Windows are
on the [releases page](https://github.com/infosave2007/cmf/releases). Pass the
draft as it is: the file builds the authors' prompt around it and starts from
their sampling settings. Keep `--max-tokens` high (the default is 256): a
rewrite runs about as long as its draft (932 tokens for a 957-token draft in
our tests).

## Files

| file | weights | size | top-1 vs bf16 | KL | perplexity | peak GPU memory | recommended for |
|---|---|---:|---:|---:|---:|---:|---|
| `humanizer-12b-q8_2f.cmf` | 8-bit | 11.97 GB | 99.4 % | 0.0008 | 2.1174 | 15.3 GB | **default**: closest to the original; Apple silicon; GPUs with 16 GB or more |
| `humanizer-12b-q4tp-ffn.cmf` | 4-bit FFN, 8-bit attention and embeddings | 7.89 GB | 95.5 % | 0.020 | 2.1555 | 10.3 GB | 31 % faster decode on the RTX PRO 4000; GPUs with 12 GB |

Quality is measured against the authors' bf16 weights run in HF transformers,
on 2,051 tokens of the authors' own bf16 rewrites of six drafts (four English,
two Chinese): the share of positions where the file's top prediction matches
bf16's, the mean KL divergence from bf16, and perplexity (bf16: 2.1168). A
second set of 2,127 tokens gives 98.9 % (`q8_2f`) and 95.6 % (`q4tp-ffn`)
top-1. In a greedy fact check on three drafts, `q8_2f` kept every fact as
written and `q4tp-ffn` wrote one amount differently ("$186,000" as "$186k").
Peak GPU memory was read with
nvidia-smi during `cortiq bench` with a 1,000-token prompt on an RTX PRO 4000;
on Apple silicon and on the CPU both files ran on a 24 GB Mac mini.

## Performance

`cortiq bench --core --ignore-eos`, one stream, cortiq 0.8.16; RTX rows are medians of three runs:

| hardware | backend | file | decode, tok/s | decode at 1k tokens | 1k-token prompt, tok/s | time to first token |
|---|---|---|---:|---:|---:|---:|
| RTX PRO 4000 Blackwell (24 GB) | Vulkan | `q4tp-ffn` | 41.6 | 36.3 | 137 | 6.9 s |
| RTX PRO 4000 Blackwell (24 GB) | Vulkan | `q8_2f` | 31.7 | 28.3 | 90.2 | 10.8 s |
| Mac mini M4 (24 GB) | Metal | `q8_2f` | 6.6 | 6.0 | 68 | 14.7 s |
| Mac mini M4 (24 GB) | Metal | `q4tp-ffn` | 8.5 | 8.3 | 72 | 14.0 s |
| Mac mini M4 (24 GB) | CPU | `q8_2f` | 4.7 | | | |
| Mac mini M4 (24 GB) | CPU | `q4tp-ffn` | 4.6 | | | |

Reading the prompt also uses the CPU; the RTX host had about 10 cores. Whole
rewrites through `cortiq serve`:

| hardware | file | 957-token draft: first token | its full rewrite | rewrite speed | email, 362 tokens in |
|---|---|---:|---:|---:|---:|
| RTX PRO 4000 Blackwell | `q4tp-ffn` | 7.5 s | 901 tokens in 37.1 s | 30.1 tok/s | 10.6 s |
| RTX PRO 4000 Blackwell | `q8_2f` | 11.2 s | 932 tokens in 48.9 s | 24.7 tok/s | 14.8 s |

RTX rows use the authors' sampling (medians of three); Mac rows are greedy.
`cortiq run` loads the weights onto the GPU at every start (about 13 s on the
RTX PRO 4000), so for more than one draft keep `cortiq serve` running. On
Apple silicon use `q8_2f`: the 4-bit file is no faster there and reads prompts
more slowly.

## Usage

### Prompt and sampling

humanizer is a text-completion model, not a chat model. Each file carries the
authors' prompt as its template: `cortiq run --prompt` and
`/v1/chat/completions` take the draft and build the instruction, a blank line,
the draft with surrounding whitespace removed, and `### Rewritten:`, token for
token the authors' format. One draft per request; system messages and earlier
turns are ignored. Generation ends at the end-of-sequence token, so don't add
stop strings, `"###"` least of all.

The files also carry the authors' sampling: temperature 1.0, top-p 0.95, top-k
and min-p off, repetition penalty 1.0. `run` and `serve` start from it, flags
and request fields override it, and `cortiq info` shows it.

```bash
cortiq run humanizer-12b-q8_2f.cmf --max-tokens 2048 --prompt "$(cat draft.txt)"              # authors' sampling
cortiq run humanizer-12b-q8_2f.cmf --max-tokens 2048 --seed 7 --prompt "$(cat draft.txt)"     # repeatable sample
cortiq run humanizer-12b-q8_2f.cmf --max-tokens 2048 --greedy --prompt "$(cat draft.txt)"     # deterministic
```

The authors note that on short, informal English drafts with technical jargon
the model now and then writes the whole rewrite in Chinese; their app checks
the language and samples again. Do the same: run it once more without a fixed
seed. The authors size the context at 8,192 tokens for instruction, draft and
rewrite together; split longer documents at paragraph breaks.

### OpenAI-compatible server

```bash
cortiq serve humanizer-12b-q8_2f.cmf --port 8080

jq -n --rawfile d draft.txt '{model: "humanizer", max_tokens: 2048,
                              messages: [{role: "user", content: $d}]}' \
  | curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d @- \
  | jq -r '.choices[0].message.content'
```

Always send `max_tokens`: the server default is 256, and
`max_completion_tokens` is not read. `temperature`, `top_p`, `top_k`, `min_p`,
`repetition_penalty` (or `repeat_penalty`) and `seed` in a request override the
file's sampling. `/v1/completions` also accepts the authors' full prompt as
plain text, as llama.cpp's `/completion` does; the model then gets the same
tokens as through the chat endpoint.
Other endpoints: `/v1/models` and a web dashboard on the same port.
`--host 127.0.0.1` keeps the server local.

### Hardware

**Choosing a GPU (Vulkan).** `cortiq gpu` lists the adapters;
`CMF_GPU_ADAPTER=1` (index or name substring) picks one. If the list is empty
on Linux, install `libvulkan1 libglvnd0 libegl1 libgl1 libglx0`; on headless
machines set `XDG_RUNTIME_DIR=/tmp`.

**CPU only.** `CMF_GPU=0` keeps everything on the CPU.

## Examples

`q8_2f` through `cortiq serve` with the authors' sampling. Both drafts were
written for this card. Shown is the best of three samples by a strict string
check of the facts; in the other four samples every flagged fact was still
there in another format ("$186K", "Sept 30", "2026.10.8").

**English email, draft:**

```text
Subject: Update on the Q3 Vendor Migration

Hi Priya,

I hope this message finds you well. I wanted to take a moment to provide you with a comprehensive update regarding the status of our Q3 vendor migration project. As you may be aware, we have been working diligently to transition our logistics operations from Northwind Freight to Calder & Sons.

First and foremost, it is important to note that the migration is currently 72% complete. To date, 1,840 of our 2,550 active shipments have been successfully moved to the new platform. Furthermore, the remaining 710 shipments are scheduled to be transferred by October 24, 2026.

Additionally, I would like to highlight that the projected cost savings remain on track. We anticipate annual savings of approximately $186,000, which represents a 14.5% reduction compared to our previous contract. However, it is worth mentioning that we encountered a brief delay on September 30 due to an API outage on the Calder side, which lasted roughly 6 hours.

Moving forward, Marcus Lee from the operations team will be coordinating the final cutover. Please do not hesitate to reach out if you have any questions or concerns.

Best regards,
Daniel
```

**Rewrite** (all 16 facts kept; "September 30" became "Sep 30"):

```text
Subject: Q3 Vendor Migration update

Hi Priya,

Hope you're doing well. Just wanted to send you a thorough update on our current status on the Q3 vendor migration. If you remember, we're working hard to switch our logistics business from Northwind Freight to Calder & Sons.

Important information - we're 72% done with the migration. We've successfully transferred 1,840 (out of 2,550) active shipments to the new system. The good news - by October 24, 2026 we will move 710 left shipments.

Also, everything's on schedule with the expected cost savings. So we'll be saving ~$186,000 (14.5% less than what we had in the old contract) a year. Though we had a little delay on Sep 30 because of the Calder's API outage (it was about 6 hours).

Anyway, operations team (Marcus Lee) will be in charge of the last cutover. If you have any questions/concerns - please, let me know.

Best regards
Daniel
```

**Chinese report, draft:**

```text
主题：关于华东区第三季度销售情况的汇报

尊敬的王总：

您好！首先，非常感谢您一直以来对我们团队工作的大力支持与悉心指导。在此，我谨代表华东区销售团队，向您详细汇报2026年第三季度的销售情况。

总体而言，本季度华东区实现销售额4,860万元，同比增长18.3%，完成年度目标的76%。其中，上海分公司贡献了2,150万元，杭州分公司贡献了1,320万元，南京分公司贡献了980万元。值得一提的是，新产品线“云帆系列”在9月份正式上市，首月销量即突破3,600台。

与此同时，我们也必须清醒地认识到存在的不足。例如，苏州地区的回款周期由原来的45天延长至62天，这在一定程度上影响了现金流。针对这一问题，我们计划于10月15日前与主要客户逐一沟通，力争将回款周期缩短至50天以内。

展望第四季度，我们将继续深耕重点客户，进一步提升市场占有率。如有任何问题，请随时与我联系。

此致
敬礼！

李明
2026年10月8日
```

**Rewrite** (all 22 facts kept; thousands separators dropped, "4,860" became "4860"):

```text
主题：华东区第三季度销售报告

尊敬的王总：

您好！首先，衷心感谢您对我们工作的大力支持和关怀指导。在此，我代表华东区销售团队向您具体汇报一下2026年第三季度销售情况。

整体上，本季度华东区销售总额为4860万元，较去年同期增长18.3%。完成年度目标的76%。主要由上海分公司贡献2150万元，杭州分公司贡献1320万元，南京分公司贡献980万元。另外，9月份新产品线“云帆系列”新上市，首月销售量便超过了3600台。

同时，我们也要看到不足。比如：在苏州地区回款周期由原来的45天增加到62天，一定程度上影响了现金流。对此，我们计划在10月15日前，和主要客户一一进行沟通，争取将回款周期控制在50天之内。

第四季度，我们将继续耕耘重点客户，进一步增加市场占有率。如有问题，请随时联系我。

此致
敬礼！

李明
2026年10月8日
```

## Limits

- Text only: the vision and audio parts of the Gemma 4 base are not included.
- It can still change a detail. Proofread numbers, dates and names before you
  send the result. Formatting (paragraph breaks, lists, headings) can change too.
- Now and then it answers an English draft in Chinese; sample again.
- The authors note that in casual genres it sometimes adds slang or profanity
  that was not in the draft.
- The authors intend it as a writing tool for your own drafts. Where a school,
  employer or publication has rules about AI assistance, follow them.

## Verify the download

```bash
hf download infosave/humanizer-cmf humanizer-12b-q8_2f.cmf.sha256 --local-dir .
sha256sum -c humanizer-12b-q8_2f.cmf.sha256      # macOS: shasum -a 256 -c humanizer-12b-q8_2f.cmf.sha256
cortiq verify humanizer-12b-q8_2f.cmf
cortiq info humanizer-12b-q8_2f.cmf               # shows the sampling defaults
```

## License and attribution

Apache-2.0, like the original. humanizer is © 2026 Stephen Yu
([jialinyyzz/humanizer](https://huggingface.co/jialinyyzz/humanizer),
[GitHub](https://github.com/sgaofen/humanizer-local-model)), fine-tuned from
[google/gemma-4-12B](https://huggingface.co/google/gemma-4-12B), which Google
releases under Apache-2.0. These files convert the authors' bf16 weights to CMF
and leave out the vision and audio parts. Quality and speed figures on this
page are ours; evaluation figures credited to the authors are theirs. This repo
is not affiliated with or endorsed by the humanizer authors or by Google.

---

## Документация на русском

[humanizer](https://huggingface.co/jialinyyzz/humanizer) от jialinyyzz
переписывает черновики, написанные ИИ (письма, эссе, отчёты, посты на
форумах; английский и китайский), так, чтобы они читались как написанные
человеком; модель обучена сохранять каждое число, единицу измерения, дату, имя
и цитату. Это дообученная модель Google
[Gemma 4 12B](https://huggingface.co/google/gemma-4-12B). Здесь её текстовая
модель упакована в отдельные файлы [CMF](https://github.com/infosave2007/cmf)
для `cortiq` — движка на Rust без Python и ML-фреймворков: GPU NVIDIA, AMD и
Intel через Vulkan/DX12, Apple silicon через Metal, а также CPU.

У авторов: [карточка модели и оценка](https://huggingface.co/jialinyyzz/humanizer#evaluation-details) ·
[приложение для Mac и Windows](https://github.com/sgaofen/humanizer-local-model/releases/latest) ·
[GGUF-файлы для llama.cpp](https://huggingface.co/jialinyyzz/humanizer-GGUF) ·
[исходники и документация](https://github.com/sgaofen/humanizer-local-model).
По их данным, Originality.ai на самой строгой настройке признал написанными
человеком 95 % из 210 английских переписанных текстов, а строгий LLM-судья не
нашёл фактических ошибок в 376 из 420 английских текстов. Это измерения
авторов на их собственных файлах, а не на CMF-файлах отсюда; методика и
оговорки — в [их оценке](https://huggingface.co/jialinyyzz/humanizer#evaluation-details).

### Быстрый старт

```bash
cargo install cortiq-cli
hf download infosave/humanizer-cmf humanizer-12b-q8_2f.cmf --local-dir .
cortiq run humanizer-12b-q8_2f.cmf --max-tokens 2048 --prompt "$(cat draft.txt)"
```

`humanizer-12b-q8_2f.cmf` (11.97 ГБ) — файл по умолчанию: 31.7 ток/с на RTX
PRO 4000 и 6.6 на Mac mini M4 (Metal). `humanizer-12b-q4tp-ffn.cmf` (7.89 ГБ)
помещается в GPU с 12 ГБ и выдаёт 41.6 ток/с на RTX PRO 4000 при чуть меньшем
качестве ([Файлы](#файлы)).

Нужен cortiq 0.8.16 или новее; готовые сборки для Linux, macOS и Windows — на
[странице релизов](https://github.com/infosave2007/cmf/releases). Передавайте
черновик как есть: файл сам строит вокруг него промпт авторов и берёт их
настройки сэмплинга. Ставьте `--max-tokens` с запасом (по умолчанию 256):
переписанный текст примерно той же длины, что и черновик (в наших тестах 932
токена для черновика из 957).

### Файлы

| файл | веса | размер | совпадение top-1 с bf16 | KL | перплексия | пик памяти GPU | назначение |
|---|---|---:|---:|---:|---:|---:|---|
| `humanizer-12b-q8_2f.cmf` | 8 бит | 11.97 ГБ | 99.4 % | 0.0008 | 2.1174 | 15.3 ГБ | **по умолчанию**: ближе всего к оригиналу; Apple silicon; GPU от 16 ГБ |
| `humanizer-12b-q4tp-ffn.cmf` | FFN 4 бита, внимание и эмбеддинги 8 бит | 7.89 ГБ | 95.5 % | 0.020 | 2.1555 | 10.3 ГБ | генерация на 31 % быстрее на RTX PRO 4000; GPU от 12 ГБ |

Качество измерено относительно bf16-весов авторов в HF transformers на 2 051
токене собственных bf16-переписываний авторов для шести черновиков (четыре
английских, два китайских): доля позиций, где лучший прогноз файла совпадает с
bf16, средняя KL-дивергенция от bf16 и перплексия (у bf16 2.1168). Второй
набор из 2 127 токенов даёт 98.9 % (`q8_2f`) и 95.6 % (`q4tp-ffn`) top-1. В
жадной проверке фактов на трёх черновиках `q8_2f` сохранил все факты дословно,
а `q4tp-ffn` записал одну сумму иначе («$186,000» как «$186k»). Пик памяти GPU
снят через nvidia-smi
во время `cortiq bench` с промптом 1000 токенов на RTX PRO 4000; на Apple
silicon и на CPU оба файла работали на Mac mini с 24 ГБ.

### Скорость

`cortiq bench --core --ignore-eos`, один поток, cortiq 0.8.16; строки RTX — медианы трёх запусков:

| железо | бэкенд | файл | генерация, ток/с | генерация на 1k токенах | промпт 1k токенов, ток/с | время до первого токена |
|---|---|---|---:|---:|---:|---:|
| RTX PRO 4000 Blackwell (24 ГБ) | Vulkan | `q4tp-ffn` | 41.6 | 36.3 | 137 | 6.9 с |
| RTX PRO 4000 Blackwell (24 ГБ) | Vulkan | `q8_2f` | 31.7 | 28.3 | 90.2 | 10.8 с |
| Mac mini M4 (24 ГБ) | Metal | `q8_2f` | 6.6 | 6.0 | 68 | 14.7 с |
| Mac mini M4 (24 ГБ) | Metal | `q4tp-ffn` | 8.5 | 8.3 | 72 | 14.0 с |
| Mac mini M4 (24 ГБ) | CPU | `q8_2f` | 4.7 | | | |
| Mac mini M4 (24 ГБ) | CPU | `q4tp-ffn` | 4.6 | | | |

Чтение промпта нагружает и CPU; у машины с RTX было около 10 ядер. Переписывание
целиком через `cortiq serve`:

| железо | файл | черновик 957 токенов: первый токен | его полное переписывание | скорость переписывания | письмо, 362 токена на входе |
|---|---|---:|---:|---:|---:|
| RTX PRO 4000 Blackwell | `q4tp-ffn` | 7.5 с | 901 токен за 37.1 с | 30.1 ток/с | 10.6 с |
| RTX PRO 4000 Blackwell | `q8_2f` | 11.2 с | 932 токена за 48.9 с | 24.7 ток/с | 14.8 с |

Строки RTX — с сэмплингом авторов (медианы трёх), строки Mac — жадные.
`cortiq run` при каждом запуске загружает веса в GPU (около 13 с на RTX PRO
4000), поэтому для нескольких черновиков держите запущенным `cortiq serve`. На
Apple silicon берите `q8_2f`: 4-битный файл там не быстрее и медленнее читает
промпт.

### Использование

**Промпт и сэмплинг.** humanizer — модель продолжения текста, а не чат-модель.
В каждом файле промпт авторов хранится как шаблон: `cortiq run --prompt` и
`/v1/chat/completions` принимают черновик и строят инструкцию, пустую строку,
черновик без пробелов по краям и `### Rewritten:` — токен в токен формат
авторов. Один черновик на запрос; системные сообщения и прежние реплики
игнорируются. Генерация заканчивается на токене конца последовательности, так
что не добавляйте стоп-строки, тем более `"###"`.

В файлах записан и сэмплинг авторов: temperature 1.0, top-p 0.95, top-k и
min-p выключены, repetition penalty 1.0. `run` и `serve` начинают с него, флаги
и поля запроса его переопределяют, `cortiq info` его показывает.

```bash
cortiq run humanizer-12b-q8_2f.cmf --max-tokens 2048 --prompt "$(cat draft.txt)"              # сэмплинг авторов
cortiq run humanizer-12b-q8_2f.cmf --max-tokens 2048 --seed 7 --prompt "$(cat draft.txt)"     # воспроизводимый сэмпл
cortiq run humanizer-12b-q8_2f.cmf --max-tokens 2048 --greedy --prompt "$(cat draft.txt)"     # детерминированно
```

По наблюдению авторов, на коротких неформальных английских черновиках с
техническим жаргоном модель иногда пишет весь ответ по-китайски; их приложение
проверяет язык и сэмплирует заново. Делайте так же — запустите ещё раз без
фиксированного seed. Авторы рассчитывают контекст на 8 192 токена на
инструкцию, черновик и ответ вместе; длинные документы делите по абзацам.

**Сервер с OpenAI-совместимым API:**

```bash
cortiq serve humanizer-12b-q8_2f.cmf --port 8080

jq -n --rawfile d draft.txt '{model: "humanizer", max_tokens: 2048,
                              messages: [{role: "user", content: $d}]}' \
  | curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d @- \
  | jq -r '.choices[0].message.content'
```

Всегда передавайте `max_tokens`: по умолчанию сервер ограничивает ответ 256
токенами, а `max_completion_tokens` не читает. Поля `temperature`, `top_p`,
`top_k`, `min_p`, `repetition_penalty` (или `repeat_penalty`) и `seed`
переопределяют сэмплинг файла. `/v1/completions` принимает и полный промпт
авторов обычным текстом, как `/completion` в llama.cpp; модель получает те же
токены, что и через чат-эндпоинт.
Ещё есть `/v1/models` и веб-панель на том же порту. `--host 127.0.0.1`
оставляет сервер локальным.

**Выбор GPU (Vulkan).** `cortiq gpu` показывает адаптеры; `CMF_GPU_ADAPTER=1`
(индекс или часть имени) закрепляет нужный. Если на Linux список пуст,
установите `libvulkan1 libglvnd0 libegl1 libgl1 libglx0`; на машинах без
дисплея задайте `XDG_RUNTIME_DIR=/tmp`. `CMF_GPU=0` оставляет всё на CPU.

### Примеры

Черновики и переписанные тексты (`q8_2f`, сэмплинг авторов, лучший из трёх
сэмплов по строгой проверке фактов) — в разделе [Examples](#examples) выше.
В английском письме сохранены все 16 фактов, «September 30» стало «Sep 30»; в
китайском отчёте сохранены все 22 факта, пропали разделители тысяч. В
остальных четырёх сэмплах каждый отмеченный проверкой факт остался на месте в
другом формате («$186K», «Sept 30», «2026.10.8»).

### Ограничения

- Только текст: зрительная и звуковая части базовой Gemma 4 не включены.
- Модель всё же может изменить деталь. Перед отправкой вычитайте числа, даты и
  имена. Форматирование (абзацы, списки, заголовки) тоже может измениться.
- Иногда отвечает на английский черновик по-китайски — сэмплируйте заново.
- По наблюдению авторов, в разговорных жанрах модель иногда добавляет сленг или
  ругательства, которых не было в черновике.
- Авторы задумали её как инструмент для ваших собственных черновиков. Если у
  учебного заведения, работодателя или издания есть правила об использовании
  ИИ, соблюдайте их.

### Проверка загрузки

```bash
hf download infosave/humanizer-cmf humanizer-12b-q8_2f.cmf.sha256 --local-dir .
sha256sum -c humanizer-12b-q8_2f.cmf.sha256      # macOS: shasum -a 256 -c humanizer-12b-q8_2f.cmf.sha256
cortiq verify humanizer-12b-q8_2f.cmf
cortiq info humanizer-12b-q8_2f.cmf               # сэмплинг по умолчанию
```

### Лицензия и авторство

Apache-2.0, как у оригинала. humanizer © 2026 Stephen Yu
([jialinyyzz/humanizer](https://huggingface.co/jialinyyzz/humanizer),
[GitHub](https://github.com/sgaofen/humanizer-local-model)), дообучен из
[google/gemma-4-12B](https://huggingface.co/google/gemma-4-12B), которую Google
распространяет под Apache-2.0. Эти файлы — bf16-веса авторов, переведённые в
CMF, без зрительной и звуковой частей. Цифры качества и скорости на этой
странице — наши; оценки, приписанные авторам, — их. Репозиторий не связан с
авторами humanizer и с Google и не одобрен ими.

---

## 中文文档

jialinyyzz 的 [humanizer](https://huggingface.co/jialinyyzz/humanizer) 用于改写
AI 写的草稿（邮件、文章、报告、论坛帖子；英文和中文），让它读起来像人写的；模型经过训练，会保留每个数字、单位、日期、名称和引文。它基于 Google 的
[Gemma 4 12B](https://huggingface.co/google/gemma-4-12B) 微调。本仓库把其中的文本模型打包为独立的 [CMF](https://github.com/infosave2007/cmf) 文件，由 `cortiq` 运行——一个不依赖 Python 和机器学习框架的 Rust 推理引擎：NVIDIA、AMD、Intel 显卡走 Vulkan/DX12，
Apple silicon 走 Metal，也可在 CPU 上运行。

原作者提供：[模型卡与评测](https://huggingface.co/jialinyyzz/humanizer#evaluation-details) ·
[Mac 与 Windows 桌面应用](https://github.com/sgaofen/humanizer-local-model/releases/latest) ·
[llama.cpp 用 GGUF 文件](https://huggingface.co/jialinyyzz/humanizer-GGUF) ·
[源码与文档](https://github.com/sgaofen/humanizer-local-model)。据作者报告，在最严格设置下
Originality.ai 将 210 篇英文改写中的 95% 判定为人写；严格的 LLM 评审在 420 篇英文改写中有 376 篇未发现事实问题。这些是作者在其自身文件上的测量，而非本仓库的 CMF 文件，方法与注意事项见
[作者的评测](https://huggingface.co/jialinyyzz/humanizer#evaluation-details)。

### 快速开始

```bash
cargo install cortiq-cli
hf download infosave/humanizer-cmf humanizer-12b-q8_2f.cmf --local-dir .
cortiq run humanizer-12b-q8_2f.cmf --max-tokens 2048 --prompt "$(cat draft.txt)"
```

默认文件为 `humanizer-12b-q8_2f.cmf`（11.97 GB）：RTX PRO 4000 上 31.7 tok/s，M4 Mac mini
上 6.6 tok/s（Metal）。`humanizer-12b-q4tp-ffn.cmf`（7.89 GB）可放入 12 GB 显存，在 RTX PRO 4000 上达到 41.6 tok/s，质量略低（见[文件](#文件)）。

需要 cortiq 0.8.16 或更新版本；Linux、macOS、Windows 预编译二进制见
[发布页面](https://github.com/infosave2007/cmf/releases)。直接传入草稿即可：文件会自动套上作者的提示词，并使用作者的采样设置。请把 `--max-tokens` 设得足够大（默认 256）：改写长度与草稿相当（我们测试中 957 token 的草稿改写为 932 token）。

### 文件

| 文件 | 权重 | 大小 | 与 bf16 的 top-1 一致率 | KL | 困惑度 | GPU 显存峰值 | 适用场景 |
|---|---|---:|---:|---:|---:|---:|---|
| `humanizer-12b-q8_2f.cmf` | 8 位 | 11.97 GB | 99.4% | 0.0008 | 2.1174 | 15.3 GB | **默认**：最接近原模型；Apple silicon；16 GB 及以上显存 |
| `humanizer-12b-q4tp-ffn.cmf` | FFN 4 位，注意力与嵌入 8 位 | 7.89 GB | 95.5% | 0.020 | 2.1555 | 10.3 GB | RTX PRO 4000 上解码快 31%；12 GB 显存即可 |

质量以作者的 bf16 权重（HF transformers）为参照，在作者自己用 bf16 对六篇草稿（四篇英文、两篇中文）生成的改写上共 2,051 个 token 测得：文件最优预测与 bf16 一致的位置比例、相对 bf16
的平均 KL 散度，以及困惑度（bf16 为 2.1168）。另一组 2,127 个 token 的样本给出 98.9%（`q8_2f`）与 95.6%（`q4tp-ffn`）的 top-1。在三篇草稿的贪心解码事实核查中，`q8_2f` 原样保留了全部事实，`q4tp-ffn`
把一个金额写成了另一种格式（“$186,000”写作“$186k”）。显存峰值是在 RTX PRO 4000 上
`cortiq bench` 处理 1,000 token 提示词时用 nvidia-smi 读取的；Apple silicon 与 CPU 上两个文件均在 24 GB 的
Mac mini 上运行。

### 性能

`cortiq bench --core --ignore-eos`，单流，cortiq 0.8.16；RTX 各行为三次运行的中位数：

| 硬件 | 后端 | 文件 | 解码 tok/s | 1k token 处解码 | 1k token 提示词 tok/s | 首 token 时间 |
|---|---|---|---:|---:|---:|---:|
| RTX PRO 4000 Blackwell（24 GB） | Vulkan | `q4tp-ffn` | 41.6 | 36.3 | 137 | 6.9 s |
| RTX PRO 4000 Blackwell（24 GB） | Vulkan | `q8_2f` | 31.7 | 28.3 | 90.2 | 10.8 s |
| Mac mini M4（24 GB） | Metal | `q8_2f` | 6.6 | 6.0 | 68 | 14.7 s |
| Mac mini M4（24 GB） | Metal | `q4tp-ffn` | 8.5 | 8.3 | 72 | 14.0 s |
| Mac mini M4（24 GB） | CPU | `q8_2f` | 4.7 | | | |
| Mac mini M4（24 GB） | CPU | `q4tp-ffn` | 4.6 | | | |

读取提示词也会用到 CPU；RTX 测试机约有 10 个核心。通过 `cortiq serve` 完整改写：

| 硬件 | 文件 | 957 token 草稿：首 token | 完整改写 | 改写速度 | 邮件，输入 362 token |
|---|---|---:|---:|---:|---:|
| RTX PRO 4000 Blackwell | `q4tp-ffn` | 7.5 s | 901 token，37.1 s | 30.1 tok/s | 10.6 s |
| RTX PRO 4000 Blackwell | `q8_2f` | 11.2 s | 932 token，48.9 s | 24.7 tok/s | 14.8 s |

RTX 各行使用作者的采样设置（三次中位数），Mac 各行为贪心解码。`cortiq run` 每次启动都要把权重载入 GPU（RTX PRO 4000 上约 13 s），处理多篇草稿时请保持 `cortiq serve` 运行。
Apple silicon 上请用 `q8_2f`：4 位文件在那里并不更快，读取提示词还更慢。

### 使用

**提示词与采样。** humanizer 是文本续写模型，不是对话模型。每个文件都把作者的提示词存为模板：
`cortiq run --prompt` 和 `/v1/chat/completions` 接收草稿，依次拼出指令、一个空行、去掉首尾空白的草稿以及 `### Rewritten:`，与作者的格式逐 token 一致。每次请求一篇草稿；系统消息和之前的对话轮次会被忽略。生成在序列结束 token 处停止，不要添加停止字符串，尤其不要用 `"###"`。

文件中也记录了作者的采样设置：temperature 1.0、top-p 0.95、关闭 top-k 与 min-p、
repetition penalty 1.0。`run` 和 `serve` 以此为默认，命令行参数和请求字段可以覆盖，
`cortiq info` 可查看。

```bash
cortiq run humanizer-12b-q8_2f.cmf --max-tokens 2048 --prompt "$(cat draft.txt)"              # 作者的采样设置
cortiq run humanizer-12b-q8_2f.cmf --max-tokens 2048 --seed 7 --prompt "$(cat draft.txt)"     # 可复现的采样
cortiq run humanizer-12b-q8_2f.cmf --max-tokens 2048 --greedy --prompt "$(cat draft.txt)"     # 确定性输出
```

据作者观察，对于含技术术语的简短非正式英文草稿，模型偶尔会把整篇改写写成中文；作者的应用会检查语言并重新采样。你也可以这样做：不固定 seed 再运行一次。作者为指令、草稿和改写合计预留 8,192 个
token 的上下文；更长的文档请按段落拆分。

**OpenAI 兼容服务：**

```bash
cortiq serve humanizer-12b-q8_2f.cmf --port 8080

jq -n --rawfile d draft.txt '{model: "humanizer", max_tokens: 2048,
                              messages: [{role: "user", content: $d}]}' \
  | curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d @- \
  | jq -r '.choices[0].message.content'
```

请始终传入 `max_tokens`：服务器默认只生成 256 个 token，且不读取 `max_completion_tokens`。请求中的 `temperature`、`top_p`、`top_k`、`min_p`、`repetition_penalty`（或
`repeat_penalty`）和 `seed` 会覆盖文件的采样设置。`/v1/completions` 也接受纯文本形式的作者完整提示词（与 llama.cpp 的 `/completion` 相同），模型收到的 token 与对话接口一致。同一端口还提供
`/v1/models` 和网页控制台；`--host 127.0.0.1` 使服务仅限本机访问。

**选择 GPU（Vulkan）。** `cortiq gpu` 列出显卡；`CMF_GPU_ADAPTER=1`（序号或名称片段）指定其中一块。若 Linux 上列表为空，请安装 `libvulkan1 libglvnd0 libegl1 libgl1 libglx0`；无显示器的机器请设置 `XDG_RUNTIME_DIR=/tmp`。`CMF_GPU=0` 只用 CPU。

### 示例

草稿与改写（`q8_2f`，作者的采样设置，按严格的事实核查从三个样本中选出最佳）见上文
[Examples](#examples)。英文邮件保留了全部 16 项事实，“September 30”写成了“Sep 30”；中文报告保留了全部 22 项事实，去掉了千位分隔符。其余四个样本中，核查标出的每项事实都仍在，只是换了格式（“$186K”“Sept 30”“2026.10.8”）。

### 局限

- 仅文本：未包含 Gemma 4 基座的视觉与音频部分。
- 仍可能改动个别细节。发送前请核对数字、日期和名称。格式（分段、列表、标题）也可能变化。
- 偶尔会用中文改写英文草稿——重新采样即可。
- 据作者观察，在口语化体裁中它有时会加入草稿里没有的俚语或粗话。
- 作者将其定位为改写你自己草稿的写作工具。学校、雇主或出版方对使用 AI 有规定的，请遵守。

### 校验下载

```bash
hf download infosave/humanizer-cmf humanizer-12b-q8_2f.cmf.sha256 --local-dir .
sha256sum -c humanizer-12b-q8_2f.cmf.sha256      # macOS：shasum -a 256 -c humanizer-12b-q8_2f.cmf.sha256
cortiq verify humanizer-12b-q8_2f.cmf
cortiq info humanizer-12b-q8_2f.cmf               # 显示默认采样设置
```

### 许可与署名

与原模型相同，采用 Apache-2.0。humanizer © 2026 Stephen Yu
（[jialinyyzz/humanizer](https://huggingface.co/jialinyyzz/humanizer)、
[GitHub](https://github.com/sgaofen/humanizer-local-model)），基于 Google 以 Apache-2.0
发布的 [google/gemma-4-12B](https://huggingface.co/google/gemma-4-12B) 微调。这些文件是把作者的 bf16 权重转换为 CMF，并去掉了视觉与音频部分。本页的质量与速度数据由我们测得；注明出自作者的评测数据归作者所有。本仓库与 humanizer 作者及 Google 均无关联，也未获其认可。
