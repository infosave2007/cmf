---
license: other
license_name: qwen
license_link: https://huggingface.co/orcarouter/OrcaCyber-Nano-1.0
library_name: cortiq
base_model: orcarouter/OrcaCyber-Nano-1.0
base_model_relation: quantized
pipeline_tag: text-generation
tags:
  - cmf
  - cortiq
  - quantized
  - q4tp
  - cybersecurity
  - qwen3.8
  - long-context
  - low-vram
  - vulkan
language:
  - en
  - ru
  - zh
---

# OrcaCyber-Nano-1.0 Turbo — CMF

The 4-bit weights of [OrcaCyber-Nano-1.0](https://huggingface.co/orcarouter/OrcaCyber-Nano-1.0), OrcaRouter's
27B model for defensive security work (Qwen3.8-27B architecture), under the header of
[Qwen3.8-27B Turbo](https://huggingface.co/infosave/Qwen3.8-27B-Turbo-cmf), which holds device memory in place on
long prompts. On a 16 GB card it reads 64 000 tokens in 14.8 GB of device memory and decodes there at
26.0 tok/s; the standard file runs out of device memory from 16 000 tokens on that card. The model's own MTP head drafts tokens: about twice the plain
speed on code. One [CMF](https://github.com/infosave2007/cmf) file for `cortiq`, a Rust inference engine with no
Python and no ML framework.

[English](#quick-start) · [Русский](#документация-на-русском) · [中文](#中文说明)

## Quick start

```bash
cargo install cortiq-cli
hf download infosave/OrcaCyber-Nano-1.0-Turbo-cmf orcacyber-nano-1.0-turbo.cmf --local-dir .
cortiq run orcacyber-nano-1.0-turbo.cmf --prompt-file incident.txt --no-think
```

Measured with cortiq built from [infosave2007/cmf#99](https://github.com/infosave2007/cmf/pull/99), which comes in the next release; cortiq 0.8.63 runs the same file and reads prompts at about 75 tok/s instead of 98–112. `--no-think` asks for a direct answer; without it the model reasons first, so raise `--max-tokens`
(default 256). Each `cortiq run` loads the model again (about 8 s for 14 GB on the card measured); for chat keep it
loaded with the server:

```bash
cortiq serve orcacyber-nano-1.0-turbo.cmf --host 127.0.0.1 --port 8080

curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "orcacyber-nano-1.0-turbo", "enable_thinking": false, "temperature": 0,
       "messages": [{"role": "user", "content": "Is strcpy(b, s) into char b[16] safe? Fix it."}]}'
```

Endpoints: `/v1/chat/completions` (streaming, `tools`, `enable_thinking`), `/v1/completions`, `/v1/models`,
`/healthz`. Without `--host` the server listens on all interfaces.

| file | quantization | size | sha256 |
|---|---|---:|---|
| `orcacyber-nano-1.0-turbo.cmf` | 4-bit (q4tp), text, MTP head | 14.3 GB | `6ff230576c09672102f1f0dbc6020721bc1bad548aa4df9bb31e3c7d1251d5f7` |

## Performance

![Standard and turbo](https://huggingface.co/infosave/OrcaCyber-Nano-1.0-Turbo-cmf/resolve/main/assets/compare_en.png)

NVIDIA RTX 2000 Ada (16 GB, 70 W), Vulkan, `cortiq bench --core`, 60 tokens after a prompt of that length.
Standard is the same weights without the turbo header.

| prompt, tokens | first token, s (standard) | first token, s (**turbo**) | decode, tok/s (standard) | decode, tok/s (**turbo**) | peak VRAM, GB (standard) | peak VRAM, GB (**turbo**) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 000 | 9.1 | **8.9** | 27.4 | **29.9** | 14.1 | **14.1** |
| 4 000 | 37.0 | **36.7** | 25.8 | **26.7** | 14.5 | **14.5** |
| 16 000 | does not fit | **159** | — | **26.7** | — | **14.8** |
| 32 000 | does not fit | **324** | — | **26.3** | — | **14.8** |
| 64 000 | does not fit | **652** | — | **26.0** | — | **14.8** |

The first token includes moving the weights to the card (about 5 s); the prompt itself is read at 98–112 tok/s. `bench` decodes with the MTP drafts on. The standard file runs out of device memory from 16 000 tokens on this 16 GB card. Host RAM of the turbo file while a long prompt is read: 4.1 GB at 16 000, 7.1 GB at 32 000, 13.3 GB at 64 000 (`--o1-spill` moves it to a disk).

Real answers, one per process, 220 tokens, greedy: without drafts against the MTP head's drafts.

| prompt | no drafts, tok/s | MTP drafts, tok/s | drafts kept |
|---|---:|---:|---:|
| a Rust function with tests | 13.5 | 28.2 | 76 % |
| a security review of a Flask handler | 13.4 | 26.5 | 70 % |
| triage of auth log lines | 13.3 | 23.6 | 64 % |
| two paragraphs of prose | 13.5 | 19.6 | 48 % |
| a Russian answer | 13.5 | 19.5 | 48 % |

A real answer: a one-paragraph summary of 100 000 characters of Wikipedia (about 24 600 tokens), correct (the Second Battle of Naktong Bulge), first token 252 s, decode 20.8 tok/s.

![A long prompt on a small card](https://huggingface.co/infosave/OrcaCyber-Nano-1.0-Turbo-cmf/resolve/main/assets/ceiling_en.png)

**A fixed ceiling.** On a 16 GB card the first 4096 rows of a prompt are read exactly (8192 on a larger card). The
keys and values of later rows leave the card as they are read, for RAM or an SSD (`--o1-spill`). Before the last
512 rows the far field is built from them one layer at a time: the first 4 rows and the last 1800 exact, the rest in
32 landmarks per head, and the 248 rows that score highest by what the layers do with them (pass keys, codes,
names) stay exact. Patent application No. 19/738,763.

**Quality.** WikiText-2, 12 windows of 512 tokens: ppl 8.89 (8.79 for Qwen3.8-27B Turbo in the same
harness); Rust source: 3.79. Facts planted at random places of 100 000 characters (a pass key, an access code, a password, a room number): 6 of 6 with the exact rows; a far field without them finds 0 of 2.

An answer of the file, `--greedy --no-think`:

> **What does this C function do, and is it safe? `void f(char *s){char b[16]; strcpy(b,s); puts(b);}`**
>
> This function copies a string into a fixed-size local buffer and then prints it.
>
> **Safety: No, it is not safe.**
>
> The core issue is `strcpy(b, s)`. The destination buffer `b` is only 16 bytes, but `strcpy` does not check the
> length of `s`. If `s` is 16 characters or longer (including the null terminator), `strcpy` will write past the end
> of `b`, causing a **buffer overflow**. This can corrupt the stack, leading to crashes or potential security
> vulnerabilities (e.g., stack smashing).
>
> **Fix:** Use `strncpy` or `snprintf` to limit the copy to 15 characters plus a null terminator, or check the
> length of `s` before copying. For example:
>
> ```c
> void f(char *s) {
>     char b[16];
>     strncpy(b, s, 15);
>     b[15] = '\0';
>     puts(b);
> }
> ```
>
> This ensures `b` is always null-terminated and never overflows.

## Options

| option | meaning |
|---|---|
| `--o1-spill DIR` | a long prompt's rows in files in `DIR`, also `CMF_O1_SPILL` |
| `--prompt-file FILE` | a prompt longer than a command line holds |
| `CMF_GRAPH_SPEC=0` | no draft tokens |
| `CMF_O1=off` | the plain model, with exact attention at any length |

## Limits

- Text only: OrcaCyber's vision projector (`mmproj`) and its DFlash2 drafter are not in the file. On a 16 GB card
  the drafter has no room beside the trunk, and the MTP head is the faster drafter there.
- OrcaCyber-Nano-1.0 is published only as GGUF. These weights are the Mix-15.7G file (an imatrix IQ4_XS/Q5_K/Q6_K
  mix) decoded and re-encoded as q4tp, so they carry both roundings.
- Past the seal the text is approximate: what lies far back fades, apart from the rows kept exact.
- Measured on one card, an RTX 2000 Ada. The far field runs on Vulkan, DX12 and Metal; on the CPU the file runs
  as the standard one.

## How the file was made

```bash
cortiq import-gguf OrcaCyber-Nano-1.0-AI-Defense-Mix-15.7G.gguf --quant q4tp \
  --tokenizer-dir qwen38-tokenizer --output orcacyber-nano-1.0-q4tp.cmf
cortiq far-turbo orcacyber-nano-1.0-q4tp.cmf orcacyber-nano-1.0-turbo.cmf --exact-weights keep27.bin
```

The GGUF comes from the source page once its terms are accepted. `qwen38-tokenizer` holds Qwen3.8-27B's `tokenizer.json` (the GGUF's vocabulary is the same; the file keeps
OrcaCyber's chat template). `keep27.bin`, the exact-row weights fitted on Qwen3.8-27B, is in the
[repository](https://github.com/infosave2007/cmf/tree/master/scripts/dory/router).

## License

Weights: OrcaCyber-Nano-1.0 by [OrcaRouter](https://huggingface.co/orcarouter/OrcaCyber-Nano-1.0), published under
the license it names "qwen"; it is built on [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache 2.0).
Read the source page before use. The far field is a subject of U.S. patent application No. 19/738,763; see
[PATENTS.md](https://github.com/infosave2007/cmf/blob/master/PATENTS.md).

---

## Документация на русском

4-битные веса [OrcaCyber-Nano-1.0](https://huggingface.co/orcarouter/OrcaCyber-Nano-1.0), модели OrcaRouter на 27B
для защитной кибербезопасности (архитектура Qwen3.8-27B), с заголовком
[Qwen3.8-27B Turbo](https://huggingface.co/infosave/Qwen3.8-27B-Turbo-cmf), который держит память карты на месте при
длинных промптах. На карте 16 ГБ файл читает 64 000 токенов в 14,8 ГБ памяти карты и декодирует там
26,0 ток/с; обычный файл не помещается на карту уже с 16 000 токенов. Собственная MTP-голова модели пишет черновые токены: на коде это
примерно вдвое быстрее обычного декода. Один [CMF](https://github.com/infosave2007/cmf)-файл для `cortiq`, движка на
Rust без Python и ML-фреймворков.

### Быстрый старт

```bash
cargo install cortiq-cli
hf download infosave/OrcaCyber-Nano-1.0-Turbo-cmf orcacyber-nano-1.0-turbo.cmf --local-dir .
cortiq run orcacyber-nano-1.0-turbo.cmf --prompt-file incident.txt --no-think
```

Замерено на cortiq, собранном из [infosave2007/cmf#99](https://github.com/infosave2007/cmf/pull/99), он войдёт в следующий релиз; cortiq 0.8.63 запускает тот же файл и читает промпт примерно со скоростью 75 ток/с вместо 98–112. `--no-think` просит прямой ответ; без него модель сначала рассуждает, так что поднимите `--max-tokens`
(по умолчанию 256). Каждый `cortiq run` загружает модель заново (около 8 с на 14 ГБ на замеренной карте); для чата
держите её загруженной в сервере:

```bash
cortiq serve orcacyber-nano-1.0-turbo.cmf --host 127.0.0.1 --port 8080
```

Эндпоинты: `/v1/chat/completions` (стриминг, `tools`, `enable_thinking`), `/v1/completions`, `/v1/models`,
`/healthz`. Без `--host` сервер слушает все интерфейсы.

| файл | квантизация | размер | sha256 |
|---|---|---:|---|
| `orcacyber-nano-1.0-turbo.cmf` | 4 бита (q4tp), текст, MTP-голова | 14,3 ГБ | `6ff230576c09672102f1f0dbc6020721bc1bad548aa4df9bb31e3c7d1251d5f7` |

### Производительность

![Обычный и турбо](https://huggingface.co/infosave/OrcaCyber-Nano-1.0-Turbo-cmf/resolve/main/assets/compare_ru.png)

NVIDIA RTX 2000 Ada (16 ГБ, 70 Вт), Vulkan, `cortiq bench --core`, 60 токенов после промпта такой длины. Обычный —
те же веса без турбо-заголовка.

| промпт, токенов | первый токен, с (обычный) | первый токен, с (**турбо**) | декод, ток/с (обычный) | декод, ток/с (**турбо**) | пик памяти карты, ГБ (обычный) | пик памяти карты, ГБ (**турбо**) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 000 | 9,1 | **8,9** | 27,4 | **29,9** | 14,1 | **14,1** |
| 4 000 | 37,0 | **36,7** | 25,8 | **26,7** | 14,5 | **14,5** |
| 16 000 | не помещается | **159** | — | **26,7** | — | **14,8** |
| 32 000 | не помещается | **324** | — | **26,3** | — | **14,8** |
| 64 000 | не помещается | **652** | — | **26,0** | — | **14,8** |

Первый токен включает перенос весов на карту (около 5 с); сам промпт читается со скоростью 98–112 ток/с. `bench` декодирует с черновиками MTP. Обычный файл с 16 000 токенов не помещается на эту карту 16 ГБ. RAM хоста у турбо-файла при чтении длинного промпта: 4,1 ГБ на 16 000, 7,1 ГБ на 32 000, 13,3 ГБ на 64 000 (`--o1-spill` переносит их на диск).

Реальные ответы, по одному на процесс, 220 токенов, жадно: без черновиков и с черновиками MTP-головы.

| промпт | без черновиков, ток/с | черновики MTP, ток/с | принято черновиков |
|---|---:|---:|---:|
| функция на Rust с тестами | 13,5 | 28,2 | 76 % |
| ревью безопасности обработчика Flask | 13,4 | 26,5 | 70 % |
| разбор строк auth-лога | 13,3 | 23,6 | 64 % |
| два абзаца прозы | 13,5 | 19,6 | 48 % |
| ответ на русском | 13,5 | 19,5 | 48 % |

Реальный ответ: краткое содержание 100 000 символов Википедии одним абзацем (около 24 600 токенов), верное (Вторая битва за выступ Нактонг), первый токен через 252 с, декод 20,8 ток/с.

![Длинный промпт на маленькой карте](https://huggingface.co/infosave/OrcaCyber-Nano-1.0-Turbo-cmf/resolve/main/assets/ceiling_ru.png)

**Постоянный потолок.** На карте 16 ГБ первые 4096 строк промпта читаются точно (8192 на карте побольше). Ключи и
значения следующих строк уходят с карты по мере чтения, в RAM или на SSD (`--o1-spill`). До последних 512 строк из
них слой за слоем строится дальнее поле: первые 4 и последние 1800 строк точные, остальное в 32 опорных точках на
голову, а 248 строк с наибольшей оценкой по тому, что с ними делают слои (ключи, коды, имена), остаются точными.
Патентная заявка № 19/738,763.

**Качество.** WikiText-2, 12 окон по 512 токенов: ppl 8,89 (8,79 у Qwen3.8-27B Turbo в том же замере);
исходники на Rust: 3,79. Факты в случайных местах 100 000 символов (ключ, код доступа, пароль, номер комнаты): 6 из 6 с точными строками; дальнее поле без них находит 0 из 2.

Ответ файла, `--greedy --no-think`:

> **What does this C function do, and is it safe? `void f(char *s){char b[16]; strcpy(b,s); puts(b);}`**
>
> This function copies a string into a fixed-size local buffer and then prints it.
>
> **Safety: No, it is not safe.**
>
> The core issue is `strcpy(b, s)`. The destination buffer `b` is only 16 bytes, but `strcpy` does not check the
> length of `s`. If `s` is 16 characters or longer (including the null terminator), `strcpy` will write past the end
> of `b`, causing a **buffer overflow**. This can corrupt the stack, leading to crashes or potential security
> vulnerabilities (e.g., stack smashing).
>
> **Fix:** Use `strncpy` or `snprintf` to limit the copy to 15 characters plus a null terminator, or check the
> length of `s` before copying. For example:
>
> ```c
> void f(char *s) {
>     char b[16];
>     strncpy(b, s, 15);
>     b[15] = '\0';
>     puts(b);
> }
> ```
>
> This ensures `b` is always null-terminated and never overflows.

### Опции

| опция | смысл |
|---|---|
| `--o1-spill DIR` | строки длинного промпта в файлах в `DIR`, также `CMF_O1_SPILL` |
| `--prompt-file FILE` | промпт длиннее командной строки |
| `CMF_GRAPH_SPEC=0` | без черновых токенов |
| `CMF_O1=off` | обычная модель с точным вниманием на любой длине |

### Ограничения

- Только текст: проектора зрения OrcaCyber (`mmproj`) и её драфтера DFlash2 в файле нет. На карте 16 ГБ драфтеру
  нет места рядом со стволом, а MTP-голова там быстрее.
- OrcaCyber-Nano-1.0 опубликована только в GGUF. Эти веса — файл Mix-15.7G (imatrix-смесь IQ4_XS/Q5_K/Q6_K),
  раскодированный и перекодированный в q4tp, так что в них оба округления.
- После печати текст приблизителен: далёкое прошлое тускнеет, кроме строк, оставленных точными.
- Замерено на одной карте, RTX 2000 Ada. Дальнее поле работает на Vulkan, DX12 и Metal; на CPU файл идёт как обычный.

### Как сделан файл

```bash
cortiq import-gguf OrcaCyber-Nano-1.0-AI-Defense-Mix-15.7G.gguf --quant q4tp \
  --tokenizer-dir qwen38-tokenizer --output orcacyber-nano-1.0-q4tp.cmf
cortiq far-turbo orcacyber-nano-1.0-q4tp.cmf orcacyber-nano-1.0-turbo.cmf --exact-weights keep27.bin
```

GGUF скачивается со страницы источника после принятия условий. `qwen38-tokenizer` содержит `tokenizer.json`
Qwen3.8-27B (словарь у GGUF тот же; шаблон чата в файле — OrcaCyber). `keep27.bin` — веса точных строк, подобранные на
Qwen3.8-27B, лежат в [репозитории](https://github.com/infosave2007/cmf/tree/master/scripts/dory/router).

### Лицензия

Веса: OrcaCyber-Nano-1.0 от [OrcaRouter](https://huggingface.co/orcarouter/OrcaCyber-Nano-1.0) под лицензией, которую
источник называет «qwen»; модель построена на [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache 2.0).
Перед использованием прочтите страницу источника. Дальнее поле — предмет патентной заявки США № 19/738,763; см.
[PATENTS.md](https://github.com/infosave2007/cmf/blob/master/PATENTS.md).

---

## 中文说明

[OrcaCyber-Nano-1.0](https://huggingface.co/orcarouter/OrcaCyber-Nano-1.0) 的 4 位权重。这是 OrcaRouter 面向防御性网络安全工作的
27B 模型（Qwen3.8-27B 架构），配上 [Qwen3.8-27B Turbo](https://huggingface.co/infosave/Qwen3.8-27B-Turbo-cmf) 的头部：长提示下显存保持不变。
在 16 GB 显卡上读取 64 000 token 时显存为 14.8 GB，此时解码 26.0 tok/s；普通文件从 16 000 token 起就放不进显卡。模型自带的 MTP 头生成草稿
token：在代码上约为普通解码的两倍。一个 [CMF](https://github.com/infosave2007/cmf) 文件，供 `cortiq` 使用；`cortiq` 是一个
Rust 推理引擎，不需要 Python，也不需要机器学习框架。

### 快速开始

```bash
cargo install cortiq-cli
hf download infosave/OrcaCyber-Nano-1.0-Turbo-cmf orcacyber-nano-1.0-turbo.cmf --local-dir .
cortiq run orcacyber-nano-1.0-turbo.cmf --prompt-file incident.txt --no-think
```

测量所用的 cortiq 由 [infosave2007/cmf#99](https://github.com/infosave2007/cmf/pull/99) 构建，将包含在下一个版本中；cortiq 0.8.63 也能运行同一文件，但读取提示约为 75 tok/s，而不是 98–112。 `--no-think` 要求直接回答；不加时模型会先推理，请调大 `--max-tokens`（默认 256）。每次 `cortiq run` 都会重新加载模型
（在实测显卡上 14 GB 约需 8 秒）；聊天请用服务器常驻：

```bash
cortiq serve orcacyber-nano-1.0-turbo.cmf --host 127.0.0.1 --port 8080
```

接口：`/v1/chat/completions`（流式、`tools`、`enable_thinking`）、`/v1/completions`、`/v1/models`、`/healthz`。不加 `--host` 时服务器监听所有网卡。

| 文件 | 量化 | 大小 | sha256 |
|---|---|---:|---|
| `orcacyber-nano-1.0-turbo.cmf` | 4 位（q4tp），文本，MTP 头 | 14.3 GB | `6ff230576c09672102f1f0dbc6020721bc1bad548aa4df9bb31e3c7d1251d5f7` |

### 性能

![普通与 Turbo](https://huggingface.co/infosave/OrcaCyber-Nano-1.0-Turbo-cmf/resolve/main/assets/compare_zh.png)

NVIDIA RTX 2000 Ada（16 GB，70 W），Vulkan，`cortiq bench --core`，在该长度的提示之后生成 60 个 token。普通文件是同样的权重，不带 turbo 头部。

| 提示 token 数 | 首个 token，秒（普通） | 首个 token，秒（turbo） | 解码 tok/s（普通） | 解码 tok/s（turbo） | 显存峰值 GB（普通） | 显存峰值 GB（turbo） |
|---:|---:|---:|---:|---:|---:|---:|
| 1 000 | 9.1 | 8.9 | 27.4 | 29.9 | 14.1 | 14.1 |
| 4 000 | 37.0 | 36.7 | 25.8 | 26.7 | 14.5 | 14.5 |
| 16 000 | 放不下 | 159 | — | 26.7 | — | 14.8 |
| 32 000 | 放不下 | 324 | — | 26.3 | — | 14.8 |
| 64 000 | 放不下 | 652 | — | 26.0 | — | 14.8 |

首个 token 包含把权重搬到显卡上的时间（约 5 秒）；提示本身的读取速度为 98–112 tok/s。`bench` 解码时开启 MTP 草稿。在这块 16 GB 显卡上，普通文件从 16 000 token 起显存不足。读取长提示时 turbo 文件占用的主机内存：16 000 时 4.1 GB，32 000 时 7.1 GB，64 000 时 13.3 GB（`--o1-spill` 可移到磁盘）。

真实回答，每个进程一个，220 个 token，贪心解码：不用草稿与使用 MTP 头的草稿。

| 提示 | 无草稿 tok/s | MTP 草稿 tok/s | 草稿接受率 |
|---|---:|---:|---:|
| 带测试的 Rust 函数 | 13.5 | 28.2 | 76% |
| Flask 处理函数的安全审查 | 13.4 | 26.5 | 70% |
| 认证日志分诊 | 13.3 | 23.6 | 64% |
| 两段散文 | 13.5 | 19.6 | 48% |
| 俄语回答 | 13.5 | 19.5 | 48% |

真实回答：用一段话概括 100 000 字符的维基百科（约 24 600 token），回答正确（第二次洛东江突出部战役），首个 token 252 秒，解码 20.8 tok/s。

![小显卡上的长提示](https://huggingface.co/infosave/OrcaCyber-Nano-1.0-Turbo-cmf/resolve/main/assets/ceiling_zh.png)

固定上限：在 16 GB 显卡上，提示的前 4096 行精确读取（更大的显卡为 8192）。之后各行的键和值一读完就离开显卡，进入内存或 SSD（`--o1-spill`）。
在最后 512 行之前，由它们逐层构建远场：前 4 行和后 1800 行精确，其余每个头用 32 个地标表示；按各层对其处理打分最高的 248 行
（密钥、编码、名字）保持精确。专利申请号 19/738,763。

质量：WikiText-2，12 个 512 token 窗口，ppl 8.89（同一测法下 Qwen3.8-27B Turbo 为 8.79）；Rust 源码：3.79。在 100 000 字符的随机位置插入事实（密钥、访问码、密码、房间号）：有精确行时 6/6；没有精确行的远场为 0/2。

文件的一个回答，`--greedy --no-think`：

> **What does this C function do, and is it safe? `void f(char *s){char b[16]; strcpy(b,s); puts(b);}`**
>
> This function copies a string into a fixed-size local buffer and then prints it.
>
> **Safety: No, it is not safe.**
>
> The core issue is `strcpy(b, s)`. The destination buffer `b` is only 16 bytes, but `strcpy` does not check the
> length of `s`. If `s` is 16 characters or longer (including the null terminator), `strcpy` will write past the end
> of `b`, causing a **buffer overflow**. This can corrupt the stack, leading to crashes or potential security
> vulnerabilities (e.g., stack smashing).
>
> **Fix:** Use `strncpy` or `snprintf` to limit the copy to 15 characters plus a null terminator, or check the
> length of `s` before copying. For example:
>
> ```c
> void f(char *s) {
>     char b[16];
>     strncpy(b, s, 15);
>     b[15] = '\0';
>     puts(b);
> }
> ```
>
> This ensures `b` is always null-terminated and never overflows.

### 选项

| 选项 | 含义 |
|---|---|
| `--o1-spill DIR` | 长提示的行存放在 `DIR` 的文件中，也可用 `CMF_O1_SPILL` |
| `--prompt-file FILE` | 超过命令行长度的提示 |
| `CMF_GRAPH_SPEC=0` | 不使用草稿 token |
| `CMF_O1=off` | 普通模型，任何长度都用精确注意力 |

### 限制

- 只有文本：文件中不含 OrcaCyber 的视觉投影器（`mmproj`）和 DFlash2 草稿模型。在 16 GB 显卡上草稿模型没有空间与主干并存，而且那里 MTP 头更快。
- OrcaCyber-Nano-1.0 只以 GGUF 形式发布。这些权重来自 Mix-15.7G 文件（imatrix 的 IQ4_XS/Q5_K/Q6_K 混合），解码后重新编码为 q4tp，因此带有两次舍入。
- 超过封存点后文本是近似的：远处的内容会变淡，保留为精确的行除外。
- 只在一块显卡 RTX 2000 Ada 上测量。远场可在 Vulkan、DX12 和 Metal 上运行；在 CPU 上该文件按普通文件运行。

### 文件是如何制作的

```bash
cortiq import-gguf OrcaCyber-Nano-1.0-AI-Defense-Mix-15.7G.gguf --quant q4tp \
  --tokenizer-dir qwen38-tokenizer --output orcacyber-nano-1.0-q4tp.cmf
cortiq far-turbo orcacyber-nano-1.0-q4tp.cmf orcacyber-nano-1.0-turbo.cmf --exact-weights keep27.bin
```

GGUF 需在源页面接受条款后下载。`qwen38-tokenizer` 中放 Qwen3.8-27B 的 `tokenizer.json`（GGUF 的词表相同；文件保留 OrcaCyber 的聊天模板）。
`keep27.bin` 是在 Qwen3.8-27B 上拟合的精确行权重，位于[仓库](https://github.com/infosave2007/cmf/tree/master/scripts/dory/router)。

### 许可

权重：OrcaCyber-Nano-1.0，作者 [OrcaRouter](https://huggingface.co/orcarouter/OrcaCyber-Nano-1.0)，采用源页面所称的 “qwen” 许可；
该模型基于 [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)（Apache 2.0）。使用前请阅读源页面。远场是美国专利申请
No. 19/738,763 的主题；见 [PATENTS.md](https://github.com/infosave2007/cmf/blob/master/PATENTS.md)。
