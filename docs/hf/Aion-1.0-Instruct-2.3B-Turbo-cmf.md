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
  - long-context
  - low-vram
  - cuda
  - vulkan
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

# Aion-1.0-Instruct-2.3B Turbo — CMF

The 4-bit weights of [Aion-1.0-Instruct-2.3B](https://huggingface.co/microsoft/Aion-1.0-Instruct-2.3B) under a
header that holds the card's memory and the decode speed in place as the prompt grows.

- **NVIDIA L4 (24 GB), Vulkan:** a 30 000-token prompt holds the card at 3.6 GB (4.7 GB for the standard 4-bit
  file) and decodes at 80.8 tok/s against 53.1, the first token in 21.0 s against 27.4.
- **NVIDIA L4, CUDA:** decode holds at 88–90 tok/s from 16 000 to 30 000 tokens (94.5 → 79.3 for the standard
  file); the card stays at 3.0–3.7 GB, as the standard file does on CUDA.
- **MacBook Air M4, Metal:** 16 384 tokens in 0.83 GB of keys and values against 1.62 GB, decode 44.7 tok/s against 42.2.
- **Planted facts** in long Wikipedia text: 4/4 at about 13 000 tokens on CUDA and Vulkan, and at about 27 000
  tokens 4/4 on CUDA (Transformers bf16: 3/4), 2/4 on Vulkan's fixed-memory reading.

It is one [CMF](https://github.com/infosave2007/cmf) file for `cortiq`, a Rust inference engine with no Python and
no ML framework. Prompts under 8192 tokens run exact attention, as in the
[standard files](https://huggingface.co/infosave/Aion-1.0-Instruct-2.3B-cmf).

[English](#quick-start) · [Русский](#документация-на-русском) · [中文](#中文文档)

## Quick start

```bash
cargo install cortiq-cli
hf download infosave/Aion-1.0-Instruct-2.3B-Turbo-cmf Aion-1.0-Instruct-2.3B-turbo.cmf --local-dir .
cortiq run Aion-1.0-Instruct-2.3B-turbo.cmf --prompt-file report.txt --greedy
```

With CUDA: `cargo install cortiq-cli --features cuda`, then `CMF_GPU=cuda cortiq run …`. Use cortiq 0.9.9 or
later; prebuilt binaries are on the [releases page](https://github.com/infosave2007/cmf/releases).

## What the file does

**A fixed ceiling.** 21 of Aion's 28 layers attend over a 2048-token window and keep only those rows on the card.
The other 7 attend over the whole prompt. In this file, once a prompt reaches 8192 tokens, those 7 layers keep the
first 4 rows and the last 1800 exact and fold the rest into 32 landmarks per head, a far field of fixed size.
Card memory and the cost of a token stop growing with the text. Patent application No. 19/738,763.

**Exact rows.** A far field alone loses a password. Each row is scored by what the layers do with it (for each of
the 7 layers' 8 KV heads: how far the key lies from the span of the landmarks, and the value's norm; a linear fit
over the 112 ranks, made on planted facts in Wikipedia: left-out prompts rank the fact's rows within the top
0.02 %), and the 248 best stay exact. The question at the end of the prompt reads them before it is answered.

## Performance

`cortiq bench --core` after a prompt of that length; card memory is the peak `nvidia-smi` shows during the run.

### NVIDIA L4 (24 GB, Google Colab), Vulkan

| prompt | file | first token | decode, tok/s | card memory |
|---|---|---:|---:|---:|
| 16 384 | Turbo | 12.2 s | 80.9 | 3.6 GB |
| 16 384 | standard `q4tp` | 10.3 s | 64.0 | 4.0 GB |
| 30 000 | Turbo | 21.0 s | 80.8 | 3.6 GB |
| 30 000 | Turbo, `CMF_O1_WHOLE=1` | 34.1 s | 64.8 | 4.6 GB |
| 30 000 | standard `q4tp` | 27.4 s | 53.1 | 4.7 GB |

### NVIDIA L4 (24 GB, Google Colab), CUDA

| prompt | file | first token | decode, tok/s | card memory |
|---|---|---:|---:|---:|
| 16 384 | Turbo | 6.4 s | 89.5 | 3.0 GB |
| 16 384 | standard `q4tp` | 4.2 s | 94.5 | 3.0 GB |
| 30 000 | Turbo | 13.5 s | 88.5 | 3.7 GB |
| 30 000 | Turbo, `CMF_O1_WHOLE=0` | 11.5 s | 112.4 | 2.8 GB |
| 30 000 | standard `q4tp` | 10.0 s | 79.3 | 3.6 GB |

### MacBook Air M4 (24 GB), Metal

| prompt | file | first token | decode, tok/s | keys and values |
|---|---|---:|---:|---:|
| 16 384 | Turbo | 43 s | 44.7 | 0.83 GB |
| 16 384 | standard `q4tp` | 44 s | 42.2 | 1.62 GB |

### Two ways to read a long prompt

On Vulkan the prompt past 8192 tokens is read through the far field, chunk by chunk, and the rows leave
the card as they are read: memory is fixed, and what those rows compute is an approximation. On CUDA the whole
prompt is read exact and the far field is built once before its last 512 rows: memory grows with the prompt as the
standard file's does, the answer starts from exact rows. `CMF_O1_WHOLE=1` reads the prompt whole on Vulkan too
(the 30 000-token row above; planted facts 4/4 instead of 2/4), and `CMF_O1_WHOLE=0` reads it in chunks on CUDA:
2.7–2.8 GB from 16 000 to 30 000 tokens and 112 tok/s, with planted facts 3/4 and 2/4.

## Quality

Perplexity is the 4-bit file's below 8192 tokens (WikiText-2 14.61, Rust 7.83, Python 7.43; bf16 14.24 / 7.66 /
7.26). Past the seal the answer depends on what the far field and the exact rows keep: planted facts in long Wikipedia text (four kinds: a pass key, an access code, a password, a
room number), greedy:

| prompt | Transformers bf16 | standard `q4tp`, CUDA | Turbo, CUDA | Turbo, Vulkan |
|---|---:|---:|---:|---:|
| about 13 000 tokens | 4/4 | 4/4 | 4/4 | 4/4 |
| about 27 000 tokens | 3/4 | 3/4 | 4/4 | 2/4 (4/4 with `CMF_O1_WHOLE=1`) |

At 27 000 tokens Vulkan's fixed-memory reading loses a password a quarter of the way into the text and misses the
question the reference misses too; read whole, the same file answers all four, one more than the reference.

## Verify the download

| file | sha256 |
|---|---|
| `Aion-1.0-Instruct-2.3B-turbo.cmf` | `fddd9d0c18423f76c96e31e875eb54d46a0bec1ffe81d3fe93ade44752971015` |
| `keep_aion.bin` (the exact-row weights, already inside the file) | `b1ef2fb742c664f115931e3106e3ef525fea16063fa7c669ce4fb8b12acaf529` |

Weights: [MIT](https://huggingface.co/microsoft/Aion-1.0-Instruct-2.3B/blob/main/LICENSE), from
[microsoft/Aion-1.0-Instruct-2.3B](https://huggingface.co/microsoft/Aion-1.0-Instruct-2.3B).

---

## Документация на русском

4-битные веса [Aion-1.0-Instruct-2.3B](https://huggingface.co/microsoft/Aion-1.0-Instruct-2.3B) с заголовком,
который держит память карты и скорость генерации на месте по мере роста промпта.

- **NVIDIA L4 (24 ГБ), Vulkan:** промпт из 30 000 токенов держит карту на 3,6 ГБ (4,7 ГБ у стандартного
  4-битного файла), генерация после него 80,8 ток/с против 53,1, первый токен за 21,0 с против 27,4.
- **NVIDIA L4, CUDA:** генерация держится на 88–90 ток/с от 16 000 до 30 000 токенов (у стандартного файла
  94,5 → 79,3); карта остаётся на 3,0–3,7 ГБ, как и у стандартного файла на CUDA.
- **MacBook Air M4, Metal:** 16 384 токена в 0,83 ГБ ключей и значений против 1,62 ГБ, генерация 44,7 ток/с против 42,2.
- **Спрятанные факты** в длинном тексте Википедии: 4/4 примерно на 13 000 токенов на CUDA и Vulkan; примерно на
  27 000 — 4/4 на CUDA (Transformers bf16 — 3/4) и 2/4 на Vulkan при чтении с фиксированной памятью.

Один файл [CMF](https://github.com/infosave2007/cmf) для `cortiq` — движка на Rust без Python и ML-фреймворков.
Промпты короче 8192 токенов идут точным вниманием, как в
[стандартных файлах](https://huggingface.co/infosave/Aion-1.0-Instruct-2.3B-cmf).

### Быстрый старт

```bash
cargo install cortiq-cli
hf download infosave/Aion-1.0-Instruct-2.3B-Turbo-cmf Aion-1.0-Instruct-2.3B-turbo.cmf --local-dir .
cortiq run Aion-1.0-Instruct-2.3B-turbo.cmf --prompt-file report.txt --greedy
```

С CUDA: установка с `--features cuda`, запуск с `CMF_GPU=cuda`. Нужен cortiq 0.9.9 или новее; готовые сборки — на
[странице релизов](https://github.com/infosave2007/cmf/releases).

### Что делает файл

**Фиксированный потолок.** 21 из 28 слоёв Aion смотрит в окно из 2048 токенов и держит на карте только эти строки.
Остальные 7 смотрят на весь промпт. В этом файле, как только промпт доходит до 8192 токенов, эти 7 слоёв держат
точными первые 4 строки и последние 1800, а остальное сворачивают в 32 опорные точки на голову — дальнее поле
фиксированного размера. Память карты и цена токена перестают расти с текстом. Заявка на патент № 19/738,763.

**Точные строки.** Одно дальнее поле теряет пароль. Каждая строка получает оценку по тому, что с ней делают слои
(для каждой из 8 KV-голов 7 слоёв: как далеко ключ от линейной оболочки опорных точек и норма значения; линейная
модель над 112 рангами, подобранная на спрятанных фактах в Википедии: на отложенных промптах строки факта попадают
в верхние 0,02 %), и 248 лучших остаются точными. Вопрос в конце промпта читает их перед ответом.

### Скорость

`cortiq bench --core` после промпта такой длины; память карты — пик по `nvidia-smi` за прогон.

**NVIDIA L4 (24 ГБ, Google Colab), Vulkan**

| промпт | файл | первый токен | генерация, ток/с | память карты |
|---|---|---:|---:|---:|
| 16 384 | Turbo | 12,2 с | 80,9 | 3,6 ГБ |
| 16 384 | стандартный `q4tp` | 10,3 с | 64,0 | 4,0 ГБ |
| 30 000 | Turbo | 21,0 с | 80,8 | 3,6 ГБ |
| 30 000 | Turbo, `CMF_O1_WHOLE=1` | 34,1 с | 64,8 | 4,6 ГБ |
| 30 000 | стандартный `q4tp` | 27,4 с | 53,1 | 4,7 ГБ |

**NVIDIA L4 (24 ГБ, Google Colab), CUDA**

| промпт | файл | первый токен | генерация, ток/с | память карты |
|---|---|---:|---:|---:|
| 16 384 | Turbo | 6,4 с | 89,5 | 3,0 ГБ |
| 16 384 | стандартный `q4tp` | 4,2 с | 94,5 | 3,0 ГБ |
| 30 000 | Turbo | 13,5 с | 88,5 | 3,7 ГБ |
| 30 000 | Turbo, `CMF_O1_WHOLE=0` | 11,5 с | 112,4 | 2,8 ГБ |
| 30 000 | стандартный `q4tp` | 10,0 с | 79,3 | 3,6 ГБ |

**MacBook Air M4 (24 ГБ), Metal**

| промпт | файл | первый токен | генерация, ток/с | ключи и значения |
|---|---|---:|---:|---:|
| 16 384 | Turbo | 43 с | 44,7 | 0,83 ГБ |
| 16 384 | стандартный `q4tp` | 44 с | 42,2 | 1,62 ГБ |

**Два способа читать длинный промпт.** На Vulkan промпт после 8192 токенов читается через дальнее поле,
кусками, и строки уходят с карты по мере чтения: память фиксирована, а вычисленное по этим строкам — приближение.
На CUDA весь промпт читается точно, и дальнее поле строится один раз перед последними 512 строками: память растёт
с промптом, как у стандартного файла, зато ответ опирается на точные строки. `CMF_O1_WHOLE=1` читает промпт целиком
и на Vulkan (строка на 30 000 токенов выше; спрятанные факты 4/4 вместо 2/4), а `CMF_O1_WHOLE=0` читает его кусками
на CUDA: 2,7–2,8 ГБ от 16 000 до 30 000 токенов и 112 ток/с, спрятанные факты — 3/4 и 2/4.

### Качество

До 8192 токенов перплексия — как у 4-битного файла (WikiText-2 14,61, Rust 7,83, Python 7,43; bf16 14,24 / 7,66 /
7,26). Дальше ответ зависит от того, что сохранят дальнее поле и точные строки. Спрятанные факты (ключ, код
доступа, пароль, номер комнаты), жадно:

| промпт | Transformers bf16 | стандартный `q4tp`, CUDA | Turbo, CUDA | Turbo, Vulkan |
|---|---:|---:|---:|---:|
| около 13 000 токенов | 4/4 | 4/4 | 4/4 | 4/4 |
| около 27 000 токенов | 3/4 | 3/4 | 4/4 | 2/4 (4/4 с `CMF_O1_WHOLE=1`) |

На 27 000 токенах чтение с фиксированной памятью на Vulkan теряет пароль, стоящий на четверти текста, и ошибается
на вопросе, на котором ошибается и эталон; при чтении целиком тот же файл отвечает на все четыре — на один больше
эталона.

Суммы sha256 — в таблице английского раздела. Веса:
[MIT](https://huggingface.co/microsoft/Aion-1.0-Instruct-2.3B/blob/main/LICENSE), от
[microsoft/Aion-1.0-Instruct-2.3B](https://huggingface.co/microsoft/Aion-1.0-Instruct-2.3B).

---

## 中文文档

[Aion-1.0-Instruct-2.3B](https://huggingface.co/microsoft/Aion-1.0-Instruct-2.3B) 的 4 位权重，配合一个让显存和
生成速度不随提示词增长而变化的文件头。

- **NVIDIA L4（24 GB），Vulkan：** 30 000 token 的提示词下显存保持 3.6 GB（标准 4 位文件 4.7 GB），之后生成
  80.8 tok/s（标准 53.1），首 token 21.0 s（标准 27.4）。
- **NVIDIA L4，CUDA：** 16 000 到 30 000 token 生成保持 88–90 tok/s（标准文件 94.5 → 79.3）；显存 3.0–3.7 GB，
  与标准文件在 CUDA 上相同。
- **MacBook Air M4，Metal：** 16 384 token 的键值占 0.83 GB（标准 1.62 GB），生成 44.7 tok/s（标准 42.2）。
- **埋入事实**（长篇维基百科文本）：约 13 000 token 时 CUDA 和 Vulkan 均为 4/4；约 27 000 token 时 CUDA 4/4
  （Transformers bf16 为 3/4），Vulkan 固定显存读取为 2/4。

这是一个供 `cortiq` 使用的 [CMF](https://github.com/infosave2007/cmf) 文件（不依赖 Python 和机器学习框架的 Rust
推理引擎）。8192 token 以内的提示词使用精确注意力，与[标准文件](https://huggingface.co/infosave/Aion-1.0-Instruct-2.3B-cmf)
相同。

### 快速开始

```bash
cargo install cortiq-cli
hf download infosave/Aion-1.0-Instruct-2.3B-Turbo-cmf Aion-1.0-Instruct-2.3B-turbo.cmf --local-dir .
cortiq run Aion-1.0-Instruct-2.3B-turbo.cmf --prompt-file report.txt --greedy
```

CUDA：安装时加 `--features cuda`，运行时设 `CMF_GPU=cuda`。需要 cortiq 0.9.9 或更新版本；预编译二进制见[发布页面](https://github.com/infosave2007/cmf/releases)。

### 原理

**固定上限。** Aion 的 28 层中有 21 层只关注 2048 token 的窗口，显卡上只保留这些行；其余 7 层关注整个提示词。
在这个文件里，提示词达到 8192 token 后，这 7 层精确保留前 4 行和最后 1800 行，其余折叠为每个头 32 个地标点，
即固定大小的远场。显存和每个 token 的开销不再随文本增长。专利申请号 19/738,763。

**精确行。** 单靠远场会丢失密码。每一行按各层对它的处理打分（7 层 × 8 个 KV 头：键离地标张成空间的距离，以及值的
范数；在维基百科埋入事实上拟合的 112 个排名的线性模型，留出提示词中事实所在行排在前 0.02 %），最好的 248 行保持
精确。提示词末尾的问题在回答前读取它们。

### 性能

`cortiq bench --core`，在该长度提示词之后测量；显存为运行期间 `nvidia-smi` 峰值。

**NVIDIA L4（24 GB，Google Colab），Vulkan**

| 提示词 | 文件 | 首 token | 生成，tok/s | 显存 |
|---|---|---:|---:|---:|
| 16 384 | Turbo | 12.2 s | 80.9 | 3.6 GB |
| 16 384 | 标准 `q4tp` | 10.3 s | 64.0 | 4.0 GB |
| 30 000 | Turbo | 21.0 s | 80.8 | 3.6 GB |
| 30 000 | Turbo，`CMF_O1_WHOLE=1` | 34.1 s | 64.8 | 4.6 GB |
| 30 000 | 标准 `q4tp` | 27.4 s | 53.1 | 4.7 GB |

**NVIDIA L4（24 GB，Google Colab），CUDA**

| 提示词 | 文件 | 首 token | 生成，tok/s | 显存 |
|---|---|---:|---:|---:|
| 16 384 | Turbo | 6.4 s | 89.5 | 3.0 GB |
| 16 384 | 标准 `q4tp` | 4.2 s | 94.5 | 3.0 GB |
| 30 000 | Turbo | 13.5 s | 88.5 | 3.7 GB |
| 30 000 | Turbo，`CMF_O1_WHOLE=0` | 11.5 s | 112.4 | 2.8 GB |
| 30 000 | 标准 `q4tp` | 10.0 s | 79.3 | 3.6 GB |

**MacBook Air M4（24 GB），Metal**

| 提示词 | 文件 | 首 token | 生成，tok/s | 键值 |
|---|---|---:|---:|---:|
| 16 384 | Turbo | 43 s | 44.7 | 0.83 GB |
| 16 384 | 标准 `q4tp` | 44 s | 42.2 | 1.62 GB |

**读取长提示词的两种方式。** 在 Vulkan 上，8192 token 之后的提示词经远场分块读取，读完的行即离开显卡：
显存固定，而这些行算出的结果是近似值。在 CUDA 上，整个提示词被精确读取，远场在最后 512 行之前一次建成：显存像
标准文件一样随提示词增长，但回答基于精确的行。`CMF_O1_WHOLE=1` 让 Vulkan 也整体读取（见上表 30 000 token 一行；
埋入事实 4/4 而非 2/4）；`CMF_O1_WHOLE=0` 让 CUDA 分块读取：16 000 到 30 000 token 显存 2.7–2.8 GB、生成 112 tok/s，
埋入事实为 3/4 和 2/4。

### 质量

8192 token 以内的困惑度与 4 位文件相同（WikiText-2 14.61，Rust 7.83，Python 7.43；bf16 14.24 / 7.66 / 7.26）。之后
回答取决于远场和精确行保留了什么。埋入事实（通行码、访问码、密码、房间号），贪心：

| 提示词 | Transformers bf16 | 标准 `q4tp`，CUDA | Turbo，CUDA | Turbo，Vulkan |
|---|---:|---:|---:|---:|
| 约 13 000 token | 4/4 | 4/4 | 4/4 | 4/4 |
| 约 27 000 token | 3/4 | 3/4 | 4/4 | 2/4（`CMF_O1_WHOLE=1` 时 4/4） |

在 27 000 token 时，Vulkan 的固定显存读取会丢失位于文本四分之一处的密码，并答错参考模型也答错的那个问题；整体
读取时同一文件四个全对，比参考模型多一个。

sha256 见英文部分的表格。权重：[MIT](https://huggingface.co/microsoft/Aion-1.0-Instruct-2.3B/blob/main/LICENSE)，
来自 [microsoft/Aion-1.0-Instruct-2.3B](https://huggingface.co/microsoft/Aion-1.0-Instruct-2.3B)。
