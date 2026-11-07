---
license: apache-2.0
library_name: cortiq
base_model: DRUPEs/CherryPit-1.2-2B
base_model_relation: quantized
pipeline_tag: text-generation
tags:
  - cmf
  - cortiq
  - quantized
  - q4tp
  - code
  - long-context
  - low-vram
  - vulkan
language:
  - en
  - ru
  - zh
---

# CherryPit-1.2-2B Turbo — CMF

The 4-bit weights of [CherryPit-1.2-2B](https://huggingface.co/DRUPEs/CherryPit-1.2-2B) under a header
that holds device memory and decode speed in place on long prompts. At 64 000 tokens it takes 3.2 GB of
the card instead of 7.4, reads the prompt in 172 s instead of 229 and decodes at 67 tok/s instead of 43,
with passwords and codes in the text still exact. Prompts under 8192 tokens run exact attention, as in the
[standard file](https://huggingface.co/infosave/CherryPit-1.2-2B-cmf). It is one
[CMF](https://github.com/infosave2007/cmf) file for `cortiq`, a Rust inference engine with no Python and
no ML framework.

[English](#quick-start) · [Русский](#документация-на-русском) · [中文](#中文说明)

## Quick start

```bash
cargo install cortiq-cli
hf download infosave/CherryPit-1.2-2B-Turbo-cmf CherryPit-1.2-2B-turbo.cmf --local-dir .
CMF_MAX_SEQ=131072 cortiq run CherryPit-1.2-2B-turbo.cmf --prompt-file report.txt --greedy --no-think
```

Short on RAM as well, with a fast SSD:

```bash
CMF_MAX_SEQ=131072 cortiq run CherryPit-1.2-2B-turbo.cmf --prompt-file report.txt --greedy --no-think --o1-spill /mnt/ssd/cortiq
```

Use cortiq 0.8.61 or later; prebuilt binaries are on the
[releases page](https://github.com/infosave2007/cmf/releases). `CMF_MAX_SEQ` lifts the default cap of
32 768 tokens to the model's 131 072.

## What the file does

![A long prompt on a small card](https://huggingface.co/infosave/CherryPit-1.2-2B-Turbo-cmf/resolve/main/assets/ceiling_en.png)

**A fixed ceiling.** The first 8192 rows of a prompt are read exactly on the card. The keys and values of
later rows leave the card as they are read, for RAM or an SSD. Before the last 512 rows, the far field is
built from them one layer at a time, and the rows go away. A far field keeps the first 4 rows and the last
1800 exact and everything else in 32 landmarks per head, so device memory and the cost of a token stop
growing with the text. Patent application No. 19/738,763.

**Exact rows.** A far field alone loses a password. So each row is scored by what the layers do with it,
and the 248 best stay exact. A question at the end of the prompt reads them before it is answered. On this
model a fact's rows land in the top 0.01 % of a prompt on average, and in the top 0.02 % at worst, on
prompts the fit did not see.

![Passwords and codes stay exact](https://huggingface.co/infosave/CherryPit-1.2-2B-Turbo-cmf/resolve/main/assets/exact_rows_en.png)

## Performance

![Standard and turbo](https://huggingface.co/infosave/CherryPit-1.2-2B-Turbo-cmf/resolve/main/assets/compare_en.png)

RTX 5090, Vulkan, `cortiq bench --core`, 60 tokens after a prompt of that length.

| prompt, tokens | first token, s (standard) | first token, s (**turbo**) | decode, tok/s (standard) | decode, tok/s (**turbo**) | peak VRAM, GB (standard) | peak VRAM, GB (**turbo**) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 000 | 0.43 | **0.45** | 108 | **109** | 2.1 | **1.9** |
| 4 000 | 2.3 | **2.3** | 112 | **108** | 2.4 | **2.4** |
| 16 000 | 18.1 | **48.8** | 90 | **67** | 3.5 | **2.9** |
| 32 000 | 61.7 | **86.4** | 74 | **67** | 4.8 | **2.7** |
| 64 000 | 228.5 | **171.9** | 43 | **67** | 7.4 | **3.2** |

At 16 000 and 32 000 tokens a card with room for the cache is faster with the standard file; from 64 000
tokens on, and on a card the cache does not fit, the turbo file is the faster and the smaller one.

**On a 4 GB card.** With the budget of such a card set on the same 5090 (`CMF_GPU_VRAM_MB=3800`), the
file stays at 2.6 GB at 64 000 tokens: first token 161 s, decode 66 tok/s.

**Host memory.** While a long prompt is read, its rows take RAM or a disk; the mapped weights are not
counted. At 64 000 tokens: 11.0 GB of RAM, or with `--o1-spill` 8.3 GB of RAM and 8.2 GB on disk, at the same speed. The rows are needed only until the far field is built.

**Quality.** A planted fact (a pass key, an access code, a password, a room number): 8 of 8 in 100 000
characters (about 22 000 tokens), 3 of 4 in 300 000 (about 67 000 tokens); the fourth answer named the
number only after the 24 tokens it was given.

## Options

| option | meaning |
|---|---|
| `--o1-spill DIR` | a long prompt's rows in files in `DIR`, also `CMF_O1_SPILL` |
| `--prompt-file FILE` | a prompt longer than a command line holds |
| `CMF_O1_PREFIX_MAX=N` | seal at `N` rows instead of 8192 |
| `CMF_O1=off` | the plain model, with exact attention at any length |

## Limits

- The far field runs on Vulkan, DX12 and Metal; on the CPU this file runs as the standard one. It was
  measured on an NVIDIA card.
- Past the seal the text is approximate: what lies far back fades, apart from the rows kept exact. The fit
  behind those rows saw planted facts; other kinds of important text were not tested. Asked to summarise the
  last of several Wikipedia articles in 100 000 characters, the turbo file mixed in an earlier one, where the
  standard file kept to the last.
- Rows that arrive after a prompt ends stay in the window or go to the far field. A turn that diverges from
  a sealed conversation reads the whole prompt again.
- A generation that starts below the seal stays exact, and its memory grows with every token.

## Building a turbo file

The weights of the exact-row fit, `keep_cherrypit.bin`, are in this repository.

```bash
cortiq far-turbo CherryPit-1.2-2B-q4tp.cmf CherryPit-1.2-2B-turbo.cmf --exact-weights keep_cherrypit.bin
```

The method is described in [the Qwen3.8-27B Turbo guide](https://github.com/infosave2007/cmf/blob/master/docs/QWEN38_TURBO.md).

## License

Weights: [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0), from
[DRUPEs/CherryPit-1.2-2B](https://huggingface.co/DRUPEs/CherryPit-1.2-2B). The far field is a subject of
U.S. patent application No. 19/738,763; see
[PATENTS.md](https://github.com/infosave2007/cmf/blob/master/PATENTS.md).

---

## Документация на русском

4-битные веса [CherryPit-1.2-2B](https://huggingface.co/DRUPEs/CherryPit-1.2-2B) с заголовком, который
держит память карты и скорость декода на месте при длинных промптах. На 64 000 токенов файл занимает
3,2 ГБ карты вместо 7,4, читает промпт за 172 с вместо 229 и декодирует 67 ток/с вместо 43, а пароли и
коды в тексте остаются точными. Промпты короче 8192 токенов идут с точным вниманием, как в
[обычном файле](https://huggingface.co/infosave/CherryPit-1.2-2B-cmf). Это один
[CMF](https://github.com/infosave2007/cmf)-файл для `cortiq`, движка на Rust без Python и ML-фреймворков.

### Быстрый старт

```bash
cargo install cortiq-cli
hf download infosave/CherryPit-1.2-2B-Turbo-cmf CherryPit-1.2-2B-turbo.cmf --local-dir .
CMF_MAX_SEQ=131072 cortiq run CherryPit-1.2-2B-turbo.cmf --prompt-file report.txt --greedy --no-think
```

Если не хватает и оперативной памяти, но есть быстрый SSD, добавьте `--o1-spill /mnt/ssd/cortiq`. Нужен
cortiq 0.8.61 или новее; `CMF_MAX_SEQ` поднимает предел по умолчанию в 32 768 токенов до 131 072 у модели.

### Что делает файл

![Длинный промпт на маленькой карте](https://huggingface.co/infosave/CherryPit-1.2-2B-Turbo-cmf/resolve/main/assets/ceiling_ru.png)

**Постоянный потолок.** Первые 8192 строки промпта читаются на карте точно. Ключи и значения следующих
строк уходят с карты по мере чтения, в RAM или на SSD. Перед последними 512 строками из них послойно
строится дальнее поле, и сами строки больше не нужны. Дальнее поле держит точными первые 4 строки и
последние 1800, а всё остальное хранит в 32 опорных точках на голову, поэтому память карты и цена токена
перестают расти вместе с текстом. Патентная заявка США № 19/738,763.

**Точные строки.** Одно дальнее поле теряет пароль. Поэтому каждая строка получает оценку по тому, что с
ней делают слои, и 248 лучших остаются точными. На этой модели строки факта попадают в верхние 0,01 %
промпта в среднем и в верхние 0,02 % в худшем случае, на промптах, которых подгонка не видела.

![Пароли и коды остаются точными](https://huggingface.co/infosave/CherryPit-1.2-2B-Turbo-cmf/resolve/main/assets/exact_rows_ru.png)

### Скорость

![Обычный и турбо](https://huggingface.co/infosave/CherryPit-1.2-2B-Turbo-cmf/resolve/main/assets/compare_ru.png)

RTX 5090, Vulkan, `cortiq bench --core`, 60 токенов после промпта такой длины.

| промпт, токенов | первый токен, с (обычный) | первый токен, с (**турбо**) | декод, ток/с (обычный) | декод, ток/с (**турбо**) | пик VRAM, ГБ (обычный) | пик VRAM, ГБ (**турбо**) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 000 | 0,43 | **0,45** | 108 | **109** | 2,1 | **1,9** |
| 4 000 | 2,3 | **2,3** | 112 | **108** | 2,4 | **2,4** |
| 16 000 | 18,1 | **48,8** | 90 | **67** | 3,5 | **2,9** |
| 32 000 | 61,7 | **86,4** | 74 | **67** | 4,8 | **2,7** |
| 64 000 | 228,5 | **171,9** | 43 | **67** | 7,4 | **3,2** |

На 16 000 и 32 000 токенов карта, где помещается кэш, быстрее с обычным файлом; с 64 000 токенов, а также
на карте, где кэш не помещается, турбо-файл и быстрее, и меньше.

**На карте 4 ГБ.** С бюджетом такой карты на той же 5090 (`CMF_GPU_VRAM_MB=3800`) файл держится в
2,6 ГБ на 64 000 токенов: первый токен 161 с, декод 66 ток/с.

**Память хоста.** Пока читается длинный промпт, его строки занимают RAM или диск; отображённые веса не
считаются. На 64 000 токенов: 11,0 ГБ RAM, а с `--o1-spill` — 8,3 ГБ RAM и 8,2 ГБ на диске при той же скорости.

**Качество.** Факт в тексте (ключ, код доступа, пароль, номер комнаты): 8 из 8 в 100 000 символов (около 22 000 токенов),
3 из 4 в 300 000 (около 67 000 токенов); четвёртый ответ назвал число
только после отведённых ему 24 токенов.

### Параметры

| параметр | что делает |
|---|---|
| `--o1-spill DIR` | строки длинного промпта в файлах в `DIR`, также `CMF_O1_SPILL` |
| `--prompt-file FILE` | промпт длиннее, чем вмещает командная строка |
| `CMF_O1_PREFIX_MAX=N` | печать на `N` строках вместо 8192 |
| `CMF_O1=off` | обычная модель с точным вниманием на любой длине |

Дальнее поле работает на Vulkan, DX12 и Metal; на CPU файл работает как обычный; замерено на карте NVIDIA.
За печатью текст приближённый: дальнее уходит, кроме точных строк; пересказ последней из нескольких статей в
100 000 символов примешал более раннюю, а обычный файл — нет. Собрать такой файл:
`cortiq far-turbo CherryPit-1.2-2B-q4tp.cmf CherryPit-1.2-2B-turbo.cmf --exact-weights keep_cherrypit.bin`
(веса лежат в этом репозитории). Веса: Apache 2.0, от DRUPEs; дальнее поле — предмет патентной заявки
США № 19/738,763.

---

## 中文说明

[CherryPit-1.2-2B](https://huggingface.co/DRUPEs/CherryPit-1.2-2B) 的 4 位权重，文件头让长提示下的显存和
解码速度保持不变。64 000 个 token 时占用显存 3.2 GB（普通文件 7.4 GB），读取提示用时 172 秒（普通文件
229 秒），解码 67 tok/s（普通文件 43 tok/s），文本中的密码和编码依然精确。短于 8192 个 token 的提示使用
精确注意力，与[普通文件](https://huggingface.co/infosave/CherryPit-1.2-2B-cmf)相同。这是一个供 `cortiq`
使用的 [CMF](https://github.com/infosave2007/cmf) 文件。

### 快速开始

```bash
cargo install cortiq-cli
hf download infosave/CherryPit-1.2-2B-Turbo-cmf CherryPit-1.2-2B-turbo.cmf --local-dir .
CMF_MAX_SEQ=131072 cortiq run CherryPit-1.2-2B-turbo.cmf --prompt-file report.txt --greedy --no-think
```

内存也不足但有高速 SSD 时，加上 `--o1-spill /mnt/ssd/cortiq`。需要 cortiq 0.8.61 或更新版本；
`CMF_MAX_SEQ` 把默认的 32 768 token 上限提高到模型的 131 072。

### 文件做了什么

![小显卡上的长提示](https://huggingface.co/infosave/CherryPit-1.2-2B-Turbo-cmf/resolve/main/assets/ceiling_zh.png)

**固定上限。** 提示的前 8192 行在显卡上精确读取，之后各行的键和值一读完就离开显卡，进入内存或 SSD。
在最后 512 行之前，用它们逐层构建远场，然后这些行就不再需要。远场精确保留前 4 行和后 1800 行，其余部分
用每个头 32 个地标表示，因此显存和每个 token 的成本不再随文本增长。美国专利申请号 19/738,763。

**精确行。** 只有远场会丢失密码，所以按各层对每一行的处理给它打分，最好的 248 行保持精确。在这个模型上，
事实所在的行平均落在提示的前 0.01%，最差也在前 0.02%（在拟合未见过的提示上）。

![密码和编码保持精确](https://huggingface.co/infosave/CherryPit-1.2-2B-Turbo-cmf/resolve/main/assets/exact_rows_zh.png)

### 性能

![普通与 Turbo](https://huggingface.co/infosave/CherryPit-1.2-2B-Turbo-cmf/resolve/main/assets/compare_zh.png)

RTX 5090，Vulkan，`cortiq bench --core`，在该长度的提示之后生成 60 个 token。

| 提示，token | 首 token，秒（普通） | 首 token，秒（**turbo**） | 解码，tok/s（普通） | 解码，tok/s（**turbo**） | 显存峰值，GB（普通） | 显存峰值，GB（**turbo**） |
|---:|---:|---:|---:|---:|---:|---:|
| 1 000 | 0.43 | **0.45** | 108 | **109** | 2.1 | **1.9** |
| 4 000 | 2.3 | **2.3** | 112 | **108** | 2.4 | **2.4** |
| 16 000 | 18.1 | **48.8** | 90 | **67** | 3.5 | **2.9** |
| 32 000 | 61.7 | **86.4** | 74 | **67** | 4.8 | **2.7** |
| 64 000 | 228.5 | **171.9** | 43 | **67** | 7.4 | **3.2** |

16 000 和 32 000 token 时，若显卡放得下缓存，普通文件更快；从 64 000 token 起，以及在放不下缓存的显卡上，
turbo 文件既更快也更小。

**4 GB 显卡。** 在同一张 5090 上设置这种显卡的预算（`CMF_GPU_VRAM_MB=3800`），64 000 token 时文件占用
2.6 GB：首 token 161 秒，解码 66 tok/s。

**主机内存。** 读取长提示时，其各行占用内存或磁盘（不计映射的权重）。64 000 token 时：内存 11.0 GB；使用 `--o1-spill` 时内存 8.3 GB、磁盘 8.2 GB，速度不变。

**质量。** 插入的事实（密钥、访问码、密码、房间号）：100 000 字符（约 22 000 token）中 8/8，300 000 字符（约 67 000 token）中
3/4；第四个回答在给定的 24 个 token 之后才说出数字。

远场在 Vulkan、DX12 和 Metal 上运行；在 CPU 上该文件按普通文件运行；测量在 NVIDIA 显卡上进行。超过封存点后文本是近似的：
在 100 000 字符中总结多篇文章的最后一篇时，turbo 文件混入了前一篇，普通文件没有。构建方法：
`cortiq far-turbo CherryPit-1.2-2B-q4tp.cmf CherryPit-1.2-2B-turbo.cmf --exact-weights keep_cherrypit.bin`
（权重文件在本仓库中）。权重：Apache 2.0，来自 DRUPEs；远场是美国专利申请 19/738,763 的主题。
