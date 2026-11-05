---
license: apache-2.0
library_name: cortiq
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: text-generation
tags:
  - cmf
  - cortiq
  - quantized
  - q4tp
  - qwen3.8
  - long-context
  - low-vram
  - vulkan
language:
  - en
  - ru
  - zh
---

# Qwen3.8-27B Turbo — CMF

The 4-bit weights of [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) under a header that holds
device memory in place on long prompts. It fits a 16 GB card at any prompt length, reads 64 000 tokens
in 177 s against 316 s, and decodes at 92 tok/s there against 51, with passwords and codes in the text
still exact. Prompts under 8192 tokens run exact attention, as in the
[standard file](https://huggingface.co/infosave/Qwen3.8-27B-cmf). It is one
[CMF](https://github.com/infosave2007/cmf) file for `cortiq`, a Rust inference engine with no Python
and no ML framework.

[English](#quick-start) · [Русский](#документация-на-русском) · [中文](#中文说明)

## Quick start

```bash
cargo install cortiq-cli
hf download infosave/Qwen3.8-27B-Turbo-cmf qwen38-27b-turbo.cmf --local-dir .
cortiq run qwen38-27b-turbo.cmf --prompt-file report.txt --greedy --no-think
```

Short on RAM as well, with a fast SSD:

```bash
cortiq run qwen38-27b-turbo.cmf --prompt-file report.txt --greedy --no-think --o1-spill /mnt/ssd/cortiq
```

Use cortiq 0.8.58 or later; prebuilt binaries are on the
[releases page](https://github.com/infosave2007/cmf/releases). `--greedy` turns on draft tokens, and
`--no-think` asks for a direct answer.

## What the file does

![A long prompt on a small card](https://huggingface.co/infosave/Qwen3.8-27B-Turbo-cmf/resolve/main/assets/ceiling_en.png)

**A fixed ceiling.** The first 8192 rows of a prompt are read exactly on the card, 4096 on a 16 GB
card. The keys and values of later rows leave the card as they are read, for RAM or an SSD. Before the
last 512 rows, the far field is built from them one layer at a time, and the rows go away. A far field
keeps the first 4 rows and the last 1800 exact and everything else in 32 landmarks per head, so device
memory and the cost of a token stop growing with the text. Patent application No. 19/738,763.

**Exact rows.** A far field alone loses a password. So each row is scored by what the layers do with
it, and the 248 best stay exact. A question at the end of the prompt reads them before it is answered.

![Passwords and codes stay exact](https://huggingface.co/infosave/Qwen3.8-27B-Turbo-cmf/resolve/main/assets/exact_rows_en.png)

**Short threads and drafts.** A sealed token and a chunk of rows take the attention apart into seven
small stages that fill the whole card, where one long thread per head leaves most of it idle. The
model's MTP head drafts tokens, the card verifies them in one batch, and the far field moves over the
accepted ones only. Greedy text stays the same with and without drafts.

## Performance

![Standard and turbo](https://huggingface.co/infosave/Qwen3.8-27B-Turbo-cmf/resolve/main/assets/compare_en.png)

RTX 5090, Vulkan, `cortiq bench --core`, 60 tokens after a prompt of that length.

| prompt, tokens | first token, s (standard) | first token, s (**turbo**) | decode, tok/s (standard) | decode, tok/s (**turbo**) | peak VRAM, GB (standard) | peak VRAM, GB (**turbo**) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 000 | 2.2 | **2.2** | 105 | **105** | 14.5 | **14.4** |
| 4 000 | 8.8 | **8.8** | 104 | **104** | 14.9 | **14.8** |
| 16 000 | 42 | **47** | 87 | **92** | 16.5 | **15.5** |
| 32 000 | 107 | **92** | 71 | **91** | 18.5 | **15.7** |
| 64 000 | 316 | **177** | 51 | **92** | 22.8 | **15.7** |

A real answer, 400 tokens of a summary of Wikipedia articles, with about half of the drafts accepted:

| prompt | first token, s (standard) | first token, s (**turbo**) | decode, tok/s (standard) | decode, tok/s (**turbo**) |
|---|---:|---:|---:|---:|
| 30 000 characters | 19.2 | **18.6** | 76.4 | **76.3** |
| 100 000 characters | 76.7 | **72.1** | 50.9 | **60.4** |

**On a 16 GB card.** With the budget of such a card set on the same 5090 (`CMF_GPU_VRAM_MB=15000`), the
file seals at 4096 rows and stays at 15.0 GB. First token 43 s at 16 000, 87 s at 32 000, 176 s at
64 000, decode 92 tok/s. A real 16 GB card has fewer shader cores, so expect lower speed at the same
memory.

**On a Mac.** MacBook Air M4, 24 GB, Metal: at 16 000 tokens the process stays at 6.3 GB beside the mapped
weights, first token 545 s, decode 4 tok/s. A planted fact in 40 000 characters came back whole. An M4 decodes this
model at about 5 tok/s at any length, bound by its matrix speed, so on a Mac the file saves memory and not time. Drafts do not pay there, 4.6 tok/s with them against 4.9 without, and stay off.

**Host memory.** While a long prompt is read, its rows take RAM or a disk. The mapped weights are not
counted here; the system can drop them from RAM.

| prompt, tokens | RAM | with `--o1-spill`: RAM | with `--o1-spill`: disk |
|---:|---:|---:|---:|
| 16 000 | 4.4 GB | 2.2 GB | 4.7 GB |
| 32 000 | 7.0 GB | 2.2 GB | 8.7 GB |
| 64 000 | 13.2 GB | 3.0 GB | 18.7 GB |

On a disk with 4.5 GB/s write and 5.2 GB/s read the speed is the same as in RAM. The rows are needed
only until the far field is built.

**Quality.** A planted fact in 100 000 characters, four kinds of them (a pass key, an access code, a
password, a room number): 12 of 12, 8 of them with a 15 GB budget and an SSD; in 230 000 characters, about 53 000 tokens, 6 of 6 on the
same path. A far field without exact rows finds 0 of 4. Perplexity on the rows past the seal, two windows of 16 384 tokens of Wikipedia: exact 4.76, far field 4.97.

## Options

| option | meaning |
|---|---|
| `--o1-spill DIR` | a long prompt's rows in files in `DIR`, also `CMF_O1_SPILL` |
| `--prompt-file FILE` | a prompt longer than a command line holds |
| `CMF_O1_PREFIX_MAX=N` | seal at `N` rows instead of the choice for the card |
| `CMF_O1=off` | the plain model, with exact attention at any length |
| `CMF_O1_SPEC=0` | no drafts over the far field |

## Limits

- The far field runs on Vulkan, DX12 and Metal. On the CPU this file runs as the standard one. It was measured
  on an NVIDIA card and an M4.
- Past the seal the text is approximate: what lies far back fades, apart from the rows kept exact. The
  fit behind those rows saw planted facts, and other kinds of important text were not tested.
- Rows that arrive after a prompt ends stay in the window or go to the far field. A turn that diverges
  from a sealed conversation reads the whole prompt again.
- On Vulkan, drafts need greedy decoding without penalties. Without `--greedy` a sealed token runs at
  about 57 tok/s.
- A generation that starts below the seal stays exact, and its memory grows with every token.

## Building a turbo file

The weights of the exact-row fit are in the
[repository](https://github.com/infosave2007/cmf/tree/master/scripts/dory/router), with the scripts that
make them.

```bash
cortiq far-turbo qwen38-27b-q4tp.cmf qwen38-27b-turbo.cmf --exact-weights keep27.bin
```

See [the guide](https://github.com/infosave2007/cmf/blob/master/docs/QWEN38_TURBO.md).

## License

Weights: [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0), from
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). The far field is a subject of U.S. patent
application No. 19/738,763; see
[PATENTS.md](https://github.com/infosave2007/cmf/blob/master/PATENTS.md).

---

## Документация на русском

4-битные веса [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) с заголовком, который держит память
карты на месте при длинных промптах. Файл помещается на карту 16 ГБ при любой длине промпта, читает
64 000 токенов за 177 с вместо 316 и декодирует там 92 ток/с вместо 51, а пароли и коды в тексте остаются
точными. Промпты короче 8192 токенов идут с точным вниманием, как в
[обычном файле](https://huggingface.co/infosave/Qwen3.8-27B-cmf). Это один
[CMF](https://github.com/infosave2007/cmf)-файл для `cortiq`, движка на Rust без Python и ML-фреймворков.

### Быстрый старт

```bash
cargo install cortiq-cli
hf download infosave/Qwen3.8-27B-Turbo-cmf qwen38-27b-turbo.cmf --local-dir .
cortiq run qwen38-27b-turbo.cmf --prompt-file report.txt --greedy --no-think
```

Если не хватает и оперативной памяти, но есть быстрый SSD:

```bash
cortiq run qwen38-27b-turbo.cmf --prompt-file report.txt --greedy --no-think --o1-spill /mnt/ssd/cortiq
```

Нужен cortiq 0.8.58 или новее; готовые сборки лежат на
[странице релизов](https://github.com/infosave2007/cmf/releases). `--greedy` включает черновые токены,
`--no-think` просит ответ без рассуждения.

### Что делает файл

![Длинный промпт на маленькой карте](https://huggingface.co/infosave/Qwen3.8-27B-Turbo-cmf/resolve/main/assets/ceiling_ru.png)

**Постоянный потолок.** Первые 8192 строки промпта читаются на карте точно, на карте 16 ГБ — 4096.
Ключи и значения следующих строк уходят с карты по мере чтения, в RAM или на SSD. Перед последними
512 строками из них послойно строится дальнее поле, и сами строки больше не нужны. Дальнее поле держит
точными первые 4 строки и последние 1800, а всё остальное хранит в 32 опорных точках на голову, поэтому
память карты и цена токена перестают расти вместе с текстом. Патентная заявка США № 19/738,763.

**Точные строки.** Одно дальнее поле теряет пароль. Поэтому каждая строка получает оценку по тому, что с
ней делают слои, и 248 лучших остаются точными. Вопрос в конце промпта читает их до ответа.

![Пароли и коды остаются точными](https://huggingface.co/infosave/Qwen3.8-27B-Turbo-cmf/resolve/main/assets/exact_rows_ru.png)

**Короткие потоки и черновики.** Запечатанный токен и пакет строк разбирают внимание на семь небольших
стадий, которые занимают всю карту; один длинный поток на голову оставляет большую её часть без дела.
Голова MTP предлагает токены, карта проверяет их одним пакетом, а дальнее поле сдвигается только на
принятые. Текст при жадном декодировании с черновиками и без них один и тот же.

### Скорость

![Обычный файл и турбо](https://huggingface.co/infosave/Qwen3.8-27B-Turbo-cmf/resolve/main/assets/compare_ru.png)

RTX 5090, Vulkan, `cortiq bench --core`, 60 токенов после промпта этой длины.

| промпт, токенов | первый токен, с (обычный) | первый токен, с (**турбо**) | декод, ток/с (обычный) | декод, ток/с (**турбо**) | пик VRAM, ГБ (обычный) | пик VRAM, ГБ (**турбо**) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 000 | 2,2 | **2,2** | 105 | **105** | 14,5 | **14,4** |
| 4 000 | 8,8 | **8,8** | 104 | **104** | 14,9 | **14,8** |
| 16 000 | 42 | **47** | 87 | **92** | 16,5 | **15,5** |
| 32 000 | 107 | **92** | 71 | **91** | 18,5 | **15,7** |
| 64 000 | 316 | **177** | 51 | **92** | 22,8 | **15,7** |

Настоящий ответ: 400 токенов пересказа статей Википедии, принята примерно половина черновиков.

| промпт | первый токен, с (обычный) | первый токен, с (**турбо**) | декод, ток/с (обычный) | декод, ток/с (**турбо**) |
|---|---:|---:|---:|---:|
| 30 000 символов | 19,2 | **18,6** | 76,4 | **76,3** |
| 100 000 символов | 76,7 | **72,1** | 50,9 | **60,4** |

**На карте 16 ГБ.** С бюджетом такой карты на той же 5090 (`CMF_GPU_VRAM_MB=15000`) файл запечатывает на
4096 строках и держится на 15,0 ГБ. Первый токен: 43 с на 16 000, 87 с на 32 000, 176 с на 64 000, декод
92 ток/с. У настоящей карты на 16 ГБ меньше вычислительных блоков, поэтому при той же памяти скорость
будет ниже.

**На Маке.** MacBook Air M4, 24 ГБ, Metal: на 16 000 токенов процесс держится на 6,3 ГБ помимо отображённых весов,
первый токен 545 с, декод 4 ток/с. Факт, вставленный в 40 000 символов, вернулся целым. M4 декодирует эту модель примерно
на 5 ток/с при любой длине, упираясь в скорость умножения матриц, поэтому на Маке файл экономит память, а не время. Черновики там не окупаются, 4,6 ток/с против 4,9 без них, и остаются выключенными.

**Память хоста.** Пока читается длинный промпт, его строки занимают RAM или диск. Отображённые в память
веса здесь не считаются: система может выгрузить их из RAM.

| промпт, токенов | RAM | с `--o1-spill`: RAM | с `--o1-spill`: диск |
|---:|---:|---:|---:|
| 16 000 | 4,4 ГБ | 2,2 ГБ | 4,7 ГБ |
| 32 000 | 7,0 ГБ | 2,2 ГБ | 8,7 ГБ |
| 64 000 | 13,2 ГБ | 3,0 ГБ | 18,7 ГБ |

На диске с записью 4,5 ГБ/с и чтением 5,2 ГБ/с скорость та же, что и в RAM. Строки нужны только до сборки
дальнего поля.

**Качество.** Факт, вставленный в 100 000 символов, четырёх видов (ключ, код доступа, пароль, номер
комнаты): 12 из 12, из них 8 с бюджетом 15 ГБ и SSD; в 230 000 символов, это около 53 000 токенов, 6 из 6 тем же путём. Дальнее поле без точных строк находит 0 из 4.
Перплексия на строках после места запечатывания, два окна по 16 384 токена Википедии: точное внимание 4,76, дальнее поле 4,97.

### Параметры

| параметр | значение |
|---|---|
| `--o1-spill DIR` | строки длинного промпта в файлах в `DIR`, также `CMF_O1_SPILL` |
| `--prompt-file FILE` | промпт длиннее, чем вмещает командная строка |
| `CMF_O1_PREFIX_MAX=N` | запечатать на `N` строках, а не по размеру карты |
| `CMF_O1=off` | обычная модель, точное внимание при любой длине |
| `CMF_O1_SPEC=0` | без черновиков над дальним полем |

### Ограничения

- Дальнее поле работает на Vulkan, DX12 и Metal. На CPU этот файл идёт как обычный. Замеры сделаны на карте NVIDIA
  и на M4.
- За местом запечатывания текст приближённый: далёкое блекнет, кроме строк, оставленных точными. Модель,
  которая их выбирает, видела вставленные факты; другие виды важного текста не проверялись.
- Строки, пришедшие после конца промпта, остаются в окне или уходят в дальнее поле. Ход, который расходится
  с запечатанным разговором, читает весь промпт заново.
- На Vulkan черновикам нужно жадное декодирование без штрафов. Без `--greedy` запечатанный токен идёт примерно
  на 57 ток/с.
- Генерация, начатая ниже места запечатывания, остаётся точной, и её память растёт с каждым токеном.

### Как собрать турбо-файл

Веса модели точных строк лежат в
[репозитории](https://github.com/infosave2007/cmf/tree/master/scripts/dory/router) вместе со скриптами, которые
их строят.

```bash
cortiq far-turbo qwen38-27b-q4tp.cmf qwen38-27b-turbo.cmf --exact-weights keep27.bin
```

Подробнее в [руководстве](https://github.com/infosave2007/cmf/blob/master/docs/QWEN38_TURBO.ru.md).

### Лицензия

Веса: [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0), из
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). Дальнее поле защищается патентной заявкой США
№ 19/738,763; см. [PATENTS.md](https://github.com/infosave2007/cmf/blob/master/PATENTS.md).

---

## 中文说明

[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) 的 4 位权重,文件头让长提示下的显存保持不变。任何长度的提示都能放进 16 GB 显卡;读取 64 000 个 token 用时 177 秒(普通文件 316 秒),此后解码 92 tok/s(普通文件 51 tok/s),文本中的密码和编码依然精确。短于 8192 个 token 的提示使用精确注意力,与[普通文件](https://huggingface.co/infosave/Qwen3.8-27B-cmf)相同。这是一个供 `cortiq` 使用的 [CMF](https://github.com/infosave2007/cmf) 文件。`cortiq` 是 Rust 推理引擎,不需要 Python,也不需要机器学习框架。

### 快速开始

```bash
cargo install cortiq-cli
hf download infosave/Qwen3.8-27B-Turbo-cmf qwen38-27b-turbo.cmf --local-dir .
cortiq run qwen38-27b-turbo.cmf --prompt-file report.txt --greedy --no-think
```

内存也不够,但有高速 SSD 时:

```bash
cortiq run qwen38-27b-turbo.cmf --prompt-file report.txt --greedy --no-think --o1-spill /mnt/ssd/cortiq
```

需要 cortiq 0.8.58 或更新版本;预编译版本在[发布页](https://github.com/infosave2007/cmf/releases)。`--greedy` 开启草稿 token,`--no-think` 要求直接回答。

### 这个文件做了什么

![小显卡上的长提示](https://huggingface.co/infosave/Qwen3.8-27B-Turbo-cmf/resolve/main/assets/ceiling_zh.png)

**固定的上限。** 提示的前 8192 行在显卡上精确读取,16 GB 显卡上为 4096 行。之后各行的键和值一读完就离开显卡,进入内存或 SSD。在最后 512 行之前,由这些行逐层构建远场,此后不再需要它们。远场让前 4 行和后 1800 行保持精确,其余内容存放在每个头 32 个地标中,所以显存和单个 token 的成本不再随文本增长。美国专利申请号 19/738,763。

**精确行。** 只用远场会丢失密码。因此根据各层对每一行的处理为其打分,最好的 248 行保持精确。提示末尾的问题在回答之前会读到它们。

![密码和编码保持精确](https://huggingface.co/infosave/Qwen3.8-27B-Turbo-cmf/resolve/main/assets/exact_rows_zh.png)

**短线程和草稿。** 封存后的 token 和成批的行把注意力拆成七个小阶段,占满整张显卡;而每个头一条长线程会让显卡大部分闲置。MTP 头提出草稿 token,显卡一批验证,远场只前移被接受的部分。贪婪解码时,有无草稿的文本相同。

### 速度

![普通文件与 turbo](https://huggingface.co/infosave/Qwen3.8-27B-Turbo-cmf/resolve/main/assets/compare_zh.png)

RTX 5090,Vulkan,`cortiq bench --core`,在该长度的提示之后生成 60 个 token。

| 提示,token 数 | 首个 token,秒(普通) | 首个 token,秒(**turbo**) | 解码,tok/s(普通) | 解码,tok/s(**turbo**) | 显存峰值,GB(普通) | 显存峰值,GB(**turbo**) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 000 | 2.2 | **2.2** | 105 | **105** | 14.5 | **14.4** |
| 4 000 | 8.8 | **8.8** | 104 | **104** | 14.9 | **14.8** |
| 16 000 | 42 | **47** | 87 | **92** | 16.5 | **15.5** |
| 32 000 | 107 | **92** | 71 | **91** | 18.5 | **15.7** |
| 64 000 | 316 | **177** | 51 | **92** | 22.8 | **15.7** |

真实回答:400 个 token 的维基百科文章摘要,约一半草稿被接受。

| 提示 | 首个 token,秒(普通) | 首个 token,秒(**turbo**) | 解码,tok/s(普通) | 解码,tok/s(**turbo**) |
|---|---:|---:|---:|---:|
| 30 000 字符 | 19.2 | **18.6** | 76.4 | **76.3** |
| 100 000 字符 | 76.7 | **72.1** | 50.9 | **60.4** |

**在 16 GB 显卡上。** 在同一张 5090 上设置这种显卡的预算(`CMF_GPU_VRAM_MB=15000`),文件在 4096 行封存,保持在 15.0 GB。首个 token:16 000 时 43 秒,32 000 时 87 秒,64 000 时 176 秒,解码 92 tok/s。真正的 16 GB 显卡计算单元更少,因此相同显存下速度会更低。

**在 Mac 上。** MacBook Air M4,24 GB,Metal:16 000 个 token 时,进程除内存映射的权重外保持在 6.3 GB,首个 token 545 秒,解码 4 tok/s。插入 40 000 字符中的事实完整返回。M4 在任何长度下解码这个模型约为 5 tok/s,受矩阵运算速度限制,所以在 Mac 上这个文件节省的是内存,而不是时间。草稿在那里不划算(有草稿 4.6 tok/s,没有 4.9 tok/s),默认保持关闭。

**主机内存。** 读取长提示时,它的行占用内存或磁盘。这里不计内存映射的权重,系统可以把它们移出内存。

| 提示,token 数 | 内存 | 使用 `--o1-spill`:内存 | 使用 `--o1-spill`:磁盘 |
|---:|---:|---:|---:|
| 16 000 | 4.4 GB | 2.2 GB | 4.7 GB |
| 32 000 | 7.0 GB | 2.2 GB | 8.7 GB |
| 64 000 | 13.2 GB | 3.0 GB | 18.7 GB |

在写入 4.5 GB/s、读取 5.2 GB/s 的磁盘上,速度与使用内存相同。这些行只在构建远场之前需要。

**质量。** 在 100 000 字符中插入事实,共四类(密钥、访问码、密码、房间号):12/12,其中 8 次使用 15 GB 预算和 SSD;在 230 000 字符(约 53 000 token)中同样路径 6/6。没有精确行的远场为 0/4。封存点之后各行的困惑度,两个 16 384 token 的维基百科窗口:精确注意力 4.76,远场 4.97。

### 参数

| 参数 | 含义 |
|---|---|
| `--o1-spill DIR` | 把长提示的行放在 `DIR` 的文件中,也可用 `CMF_O1_SPILL` |
| `--prompt-file FILE` | 比命令行能容纳的更长的提示 |
| `CMF_O1_PREFIX_MAX=N` | 在 `N` 行封存,不按显卡选择 |
| `CMF_O1=off` | 普通模型,任何长度都用精确注意力 |
| `CMF_O1_SPEC=0` | 远场上不用草稿 |

### 限制

- 远场运行在 Vulkan、DX12 和 Metal 上。在 CPU 上,这个文件按普通文件运行。测量在 NVIDIA 显卡和 M4 上进行。
- 封存点之后的文本是近似的:远处的内容会淡化,保持精确的行除外。选择这些行的拟合只见过插入的事实,其他类型的重要文本没有测试。
- 提示结束后到达的行留在窗口中或进入远场。与已封存对话不一致的回合会重新读取整个提示。
- 在 Vulkan 上,草稿需要不带惩罚的贪婪解码。不加 `--greedy` 时,封存后的 token 约为 57 tok/s。
- 从封存点以下开始的生成保持精确,其内存随每个 token 增长。

### 构建 turbo 文件

精确行拟合的权重及其生成脚本在[仓库](https://github.com/infosave2007/cmf/tree/master/scripts/dory/router)中。

```bash
cortiq far-turbo qwen38-27b-q4tp.cmf qwen38-27b-turbo.cmf --exact-weights keep27.bin
```

详见[指南](https://github.com/infosave2007/cmf/blob/master/docs/QWEN38_TURBO.zh.md)。

### 许可证

权重:[Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0),来自 [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)。远场受美国专利申请号 19/738,763 保护;见 [PATENTS.md](https://github.com/infosave2007/cmf/blob/master/PATENTS.md)。
