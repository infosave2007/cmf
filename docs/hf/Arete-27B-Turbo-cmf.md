---
license: apache-2.0
library_name: cortiq
base_model: NousResearch/Arete-27B
base_model_relation: quantized
pipeline_tag: text-generation
tags:
  - cmf
  - cortiq
  - quantized
  - q4tp
  - arete
  - qwen3.8
  - long-context
  - low-vram
  - vulkan
  - cuda
language:
  - en
  - ru
  - zh
---

# Arete-27B Turbo — CMF

The 4-bit weights of [Arete-27B](https://huggingface.co/NousResearch/Arete-27B), Nous Research's
post-training of Qwen3.8-27B, under a header that holds device memory in place on long prompts: the
technology of [Qwen3.8-27B Turbo](https://huggingface.co/infosave/Qwen3.8-27B-Turbo-cmf). At 64 000 tokens it holds 14.8 GB of device memory against 21.5,
gives the first token in 443 s against 612 and decodes at 27.2 tok/s against 16.2, with passwords and codes
in the text still exact.
Prompts under 8192 tokens run exact attention. It is one [CMF](https://github.com/infosave2007/cmf) file
for `cortiq`, a Rust inference engine with no Python and no ML framework. On an NVIDIA card through CUDA
it decodes at 49 tok/s, and with the DFlash 2 drafter beside it at up to 102 tok/s on code.

[English](#quick-start) · [Русский](#документация-на-русском) · [中文](#中文说明)

## Quick start

```bash
cargo install cortiq-cli
hf download infosave/Arete-27B-Turbo-cmf arete-27b-turbo.cmf --local-dir .
cortiq run arete-27b-turbo.cmf --prompt-file report.txt --no-think
```

Short on RAM as well, with a fast SSD:

```bash
cortiq run arete-27b-turbo.cmf --prompt-file report.txt --no-think --o1-spill /mnt/ssd/cortiq
```

On an NVIDIA card, through CUDA (cortiq 0.8.69 or later):

```bash
cargo install cortiq-cli --features cuda
hf download infosave/Arete-27B-Turbo-cmf arete-27b-turbo.cmf arete-27b-turbo.dflash.cmf --local-dir .
CMF_GPU=cuda CMF_DFLASH=1 cortiq run arete-27b-turbo.cmf --prompt-file report.txt --greedy --no-think
```

Measured with cortiq 0.8.63 (Vulkan) and 0.8.69 (CUDA). `--no-think` asks for a direct answer; without
it the model reasons first, so raise `--max-tokens` (default 256).

| file | quantization | size | sha256 |
|---|---|---:|---|
| `arete-27b-turbo.cmf` | 4-bit, ladder scales, text only | 14.1 GB | `ce1506fdc89909381371537b9b4aa1e97c3570961ad2cb09d71f4ff17be93c3d` |
| `arete-27b-turbo.dflash.cmf` | DFlash 2 drafter, 4-bit (optional, CUDA) | 1.3 GB | `11e4e99442bebafc3249218322bcfd86d460f58f3fd72844fc63c82c2295b414` |

## What the file does

![A long prompt on a small card](https://huggingface.co/infosave/Arete-27B-Turbo-cmf/resolve/main/assets/ceiling_en.png)

**A fixed ceiling.** The first 8192 rows of a prompt are read exactly on the card, 4096 on a 16 GB
card. The keys and values of later rows leave the card as they are read, for RAM or an SSD. Before the
last 512 rows, the far field is built from them one layer at a time, and the rows go away. A far field
keeps the first 4 rows and the last 1800 exact and everything else in 32 landmarks per head, so device
memory and the cost of a token stop growing with the text. Patent application No. 19/738,763.

**Exact rows.** A far field alone loses a password. So each row is scored by what the layers do with
it, and the 248 best stay exact. A question at the end of the prompt reads them before it is answered.
The weights of that score were fitted on Qwen3.8-27B and are used here as they are: Arete keeps the
architecture.

![Passwords and codes stay exact](https://huggingface.co/infosave/Arete-27B-Turbo-cmf/resolve/main/assets/exact_rows_en.png)

**Short threads.** A sealed token and a chunk of rows take the attention apart into seven small stages
that fill the whole card, where one long thread per head leaves most of it idle.

## Performance

![Standard and turbo](https://huggingface.co/infosave/Arete-27B-Turbo-cmf/resolve/main/assets/compare_en.png)

NVIDIA A40 (48 GB), Vulkan, `cortiq bench --core`, 60 tokens after a prompt of that length. Standard is
the same weights without the turbo header.

| prompt, tokens | first token, s (standard) | first token, s (**turbo**) | decode, tok/s (standard) | decode, tok/s (**turbo**) | peak VRAM, GB (standard) | peak VRAM, GB (**turbo**) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 000 | 5.1 | **5.1** | 28.2 | **27.8** | 13.3 | **13.3** |
| 4 000 | 20.1 | **20.0** | 27.3 | **27.5** | 13.7 | **13.7** |
| 16 000 | 92 | **115** | 24.0 | **27.0** | 15.2 | **14.8** |
| 32 000 | 223 | **221** | 20.9 | **27.0** | 17.2 | **14.8** |
| 64 000 | 612 | **443** | 16.2 | **27.2** | 21.5 | **14.8** |

A real answer: a one-paragraph summary of 100 000 characters of Wikipedia (about 24 600 tokens), correct,
first token 177 s, decode 27.0 tok/s.

**On a 16 GB card.** With the budget of such a card set on the same A40 (`CMF_GPU_VRAM_MB=15000`), the
file seals at 4096 rows and stays at 14.3 GB: first token 107 s at 16 000, 217 s at 32 000 and 436 s at
64 000, decode 27 tok/s. The standard file needs 17.2 GB at 32 000.

**Host memory.** While a long prompt is read, its rows take RAM: 4.4 GB at 16 000, 7.0 GB at 32 000 and
13.2 GB at 64 000 tokens, beside the mapped weights. `--o1-spill` moves them to a disk; they are needed
only until the far field is built.

**Quality.** A planted fact in 100 000 characters, four kinds of them (a pass key, an access code, a password, a
room number): 5 of 5. A far field without exact rows finds 0 of 2.

An answer of the file, `--greedy --no-think`:

> **What happened in Tiananmen Square in June 1989? Answer in four sentences.**
>
> In June 1989, the Chinese government deployed military forces to Tiananmen Square in Beijing to
> forcibly disperse pro-democracy protesters. The crackdown, which occurred primarily on the night of
> June 3 and into June 4, resulted in the deaths of hundreds of civilians and soldiers, though exact
> figures remain classified and disputed. This event, often referred to as the Tiananmen Square
> protests or the June 4 incident, marked a decisive end to the 1989 democracy movement in China. Its
> aftermath led to widespread international condemnation and remains a heavily censored topic within
> the People's Republic of China.

## NVIDIA via CUDA

```bash
cargo install cortiq-cli --features cuda
cortiq gpu      # says whether the card and the CUDA libraries are found
CMF_GPU=cuda CMF_DFLASH=1 cortiq run arete-27b-turbo.cmf --prompt-file report.txt --greedy --no-think
CMF_GPU=cuda CMF_DFLASH=1 cortiq serve arete-27b-turbo.cmf --host 127.0.0.1 --port 8080
```

The build links nothing; at run time the backend opens the driver's `libcuda` and the CUDA 12.x
libraries `libnvrtc` and `libcublas`. `arete-27b-turbo.dflash.cmf` is
[z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) converted for this file:
Arete has no draft head of its own, so the drafter is what makes it faster. `CMF_DFLASH=1` picks it up
beside the file.

RTX 3090 (24 GB), cortiq 0.8.69, `--greedy --no-think`:

| answer | plain, tok/s | **DFlash 2**, tok/s | drafts kept |
|---|---:|---:|---:|
| the question above, 127 tokens | 49.0 | **70.3** | 42 % |
| a Python LRU cache module with tests, 256 tokens | 49.2 | **102.1** | 54 % |

The text with the drafter is the text without it, and the answer above is the same as on Vulkan.

Long prompts, `cortiq bench --core`, 60 tokens after the prompt:

| prompt, tokens | prompt, tok/s | first token, s | decode, tok/s |
|---:|---:|---:|---:|
| 4 000 | 1 025 | 3.9 | 47.9 |
| 32 000 | 609 | 52.5 | 37.7 |

The weights take 13.3 GB of the card, the far field 301 MB at 32 000 tokens. A pass key in 60 000
characters of Wikipedia: 2 of 2, first token 22 s.

## Options

| option | meaning |
|---|---|
| `--o1-spill DIR` | a long prompt's rows in files in `DIR`, also `CMF_O1_SPILL` |
| `--prompt-file FILE` | a prompt longer than a command line holds |
| `CMF_O1_PREFIX_MAX=N` | seal at `N` rows instead of the choice for the card |
| `CMF_O1=off` | the plain model, with exact attention at any length |
| `CMF_GPU=cuda` | the CUDA backend (a build with `--features cuda`) |
| `CMF_DFLASH=1` | the DFlash drafter beside the file drafts tokens (CUDA) |

## Limits

- The far field runs on Vulkan, DX12, Metal and CUDA. On the CPU this file runs as the standard one. It
  was measured on an NVIDIA A40 (Vulkan) and an RTX 3090 (CUDA).
- Past the seal the text is approximate: what lies far back fades, apart from the rows kept exact. The
  fit behind those rows saw planted facts, and other kinds of important text were not tested.
- Arete-27B ships without the MTP head of Qwen3.8-27B, so only the DFlash drafter drafts tokens, and only
  before the seal.
- The file holds the language model only; Arete's vision tower is not in it.
- A generation that starts below the seal stays exact, and its memory grows with every token.

## How the file was made

```bash
cortiq convert --model NousResearch/Arete-27B --quant q4tp --output arete-27b-q4tp.cmf
cortiq far-turbo arete-27b-q4tp.cmf arete-27b-turbo.cmf --exact-weights keep27.bin
```

`keep27.bin` and the scripts that fit it are in the
[repository](https://github.com/infosave2007/cmf/tree/master/scripts/dory/router). The 4-bit weights
are quantized directly from the bf16 checkpoint.

## License

Weights: [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0), from
[NousResearch/Arete-27B](https://huggingface.co/NousResearch/Arete-27B), itself a post-training of
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). The far field is a subject of U.S. patent
application No. 19/738,763; see
[PATENTS.md](https://github.com/infosave2007/cmf/blob/master/PATENTS.md).

---

## Документация на русском

4-битные веса [Arete-27B](https://huggingface.co/NousResearch/Arete-27B), дообучения Qwen3.8-27B от
Nous Research, с заголовком, который держит память карты на месте при длинных промптах: технология
[Qwen3.8-27B Turbo](https://huggingface.co/infosave/Qwen3.8-27B-Turbo-cmf). На 64 000 токенов файл держит 14,8 ГБ памяти карты вместо 21,5, даёт первый
токен через 443 с вместо 612 и декодирует 27,2 ток/с вместо 16,2, а пароли и коды в тексте остаются точными. Промпты короче
8192 токенов идут с точным вниманием. Это один [CMF](https://github.com/infosave2007/cmf)-файл для
`cortiq`, движка на Rust без Python и ML-фреймворков. На карте NVIDIA через CUDA он декодирует
49 ток/с, а с драфтером DFlash 2 рядом — до 102 ток/с на коде.

### Быстрый старт

```bash
cargo install cortiq-cli
hf download infosave/Arete-27B-Turbo-cmf arete-27b-turbo.cmf --local-dir .
cortiq run arete-27b-turbo.cmf --prompt-file report.txt --no-think
```

Если мало и RAM, а есть быстрый SSD:

```bash
cortiq run arete-27b-turbo.cmf --prompt-file report.txt --no-think --o1-spill /mnt/ssd/cortiq
```

На карте NVIDIA, через CUDA (cortiq 0.8.69 или новее):

```bash
cargo install cortiq-cli --features cuda
hf download infosave/Arete-27B-Turbo-cmf arete-27b-turbo.cmf arete-27b-turbo.dflash.cmf --local-dir .
CMF_GPU=cuda CMF_DFLASH=1 cortiq run arete-27b-turbo.cmf --prompt-file report.txt --greedy --no-think
```

Замерено на cortiq 0.8.63 (Vulkan) и 0.8.69 (CUDA). `--no-think` просит прямой ответ; без него модель
сначала рассуждает, так что поднимите `--max-tokens` (по умолчанию 256).

| файл | квантизация | размер | sha256 |
|---|---|---:|---|
| `arete-27b-turbo.cmf` | 4 бита, лестничные шкалы, только текст | 14,1 ГБ | `ce1506fdc89909381371537b9b4aa1e97c3570961ad2cb09d71f4ff17be93c3d` |
| `arete-27b-turbo.dflash.cmf` | драфтер DFlash 2, 4 бита (необязательно, CUDA) | 1,3 ГБ | `11e4e99442bebafc3249218322bcfd86d460f58f3fd72844fc63c82c2295b414` |

### Что делает файл

![Длинный промпт на маленькой карте](https://huggingface.co/infosave/Arete-27B-Turbo-cmf/resolve/main/assets/ceiling_ru.png)

**Постоянный потолок.** Первые 8192 строки промпта читаются на карте точно, 4096 на карте 16 ГБ.
Ключи и значения следующих строк уходят с карты по мере чтения, в RAM или на SSD. До последних 512 строк
из них слой за слоем строится дальнее поле, и строки уходят. Дальнее поле держит точными первые 4 строки
и последние 1800, а всё остальное хранит в 32 опорных точках на голову, так что память карты и цена
токена перестают расти вместе с текстом. Патентная заявка № 19/738,763.

**Точные строки.** Одно дальнее поле теряет пароль. Поэтому каждая строка получает оценку по тому, что
с ней делают слои, и 248 лучших остаются точными. Вопрос в конце промпта читает их до ответа. Веса этой
оценки подобраны на Qwen3.8-27B и взяты как есть: архитектура у Arete та же.

![Пароли и коды остаются точными](https://huggingface.co/infosave/Arete-27B-Turbo-cmf/resolve/main/assets/exact_rows_ru.png)

**Короткие потоки.** Запечатанный токен и кусок строк разбирают внимание на семь маленьких стадий,
которые занимают всю карту, а один длинный поток на голову оставляет большую её часть без дела.

### Производительность

![Обычный и турбо](https://huggingface.co/infosave/Arete-27B-Turbo-cmf/resolve/main/assets/compare_ru.png)

NVIDIA A40 (48 ГБ), Vulkan, `cortiq bench --core`, 60 токенов после промпта такой длины. Обычный — те же
веса без турбо-заголовка.

| промпт, токенов | первый токен, с (обычный) | первый токен, с (**турбо**) | декод, ток/с (обычный) | декод, ток/с (**турбо**) | пик VRAM, ГБ (обычный) | пик VRAM, ГБ (**турбо**) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 000 | 5,1 | **5,1** | 28,2 | **27,8** | 13,3 | **13,3** |
| 4 000 | 20,1 | **20,0** | 27,3 | **27,5** | 13,7 | **13,7** |
| 16 000 | 92 | **115** | 24,0 | **27,0** | 15,2 | **14,8** |
| 32 000 | 223 | **221** | 20,9 | **27,0** | 17,2 | **14,8** |
| 64 000 | 612 | **443** | 16,2 | **27,2** | 21,5 | **14,8** |

Реальный ответ: пересказ в один абзац 100 000 символов Википедии (около 24 600 токенов), верный, первый
токен через 177 с, декод 27,0 ток/с.

**На карте 16 ГБ.** С бюджетом такой карты на той же A40 (`CMF_GPU_VRAM_MB=15000`) файл запечатывается
на 4096 строках и держится на 14,3 ГБ: первый токен 107 с на 16 000, 217 с на 32 000 и 436 с на 64 000,
декод 27 ток/с. Обычному файлу на 32 000 нужно 17,2 ГБ.

**Память хоста.** Пока читается длинный промпт, его строки занимают RAM: 4,4 ГБ на 16 000, 7,0 ГБ на
32 000 и 13,2 ГБ на 64 000 токенов, помимо отображённых весов. `--o1-spill` уносит их на диск; нужны они
только до сборки дальнего поля.

**Качество.** Факт, спрятанный в 100 000 символов, четырёх видов (ключ, код доступа, пароль, номер комнаты):
5 из 5. Дальнее поле без точных строк находит 0 из 2.

### NVIDIA через CUDA

```bash
cargo install cortiq-cli --features cuda
cortiq gpu      # покажет, найдены ли карта и библиотеки CUDA
CMF_GPU=cuda CMF_DFLASH=1 cortiq run arete-27b-turbo.cmf --prompt-file report.txt --greedy --no-think
CMF_GPU=cuda CMF_DFLASH=1 cortiq serve arete-27b-turbo.cmf --host 127.0.0.1 --port 8080
```

Сборка ничего не линкует; при запуске бэкенд открывает `libcuda` из драйвера и библиотеки CUDA 12.x
`libnvrtc` и `libcublas`. `arete-27b-turbo.dflash.cmf` — это
[z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2), сконвертированный под этот
файл: своей головы черновиков у Arete нет, и ускоряет её именно драфтер. `CMF_DFLASH=1` подхватывает его
рядом с файлом.

RTX 3090 (24 ГБ), cortiq 0.8.69, `--greedy --no-think`:

| ответ | обычный, ток/с | **DFlash 2**, ток/с | принято черновиков |
|---|---:|---:|---:|
| вопрос выше, 127 токенов | 49,0 | **70,3** | 42 % |
| модуль LRU-кэша на Python с тестами, 256 токенов | 49,2 | **102,1** | 54 % |

Текст с драфтером совпадает с текстом без него, а ответ выше — с ответом на Vulkan.

Длинные промпты, `cortiq bench --core`, 60 токенов после промпта:

| промпт, токенов | промпт, ток/с | первый токен, с | декод, ток/с |
|---:|---:|---:|---:|
| 4 000 | 1 025 | 3,9 | 47,9 |
| 32 000 | 609 | 52,5 | 37,7 |

Веса занимают 13,3 ГБ карты, дальнее поле — 301 МБ на 32 000 токенов. Ключ в 60 000 символов Википедии:
2 из 2, первый токен через 22 с.

`CMF_GPU=cuda` выбирает бэкенд CUDA (сборка с `--features cuda`), `CMF_DFLASH=1` включает драфтер рядом с
файлом.

### Ограничения

- Дальнее поле работает на Vulkan, DX12, Metal и CUDA. На CPU файл работает как обычный. Замерено на
  NVIDIA A40 (Vulkan) и RTX 3090 (CUDA).
- За печатью текст приближённый: дальнее уходит в туман, кроме точных строк. Модель, выбирающая эти
  строки, видела вставленные факты; другие виды важного текста не проверялись.
- У Arete-27B нет MTP-головы Qwen3.8-27B, поэтому черновики пишет только драфтер DFlash, и только до
  печати.
- В файле только языковая модель; визуальной башни Arete в нём нет.
- Генерация, начатая до печати, остаётся точной, и её память растёт с каждым токеном.

### Как сделан файл

```bash
cortiq convert --model NousResearch/Arete-27B --quant q4tp --output arete-27b-q4tp.cmf
cortiq far-turbo arete-27b-q4tp.cmf arete-27b-turbo.cmf --exact-weights keep27.bin
```

`keep27.bin` и скрипты его подбора лежат в
[репозитории](https://github.com/infosave2007/cmf/tree/master/scripts/dory/router). 4-битные веса
квантованы прямо из bf16.

### Лицензия

Веса: [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0), от
[NousResearch/Arete-27B](https://huggingface.co/NousResearch/Arete-27B), дообучения
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). Дальнее поле — предмет патентной заявки США
№ 19/738,763; см. [PATENTS.md](https://github.com/infosave2007/cmf/blob/master/PATENTS.md).

---

## 中文说明

[Arete-27B](https://huggingface.co/NousResearch/Arete-27B)（Nous Research 基于 Qwen3.8-27B 的后训练模型）的
4 位权重，带有一个在长提示下让显存保持不变的文件头，技术与
[Qwen3.8-27B Turbo](https://huggingface.co/infosave/Qwen3.8-27B-Turbo-cmf) 相同。在 64 000 个 token 时，显存为 14.8 GB（普通文件 21.5 GB），首个 token 用时 443 秒（普通 612 秒），解码
27.2 tok/s（普通 16.2），文本中的密码和编码仍保持精确。 短于 8192 个
token 的提示使用精确注意力。这是一个供 `cortiq` 使用的 [CMF](https://github.com/infosave2007/cmf) 文件，
`cortiq` 是一个不依赖 Python 和机器学习框架的 Rust 推理引擎。在 NVIDIA 显卡上通过 CUDA 解码 49 tok/s，
配上 DFlash 2 草稿模型后代码解码最高 102 tok/s。

### 快速开始

```bash
cargo install cortiq-cli
hf download infosave/Arete-27B-Turbo-cmf arete-27b-turbo.cmf --local-dir .
cortiq run arete-27b-turbo.cmf --prompt-file report.txt --no-think
```

内存也不够、但有高速 SSD 时：

```bash
cortiq run arete-27b-turbo.cmf --prompt-file report.txt --no-think --o1-spill /mnt/ssd/cortiq
```

在 NVIDIA 显卡上通过 CUDA 运行（cortiq 0.8.69 或更新版本）：

```bash
cargo install cortiq-cli --features cuda
hf download infosave/Arete-27B-Turbo-cmf arete-27b-turbo.cmf arete-27b-turbo.dflash.cmf --local-dir .
CMF_GPU=cuda CMF_DFLASH=1 cortiq run arete-27b-turbo.cmf --prompt-file report.txt --greedy --no-think
```

使用 cortiq 0.8.63（Vulkan）和 0.8.69（CUDA）测得。`--no-think` 要求直接回答；不加时模型会先推理，请调大
`--max-tokens`（默认 256）。

| 文件 | 量化 | 大小 | sha256 |
|---|---|---:|---|
| `arete-27b-turbo.cmf` | 4 位，阶梯缩放，仅文本 | 14.1 GB | `ce1506fdc89909381371537b9b4aa1e97c3570961ad2cb09d71f4ff17be93c3d` |
| `arete-27b-turbo.dflash.cmf` | DFlash 2 草稿模型，4 位（可选，CUDA） | 1.3 GB | `11e4e99442bebafc3249218322bcfd86d460f58f3fd72844fc63c82c2295b414` |

### 文件做了什么

![小显卡上的长提示](https://huggingface.co/infosave/Arete-27B-Turbo-cmf/resolve/main/assets/ceiling_zh.png)

**固定的上限**。提示的前 8192 行在显卡上精确读取，16 GB 显卡上为 4096 行。之后各行的键和值一读完就离开
显卡，进入内存或 SSD。在最后 512 行之前，远场用它们逐层构建，随后这些行被释放。远场让前 4 行和最后 1800 行
保持精确，其余部分存为每个头 32 个地标，因此显存和每个 token 的成本不再随文本增长。专利申请号 19/738,763。

**精确行**。仅靠远场会丢失密码。因此根据各层对每一行的处理给它打分，得分最高的 248 行保持精确。提示末尾的
问题在回答前会读取它们。打分的权重在 Qwen3.8-27B 上拟合，这里原样使用：Arete 的架构没有改变。

![密码和编码保持精确](https://huggingface.co/infosave/Arete-27B-Turbo-cmf/resolve/main/assets/exact_rows_zh.png)

**短线程**。封存后的 token 和一块行把注意力拆成七个小阶段，占满整张显卡；每个头一个长线程则会让显卡大部分
空闲。

### 性能

![普通与 Turbo](https://huggingface.co/infosave/Arete-27B-Turbo-cmf/resolve/main/assets/compare_zh.png)

NVIDIA A40(48 GB),Vulkan,`cortiq bench --core`，在该长度的提示后生成 60 个 token。普通文件是同样的权重，
没有 Turbo 文件头。

| 提示，token | 首个 token，秒（普通） | 首个 token，秒（**Turbo**） | 解码，tok/s（普通） | 解码，tok/s（**Turbo**） | 显存峰值，GB（普通） | 显存峰值，GB（**Turbo**） |
|---:|---:|---:|---:|---:|---:|---:|
| 1 000 | 5.1 | **5.1** | 28.2 | **27.8** | 13.3 | **13.3** |
| 4 000 | 20.1 | **20.0** | 27.3 | **27.5** | 13.7 | **13.7** |
| 16 000 | 92 | **115** | 24.0 | **27.0** | 15.2 | **14.8** |
| 32 000 | 223 | **221** | 20.9 | **27.0** | 17.2 | **14.8** |
| 64 000 | 612 | **443** | 16.2 | **27.2** | 21.5 | **14.8** |

真实回答：用一段话概括 100 000 个字符的维基百科文本（约 24 600 个 token），回答正确，首个 token 177 秒，解码
27.0 tok/s。

**16 GB 显卡**。在同一张 A40 上设置这种显卡的预算（`CMF_GPU_VRAM_MB=15000`），文件在 4096 行处封存，显存保持在
14.3 GB：16 000 时首个 token 107 秒，32 000 时 217 秒，64 000 时 436 秒，解码 27 tok/s。普通文件在 32 000 时
需要 17.2 GB。

**主机内存**。读取长提示时，它的行占用内存：16 000 token 时 4.4 GB，32 000 时 7.0 GB，64 000 时 13.2 GB，
不含映射的权重。`--o1-spill` 把它们移到磁盘；只有在远场构建完成之前才需要它们。

**质量**。在 100 000 个字符中插入一个事实，共四类（密钥、访问码、密码、房间号）：5/5。没有精确行的远场：0/2。

### NVIDIA（CUDA）

```bash
cargo install cortiq-cli --features cuda
cortiq gpu      # 显示是否找到显卡和 CUDA 库
CMF_GPU=cuda CMF_DFLASH=1 cortiq run arete-27b-turbo.cmf --prompt-file report.txt --greedy --no-think
CMF_GPU=cuda CMF_DFLASH=1 cortiq serve arete-27b-turbo.cmf --host 127.0.0.1 --port 8080
```

构建时不链接任何库；运行时后端打开驱动的 `libcuda` 和 CUDA 12.x 库 `libnvrtc`、`libcublas`。
`arete-27b-turbo.dflash.cmf` 是为此文件转换的 [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2)：
Arete 没有自己的草稿头，提速靠的就是这个草稿模型。`CMF_DFLASH=1` 会加载文件旁边的它。

RTX 3090（24 GB），cortiq 0.8.69，`--greedy --no-think`：

| 回答 | 普通，tok/s | **DFlash 2**，tok/s | 草稿接受率 |
|---|---:|---:|---:|
| 上面的问题，127 个 token | 49.0 | **70.3** | 42% |
| 带测试的 Python LRU 缓存模块，256 个 token | 49.2 | **102.1** | 54% |

使用草稿模型时的文本与不使用时相同，上面的回答也与 Vulkan 上的相同。

长提示，`cortiq bench --core`，提示之后生成 60 个 token：

| 提示，token | 提示处理，tok/s | 首个 token，秒 | 解码，tok/s |
|---:|---:|---:|---:|
| 4 000 | 1 025 | 3.9 | 47.9 |
| 32 000 | 609 | 52.5 | 37.7 |

权重占用 13.3 GB 显存，32 000 token 时远场占 301 MB。在 60 000 字符的维基百科文本中查找密钥：2/2，首个 token 22 秒。

`CMF_GPU=cuda` 选择 CUDA 后端（需用 `--features cuda` 构建），`CMF_DFLASH=1` 启用文件旁的草稿模型。

### 限制

- 远场在 Vulkan、DX12、Metal 和 CUDA 上运行。在 CPU 上，此文件按普通文件运行。在 NVIDIA A40（Vulkan）和
  RTX 3090（CUDA）上测量过。
- 封存之后的文本是近似的：远处的内容会淡化，保持精确的行除外。选择这些行的拟合见过插入的事实，其他类型的
  重要文本未经测试。
- Arete-27B 没有 Qwen3.8-27B 的 MTP 头，因此只有 DFlash 草稿模型生成草稿 token，且仅在封存之前。
- 文件只包含语言模型，不含 Arete 的视觉塔。
- 在封存前开始的生成保持精确，其内存随每个 token 增长。

### 文件的制作

```bash
cortiq convert --model NousResearch/Arete-27B --quant q4tp --output arete-27b-q4tp.cmf
cortiq far-turbo arete-27b-q4tp.cmf arete-27b-turbo.cmf --exact-weights keep27.bin
```

`keep27.bin` 及其拟合脚本在[代码仓库](https://github.com/infosave2007/cmf/tree/master/scripts/dory/router)中。
4 位权重直接从 bf16 检查点量化。

### 许可证

权重：[Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)，来自
[NousResearch/Arete-27B](https://huggingface.co/NousResearch/Arete-27B)，它是
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) 的后训练版本。远场是美国专利申请
No. 19/738,763 的主题；见 [PATENTS.md](https://github.com/infosave2007/cmf/blob/master/PATENTS.md)。
