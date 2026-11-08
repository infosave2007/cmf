---
license: apache-2.0
library_name: cortiq
base_model:
  - XHToken/Spark-X2.6-4B
base_model_relation: quantized
pipeline_tag: text-generation
tags:
  - cmf
  - cortiq
  - quantized
  - q4tp
  - long-context
  - low-vram
  - metal
  - vulkan
  - sliding-window-attention
  - tool-calling
language:
  - multilingual
  - en
  - zh
  - ru
---

# Spark-X2.6-4B Turbo — CMF

The 4-bit weights of [Spark-X2.6-4B](https://huggingface.co/XHToken/Spark-X2.6-4B) under a header that
holds device memory and decode speed in place on long prompts. On an RTX 2000 Ada at 64 000 tokens it
decodes at 56.2 tok/s instead of 18.5 and takes 4.2 GB of the card instead of 7.9. On a MacBook Air M4
after a 14 000-token prompt it decodes at 33.0 tok/s where the standard file gives 31.0 and stock mlx-lm
30.9, in 2.9 GB of memory against mlx-lm's 5.1. Passwords and codes in the text stay exact. Prompts under
8192 tokens run exact attention, as in the
[standard file](https://huggingface.co/infosave/Spark-X2.6-4B-cmf). It is one
[CMF](https://github.com/infosave2007/cmf) file for `cortiq`, a Rust inference engine with no Python and
no ML framework, on Metal, Vulkan, DX12 and the CPU.

[English](#quick-start) · [Русский](#документация-на-русском) · [中文](#中文说明)

## Quick start

```bash
cargo install cortiq-cli
hf download infosave/Spark-X2.6-4B-turbo-cmf spark-x2.6-4b-q4tp-turbo.cmf --local-dir .
CMF_MAX_SEQ=131072 cortiq run spark-x2.6-4b-q4tp-turbo.cmf --prompt-file report.txt --greedy --no-think
```

An OpenAI-compatible server, with the prompt and decode speed of each request printed as llama-server
does:

```bash
CMF_MAX_SEQ=131072 cortiq serve spark-x2.6-4b-q4tp-turbo.cmf --port 8080
```

Use cortiq 0.8.64 or later; prebuilt binaries are on the
[releases page](https://github.com/infosave2007/cmf/releases). `CMF_MAX_SEQ` lifts the default cap of
32 768 tokens. On a Mac, `CMF_ANE=1` sends the prompt's FFNs to the Neural Engine: a 14 000-token prompt
in 33.6 s instead of 54.8.

## What the file does

![A long prompt on a small card](https://huggingface.co/infosave/Spark-X2.6-4B-turbo-cmf/resolve/main/assets/ceiling_en.png)

**A fixed ceiling.** 27 of the model's 36 layers look back 512 rows and never grow. The 9 full-attention
layers read the first 8192 rows of a prompt exactly on the device; the keys and values of later rows leave
it as they are read, for RAM. Before the last 512 rows, the far field is built from them one layer at a
time, and the rows go away. A far field keeps the first 4 rows and the last 1800 exact and everything else
in 32 landmarks per head, so device memory and the cost of a token stop growing with the text. Patent
application No. 19/738,763.

**Exact rows.** A far field alone loses a password. So each row is scored by what the layers do with it,
and the 248 best stay exact. A question at the end of the prompt reads them before it is answered.

![Passwords and codes stay exact](https://huggingface.co/infosave/Spark-X2.6-4B-turbo-cmf/resolve/main/assets/exact_rows_en.png)

## On a Mac

MacBook Air M4 (10-core GPU, 24 GB, fanless), the same prompts (Wikipedia text and one instruction),
96 greedy tokens after each, one run at a time with a minute of rest between runs. cortiq on Metal;
[mlx-lm 0.32.2](https://github.com/ml-explore/mlx-lm) on a 4-bit MLX build of the model (group size 64,
2.2 GB); llama.cpp b12202 built for arm64 with Metal, Q4_K_M (3.5 GB), `-c` sized to the prompt. The
turbo and the standard file hold the same 2.16 GB of weights.

![Four engines on a MacBook Air M4](https://huggingface.co/infosave/Spark-X2.6-4B-turbo-cmf/resolve/main/assets/mac_depth_en.png)

| prompt, tokens | first token, s | | | | | decode, tok/s | | | |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| | cortiq | **turbo** | cortiq + ANE | mlx-lm | llama.cpp | cortiq | **turbo** | mlx-lm | llama.cpp |
| 2 000 | 6.6 | **6.5** | 3.4 | 6.6 | 6.2 | 36.4 | **36.3** | 36.4 | 28.8 |
| 14 000 | 54.8 | **56.4** | 33.6 | 53.1 | 52.5 | 31.0 | **33.0** | 30.9 | 22.3 |
| 32 000 | 157.9 | **131.2** | 109.8 | 144.2 | 170.1 | 24.2 | **33.1** | 20.8 | 15.0 |
| 64 000 | 426.5 | **262.1** | 334.7 | 351.5 | 494.2 | 24.5 | **33.5** | 12.6 | 8.8 |

Up to 14 000 tokens the four engines read a prompt at one speed, and cortiq, the turbo file and mlx-lm
decode at one speed, llama.cpp a quarter behind. Past that the turbo file holds 33 tok/s: at 64 000
tokens mlx-lm is at 12.6, llama.cpp at 8.8, the standard cortiq file at 24.5. It reads the 64 000-token
prompt in 262 s against 352 (mlx-lm) and 494 (llama.cpp). The Neural Engine takes the first token of a
14 000-token prompt from 54.8 s to 33.6 s and leaves decode as it is. mlx-lm's first-token times are
its prompt length over the prompt tok/s it reports; the others are measured at the first token.

![Memory after the prompt](https://huggingface.co/infosave/Spark-X2.6-4B-turbo-cmf/resolve/main/assets/memory_en.png)

| after a prompt of | cortiq | **turbo** | mlx-lm 4-bit | llama.cpp Q4_K_M |
|---|---:|---:|---:|---:|
| 2 000 tokens: footprint / resident, GB | 1.20 / 3.06 | **1.20 / 3.05** | 3.74 / 2.40 | 0.33 / 3.61 |
| 14 000 tokens: footprint / resident, GB | 2.60 / 4.37 | **2.85 / 4.70** | 5.09 / 2.43 | 0.83 / 4.11 |

The footprint is what Activity Monitor shows as Memory: the pages a process holds for itself, its GPU
buffers included. cortiq and llama.cpp map the weights file, so those pages are the system's file
cache — in the resident set, not in the footprint, and the first to go when memory is short. mlx-lm
copies the weights into its own buffers, so its footprint carries them. Both numbers are the peak over
the run, sampled every 0.25 s.

**Perplexity.** WikiText-2 test, the walk of `llama-perplexity -c 512`: 64 chunks of 512 raw tokens
from the start of the text, the second half of each chunk scored (16 320 predictions). cortiq
`ppl --window-len 512 --window-stride 512 --window-score-from 256`, mlx-lm with the same chunks, all on
the Mac.

| file | size, GB | perplexity |
|---|---:|---:|
| cortiq q4tp (this file and the standard one) | 2.16 | 32.95 |
| mlx-lm 4-bit, group 64 | 2.2 | 31.91 |
| llama.cpp Q4_K_M | 3.54 | 26.75 |
| cortiq q8_2f | 4.14 | 26.06 |
| mlx-lm 8-bit | 4.1 | 25.46 |
| llama.cpp Q8_0 | 4.37 | 26.43 |

At the same 4-bit size the q4tp file is 3 % above the MLX one; llama.cpp's Q4_K_M keeps its attention
and part of the FFNs at 6 and 8 bits, is 1.4 GB larger and sits where the 8-bit files sit. The turbo
header does not change these numbers: a 512-token chunk is read exactly.

## On an NVIDIA card

![Standard and turbo](https://huggingface.co/infosave/Spark-X2.6-4B-turbo-cmf/resolve/main/assets/compare_en.png)

RTX 2000 Ada (16 GB), Vulkan, `cortiq bench --core`, 60 tokens after a prompt of that length.

| prompt, tokens | first token, s (standard) | first token, s (**turbo**) | decode, tok/s (standard) | decode, tok/s (**turbo**) | peak VRAM, GB (standard) | peak VRAM, GB (**turbo**) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 000 | 1.4 | **1.4** | 57.2 | **56.2** | 3.5 | **3.5** |
| 16 000 | 17.9 | **29.4** | 37.9 | **56.2** | 4.5 | **4.2** |
| 32 000 | 43.3 | **57.1** | 28.0 | **56.0** | 5.6 | **4.2** |
| 64 000 | 119.6 | **116.3** | 18.5 | **56.2** | 7.9 | **4.2** |

Past the seal the turbo file decodes at 56 tok/s at any length and keeps the card at 4.2 GB. Where its
cache fits, the standard file reads a 16 000- or 32 000-token prompt faster; from 64 000 tokens on the
turbo file is the faster on the prompt too. The far field is built on the host, so the process holds
more RAM while a long prompt is read: 4.9 GB at 16 000 tokens, 7.2 at 32 000 and 11.9 at 64 000, against
2.4 GB for the standard file at any length.

![Spark-X2.6-4B in four engines](https://huggingface.co/infosave/Spark-X2.6-4B-turbo-cmf/resolve/main/assets/engines_en.png)

The same card against llama.cpp b12202 (`llama-bench`, Q4_K_M, 3.5 GB):

| | prompt of 16 000 tokens, tok/s | decode at 16 000 tokens, tok/s | decode at 64 000 tokens, tok/s | peak VRAM at 64 000, GB |
|---|---:|---:|---:|---:|
| cortiq q4tp, Vulkan | 895 | 37.9 | 18.5 | 7.9 |
| **cortiq Turbo, Vulkan** | **545** | **56.2** | **56.2** | **4.2** |
| llama.cpp Q4_K_M, Vulkan | 1576 | 51.7 | 35.6 | 5.5 |
| llama.cpp Q4_K_M, CUDA | 2711 | 49.4 | 28.3 | 5.6 |

llama.cpp reads a 16 000-token prompt on this card 1.8 times faster than cortiq does over Vulkan (3.0
times over CUDA), and its decode after a short prompt is faster too (63.7 tok/s against 57.2 at 1 000
tokens). On decode the turbo file is ahead of llama.cpp at 16 000 tokens (56.2 against 51.7 on Vulkan
and 49.4 on CUDA) and at 64 000, where it holds 56.2 tok/s against 35.6 (Vulkan) and 28.3 (CUDA) at
three quarters of the memory.

## Quality

A planted fact (a pass key, an access code, a password, a room number) at a random place in Wikipedia
text, eight trials a length, the question at the end, Vulkan:

| text | standard | **turbo** |
|---|---:|---:|
| 60 000 characters (about 14 000 tokens) | 8 of 8 | **6 of 8** |
| 120 000 characters (about 28 000 tokens) | 8 of 8 | **6 of 8** |
| 240 000 characters (about 56 000 tokens) | §S240 | **§F240** |

The misses at 60 000 and 120 000 characters were all the password kind (`qX…vT`, a token the model had not seen as
a word): the turbo file answered with text instead. §SUMMARY

## Options

| option | meaning |
|---|---|
| `--prompt-file FILE` | a prompt longer than a command line holds |
| `CMF_MAX_SEQ=N` | the context cap (32 768 by default) |
| `CMF_ANE=1` | on a Mac, the prompt's FFNs on the Neural Engine |
| `CMF_O1_PREFIX_MAX=N` | seal at `N` rows instead of 8192 |
| `CMF_O1=off` | the plain model, with exact attention at any length |

## Limits

- The far field runs on Metal, Vulkan and DX12; on the CPU this file runs as the standard one.
- Past the seal the text is approximate: what lies far back fades, apart from the rows kept exact. The
  fit behind those rows saw planted facts; other kinds of important text were not tested.
- Rows that arrive after a prompt ends stay in the window or go to the far field. A turn that diverges
  from a sealed conversation reads the whole prompt again.
- A generation that starts below the seal stays exact, and its memory grows with every token.
- cortiq reads a prompt on an NVIDIA card slower than llama.cpp: 895 against 1576 tok/s at 16 000 tokens
  over Vulkan, 2711 over CUDA (table above); the turbo file's prompt is slower still (545) while the far
  field is built on the host. On a Mac the engines are close, and the Neural Engine path is the fastest
  of the three on the prompt.

## Building a turbo file

The weights of the exact-row fit, `keep_spark26.bin`, are in this repository.

```bash
cortiq far-turbo spark-x2.6-4b-q4tp.cmf spark-x2.6-4b-q4tp-turbo.cmf --exact-weights keep_spark26.bin
```

The method is described in [the Qwen3.8-27B Turbo guide](https://github.com/infosave2007/cmf/blob/master/docs/QWEN38_TURBO.md).

## License

Weights: [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0), from
[XHToken/Spark-X2.6-4B](https://huggingface.co/XHToken/Spark-X2.6-4B). The far field is a subject of
U.S. patent application No. 19/738,763; see
[PATENTS.md](https://github.com/infosave2007/cmf/blob/master/PATENTS.md).

---

## Документация на русском

4-битные веса [Spark-X2.6-4B](https://huggingface.co/XHToken/Spark-X2.6-4B) под заголовком, который держит
память устройства и скорость генерации на месте на длинных промптах. На RTX 2000 Ada при 64 000 токенов файл
генерирует 56,2 tok/s вместо 18,5 и занимает 4,2 ГБ карты вместо 7,9. На MacBook Air M4 после промпта в
14 000 токенов — 33,0 tok/s там, где стандартный файл даёт 31,0, а стоковый mlx-lm 30,9, в 2,9 ГБ памяти
против 5,1 у mlx-lm. Пароли и коды в тексте остаются точными. Промпты короче 8192 токенов идут через точное
внимание, как в [стандартном файле](https://huggingface.co/infosave/Spark-X2.6-4B-cmf). Это один файл
[CMF](https://github.com/infosave2007/cmf) для `cortiq` — движка вывода на Rust без Python и без ML-фреймворка,
на Metal, Vulkan, DX12 и CPU.

### Быстрый старт

```bash
cargo install cortiq-cli
hf download infosave/Spark-X2.6-4B-turbo-cmf spark-x2.6-4b-q4tp-turbo.cmf --local-dir .
CMF_MAX_SEQ=131072 cortiq run spark-x2.6-4b-q4tp-turbo.cmf --prompt-file report.txt --greedy --no-think
```

OpenAI-совместимый сервер; скорость промпта и генерации каждого запроса печатается, как у llama-server:

```bash
CMF_MAX_SEQ=131072 cortiq serve spark-x2.6-4b-q4tp-turbo.cmf --port 8080
```

Нужен cortiq 0.8.64 или новее; собранные бинарники — на
[странице релизов](https://github.com/infosave2007/cmf/releases). `CMF_MAX_SEQ` снимает потолок по умолчанию
в 32 768 токенов. На Mac `CMF_ANE=1` отправляет FFN промпта на Neural Engine: промпт в 14 000 токенов за 33,6 с
вместо 54,8.

### Что делает файл

![Длинный промпт на маленькой карте](https://huggingface.co/infosave/Spark-X2.6-4B-turbo-cmf/resolve/main/assets/ceiling_ru.png)

**Фиксированный потолок.** 27 из 36 слоёв модели смотрят назад на 512 строк и не растут. 9 слоёв полного
внимания читают первые 8192 строки промпта точно на устройстве; ключи и значения дальнейших строк уходят с него
по мере чтения — в ОЗУ. Перед последними 512 строками из них слой за слоем строится дальнее поле, и строки
уходят. Дальнее поле хранит первые 4 строки и последние 1800 точно, а всё остальное — в 32 ориентирах на голову,
так что память устройства и цена токена перестают расти с текстом. Патентная заявка № 19/738,763.

**Точные строки.** Одно дальнее поле теряет пароль. Поэтому каждая строка оценивается по тому, что с ней делают
слои, и 248 лучших остаются точными. Вопрос в конце промпта читает их перед ответом.

![Пароли и коды остаются точными](https://huggingface.co/infosave/Spark-X2.6-4B-turbo-cmf/resolve/main/assets/exact_rows_ru.png)

### На Mac

MacBook Air M4 (10-ядерный GPU, 24 ГБ, без вентилятора), одни и те же промпты (текст Википедии и одна
инструкция), 96 жадных токенов после каждого, по одному прогону за раз с минутой паузы между прогонами. cortiq
на Metal; [mlx-lm 0.32.2](https://github.com/ml-explore/mlx-lm) на 4-битной MLX-сборке модели (группа 64,
2,2 ГБ); llama.cpp b12202, собранный под arm64 с Metal, Q4_K_M (3,5 ГБ), `-c` по размеру промпта. Турбо и
стандартный файл держат одни и те же 2,16 ГБ весов.

![Четыре движка на MacBook Air M4](https://huggingface.co/infosave/Spark-X2.6-4B-turbo-cmf/resolve/main/assets/mac_depth_ru.png)

| промпт, токенов | первый токен, с | | | | | генерация, tok/s | | | |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| | cortiq | **турбо** | cortiq + ANE | mlx-lm | llama.cpp | cortiq | **турбо** | mlx-lm | llama.cpp |
| 2 000 | 6,6 | **6,5** | 3,4 | 6,6 | 6,2 | 36,4 | **36,3** | 36,4 | 28,8 |
| 14 000 | 54,8 | **56,4** | 33,6 | 53,1 | 52,5 | 31,0 | **33,0** | 30,9 | 22,3 |
| 32 000 | 157,9 | **131,2** | 109,8 | 144,2 | 170,1 | 24,2 | **33,1** | 20,8 | 15,0 |
| 64 000 | 426,5 | **262,1** | 334,7 | 351,5 | 494,2 | 24,5 | **33,5** | 12,6 | 8,8 |

До 14 000 токенов четыре движка читают промпт с одной скоростью, а cortiq, турбо-файл и mlx-lm генерируют с
одной скоростью, llama.cpp — на четверть медленнее. Дальше турбо-файл держит 33 tok/s: при 64 000 токенов
mlx-lm на 12,6, llama.cpp на 8,8, стандартный файл cortiq на 24,5. Промпт в 64 000 токенов он читает за 262 с
против 352 (mlx-lm) и 494 (llama.cpp). Neural Engine сокращает первый токен промпта в 14 000 токенов с 54,8 с
до 33,6 и не меняет генерацию. Время первого токена у mlx-lm — длина промпта, делённая на заявленные им tok/s
промпта; у остальных измерено по первому токену.

![Память после промпта](https://huggingface.co/infosave/Spark-X2.6-4B-turbo-cmf/resolve/main/assets/memory_ru.png)

| после промпта в | cortiq | **турбо** | mlx-lm 4-bit | llama.cpp Q4_K_M |
|---|---:|---:|---:|---:|
| 2 000 токенов: footprint / резидентная, ГБ | 1,20 / 3,06 | **1,20 / 3,05** | 3,74 / 2,40 | 0,33 / 3,61 |
| 14 000 токенов: footprint / резидентная, ГБ | 2,60 / 4,37 | **2,85 / 4,70** | 5,09 / 2,43 | 0,83 / 4,11 |

Footprint — то, что Мониторинг системы показывает как «Память»: страницы, которые процесс держит для себя,
включая его буферы GPU. cortiq и llama.cpp отображают файл весов в память, так что эти страницы — файловый кэш
системы: в резидентном наборе, но не в footprint, и первыми уходят, когда памяти не хватает. mlx-lm копирует
веса в собственные буферы, так что его footprint несёт их. Оба числа — пик за прогон, отсчёты каждые 0,25 с.

**Перплексия.** Тест WikiText-2, обход `llama-perplexity -c 512`: 64 куска по 512 сырых токенов с начала
текста, оценивается вторая половина каждого куска (16 320 предсказаний). cortiq
`ppl --window-len 512 --window-stride 512 --window-score-from 256`, mlx-lm на тех же кусках, всё на Mac.

| файл | размер, ГБ | перплексия |
|---|---:|---:|
| cortiq q4tp (этот файл и стандартный) | 2,16 | 32,95 |
| mlx-lm 4-bit, группа 64 | 2,2 | 31,91 |
| llama.cpp Q4_K_M | 3,54 | 26,75 |
| cortiq q8_2f | 4,14 | 26,06 |
| mlx-lm 8-bit | 4,1 | 25,46 |
| llama.cpp Q8_0 | 4,37 | 26,43 |

При одном 4-битном размере файл q4tp на 3 % выше MLX; Q4_K_M у llama.cpp держит внимание и часть FFN в 6 и 8
битах, на 1,4 ГБ больше и стоит там же, где 8-битные файлы. Турбо-заголовок этих чисел не меняет: кусок в 512
токенов читается точно.

### На карте NVIDIA

![Стандартный и турбо](https://huggingface.co/infosave/Spark-X2.6-4B-turbo-cmf/resolve/main/assets/compare_ru.png)

RTX 2000 Ada (16 ГБ), Vulkan, `cortiq bench --core`, 60 токенов после промпта такой длины.

| промпт, токенов | первый токен, с (стандарт) | первый токен, с (**турбо**) | генерация, tok/s (стандарт) | генерация, tok/s (**турбо**) | пик VRAM, ГБ (стандарт) | пик VRAM, ГБ (**турбо**) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 000 | 1,4 | **1,4** | 57,2 | **56,2** | 3,5 | **3,5** |
| 16 000 | 17,9 | **29,4** | 37,9 | **56,2** | 4,5 | **4,2** |
| 32 000 | 43,3 | **57,1** | 28,0 | **56,0** | 5,6 | **4,2** |
| 64 000 | 119,6 | **116,3** | 18,5 | **56,2** | 7,9 | **4,2** |

После печати турбо-файл генерирует 56 tok/s на любой длине и держит карту на 4,2 ГБ. Пока кэш помещается,
стандартный файл читает промпт в 16 000 и 32 000 токенов быстрее; с 64 000 токенов турбо-файл быстрее и на
промпте. Дальнее поле строится на хосте, поэтому пока читается длинный промпт, процесс держит больше RAM:
4,9 ГБ на 16 000 токенов, 7,2 на 32 000 и 11,9 на 64 000 против 2,4 ГБ у стандартного файла на любой длине.

![Spark-X2.6-4B в четырёх движках](https://huggingface.co/infosave/Spark-X2.6-4B-turbo-cmf/resolve/main/assets/engines_ru.png)

Та же карта против llama.cpp b12202 (`llama-bench`, Q4_K_M, 3,5 ГБ):

| | промпт в 16 000 токенов, tok/s | генерация на 16 000 токенов, tok/s | генерация на 64 000 токенов, tok/s | пик VRAM на 64 000, ГБ |
|---|---:|---:|---:|---:|
| cortiq q4tp, Vulkan | 895 | 37,9 | 18,5 | 7,9 |
| **cortiq Turbo, Vulkan** | **545** | **56,2** | **56,2** | **4,2** |
| llama.cpp Q4_K_M, Vulkan | 1576 | 51,7 | 35,6 | 5,5 |
| llama.cpp Q4_K_M, CUDA | 2711 | 49,4 | 28,3 | 5,6 |

llama.cpp читает промпт в 16 000 токенов на этой карте в 1,8 раза быстрее cortiq через Vulkan (в 3,0 раза
через CUDA), и после короткого промпта его генерация тоже быстрее (63,7 tok/s против 57,2 на 1 000 токенов).
На генерации турбо-файл впереди llama.cpp уже на 16 000 токенов (56,2 против 51,7 на Vulkan и 49,4 на
CUDA) и на 64 000: 56,2 tok/s против 35,6 (Vulkan) и 28,3 (CUDA) при трёх четвертях памяти.

### Качество

Подставленный факт (ключ, код доступа, пароль, номер комнаты) в случайном месте текста Википедии, восемь
попыток на длину, вопрос в конце, Vulkan:

| текст | стандартный | **турбо** |
|---|---:|---:|
| 60 000 символов (около 14 000 токенов) | 8 из 8 | **6 из 8** |
| 120 000 символов (около 28 000 токенов) | 8 из 8 | **6 из 8** |
| 240 000 символов (около 56 000 токенов) | §S240 | **§F240** |

Все промахи на 60 000 и 120 000 символах — пароли (`qX…vT`, токен, которого модель не видела словом): турбо-файл отвечал
текстом вместо него. §RU_SUMMARY

### Параметры

| параметр | смысл |
|---|---|
| `--prompt-file FILE` | промпт длиннее командной строки |
| `CMF_MAX_SEQ=N` | потолок контекста (32 768 по умолчанию) |
| `CMF_ANE=1` | на Mac — FFN промпта на Neural Engine |
| `CMF_O1_PREFIX_MAX=N` | запечатать на `N` строках вместо 8192 |
| `CMF_O1=off` | обычная модель с точным вниманием на любой длине |

### Ограничения

- Дальнее поле работает на Metal, Vulkan и DX12; на CPU этот файл работает как стандартный.
- После печати текст приближённый: далёкое выцветает, кроме строк, оставленных точными. Подгонка за этими
  строками видела подставленные факты; другие виды важного текста не проверялись.
- Строки, пришедшие после конца промпта, остаются в окне или уходят в дальнее поле. Реплика, расходящаяся с
  запечатанным диалогом, перечитывает весь промпт.
- Генерация, начавшаяся ниже печати, остаётся точной, и её память растёт с каждым токеном.
- На карте NVIDIA cortiq читает промпт медленнее llama.cpp: 895 против 1576 tok/s на 16 000 токенов через
  Vulkan, 2711 через CUDA (таблица выше); промпт турбо-файла ещё медленнее (545), пока дальнее поле строится
  на хосте. На Mac движки близки, а путь через Neural Engine — самый быстрый из трёх на промпте.

### Как собрать турбо-файл

Веса подгонки точных строк, `keep_spark26.bin`, лежат в этом репозитории.

```bash
cortiq far-turbo spark-x2.6-4b-q4tp.cmf spark-x2.6-4b-q4tp-turbo.cmf --exact-weights keep_spark26.bin
```

Метод описан в [руководстве по Qwen3.8-27B Turbo](https://github.com/infosave2007/cmf/blob/master/docs/QWEN38_TURBO.md).

### Лицензия

Веса: [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0), от
[XHToken/Spark-X2.6-4B](https://huggingface.co/XHToken/Spark-X2.6-4B). Дальнее поле — предмет патентной заявки
США № 19/738,763; см. [PATENTS.md](https://github.com/infosave2007/cmf/blob/master/PATENTS.md).

---

## 中文说明

[Spark-X2.6-4B](https://huggingface.co/XHToken/Spark-X2.6-4B) 的 4 位权重，加上一个在长提示下把设备内存和生成速度
固定住的文件头。在 RTX 2000 Ada 上，64 000 词元时它以 56.2 tok/s 生成而不是 18.5，占显卡 4.2 GB 而不是 7.9。在
MacBook Air M4 上，读完 14 000 词元的提示后它以 33.0 tok/s 生成，标准文件为 31.0，原版 mlx-lm 为 30.9，内存 2.9 GB
对 mlx-lm 的 5.1。文本里的密码和编码保持精确。短于 8192 词元的提示走精确注意力，与
[标准文件](https://huggingface.co/infosave/Spark-X2.6-4B-cmf)相同。这是一个供 `cortiq` 使用的
[CMF](https://github.com/infosave2007/cmf) 文件：cortiq 是一个 Rust 推理引擎，不依赖 Python 和任何 ML 框架，运行于
Metal、Vulkan、DX12 和 CPU。

### 快速开始

```bash
cargo install cortiq-cli
hf download infosave/Spark-X2.6-4B-turbo-cmf spark-x2.6-4b-q4tp-turbo.cmf --local-dir .
CMF_MAX_SEQ=131072 cortiq run spark-x2.6-4b-q4tp-turbo.cmf --prompt-file report.txt --greedy --no-think
```

OpenAI 兼容服务器，每个请求的提示速度和生成速度像 llama-server 一样打印出来：

```bash
CMF_MAX_SEQ=131072 cortiq serve spark-x2.6-4b-q4tp-turbo.cmf --port 8080
```

需要 cortiq 0.8.64 或更新版本；预编译二进制在[发布页](https://github.com/infosave2007/cmf/releases)。`CMF_MAX_SEQ`
解除默认的 32 768 词元上限。在 Mac 上，`CMF_ANE=1` 把提示阶段的 FFN 送到 Neural Engine：14 000 词元的提示 33.6 秒
而不是 54.8 秒。

### 这个文件做什么

![小显卡上的长提示](https://huggingface.co/infosave/Spark-X2.6-4B-turbo-cmf/resolve/main/assets/ceiling_zh.png)

**固定的上限。** 模型 36 层中有 27 层只回看 512 行，永不增长。9 个全注意力层在设备上精确读取提示的前 8192 行；之后各行
的键和值一边读取一边离开设备，进入内存。在最后 512 行之前，由它们逐层构建远场，然后这些行被丢弃。远场精确保留前 4 行和
最后 1800 行，其余全部压入每个头 32 个地标，于是设备内存和每个词元的代价不再随文本增长。专利申请号 19/738,763。

**精确行。** 仅有远场会丢掉密码。因此每一行按各层对它的处理打分，最好的 248 行保持精确。提示末尾的问题在回答前先读取
它们。

![密码和编码保持精确](https://huggingface.co/infosave/Spark-X2.6-4B-turbo-cmf/resolve/main/assets/exact_rows_zh.png)

### 在 Mac 上

MacBook Air M4（10 核 GPU，24 GB，无风扇），相同的提示（维基百科文本加一条指令），每个之后贪心生成 96 个词元，一次只跑
一个，两次之间休息一分钟。cortiq 用 Metal；[mlx-lm 0.32.2](https://github.com/ml-explore/mlx-lm) 用模型的 4 位 MLX
版本（组大小 64，2.2 GB）；llama.cpp b12202 为 arm64 带 Metal 编译，Q4_K_M（3.5 GB），`-c` 按提示长度设置。turbo 文件和
标准文件持有同样的 2.16 GB 权重。

![MacBook Air M4 上的四个引擎](https://huggingface.co/infosave/Spark-X2.6-4B-turbo-cmf/resolve/main/assets/mac_depth_zh.png)

| 提示，词元 | 首词元，秒 | | | | | 生成，tok/s | | | |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| | cortiq | **turbo** | cortiq + ANE | mlx-lm | llama.cpp | cortiq | **turbo** | mlx-lm | llama.cpp |
| 2 000 | 6.6 | **6.5** | 3.4 | 6.6 | 6.2 | 36.4 | **36.3** | 36.4 | 28.8 |
| 14 000 | 54.8 | **56.4** | 33.6 | 53.1 | 52.5 | 31.0 | **33.0** | 30.9 | 22.3 |
| 32 000 | 157.9 | **131.2** | 109.8 | 144.2 | 170.1 | 24.2 | **33.1** | 20.8 | 15.0 |
| 64 000 | 426.5 | **262.1** | 334.7 | 351.5 | 494.2 | 24.5 | **33.5** | 12.6 | 8.8 |

到 14 000 词元为止，四个引擎读提示的速度相同，cortiq、turbo 文件和 mlx-lm 的生成速度相同，llama.cpp 慢四分之一。再往
后 turbo 文件保持 33 tok/s：64 000 词元时 mlx-lm 为 12.6，llama.cpp 为 8.8，cortiq 标准文件为 24.5。它读 64 000 词元
的提示用 262 秒，mlx-lm 352 秒，llama.cpp 494 秒。Neural Engine 把 14 000 词元提示的首词元从 54.8 秒缩到 33.6 秒，
生成不变。mlx-lm 的首词元时间是提示长度除以它报告的提示 tok/s；其余按首词元实测。

![提示之后的内存](https://huggingface.co/infosave/Spark-X2.6-4B-turbo-cmf/resolve/main/assets/memory_zh.png)

| 提示之后 | cortiq | **turbo** | mlx-lm 4-bit | llama.cpp Q4_K_M |
|---|---:|---:|---:|---:|
| 2 000 词元：footprint / 常驻，GB | 1.20 / 3.06 | **1.20 / 3.05** | 3.74 / 2.40 | 0.33 / 3.61 |
| 14 000 词元：footprint / 常驻，GB | 2.60 / 4.37 | **2.85 / 4.70** | 5.09 / 2.43 | 0.83 / 4.11 |

footprint 是活动监视器显示为"内存"的数字：进程为自己持有的页面，包括 GPU 缓冲区。cortiq 和 llama.cpp 把权重文件映射进
内存，这些页面属于系统的文件缓存：算在常驻集里，不算在 footprint 里，内存紧张时最先被回收。mlx-lm 把权重复制到自己的
缓冲区，所以它的 footprint 包含权重。两个数字都是整次运行的峰值，每 0.25 秒采样一次。

**困惑度。** WikiText-2 测试集，按 `llama-perplexity -c 512` 的方式：从文本开头取 64 个 512 原始词元的块，对每块的后半
计分（16 320 次预测）。cortiq `ppl --window-len 512 --window-stride 512 --window-score-from 256`，mlx-lm 用相同的块，
全部在 Mac 上。

| 文件 | 大小，GB | 困惑度 |
|---|---:|---:|
| cortiq q4tp（本文件与标准文件） | 2.16 | 32.95 |
| mlx-lm 4-bit，组 64 | 2.2 | 31.91 |
| llama.cpp Q4_K_M | 3.54 | 26.75 |
| cortiq q8_2f | 4.14 | 26.06 |
| mlx-lm 8-bit | 4.1 | 25.46 |
| llama.cpp Q8_0 | 4.37 | 26.43 |

同为 4 位大小时，q4tp 文件比 MLX 高 3 %；llama.cpp 的 Q4_K_M 把注意力和部分 FFN 保留在 6 位和 8 位，大 1.4 GB，位置
与 8 位文件相当。turbo 文件头不改变这些数字：512 词元的块是精确读取的。

### 在 NVIDIA 显卡上

![标准与 turbo](https://huggingface.co/infosave/Spark-X2.6-4B-turbo-cmf/resolve/main/assets/compare_zh.png)

RTX 2000 Ada（16 GB），Vulkan，`cortiq bench --core`，该长度的提示之后生成 60 个词元。

| 提示，词元 | 首词元，秒（标准） | 首词元，秒（**turbo**） | 生成，tok/s（标准） | 生成，tok/s（**turbo**） | 显存峰值，GB（标准） | 显存峰值，GB（**turbo**） |
|---:|---:|---:|---:|---:|---:|---:|
| 1 000 | 1.4 | **1.4** | 57.2 | **56.2** | 3.5 | **3.5** |
| 16 000 | 17.9 | **29.4** | 37.9 | **56.2** | 4.5 | **4.2** |
| 32 000 | 43.3 | **57.1** | 28.0 | **56.0** | 5.6 | **4.2** |
| 64 000 | 119.6 | **116.3** | 18.5 | **56.2** | 7.9 | **4.2** |

封印之后，turbo 文件在任何长度上以 56 tok/s 生成，显存保持在 4.2 GB。只要缓存放得下，标准文件读 16 000 和 32 000
词元的提示更快；从 64 000 词元起，turbo 文件连提示也更快。远场在主机上构建，因此读取长提示时进程占用更多内存：
16 000 词元时 4.9 GB，32 000 时 7.2 GB，64 000 时 11.9 GB；标准文件在任何长度上都是 2.4 GB。

![四个引擎中的 Spark-X2.6-4B](https://huggingface.co/infosave/Spark-X2.6-4B-turbo-cmf/resolve/main/assets/engines_zh.png)

同一张卡对比 llama.cpp b12202（`llama-bench`，Q4_K_M，3.5 GB）：

| | 16 000 词元提示，tok/s | 16 000 词元处生成，tok/s | 64 000 词元处生成，tok/s | 64 000 处显存峰值，GB |
|---|---:|---:|---:|---:|
| cortiq q4tp，Vulkan | 895 | 37.9 | 18.5 | 7.9 |
| **cortiq Turbo，Vulkan** | **545** | **56.2** | **56.2** | **4.2** |
| llama.cpp Q4_K_M，Vulkan | 1576 | 51.7 | 35.6 | 5.5 |
| llama.cpp Q4_K_M，CUDA | 2711 | 49.4 | 28.3 | 5.6 |

在这张卡上，llama.cpp 经 Vulkan 读 16 000 词元的提示比 cortiq 快 1.8 倍（经 CUDA 快 3.0 倍），短提示之后的生成也更快
（1 000 词元处 63.7 对 57.2 tok/s）。生成方面，turbo 文件在 16 000 词元处已领先 llama.cpp（56.2 对 Vulkan 的 51.7 和
CUDA 的 49.4），在 64 000 词元处同样领先：56.2 tok/s 对 35.6（Vulkan）和 28.3（CUDA），显存为其四分之三。

### 质量

在维基百科文本的随机位置植入一个事实（通行密钥、访问码、密码、房间号），每个长度八次尝试，问题放在末尾，Vulkan：

| 文本 | 标准 | **turbo** |
|---|---:|---:|
| 60 000 字符（约 14 000 词元） | 8 / 8 | **6 / 8** |
| 120 000 字符（约 28 000 词元） | 8 / 8 | **6 / 8** |
| 240 000 字符（约 56 000 词元） | §S240 | **§F240** |

60 000 和 120 000 字符上的失误都是密码类（`qX…vT`，模型没见过的词元）：turbo 文件用文字作答而没有给出它。§ZH_SUMMARY

### 选项

| 选项 | 含义 |
|---|---|
| `--prompt-file FILE` | 命令行放不下的长提示 |
| `CMF_MAX_SEQ=N` | 上下文上限（默认 32 768） |
| `CMF_ANE=1` | Mac 上，提示阶段的 FFN 走 Neural Engine |
| `CMF_O1_PREFIX_MAX=N` | 在 `N` 行处封印，而不是 8192 |
| `CMF_O1=off` | 普通模型，任何长度都用精确注意力 |

### 限制

- 远场运行于 Metal、Vulkan 和 DX12；在 CPU 上这个文件与标准文件一样工作。
- 封印之后文本是近似的：远处的内容会淡去，保持精确的行除外。这些行背后的拟合只见过植入的事实；其他类型的重要文本未经
  测试。
- 提示结束后到达的行留在窗口里或进入远场。偏离已封印对话的一轮会重新读取整个提示。
- 在封印以下开始的生成保持精确，其内存随每个词元增长。
- 在 NVIDIA 显卡上 cortiq 读提示比 llama.cpp 慢：16 000 词元处经 Vulkan 为 895 对 1576 tok/s，经 CUDA 为 2711（见上表）；
  turbo 文件的提示更慢（545），因为远场在主机上构建。在 Mac 上几个引擎接近，Neural Engine 路径在提示上是三者中最快的。

### 如何构建 turbo 文件

精确行拟合的权重 `keep_spark26.bin` 在本仓库中。

```bash
cortiq far-turbo spark-x2.6-4b-q4tp.cmf spark-x2.6-4b-q4tp-turbo.cmf --exact-weights keep_spark26.bin
```

方法见 [Qwen3.8-27B Turbo 指南](https://github.com/infosave2007/cmf/blob/master/docs/QWEN38_TURBO.md)。

### 许可

权重：[Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)，来自
[XHToken/Spark-X2.6-4B](https://huggingface.co/XHToken/Spark-X2.6-4B)。远场是美国专利申请 No. 19/738,763 的主题；见
[PATENTS.md](https://github.com/infosave2007/cmf/blob/master/PATENTS.md)。
