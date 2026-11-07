---
license: apache-2.0
library_name: cortiq
base_model: ricursiveAI/Ripple-65B-Preview
base_model_relation: quantized
pipeline_tag: text-generation
tags:
  - cmf
  - cortiq
  - quantized
  - q4tp
  - q2tp
  - ripple
  - long-context
  - low-vram
  - speculative-decoding
  - vulkan
language:
  - en
  - ru
  - zh
---

# Ripple-65B Preview — CMF

[Ripple-65B Preview](https://huggingface.co/ricursiveAI/Ripple-65B-Preview), a dense 65B model of 80
layers, in three files for `cortiq`, a Rust inference engine with no Python and no ML framework: four
bits in 34.6 GB, two bits in 29.2 GB, eight bits in 64.8 GB. Every file carries a header that holds
device memory in place on long prompts and keeps passwords and codes in the text exact. The two-bit
file keeps the busiest 5 % of each layer's FFN neurons and the K and V projections at eight bits, and
reaches the four-bit file's perplexity at 85 % of its size. The published
[DFlash2 drafter](https://huggingface.co/ricursiveAI/Ripple-65B-Preview-DFlash2) rides beside the
four-bit file as a sidecar and gives the plain run's text token for token; a round does not yet pay for itself (below).

[English](#quick-start) · [Русский](#документация-на-русском) · [中文](#中文说明)

## Quick start

```bash
cargo install cortiq-cli
hf download infosave/Ripple-65B-Preview-cmf ripple-65b-q4tp.cmf --local-dir .
cortiq run ripple-65b-q4tp.cmf --prompt "Explain the Fourier transform in three paragraphs." --greedy
```

With the drafter (one more file beside the model, `CMF_DFLASH=1`):

```bash
hf download infosave/Ripple-65B-Preview-cmf ripple-65b-q4tp.dflash.cmf --local-dir .
CMF_DFLASH=1 cortiq run ripple-65b-q4tp.cmf --prompt-file task.txt --greedy
```

Use cortiq 0.8.60 or later; prebuilt binaries are on the
[releases page](https://github.com/infosave2007/cmf/releases). The model reasons inside `<think>` before
it answers. `--greedy` turns on the drafter's rounds.

### OpenAI-compatible server

```bash
cortiq serve ripple-65b-q4tp.cmf --port 8080

curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "ripple", "max_tokens": 4096,
       "messages": [{"role": "user", "content": "Explain how a B-tree splits a full node."}]}'
```

Endpoints: `/v1/chat/completions` (streamed or not), `/v1/completions`, `/v1/models`, and a web dashboard
on the same port. `--host 127.0.0.1` keeps the server local. A conversation keeps its rows on the card
between turns, so a turn reads only what is new; the far field holds the ceiling the same way as in
`cortiq run`.

## Files

| file | GB | fits | perplexity, wikitext-2 test, first 2048 tokens (BF16 6.94) | top-1 as BF16 |
|---|---:|---|---:|---:|
| `ripple-65b-q4tp.cmf` | 34.6 | a 40 GB card; 48 GB with the drafter | 7.09 | 91.1 % |
| `ripple-65b-q2tp.cmf` | 29.2 | a 32 GB card | 7.43 | 86.2 % |
| `ripple-65b-q8_2f.cmf` | 64.8 | an 80 GB card | 6.97 | 98.3 % |
| `ripple-65b-q4tp.dflash.cmf` | 2.3 | beside the q4tp file | | |
| `ripple-65b-q2tp.dflash.cmf` | 2.3 | beside the q2tp file | | |

`q4tp` is four-bit tiled weights with the attention's K and V projections at eight bits; `q2tp` is a
two-bit mix: layer 0 and the last two layers at eight bits, the FFN's down matrix and the attention at
four, and in every layer the 5 % of FFN neurons that carry half of the activation energy at eight. The
head is eight-bit in every file. Measured on the A40 below, the device holds the q4tp file in 35.3 GB
and the q2tp file in 30.3 GB on a 4 000-token prompt.

## What the files do

![A long prompt on a 46 GB card](https://huggingface.co/infosave/Ripple-65B-Preview-cmf/resolve/main/assets/ceiling_en.png)

**A fixed ceiling.** The first 8192 rows of a prompt are read exactly on the card. The keys and values
of later rows leave the card as they are read, for RAM or an SSD. Before the last 512 rows, the far
field is built from them one layer at a time, and the rows go away. A far field keeps the first 4
rows and the last 1800 exact and everything else in 32 landmarks per head of the 16 full-attention
layers; the 64 sliding layers hold their 1024 rows anyway. Device memory and the cost of a token stop
growing with the text. Patent application No. 19/738,763.

**Exact rows.** A far field alone loses a password. So each row is scored by what the layers do with
it, and the 248 best stay exact. A question at the end of the prompt reads them before it is answered.

![Passwords and codes stay exact](https://huggingface.co/infosave/Ripple-65B-Preview-cmf/resolve/main/assets/exact_rows_en.png)

**Hot neurons.** An FFN neuron is a gate row, an up row and a down column. Over text, a few neurons do
most of the work: 5 % of them carry half of the squared activation of a layer. In the two-bit file
those neurons stay at eight bits as a tube beside the two-bit core, and the engine reads both in one
pass. The same 5 % chosen at random recover 41 % of the gap to four bits; the busiest recover 86 %,
and the K and V projections at eight bits close most of the rest.

![Two bits, with the busiest neurons at eight](https://huggingface.co/infosave/Ripple-65B-Preview-cmf/resolve/main/assets/hot_en.png)

**Drafts.** Ripple has no draft head of its own. The DFlash2 drafter, five layers over the trunk's rows
after layers 1, 20, 39, 58 and 77, proposes up to seven tokens from one block; the card verifies them
with the next token in one batch, and the far field moves over the accepted ones only. Greedy text
stays the same with and without drafts; what a round costs is in the table below.

## Performance

![Standard and turbo](https://huggingface.co/infosave/Ripple-65B-Preview-cmf/resolve/main/assets/compare_en.png)

NVIDIA A40 (46 GB), Vulkan, `cortiq bench --core`, 60 tokens after a prompt of that length, the q4tp
file. The standard column is the same file with the far field off (`CMF_O1=off`).

| prompt, tokens | first token, s (standard) | first token, s (**turbo**) | decode, tok/s (standard) | decode, tok/s (**turbo**) | peak VRAM, GB (standard) | peak VRAM, GB (**turbo**) |
|---:|---:|---:|---:|---:|---:|---:|
| 4 000 | 48.4 | 48.0 | 12.1 | 12.0 | 35.3 | 35.3 |
| 16 000 | 212.9 | 251.2 | 11.1 | 11.7 | 36.8 | 36.6 |
| 32 000 | 473.2 | 491.1 | 10.2 | 11.8 | 39.1 | 36.6 |
| 64 000 | 1138.2 | 987.8 | 8.8 | 11.8 | 43.2 | 36.6 |

**A larger card.** NVIDIA RTX PRO 6000 (96 GB), Vulkan, a 4 000-token prompt, 60 tokens:

| file | prompt, tok/s | first token, s | decode, tok/s |
|---|---:|---:|---:|
| q4tp | 209 | 19.2 | 29.8 |
| q2tp | 155 | 25.9 | 27.8 |
| q8_2f | 156 | 25.7 | 18.3 |

The q2tp file on the A40:

| prompt, tokens | first token, s | decode, tok/s | peak VRAM, GB |
|---:|---:|---:|---:|
| 4 000 | 77.8 | 11.8 | 30.3 |
| 64 000 | 1459.2 | 11.5 | 31.6 |

**With the drafter.** `cortiq run --greedy`, 160 tokens of an answer, the q4tp file:

| answer | without drafts, tok/s | with drafts, tok/s | drafts kept a round |
|---|---:|---:|---:|
| code, A40 | 12.1 | 11.6 | 4.8 |
| prose, A40 | 12.2 | 9.0 | 1.3 |
| code, RTX PRO 6000 | 30.2 | 26.9 | 4.8 |
| prose, RTX PRO 6000 | 30.6 | 21.6 | 1.2 |

The text is the plain run's token for token, and a round costs more than the tokens it keeps: on the
RTX PRO 6000 a round takes 160 ms against 32 ms for a plain token, 75 ms of it the block itself, whose
five layers go through fifteen device calls with the convolutions and the attention on the host between
them. The rounds stay off unless `CMF_DFLASH=1` asks for them; a round of seven drafts
(`CMF_GRAPH_SPEC_K=7`) keeps 4.6 of them on code.

**Host memory.** While a long prompt is read, its rows take RAM or a disk (`--o1-spill DIR`). The mapped
weights are not counted; the system can drop them from RAM.

| prompt, tokens | RAM |
|---:|---:|
| 4 000 | 0.2 GB |
| 16 000 | 4.4 GB |
| 32 000 | 7.7 GB |
| 64 000 | 14.9 GB |

**Quality.** A planted fact in 60 000 characters, four kinds of them (a pass key, an access code, a
password, a room number), the model reasoning before it answers: 3 of 4 found; the fourth, a password, was still reasoning when its 300-token answer ran out.

## Options

| option | meaning |
|---|---|
| `CMF_DFLASH=1` | the drafter's rounds, with `--greedy` |
| `CMF_GRAPH_SPEC_K=n` | a round of `n` drafts (1 to 7) instead of the depth the rounds settle on |
| `--o1-spill DIR` | a long prompt's rows in files in `DIR`, also `CMF_O1_SPILL` |
| `--prompt-file FILE` | a prompt longer than a command line holds |
| `CMF_O1_PREFIX_MAX=N` | seal at `N` rows instead of 8192 |
| `CMF_O1=off` | the plain model, with exact attention at any length |
| `CMF_O1_SPEC=0` | no drafts over the far field |

## Limits

- The far field and the drafter's rounds run on Vulkan and DX12; on Metal and the CPU these files run
  as standard ones. The long-prompt tables were measured on one NVIDIA A40, the eight-bit file and the
  drafter's rounds on an RTX PRO 6000.
- Past the seal the text is approximate: what lies far back fades, apart from the rows kept exact. The
  fit behind those rows saw planted facts, and other kinds of important text were not tested.
- Rows that arrive after a prompt ends stay in the window or go to the far field. A turn that diverges
  from a sealed conversation reads the whole prompt again.
- Drafts need greedy decoding without penalties, and do not yet make the answer faster.

## Building the files

```bash
cortiq convert ricursiveAI/Ripple-65B-Preview ripple-65b-q4tp.cmf --quant q4tp --tensor-quant 'model.layers.*.self_attn.k_proj.weight=q8_2f' --tensor-quant 'model.layers.*.self_attn.v_proj.weight=q8_2f'
cortiq far-turbo ripple-65b-q4tp.cmf ripple-65b-q4tp.cmf --exact-weights keep_ripple.bin
cortiq convert --model ricursiveAI/Ripple-65B-Preview-DFlash2 --output ripple-65b-q4tp.cmf --dflash-sidecar --quant q4tp
```

The two-bit file: `cortiq hot-probe` over text gives each neuron's energy, and `cortiq hot-bake COLD
HOT energy.mass OUT --frac 0.05 --take self_attn.k_proj --take self_attn.v_proj` puts the top 5 % of
each layer and the K, V projections from the eight-bit file into the two-bit one. The exact-row
weights (`keep_ripple.bin`) and the scripts are in the
[repository](https://github.com/infosave2007/cmf/tree/master/scripts/ripple).

## License

Weights: [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0), from
[ricursiveAI/Ripple-65B-Preview](https://huggingface.co/ricursiveAI/Ripple-65B-Preview) and
[ricursiveAI/Ripple-65B-Preview-DFlash2](https://huggingface.co/ricursiveAI/Ripple-65B-Preview-DFlash2).
The far field is a subject of U.S. patent application No. 19/738,763; see
[PATENTS.md](https://github.com/infosave2007/cmf/blob/master/PATENTS.md).

---

## Документация на русском

[Ripple-65B Preview](https://huggingface.co/ricursiveAI/Ripple-65B-Preview), плотная модель на 65B
параметров и 80 слоёв, в трёх файлах для `cortiq` — движка на Rust без Python и ML-фреймворков: четыре
бита в 34,6 ГБ, два бита в 29,2 ГБ, восемь бит в 64,8 ГБ. У каждого файла заголовок, который держит
память карты на месте при длинных промптах и сохраняет пароли и коды в тексте точными. Двухбитный файл
держит самые занятые 5 % нейронов FFN каждого слоя и проекции K и V в восьми битах и выходит на
перплексию четырёхбитного при 85 % его размера. Опубликованный
[черновик DFlash2](https://huggingface.co/ricursiveAI/Ripple-65B-Preview-DFlash2) идёт рядом с
четырёхбитным файлом отдельным файлом и даёт текст обычного прогона токен в токен; раунд пока не окупает себя (ниже).

### Быстрый старт

```bash
cargo install cortiq-cli
hf download infosave/Ripple-65B-Preview-cmf ripple-65b-q4tp.cmf --local-dir .
cortiq run ripple-65b-q4tp.cmf --prompt "Объясни преобразование Фурье в трёх абзацах." --greedy
```

С черновиком (ещё один файл рядом с моделью, `CMF_DFLASH=1`):

```bash
hf download infosave/Ripple-65B-Preview-cmf ripple-65b-q4tp.dflash.cmf --local-dir .
CMF_DFLASH=1 cortiq run ripple-65b-q4tp.cmf --prompt-file task.txt --greedy
```

Нужен cortiq 0.8.60 или новее; собранные бинарники — на
[странице релизов](https://github.com/infosave2007/cmf/releases). Модель рассуждает внутри `<think>`,
прежде чем ответить. `--greedy` включает раунды черновика.

#### OpenAI-совместимый сервер

```bash
cortiq serve ripple-65b-q4tp.cmf --port 8080

curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "ripple", "max_tokens": 4096,
       "messages": [{"role": "user", "content": "Объясни, как B-дерево расщепляет полный узел."}]}'
```

Эндпоинты: `/v1/chat/completions` (с потоком и без), `/v1/completions`, `/v1/models` и веб-панель на том
же порту. `--host 127.0.0.1` оставляет сервер локальным. Разговор держит свои строки на карте между
ходами, так что ход читает только новое; дальнее поле держит потолок так же, как в `cortiq run`.

### Файлы

| файл | ГБ | помещается | перплексия, wikitext-2 test, первые 2048 токенов (BF16 6,94) | top-1 как BF16 |
|---|---:|---|---:|---:|
| `ripple-65b-q4tp.cmf` | 34,6 | карта 40 ГБ; 48 ГБ с черновиком | 7,09 | 91,1 % |
| `ripple-65b-q2tp.cmf` | 29,2 | карта 32 ГБ | 7,43 | 86,2 % |
| `ripple-65b-q8_2f.cmf` | 64,8 | карта 80 ГБ | 6,97 | 98,3 % |
| `ripple-65b-q4tp.dflash.cmf` | 2,3 | рядом с файлом q4tp | | |
| `ripple-65b-q2tp.dflash.cmf` | 2,3 | рядом с файлом q2tp | | |

`q4tp` — четырёхбитные тайловые веса с проекциями K и V внимания в восьми битах; `q2tp` — двухбитная
смесь: слой 0 и два последних слоя в восьми битах, матрица down FFN и внимание в четырёх, и в каждом
слое те 5 % нейронов FFN, что несут половину энергии активаций, в восьми. Голова в каждом файле
восьмибитная. На A40 ниже карта держит файл q4tp в 35,3 ГБ, а файл q2tp в 30,3 ГБ на промпте в
4 000 токенов.

### Что делают файлы

![Длинный промпт на карте 46 ГБ](https://huggingface.co/infosave/Ripple-65B-Preview-cmf/resolve/main/assets/ceiling_ru.png)

**Фиксированный потолок.** Первые 8192 строки промпта читаются на карте точно. Ключи и значения
следующих строк уходят с карты по мере чтения — в RAM или на SSD. Перед последними 512 строками из них
слой за слоем строится дальнее поле, и строки уходят. Дальнее поле держит первые 4 строки и последние
1800 точными, а всё остальное — в 32 опорных точках на голову в 16 слоях полного внимания; 64 скользящих
слоя и так держат свои 1024 строки. Память карты и цена токена перестают расти с текстом. Патентная
заявка № 19/738,763.

**Точные строки.** Дальнее поле само по себе теряет пароль. Поэтому каждая строка получает оценку по
тому, что с ней делают слои, и 248 лучших остаются точными. Вопрос в конце промпта читает их до ответа.

![Пароли и коды остаются точными](https://huggingface.co/infosave/Ripple-65B-Preview-cmf/resolve/main/assets/exact_rows_ru.png)

**Горячие нейроны.** Нейрон FFN — это строка gate, строка up и столбец down. На тексте основную работу
делают немногие: 5 % нейронов несут половину квадрата активаций слоя. В двухбитном файле эти нейроны
остаются в восьми битах отдельной «трубой» рядом с двухбитным ядром, и движок читает обе за один
проход. Те же 5 %, выбранные случайно, закрывают 41 % разрыва до четырёх бит; самые занятые — 86 %, а
проекции K и V в восьми битах — большую часть остатка.

![Два бита, а самые занятые нейроны — восемь](https://huggingface.co/infosave/Ripple-65B-Preview-cmf/resolve/main/assets/hot_ru.png)

**Черновики.** У Ripple нет своей черновой головы. Черновик DFlash2 — пять слоёв над строками ствола
после слоёв 1, 20, 39, 58 и 77 — предлагает до семи токенов из одного блока; карта проверяет их вместе
со следующим токеном за один пакет, а дальнее поле сдвигается только по принятым. Жадный текст с
черновиками и без них одинаков; сколько стоит раунд — в таблице ниже.

### Скорость

![Обычный и турбо](https://huggingface.co/infosave/Ripple-65B-Preview-cmf/resolve/main/assets/compare_ru.png)

NVIDIA A40 (46 ГБ), Vulkan, `cortiq bench --core`, 60 токенов после промпта такой длины, файл q4tp.
Колонка «обычный» — тот же файл с выключенным дальним полем (`CMF_O1=off`).

| промпт, токенов | первый токен, с (обычный) | первый токен, с (**турбо**) | декод, ток/с (обычный) | декод, ток/с (**турбо**) | пик VRAM, ГБ (обычный) | пик VRAM, ГБ (**турбо**) |
|---:|---:|---:|---:|---:|---:|---:|
| 4 000 | 48,4 | 48,0 | 12,1 | 12,0 | 35,3 | 35,3 |
| 16 000 | 212,9 | 251,2 | 11,1 | 11,7 | 36,8 | 36,6 |
| 32 000 | 473,2 | 491,1 | 10,2 | 11,8 | 39,1 | 36,6 |
| 64 000 | 1138,2 | 987,8 | 8,8 | 11,8 | 43,2 | 36,6 |

**Карта побольше.** NVIDIA RTX PRO 6000 (96 ГБ), Vulkan, промпт 4 000 токенов, 60 токенов:

| файл | промпт, ток/с | первый токен, с | декод, ток/с |
|---|---:|---:|---:|
| q4tp | 209 | 19,2 | 29,8 |
| q2tp | 155 | 25,9 | 27,8 |
| q8_2f | 156 | 25,7 | 18,3 |

Файл q2tp на A40:

| промпт, токенов | первый токен, с | декод, ток/с | пик VRAM, ГБ |
|---:|---:|---:|---:|
| 4 000 | 77,8 | 11,8 | 30,3 |
| 64 000 | 1459,2 | 11,5 | 31,6 |

**С черновиком.** `cortiq run --greedy`, 160 токенов ответа, файл q4tp:

| ответ | без черновиков, ток/с | с черновиками, ток/с | принято за раунд |
|---|---:|---:|---:|
| код, A40 | 12,1 | 11,6 | 4,8 |
| проза, A40 | 12,2 | 9,0 | 1,3 |
| код, RTX PRO 6000 | 30,2 | 26,9 | 4,8 |
| проза, RTX PRO 6000 | 30,6 | 21,6 | 1,2 |

Текст — токен в токен текст обычного прогона, а раунд стоит дороже токенов, которые приносит: на
RTX PRO 6000 раунд занимает 160 мс против 32 мс на обычный токен, 75 мс из них — сам блок, пять слоёв
которого проходят через пятнадцать вызовов карты со свёртками и вниманием на хосте между ними. Раунды
выключены, пока их не попросит `CMF_DFLASH=1`; раунд из семи черновиков (`CMF_GRAPH_SPEC_K=7`)
принимает 4,6 из них на коде.

**Память хоста.** Пока читается длинный промпт, его строки занимают RAM или диск (`--o1-spill DIR`).
Отображённые веса не считаются: система может вытеснить их из RAM.

| промпт, токенов | RAM |
|---:|---:|
| 4 000 | 0,2 ГБ |
| 16 000 | 4,4 ГБ |
| 32 000 | 7,7 ГБ |
| 64 000 | 14,9 ГБ |

**Качество.** Факт, спрятанный в 60 000 символов, четырёх видов (ключ, код доступа, пароль, номер
комнаты), модель рассуждает перед ответом: 3 из 4 найдены; четвёртый, пароль, модель ещё обдумывала, когда кончились 300 токенов ответа.

### Параметры

| параметр | смысл |
|---|---|
| `CMF_DFLASH=1` | раунды черновика, вместе с `--greedy` |
| `CMF_GRAPH_SPEC_K=n` | раунд из `n` черновиков (от 1 до 7) вместо глубины, на которой раунды сходятся сами |
| `--o1-spill DIR` | строки длинного промпта в файлах в `DIR`, также `CMF_O1_SPILL` |
| `--prompt-file FILE` | промпт длиннее командной строки |
| `CMF_O1_PREFIX_MAX=N` | запечатывать на `N` строках вместо 8192 |
| `CMF_O1=off` | обычная модель с точным вниманием на любой длине |
| `CMF_O1_SPEC=0` | без черновиков над дальним полем |

### Ограничения

- Дальнее поле и раунды черновика работают на Vulkan и DX12; на Metal и CPU эти файлы работают как
  обычные. Таблицы длинных промптов измерены на одной NVIDIA A40, восьмибитный файл и раунды
  черновика — на RTX PRO 6000.
- За печатью текст приближённый: далёкое тускнеет, кроме строк, оставленных точными. Подгонка за этими
  строками видела спрятанные факты; другие виды важного текста не проверялись.
- Строки, пришедшие после конца промпта, остаются в окне или уходят в дальнее поле. Реплика, которая
  расходится с запечатанным разговором, читает весь промпт заново.
- Черновикам нужно жадное декодирование без штрафов, и ответ они пока не ускоряют.

### Как собрать файлы

```bash
cortiq convert ricursiveAI/Ripple-65B-Preview ripple-65b-q4tp.cmf --quant q4tp --tensor-quant 'model.layers.*.self_attn.k_proj.weight=q8_2f' --tensor-quant 'model.layers.*.self_attn.v_proj.weight=q8_2f'
cortiq far-turbo ripple-65b-q4tp.cmf ripple-65b-q4tp.cmf --exact-weights keep_ripple.bin
cortiq convert --model ricursiveAI/Ripple-65B-Preview-DFlash2 --output ripple-65b-q4tp.cmf --dflash-sidecar --quant q4tp
```

Двухбитный файл: `cortiq hot-probe` по тексту даёт энергию каждого нейрона, а `cortiq hot-bake COLD
HOT energy.mass OUT --frac 0.05 --take self_attn.k_proj --take self_attn.v_proj` переносит верхние 5 %
каждого слоя и проекции K, V из восьмибитного файла в двухбитный. Веса точных строк (`keep_ripple.bin`)
и скрипты — в [репозитории](https://github.com/infosave2007/cmf/tree/master/scripts/ripple).

### Лицензия

Веса: [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0), от
[ricursiveAI/Ripple-65B-Preview](https://huggingface.co/ricursiveAI/Ripple-65B-Preview) и
[ricursiveAI/Ripple-65B-Preview-DFlash2](https://huggingface.co/ricursiveAI/Ripple-65B-Preview-DFlash2).
Дальнее поле — предмет патентной заявки США № 19/738,763; см.
[PATENTS.md](https://github.com/infosave2007/cmf/blob/master/PATENTS.md).

---

## 中文说明

[Ripple-65B Preview](https://huggingface.co/ricursiveAI/Ripple-65B-Preview) 是一个 80 层、650 亿参数的稠密模型,
这里以三个文件提供给 `cortiq`——一个不依赖 Python 和 ML 框架的 Rust 推理引擎:四比特 34.6 GB,两比特 29.2 GB,
八比特 64.8 GB。每个文件都带有一个头部,让长提示下的显存保持不变,并让文本中的密码和编码保持精确。两比特文件把
每层最活跃的 5 % FFN 神经元以及 K、V 投影保留为八比特,以四比特文件 85 % 的体积达到同样的困惑度。公开的
[DFlash2 草稿模型](https://huggingface.co/ricursiveAI/Ripple-65B-Preview-DFlash2)作为附属文件放在四比特文件旁边,
输出与普通运行逐 token 相同的文本;一轮草稿目前还不能收回自己的成本(见下文)。

### 快速开始

```bash
cargo install cortiq-cli
hf download infosave/Ripple-65B-Preview-cmf ripple-65b-q4tp.cmf --local-dir .
cortiq run ripple-65b-q4tp.cmf --prompt "用三段话解释傅里叶变换。" --greedy
```

使用草稿模型(模型旁边再放一个文件,`CMF_DFLASH=1`):

```bash
hf download infosave/Ripple-65B-Preview-cmf ripple-65b-q4tp.dflash.cmf --local-dir .
CMF_DFLASH=1 cortiq run ripple-65b-q4tp.cmf --prompt-file task.txt --greedy
```

需要 cortiq 0.8.60 或更新版本;预编译二进制在
[发布页](https://github.com/infosave2007/cmf/releases)。模型先在 `<think>` 中推理再回答。`--greedy` 开启草稿轮次。

#### OpenAI 兼容服务器

```bash
cortiq serve ripple-65b-q4tp.cmf --port 8080

curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "ripple", "max_tokens": 4096,
       "messages": [{"role": "user", "content": "解释 B 树如何分裂一个满节点。"}]}'
```

接口:`/v1/chat/completions`(流式或非流式)、`/v1/completions`、`/v1/models`,以及同一端口上的网页面板。`--host 127.0.0.1` 让服务器只在本机监听。对话在轮次之间把自己的行留在显卡上,每一轮只读取新内容;远场像 `cortiq run` 一样保持上限。

### 文件

| 文件 | GB | 适合 | 困惑度,wikitext-2 test 前 2048 个 token(BF16 6.94) | top-1 与 BF16 一致 |
|---|---:|---|---:|---:|
| `ripple-65b-q4tp.cmf` | 34.6 | 40 GB 显卡;带草稿模型需 48 GB | 7.09 | 91.1 % |
| `ripple-65b-q2tp.cmf` | 29.2 | 32 GB 显卡 | 7.43 | 86.2 % |
| `ripple-65b-q8_2f.cmf` | 64.8 | 80 GB 显卡 | 6.97 | 98.3 % |
| `ripple-65b-q4tp.dflash.cmf` | 2.3 | 放在 q4tp 文件旁 | | |
| `ripple-65b-q2tp.dflash.cmf` | 2.3 | 放在 q2tp 文件旁 | | |

`q4tp` 是四比特分块权重,注意力的 K、V 投影为八比特;`q2tp` 是两比特混合:第 0 层和最后两层为八比特,FFN 的 down
矩阵和注意力为四比特,每层中承载一半激活能量的 5 % FFN 神经元为八比特。所有文件的输出头都是八比特。在下文的 A40 上,
4 000 token 的提示下 q4tp 文件占 35.3 GB 显存,q2tp 文件占 30.3 GB。

### 这些文件做了什么

![46 GB 显卡上的长提示](https://huggingface.co/infosave/Ripple-65B-Preview-cmf/resolve/main/assets/ceiling_zh.png)

**固定的上限。** 提示的前 8192 行在显卡上精确读取。之后各行的键和值一读完就离开显卡,进入内存或 SSD。在最后 512 行
之前,用它们逐层构建远场,然后这些行被释放。远场保留前 4 行和后 1800 行的精确值,其余内容压缩为 16 个全注意力层每个头
32 个地标;64 个滑动层本来只保留 1024 行。显存和每个 token 的成本不再随文本增长。专利申请号 19/738,763。

**精确行。** 只有远场会丢掉密码。因此每一行按各层对它的处理打分,最好的 248 行保持精确。提示末尾的问题在回答前读取它们。

![密码和编码保持精确](https://huggingface.co/infosave/Ripple-65B-Preview-cmf/resolve/main/assets/exact_rows_zh.png)

**热神经元。** 一个 FFN 神经元是一行 gate、一行 up 和一列 down。在文本上,少数神经元承担了大部分工作:5 % 的神经元
承载一层一半的激活平方和。在两比特文件中,这些神经元以八比特作为"管"放在两比特核心旁边,引擎一次读取两者。随机选出
同样的 5 % 只能弥补到四比特差距的 41 %;最活跃的 5 % 弥补 86 %,K、V 投影改为八比特再补上大部分余下的差距。

![两比特权重,最活跃的神经元用八比特](https://huggingface.co/infosave/Ripple-65B-Preview-cmf/resolve/main/assets/hot_zh.png)

**草稿。** Ripple 没有自己的草稿头。DFlash2 草稿模型是建立在主干第 1、20、39、58、77 层输出之上的五层网络,一个块最多
提出七个 token;显卡把它们和下一个 token 一起在一个批次中验证,远场只沿被接受的 token 前移。贪心解码的文本有无草稿
完全相同;一轮的开销见下表。

### 速度

![普通与 Turbo](https://huggingface.co/infosave/Ripple-65B-Preview-cmf/resolve/main/assets/compare_zh.png)

NVIDIA A40(46 GB),Vulkan,`cortiq bench --core`,在该长度的提示之后生成 60 个 token,q4tp 文件。"普通"一列是
同一文件关闭远场(`CMF_O1=off`)。

| 提示,token 数 | 首个 token,秒(普通) | 首个 token,秒(**turbo**) | 解码,tok/s(普通) | 解码,tok/s(**turbo**) | 显存峰值,GB(普通) | 显存峰值,GB(**turbo**) |
|---:|---:|---:|---:|---:|---:|---:|
| 4 000 | 48.4 | 48.0 | 12.1 | 12.0 | 35.3 | 35.3 |
| 16 000 | 212.9 | 251.2 | 11.1 | 11.7 | 36.8 | 36.6 |
| 32 000 | 473.2 | 491.1 | 10.2 | 11.8 | 39.1 | 36.6 |
| 64 000 | 1138.2 | 987.8 | 8.8 | 11.8 | 43.2 | 36.6 |

**更大的显卡。** NVIDIA RTX PRO 6000(96 GB),Vulkan,4 000 token 的提示,60 个 token:

| 文件 | 提示,tok/s | 首个 token,秒 | 解码,tok/s |
|---|---:|---:|---:|
| q4tp | 209 | 19.2 | 29.8 |
| q2tp | 155 | 25.9 | 27.8 |
| q8_2f | 156 | 25.7 | 18.3 |

A40 上的 q2tp 文件:

| 提示,token 数 | 首个 token,秒 | 解码,tok/s | 显存峰值,GB |
|---:|---:|---:|---:|
| 4 000 | 77.8 | 11.8 | 30.3 |
| 64 000 | 1459.2 | 11.5 | 31.6 |

**使用草稿模型。** `cortiq run --greedy`,160 个 token 的回答,q4tp 文件:

| 回答 | 无草稿,tok/s | 有草稿,tok/s | 每轮接受的草稿数 |
|---|---:|---:|---:|
| 代码,A40 | 12.1 | 11.6 | 4.8 |
| 散文,A40 | 12.2 | 9.0 | 1.3 |
| 代码,RTX PRO 6000 | 30.2 | 26.9 | 4.8 |
| 散文,RTX PRO 6000 | 30.6 | 21.6 | 1.2 |

文本与普通运行逐 token 相同,但一轮的开销超过它换来的 token:在 RTX PRO 6000 上一轮耗时 160 毫秒,而普通一个 token 只需
32 毫秒,其中 75 毫秒是草稿块本身——它的五层要经过十五次显卡调用,中间的卷积和注意力在主机上完成。除非 `CMF_DFLASH=1`
要求,轮次保持关闭;固定七个草稿的一轮(`CMF_GRAPH_SPEC_K=7`)在代码上接受其中 4.6 个。

**主机内存。** 读取长提示时,它的行占用内存或磁盘(`--o1-spill DIR`)。映射的权重不计在内,系统可以把它们从内存中
换出。

| 提示,token 数 | 内存 |
|---:|---:|
| 4 000 | 0.2 GB |
| 16 000 | 4.4 GB |
| 32 000 | 7.7 GB |
| 64 000 | 14.9 GB |

**质量。** 在 60 000 个字符中埋入一个事实,四种类型(密钥、访问码、密码、房间号),模型先推理再回答:4 个中找到 3 个;第四个(密码)在 300 个 token 的回答用完时模型仍在推理。

### 参数

| 参数 | 含义 |
|---|---|
| `CMF_DFLASH=1` | 草稿轮次,配合 `--greedy` |
| `CMF_GRAPH_SPEC_K=n` | 每轮固定 `n` 个草稿(1 到 7),而不是由轮次自行确定的深度 |
| `--o1-spill DIR` | 长提示的行存入 `DIR` 中的文件,也可用 `CMF_O1_SPILL` |
| `--prompt-file FILE` | 超过命令行长度的提示 |
| `CMF_O1_PREFIX_MAX=N` | 在 `N` 行处封存,而不是 8192 |
| `CMF_O1=off` | 普通模型,任意长度下精确注意力 |
| `CMF_O1_SPEC=0` | 远场之上不用草稿 |

### 限制

- 远场和草稿轮次在 Vulkan 和 DX12 上运行;在 Metal 和 CPU 上这些文件按普通文件运行。长提示表格在一块 NVIDIA A40 上测得,八比特文件和草稿轮次在 RTX PRO 6000 上测得。
- 封存之后文本是近似的:久远的内容会淡化,保持精确的行除外。这些行背后的拟合见过埋入的事实,其他类型的重要文本未经测试。
- 提示结束后到来的行留在窗口内或进入远场。与已封存对话分叉的一轮会重新读取整个提示。
- 草稿需要无惩罚的贪心解码,目前还不能加快回答。

### 构建文件

```bash
cortiq convert ricursiveAI/Ripple-65B-Preview ripple-65b-q4tp.cmf --quant q4tp --tensor-quant 'model.layers.*.self_attn.k_proj.weight=q8_2f' --tensor-quant 'model.layers.*.self_attn.v_proj.weight=q8_2f'
cortiq far-turbo ripple-65b-q4tp.cmf ripple-65b-q4tp.cmf --exact-weights keep_ripple.bin
cortiq convert --model ricursiveAI/Ripple-65B-Preview-DFlash2 --output ripple-65b-q4tp.cmf --dflash-sidecar --quant q4tp
```

两比特文件:`cortiq hot-probe` 在文本上给出每个神经元的能量,`cortiq hot-bake COLD HOT energy.mass OUT --frac 0.05
--take self_attn.k_proj --take self_attn.v_proj` 把每层最活跃的 5 % 和 K、V 投影从八比特文件放入两比特文件。精确行
的权重(`keep_ripple.bin`)和脚本在[仓库](https://github.com/infosave2007/cmf/tree/master/scripts/ripple)中。

### 许可证

权重:[Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0),来自
[ricursiveAI/Ripple-65B-Preview](https://huggingface.co/ricursiveAI/Ripple-65B-Preview) 和
[ricursiveAI/Ripple-65B-Preview-DFlash2](https://huggingface.co/ricursiveAI/Ripple-65B-Preview-DFlash2)。
远场是美国专利申请号 19/738,763 的主题;见 [PATENTS.md](https://github.com/infosave2007/cmf/blob/master/PATENTS.md)。
