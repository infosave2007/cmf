---
license: other
license_name: openmdw-1.1
license_link: https://openmdw.ai/license/1-1/
library_name: cortiq
base_model: nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16
base_model_relation: quantized
pipeline_tag: text-generation
tags:
  - cmf
  - cortiq
  - quantized
  - q4tp
  - q8_2f
  - dory
  - nemotron
  - looped-transformer
  - vulkan
  - metal
language:
  - en
  - ru
  - zh
---

# Nemotron-Labs Dory 1.0 4B — CMF

[Dory 1.0 4B](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16) is NVIDIA's
looped transformer: a recurrent block of 40 sub-layers runs several times over the same weights,
so one checkpoint serves 1, 2 or 4 loops. It reasons in `<think>` and calls tools. Here it comes as
single [CMF](https://github.com/infosave2007/cmf) files for `cortiq`, a Rust inference engine with
no Python and no ML framework: NVIDIA, AMD and Intel GPUs through Vulkan/DX12, Apple silicon through
Metal, and the CPU.

[English](#quick-start) · [Русский](#документация-на-русском) · [中文](#中文说明)

## Quick start

```bash
cargo install cortiq-cli
hf download infosave/Nemotron-Labs-Dory1.0-4B-cmf dory-turbo.cmf --local-dir .
cortiq run dory-turbo.cmf --prompt "Write a Rust function that reverses a list." --no-think
```

Use cortiq 0.8.57 or later; prebuilt binaries are on the
[releases page](https://github.com/infosave2007/cmf/releases). `--no-think` asks for a direct
answer. Without it the model reasons first, so raise `--max-tokens`.

## Files

| file | weights | size | wikitext-2 ppl | use |
|---|---|---:|---:|---|
| `dory-turbo.cmf` | 4-bit GPTQ | 2.09 GB | 14.22 | the same weights as `q4tp-v2` with a router and a far field; **default** |
| `dory-q4tp-v2.cmf` | 4-bit GPTQ | 2.09 GB | 13.96 | two loops on every token |
| `dory-q8_2f.cmf` | 8-bit | 4.01 GB | 13.44 | closest to the original |

Perplexity: twelve 512-token windows of wikitext-2 test, token by token. The bf16 original scores
13.42 on them. GSM8K, 60 questions, direct answers, greedy: `q4tp-v2` 53, turbo 52, `q8_2f` 51.

## What the turbo file does

**One pass or two.** After the first pass over the recurrent block, the state is rebuilt by two
isolated PCA subspaces, and the better one decides whether the token leaves the loop. Nothing is
trained by gradient, and the decision needs no read-out. Patent application No. 19/452,440.

![One pass or two, token by token](https://huggingface.co/infosave/Nemotron-Labs-Dory1.0-4B-cmf/resolve/main/assets/router_en.png)

**A far field.** Full-attention layers keep the first rows and the last 1536 rows exact, and all
the others in 32 landmarks per head. Memory and the cost of a token stop growing with the text, and this
holds while a prompt is read: 3.76 GB on the card at 12 000, 32 000 and 64 000 tokens (RTX 5090). The rows wait
in RAM, or in files with `--o1-spill DIR`. This needs cortiq 0.8.59 or later.
Patent application No. 19/738,763.

![What a long prompt keeps on the card](https://huggingface.co/infosave/Nemotron-Labs-Dory1.0-4B-cmf/resolve/main/assets/far_field_en.png)

**Exact rows.** A far field alone loses a password. Rows it would lose are scored by what the layers
do with them, with no word lists, and the top 512 stay exact. A question at the end of the prompt
reads them before it is answered.

![How passwords stay exact](https://huggingface.co/infosave/Nemotron-Labs-Dory1.0-4B-cmf/resolve/main/assets/exact_rows_en.png)

**A prompt in one pass.** Its rows leave only their cache rows behind, and nobody reads their logits.
The last chunk takes two passes. This costs 1 % in the perplexity of what follows.

## Performance

![Standard and turbo](https://huggingface.co/infosave/Nemotron-Labs-Dory1.0-4B-cmf/resolve/main/assets/compare_en.png)

Decode in tok/s; the first row is 12 real prompts of 200 tokens through `cortiq run`, the others
`cortiq bench --core` after a prompt of that length.

| | `dory-q4tp-v2` | `dory-turbo` |
|---|---:|---:|
| **Vulkan, NVIDIA L4 (24 GB)** | | |
| real prompts | 49.6 | **60.7** |
| context 1 000 | 41.1 | **60.6** |
| context 4 000 | 35.1 | **51.8** |
| context 16 000 | 21.5 | **52.2** |
| context 32 000 | 14.1 | **54.0** |
| context 64 000 | prompt not read in 25 min | **53.0** |
| first token at 16 000 / 32 000 / 64 000 | 78 s / 360 s / over 25 min | **53 s / 112 s / 228 s** |
| device KV memory at 4 000 / 32 000 | 1.0 GB / 4.3 GB | **0.64 GB / 0.50 GB** |
| **Metal, MacBook Air M4 (24 GB)** | | |
| real prompts | 25.7 | **32.9** |
| context 1 000 | 22.1 | **34.6** |
| context 4 000 | 20.3 | **31.0** |
| context 16 000 | 14.8 | **29.3** |
| first token at 16 000 | 211 s | **134 s** |
| **CPU** | | |
| M4, context 256 | 18.0 | **27.7** |
| M4, context 2 000 | 12.1 | **18.3** |
| EPYC 9254, 24 cores, context 256 | 7.6 | **11.6** |
| EPYC 9254, 24 cores, context 4 000 | 5.3 | **8.2** |

About 85 % of tokens take one pass, on Wikipedia and on a code answer alike. `dory-q8_2f.cmf` on
real prompts: 32.0 tok/s on Vulkan, 15.4 on Metal, 13.8 on the M4 CPU. Long windows, perplexity of
two windows of 16384 tokens: exact 8.10, turbo's near and far fields 8.85, a cut to 2048 rows 11.03.

## Limits

- A fact is safe as long as it ranks in the 512 rows. Facts in 100 000 characters of text: 8 of 8 on
  Vulkan, the standard file 4 of 4. In 250 000 characters: 2 of 3 whole, the third without its last digit.
- A turn that goes back over a sealed conversation reads the whole prompt again, and chooses again.
- Rows that arrive after a prompt ends can only stay in the last 1536 rows or go to the far field.
- `--loops 2` on a turbo file turns the router, the far field and the rest off. See the [guide](https://github.com/infosave2007/cmf/blob/master/docs/DORY.md).

## Usage

```bash
cortiq run dory-turbo.cmf                                   # interactive chat
cortiq run dory-turbo.cmf --prompt "..." --greedy --no-think
cortiq run dory-q4tp-v2.cmf --loops 4 --prompt "..."        # four loops on every token
cortiq serve dory-turbo.cmf --port 8080                     # OpenAI-compatible, with tool calls
```

## License

Weights: [OpenMDW-1.1](https://openmdw.ai/license/1-1/). U.S. patent applications filed with the
USPTO: No. 19/452,440 (January 19, 2026), No. 19/738,763 (July 13, 2026). Titles and the patent
grant: [PATENTS.md](https://github.com/infosave2007/cmf/blob/master/PATENTS.md).

---

## Документация на русском

[Dory 1.0 4B](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16) — циклический
трансформер NVIDIA: рекуррентный блок из 40 подслоёв проходит несколько раз по одним и тем же весам,
поэтому один чекпоинт работает с 1, 2 или 4 циклами. Модель рассуждает в `<think>` и вызывает
инструменты. Здесь она лежит одиночными файлами [CMF](https://github.com/infosave2007/cmf) для
`cortiq`, движка на Rust без Python и ML-фреймворков: GPU NVIDIA, AMD и Intel через Vulkan/DX12,
Apple silicon через Metal и CPU.

```bash
cargo install cortiq-cli
hf download infosave/Nemotron-Labs-Dory1.0-4B-cmf dory-turbo.cmf --local-dir .
cortiq run dory-turbo.cmf --prompt "Напиши функцию на Rust, которая разворачивает список." --no-think
```

Нужен cortiq 0.8.57 или новее. `--no-think` просит прямой ответ; без него модель сначала рассуждает,
поэтому увеличьте `--max-tokens`.

### Файлы

| файл | веса | размер | ppl на wikitext-2 | назначение |
|---|---|---:|---:|---|
| `dory-turbo.cmf` | 4 бита, GPTQ | 2,09 ГБ | 14,22 | те же веса, что в `q4tp-v2`, плюс роутер и дальнее поле; **по умолчанию** |
| `dory-q4tp-v2.cmf` | 4 бита, GPTQ | 2,09 ГБ | 13,96 | два цикла на каждом токене |
| `dory-q8_2f.cmf` | 8 бит | 4,01 ГБ | 13,44 | ближе всего к оригиналу |

Перплексия: двенадцать окон по 512 токенов из wikitext-2 test, по одному токену. Оригинал bf16 даёт
на них 13,42. GSM8K, 60 вопросов, прямые ответы, жадно: `q4tp-v2` 53, турбо 52, `q8_2f` 51.

### Что делает турбо-файл

**Один проход или два.** После первого прохода по рекуррентному блоку состояние восстанавливают два
изолированных подпространства PCA, и то, что справилось лучше, решает, выйдет ли токен из цикла.
Обучения градиентом нет, чтения выхода для решения не нужно. Заявка на патент № 19/452,440.

![Один проход или два, для каждого токена](https://huggingface.co/infosave/Nemotron-Labs-Dory1.0-4B-cmf/resolve/main/assets/router_ru.png)

**Дальнее поле.** Слои полного внимания держат точно первые строки и последние 1536, а все остальные
сводят к 32 опорным точкам на голову. Память и цена токена перестают расти вместе с текстом, и при чтении промпта
тоже: 3,76 ГБ на карте при 12 000, 32 000 и 64 000 токенов (RTX 5090). Строки ждут в RAM или, с `--o1-spill DIR`, в файлах. Для этого нужен cortiq 0.8.59 или новее. Заявка
на патент № 19/738,763.

![Что длинный промпт хранит на карте](https://huggingface.co/infosave/Nemotron-Labs-Dory1.0-4B-cmf/resolve/main/assets/far_field_ru.png)

**Точные строки.** Одно дальнее поле теряет пароль. Строки, которые оно потеряло бы, получают оценку
по работе самих слоёв, без списков слов, и лучшие 512 остаются точными. Вопрос в конце промпта читает
их ещё до ответа.

![Как пароли остаются точными](https://huggingface.co/infosave/Nemotron-Labs-Dory1.0-4B-cmf/resolve/main/assets/exact_rows_ru.png)

**Промпт за один проход.** От его строк остаются только строки кеша, а логиты никто не читает. Последний
кусок идёт за два прохода. Это стоит 1 % перплексии того, что идёт дальше.

### Скорость

![Обычная и турбо](https://huggingface.co/infosave/Nemotron-Labs-Dory1.0-4B-cmf/resolve/main/assets/compare_ru.png)

Декод в ток/с; первая строка — 12 реальных запросов по 200 токенов через `cortiq run`, остальные —
`cortiq bench --core` после промпта такой длины.

| | `dory-q4tp-v2` | `dory-turbo` |
|---|---:|---:|
| **Vulkan, NVIDIA L4 (24 ГБ)** | | |
| реальные запросы | 49,6 | **60,7** |
| контекст 1 000 | 41,1 | **60,6** |
| контекст 4 000 | 35,1 | **51,8** |
| контекст 16 000 | 21,5 | **52,2** |
| контекст 32 000 | 14,1 | **54,0** |
| контекст 64 000 | промпт не прочитан за 25 мин | **53,0** |
| первый токен при 16 000 / 32 000 / 64 000 | 78 с / 360 с / больше 25 мин | **53 с / 112 с / 228 с** |
| память KV при 4 000 / 32 000 | 1,0 ГБ / 4,3 ГБ | **0,64 ГБ / 0,50 ГБ** |
| **Metal, MacBook Air M4 (24 ГБ)** | | |
| реальные запросы | 25,7 | **32,9** |
| контекст 1 000 | 22,1 | **34,6** |
| контекст 4 000 | 20,3 | **31,0** |
| контекст 16 000 | 14,8 | **29,3** |
| первый токен при 16 000 | 211 с | **134 с** |
| **CPU** | | |
| M4, контекст 256 | 18,0 | **27,7** |
| M4, контекст 2 000 | 12,1 | **18,3** |
| EPYC 9254, 24 ядра, контекст 256 | 7,6 | **11,6** |
| EPYC 9254, 24 ядра, контекст 4 000 | 5,3 | **8,2** |

Примерно 85 % токенов идут за один проход: и на Википедии, и в ответе с кодом. `dory-q8_2f.cmf` на
реальных запросах: 32,0 ток/с на Vulkan, 15,4 на Metal, 13,8 на CPU M4. Длинные окна, перплексия двух
окон по 16384 токена: точно 8,10, ближнее и дальнее поле турбо 8,85, обрезка до 2048 строк 11,03.

### Ограничения

- Факт в безопасности, пока попадает в 512 строк. Факты в 100 000 символов текста: 8 из 8 на Vulkan,
  обычный файл 4 из 4. В 250 000 символов: 2 из 3 целиком, в третьем нет последней цифры.
- Реплика, расходящаяся с запечатанной историей, читает весь промпт заново и выбирает строки снова.
- Строки, пришедшие после конца промпта, могут быть только среди последних 1536 или в дальнем поле.
- `--loops 2` на турбо-файле выключает роутер, дальнее поле и остальное. Подробнее в [руководстве](https://github.com/infosave2007/cmf/blob/master/docs/DORY.ru.md).

### Лицензия

Веса: [OpenMDW-1.1](https://openmdw.ai/license/1-1/). Заявки на патенты США, поданные в USPTO:
№ 19/452,440 (19 января 2026), № 19/738,763 (13 июля 2026). Названия и патентный грант:
[PATENTS.md](https://github.com/infosave2007/cmf/blob/master/PATENTS.md).

---

## 中文说明

[Dory 1.0 4B](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16) 是 NVIDIA 的循环
Transformer:由 40 个子层组成的循环块在同一份权重上运行多次,因此同一个权重文件可以跑 1、2 或 4
次循环。模型在 `<think>` 中推理,并能调用工具。这里提供单文件 [CMF](https://github.com/infosave2007/cmf),
配合 `cortiq` 使用。`cortiq` 是 Rust 推理引擎,不需要 Python 和机器学习框架:NVIDIA、AMD、Intel
GPU 走 Vulkan/DX12,Apple 芯片走 Metal,也可以用 CPU。

```bash
cargo install cortiq-cli
hf download infosave/Nemotron-Labs-Dory1.0-4B-cmf dory-turbo.cmf --local-dir .
cortiq run dory-turbo.cmf --prompt "写一个反转列表的 Rust 函数。" --no-think
```

需要 cortiq 0.8.57 或更新版本。`--no-think` 要求直接回答;不加时模型会先推理,请调大 `--max-tokens`。

### 文件

| 文件 | 权重 | 大小 | wikitext-2 ppl | 用途 |
|---|---|---:|---:|---|
| `dory-turbo.cmf` | 4 位 GPTQ | 2.09 GB | 14.22 | 与 `q4tp-v2` 相同的权重,加上路由器和远场;默认 |
| `dory-q4tp-v2.cmf` | 4 位 GPTQ | 2.09 GB | 13.96 | 每个 token 两次循环 |
| `dory-q8_2f.cmf` | 8 位 | 4.01 GB | 13.44 | 最接近原始模型 |

困惑度:wikitext-2 test 的 12 个 512-token 窗口,逐 token 计算。bf16 原始模型为 13.42。GSM8K,60
题,直接回答,贪心解码:`q4tp-v2` 53,Turbo 52,`q8_2f` 51。

### Turbo 文件做了什么

一次或两次循环。第一次经过循环块后,由两个相互隔离的 PCA 子空间重建状态,重建得更好的那个决定 token
是否离开循环。没有梯度训练,决策也不需要读取输出。专利申请号 19/452,440。

![每个 token 一次或两次循环](https://huggingface.co/infosave/Nemotron-Labs-Dory1.0-4B-cmf/resolve/main/assets/router_zh.png)

远场。全注意力层精确保留最前几行和最后 1536 行,其余全部压缩到每个头 32 个地标。内存和每个 token
的代价不再随文本增长,读取提示词时也是如此:12 000、32 000 和 64 000 token 时显卡上都是 3.76 GB(RTX 5090)。这些行留在内存中,使用 `--o1-spill DIR` 时留在文件中。此功能需要 cortiq 0.8.59 或更新版本。专利申请号 19/738,763。

![长提示词在显卡上保留什么](https://huggingface.co/infosave/Nemotron-Labs-Dory1.0-4B-cmf/resolve/main/assets/far_field_zh.png)

精确行。只靠远场会丢失密码。远场会丢失的行,靠各层自身的行为打分,不使用词表,得分最高的 512 行保持
精确。提示词末尾的问题在回答之前就会读到它们。

![密码如何保持精确](https://huggingface.co/infosave/Nemotron-Labs-Dory1.0-4B-cmf/resolve/main/assets/exact_rows_zh.png)

单次循环读取提示词。提示词的行只留下缓存行,没有人读取它们的 logits。最后一块走两次循环。代价是后续
内容的困惑度上升 1%。

### 速度

![普通版与 Turbo](https://huggingface.co/infosave/Nemotron-Labs-Dory1.0-4B-cmf/resolve/main/assets/compare_zh.png)

解码速度,单位 tok/s。第一行是 12 个真实提问,每个 200 token,通过 `cortiq run` 运行;其余是相应长度
提示词之后的 `cortiq bench --core`。

| | `dory-q4tp-v2` | `dory-turbo` |
|---|---:|---:|
| Vulkan,NVIDIA L4((24 GB) | | |
| 真实提问 | 49.6 | 60.7 |
| 上下文 1 000 | 41.1 | 60.6 |
| 上下文 4 000 | 35.1 | 51.8 |
| 上下文 16 000 | 21.5 | 52.2 |
| 上下文 32 000 | 14.1 | 54.0 |
| 上下文 64 000 | 25 分钟未读完提示词 | 53.0 |
| 首 token,16 000 / 32 000 / 64 000 | 78 秒 / 360 秒 / 超过 25 分钟 | 53 秒 / 112 秒 / 228 秒 |
| 设备 KV 内存,4 000 / 32 000 | 1.0 GB / 4.3 GB | 0.64 GB / 0.50 GB |
| Metal,MacBook Air M4((24 GB) | | |
| 真实提问 | 25.7 | 32.9 |
| 上下文 1 000 | 22.1 | 34.6 |
| 上下文 4 000 | 20.3 | 31.0 |
| 上下文 16 000 | 14.8 | 29.3 |
| 首 token,16 000 | 211 秒 | 134 秒 |
| CPU | | |
| M4,上下文 256 | 18.0 | 27.7 |
| M4,上下文 2 000 | 12.1 | 18.3 |
| EPYC 9254,24 核,上下文 256 | 7.6 | 11.6 |
| EPYC 9254,24 核,上下文 4 000 | 5.3 | 8.2 |

约 85% 的 token 只走一次循环,维基百科文本和代码回答都是如此。`dory-q8_2f.cmf` 的真实提问速度:Vulkan
32.0 tok/s,Metal 15.4,M4 CPU 13.8。长窗口,两个 16384 token 窗口的困惑度:精确 8.10,Turbo 的近场加
远场 8.85,截断到 2048 行 11.03。

### 限制

- 事实只要进入这 512 行就是安全的。100 000 字符文本中的事实:Vulkan 8 中 8,普通文件 4 中 4。250 000 字符:3 中 2 完整,第 3 个缺最后一位。
- 与已封存历史不一致的一轮对话会重新读取整个提示词,并重新选择。
- 提示词结束后到达的行,只能留在最后 1536 行内或进入远场。
- 在 Turbo 文件上使用 `--loops 2` 会关闭路由器、远场等功能。详见[指南](https://github.com/infosave2007/cmf/blob/master/docs/DORY.zh.md)。

### 许可证

权重:[OpenMDW-1.1](https://openmdw.ai/license/1-1/)。已向 USPTO 提交的美国专利申请:No. 19/452,440
((2026 年 1 月 19 日)、No. 19/738,763((2026 年 7 月 13 日)。名称和专利授权见
[PATENTS.md](https://github.com/infosave2007/cmf/blob/master/PATENTS.md)。
