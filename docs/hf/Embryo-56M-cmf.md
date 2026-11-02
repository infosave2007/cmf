---
license: apache-2.0
language: [en, ru]
tags: [cmf, cortiq, embryo, small-model, o1-attention, resonance-routing]
---

# Embryo-56M

A language model of 56 million parameters, in its own architecture and its own format, born and trained from scratch on one laptop GPU (Apple M4, no foreign weights). It chats in English and Russian in short answers, and it answers facts exactly from records inside the same file: 4,265 medicinal plants and 663 cards on countries, chemical elements, science, computing, history, books and everyday how-to. The network itself is small and does not know facts, so a question for a fact outside the records gets an honest "I don't know" instead of a guess. Talk and small tasks go to the network: short, grammatical, and weak.

> This is the first experimental version of a model built entirely on the technologies of Oleg Kirichenko: architecture, file format, router, runtime and trainer. Larger models for different tasks and the mechanisms for training models will follow.

Try it without installing: tests and chat in the [Space](https://huggingface.co/spaces/infosave/embryo-56m).

## Quick start

```bash
cargo install cortiq-cli --features gpu   # needs cortiq 0.8.53 or newer
hf download infosave/Embryo-56M-cmf embryo-56m.cmf --local-dir .
cortiq run embryo-56m.cmf -p "What is valerian used for?"
cortiq serve embryo-56m.cmf --port 8000   # OpenAI-compatible API
```

## What it answers

![How a message finds its answer](https://huggingface.co/infosave/Embryo-56M-cmf/resolve/main/assets/routing_en.png)

| question | answer | from |
|---|---|---|
| What is valerian used for? | Sedative (calming) medicinal agent and antispasmodic acting on the smooth muscle of the gastrointestinal tract and urinary system | record `herbs` |
| Какая столица Австралии? | Австралия — страна в Океании. Столица: Канберра. Валюта: австралийский доллар. Официальный язык: английский. | record `countries` |
| Why is the sky blue? | The sky looks blue because air molecules scatter short-wavelength blue light more than red light. This is called Rayleigh scattering. | record `science` |
| Give me a tip for sleeping better. | Go to bed and get up at the same time every day, keep the room dark and cool, avoid screens and caffeine in the evening, and get daylight and movement during the day. | record `howto` |
| What is 2+2? | I don't know that: it is not in my records, and the small network behind me would only guess. Ask me about medicinal plants, countries, chemical elements, science, computing, history or books. | record `dontknow` |
| Tell me a story about a dragon. | A dragon is a unique and intriguing character, with a complex web of characters and characters that make it a fascinating story. | the network |

Answers from a record are verbatim, and so is the refusal. `x_cortiq_route` in the server response and the last line of `cortiq run` say which record answered.

## Measured

![Measured numbers: speed and state](https://huggingface.co/infosave/Embryo-56M-cmf/resolve/main/assets/advantages_en.png)

| | |
|---|---|
| parameters | 55.9 M (34.7 M active per token) |
| file | 91 MB, q8_2f weights plus the records |
| decode, Apple M4 GPU | about 650 tok/s |
| decode, Apple M4 CPU | about 470 tok/s |
| decode, NVIDIA RTX 2000 Ada / its EPYC 7443P host | 925 / 612 tok/s |
| decode after a 2000-token prompt, M4 GPU / CPU | 535 / 379 tok/s |
| decode after a 2000-token prompt, RTX 2000 Ada / EPYC | 796 / 405 tok/s |
| 3000 tokens in one run on the M4, GPU / CPU | 389 / 398 tok/s: the GPU slows down after a few seconds of continuous work |
| state per sequence | 10.1 MB, the same at any context length |
| plain questions about plants, phrasing not used in the fit | 155 of 180 |
| plant facts, held-out questions | 218 of 223 |
| follow-ups about a plant in plain words, phrasing not used in the fit | 900 of 1200 |
| unrelated messages taken for a follow-up | 41 of 1380 |
| factual questions outside the records, refused | 38 of 44, a record answered the other 6 |
| ordinary requests refused by mistake | 1 of 40 |
| questions about world entities left out of the fit | 367 of 445 |
| general prompts answered from a record | 86 of 1444, about a fifth of them a correct card that misses the question |
| general prompts refused | 243 of 1444, nearly all of them questions for a fact |
| recall of a code word in text, K = 4, distance 128 / 512 / 1024 / 1800 tokens | 62 / 70 / 56 / 47 % |

## How it is built

![One token through the network](https://huggingface.co/infosave/Embryo-56M-cmf/resolve/main/assets/architecture_en.png)

| method | what it does here | U.S. patent application |
|---|---|---|
| Resonance routing | Each record is an isolated subspace of hidden states. A request goes to the record with the smallest reconstruction error, measured over the tokens of the name the question is about plus the rest of the message. Follow-up questions have subspaces of their own, and so has the decision that a message asks for a fact the network does not have. No classifier, no keyword rules. The same principle picks an FFN expert per token. | No. 19/452,440 |
| Unified execution | One container: shared backbone, records appended without touching a single byte of it. | No. 19/731,402 |
| Constant-memory attention | Gated delta mixers and two bounded anchors: state does not grow with context, no KV cache to evict. | No. 19/738,763 |

Training ran on the M4 GPU with our own Metal kernels: raw text first, then response-only chat with raw replay. Data: FineWeb-edu, FineWeb-2, code and math corpora; SmolTalk, OASST2 and GrandMaster-PRO-MAX (its answers were written by GPT-4-Turbo). Plant cards come from our herbal reference.

## Limits

Outside the records the answers are the words of a very small model: grammatical, short, often empty. Do not use it for medical decisions: the cards describe traditional use and the evidence, they are not advice.

Apache-2.0. U.S. patent applications filed with the USPTO: No. 19/452,440 (January 19, 2026), No. 19/731,402 (July 6, 2026), No. 19/738,763 (July 13, 2026). Titles and the patent grant: [PATENTS.md](https://github.com/infosave2007/cmf/blob/master/PATENTS.md).

---

# Embryo-56M (русский)

Языковая модель на 56 миллионов параметров в собственной архитектуре и собственном формате, рождена и обучена с нуля на GPU одного ноутбука (Apple M4, без чужих весов). Коротко отвечает на английском и русском, а факты отдаёт дословно из записей в том же файле: 4265 лекарственных растений и 663 карточек о странах, химических элементах, науке, компьютерах, истории, книгах и бытовых советах. Сама сеть маленькая и фактов не знает, поэтому на вопрос о факте вне записей модель честно отвечает «не знаю», а не угадывает. Разговор и простые просьбы идут в сеть: коротко, грамотно и слабо.

> Это первый экспериментальный вариант модели, полностью основанной на технологиях Олега Кириченко: архитектура, формат файла, роутер, рантайм и обучение. В дальнейшем выйдут более крупные модели для различных задач и механизмы обучения моделей.

Попробовать без установки: тесты и чат в [Space](https://huggingface.co/spaces/infosave/embryo-56m).

```bash
cargo install cortiq-cli --features gpu
hf download infosave/Embryo-56M-cmf embryo-56m.cmf --local-dir .
cortiq run embryo-56m.cmf -p "Для чего применяют валериану?"
```

![Как сообщение находит ответ](https://huggingface.co/infosave/Embryo-56M-cmf/resolve/main/assets/routing_ru.png)

![Измеренные числа: скорость и состояние](https://huggingface.co/infosave/Embryo-56M-cmf/resolve/main/assets/advantages_ru.png)

Скорость: GPU Apple M4 около 650 ток/с, CPU около 470 ток/с. Файл 91 МБ. Состояние 10.1 МБ при любой длине контекста. Простые вопросы о растениях 155 из 180, факты 218 из 223, уточнения простыми словами 900 из 1200, вопросы о фактах вне записей с честным отказом 38 из 44.

Выбор записи делает резонансный роутер (заявка США № 19/452,440): ошибка реконструкции по токенам названия и остальному сообщению, без классификатора и правил. Свои подпространства есть у уточняющих вопросов и у решения, что в сообщении спрашивают факт, которого у сети нет. Записи дописываются в файл, не меняя ни байта сети (заявка США № 19/731,402). Память постоянна (заявка США № 19/738,763). Не используйте для медицинских решений.

![Один токен через сеть](https://huggingface.co/infosave/Embryo-56M-cmf/resolve/main/assets/architecture_ru.png)

Заявки на патенты США, поданные в USPTO: № 19/452,440 (19 января 2026), № 19/731,402 (6 июля 2026), № 19/738,763 (13 июля 2026). Названия и патентная лицензия: [PATENTS.md](https://github.com/infosave2007/cmf/blob/master/PATENTS.md).

---

# Embryo-56M（中文）

一个 5600 万参数的语言模型，采用自有架构和自有格式，在一台笔记本的 GPU（Apple M4）上从零训练，没有使用他人权重。用英语和俄语简短回答，并从同一文件内的记录中逐字给出事实：4265 种药用植物，以及 663 张关于国家、化学元素、科学、计算机、历史、书籍和日常操作的卡片。网络本身很小，不掌握事实，因此对记录之外的事实问题会如实回答“不知道”，而不是猜测。闲聊和简单任务交给网络：简短、合乎语法，但能力很弱。

> 这是第一个实验版本的模型，完全基于 Oleg Kirichenko 的技术：架构、文件格式、路由器、运行时和训练器。之后将发布面向不同任务的更大模型以及模型训练机制。

无需安装即可试用：[Space](https://huggingface.co/spaces/infosave/embryo-56m) 中有测试和对话。

```bash
cargo install cortiq-cli --features gpu
hf download infosave/Embryo-56M-cmf embryo-56m.cmf --local-dir .
cortiq run embryo-56m.cmf -p "What is valerian used for?"
```

![消息如何找到答案](https://huggingface.co/infosave/Embryo-56M-cmf/resolve/main/assets/routing_zh.png)

![实测数字：速度与状态](https://huggingface.co/infosave/Embryo-56M-cmf/resolve/main/assets/advantages_zh.png)

速度：Apple M4 GPU 约 650 tok/s，CPU 约 470 tok/s。文件 91 MB。任意上下文长度下状态均为 10.1 MB。植物简单问题 180 个中 155 个，事实问题 223 个中 218 个，用平实语言的追问 1200 个中 900 个，记录之外的事实问题如实拒答 44 个中 38 个。

记录由共振路由选择（美国专利申请号 19/452,440）：按名称词元和其余消息的重构误差决定，不用分类器和规则。追问以及“消息在询问网络没有的事实”这一判断也有各自的子空间。记录追加写入文件，不改动网络的任何字节（美国专利申请号 19/731,402）。内存恒定（美国专利申请号 19/738,763）。请勿用于医疗决策。

![一个 token 穿过网络](https://huggingface.co/infosave/Embryo-56M-cmf/resolve/main/assets/architecture_zh.png)

已向 USPTO 提交的美国专利申请：No. 19/452,440（2026 年 1 月 19 日）、No. 19/731,402（2026 年 7 月 6 日）、No. 19/738,763（2026 年 7 月 13 日）。名称和专利授权见 [PATENTS.md](https://github.com/infosave2007/cmf/blob/master/PATENTS.md)。
