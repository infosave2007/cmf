---
license: apache-2.0
language: [en, ru]
tags: [cmf, cortiq, embryo, small-model, o1-attention, resonance-routing]
---

# Embryo-56M

A language model of 56 million parameters, in its own architecture and its own format, born and trained from scratch on one laptop GPU (Apple M4, no foreign weights). It chats in English and Russian in short answers, and it answers facts exactly from records inside the same file: 4,265 medicinal plants and 663 cards on countries, chemical elements, science, computing, history, books and everyday how-to. The network itself is small and does not know facts, so a question for a fact outside the records gets an honest "I don't know" instead of a guess. Talk and small tasks go to the network: short, grammatical, and weak.

## Quick start

```bash
cargo install cortiq-cli --features gpu   # needs cortiq 0.8.53 or newer
hf download infosave/Embryo-56M-cmf embryo-56m.cmf --local-dir .
cortiq run embryo-56m.cmf -p "What is valerian used for?"
cortiq serve embryo-56m.cmf --port 8000   # OpenAI-compatible API
```

## What it answers

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

| method | what it does here | application |
|---|---|---|
| Resonance routing | Each record is an isolated subspace of hidden states. A request goes to the record with the smallest reconstruction error, measured over the tokens of the name the question is about plus the rest of the message. Follow-up questions have subspaces of their own, and so has the decision that a message asks for a fact the network does not have. No classifier, no keyword rules. The same principle picks an FFN expert per token. | US 19/452,440 |
| Unified execution | One container: shared backbone, records appended without touching a single byte of it. | US 19/731,402 |
| Constant-memory attention | Gated delta mixers and two bounded anchors: state does not grow with context, no KV cache to evict. | US 19/738,763 |

Training ran on the M4 GPU with our own Metal kernels: raw text first, then response-only chat with raw replay. Data: FineWeb-edu, FineWeb-2, code and math corpora; SmolTalk, OASST2 and GrandMaster-PRO-MAX (its answers were written by GPT-4-Turbo). Plant cards come from our herbal reference.

## Limits

Outside the records the answers are the words of a very small model: grammatical, short, often empty. Do not use it for medical decisions: the cards describe traditional use and the evidence, they are not advice.

Apache-2.0. Patents: see [PATENTS.md](https://github.com/infosave2007/cmf/blob/master/PATENTS.md).

---

# Embryo-56M (русский)

Языковая модель на 56 миллионов параметров в собственной архитектуре и собственном формате, рождена и обучена с нуля на GPU одного ноутбука (Apple M4, без чужих весов). Коротко отвечает на английском и русском, а факты отдаёт дословно из записей в том же файле: 4265 лекарственных растений и 663 карточек о странах, химических элементах, науке, компьютерах, истории, книгах и бытовых советах. Сама сеть маленькая и фактов не знает, поэтому на вопрос о факте вне записей модель честно отвечает «не знаю», а не угадывает. Разговор и простые просьбы идут в сеть: коротко, грамотно и слабо.

```bash
cargo install cortiq-cli --features gpu
hf download infosave/Embryo-56M-cmf embryo-56m.cmf --local-dir .
cortiq run embryo-56m.cmf -p "Для чего применяют валериану?"
```

Скорость: GPU Apple M4 около 650 ток/с, CPU около 470 ток/с. Файл 91 МБ. Состояние 10.1 МБ при любой длине контекста. Простые вопросы о растениях 155 из 180, факты 218 из 223, уточнения простыми словами 900 из 1200, вопросы о фактах вне записей с честным отказом 38 из 44.

Выбор записи делает резонансный роутер (US 19/452,440): ошибка реконструкции по токенам названия и остальному сообщению, без классификатора и правил. Свои подпространства есть у уточняющих вопросов и у решения, что в сообщении спрашивают факт, которого у сети нет. Записи дописываются в файл, не меняя ни байта сети (US 19/731,402). Память постоянна (US 19/738,763). Не используйте для медицинских решений.

---

# Embryo-56M（中文）

一个 5600 万参数的语言模型，采用自有架构和自有格式，在一台笔记本的 GPU（Apple M4）上从零训练，没有使用他人权重。用英语和俄语简短回答，并从同一文件内的记录中逐字给出事实：4265 种药用植物，以及 663 张关于国家、化学元素、科学、计算机、历史、书籍和日常操作的卡片。网络本身很小，不掌握事实，因此对记录之外的事实问题会如实回答“不知道”，而不是猜测。闲聊和简单任务交给网络：简短、合乎语法，但能力很弱。

```bash
cargo install cortiq-cli --features gpu
hf download infosave/Embryo-56M-cmf embryo-56m.cmf --local-dir .
cortiq run embryo-56m.cmf -p "What is valerian used for?"
```

速度：Apple M4 GPU 约 650 tok/s，CPU 约 470 tok/s。文件 91 MB。任意上下文长度下状态均为 10.1 MB。植物简单问题 180 个中 155 个，事实问题 223 个中 218 个，用平实语言的追问 1200 个中 900 个，记录之外的事实问题如实拒答 44 个中 38 个。

记录由共振路由选择（US 19/452,440）：按名称词元和其余消息的重构误差决定，不用分类器和规则。追问以及“消息在询问网络没有的事实”这一判断也有各自的子空间。记录追加写入文件，不改动网络的任何字节（US 19/731,402）。内存恒定（US 19/738,763）。请勿用于医疗决策。
