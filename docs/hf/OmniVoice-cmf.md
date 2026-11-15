---
license: cc-by-nc-4.0
library_name: cortiq
base_model: k2-fsa/OmniVoice
base_model_relation: quantized
pipeline_tag: text-to-speech
tags:
  - cmf
  - cortiq
  - text-to-speech
  - voice-cloning
  - multilingual
language:
  - multilingual
  - en
  - ru
  - zh
---

# OmniVoice — CMF

[OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) by k2-fsa speaks 600+ languages, clones a voice from a few
seconds of audio and designs one from attributes. This is the model and its Higgs audio tokenizer in one
[CMF](https://github.com/infosave2007/cmf) file for `cortiq`, a Rust inference engine with no Python.

- On an NVIDIA A40, CUDA: 4.5 s of speech in 4.4 s, 34 s of a long text in 32 s (32 steps) — real time.
- The same file on Vulkan (any vendor's card), on Apple silicon (Metal) and on the CPU.
- Checked against the float32 release: the same prompt and output length, Whisper reads our audio as the
  release's in English, Russian and Chinese, for designed and cloned voices.

[English](#quick-start) · [Русский](#документация-на-русском) · [中文](#中文说明)

## Quick start

```bash
cargo install cortiq-cli --features cuda   # NVIDIA; without the feature: Vulkan / CPU
hf download infosave/OmniVoice-cmf omnivoice-q8_2f.cmf --local-dir .
CMF_GPU=cuda cortiq tts omnivoice-q8_2f.cmf "Hello! This voice comes from a single file." --lang en --out hello.wav
```

Use cortiq 0.8.99 or later. `CMF_GPU=cuda` needs the NVIDIA driver and the CUDA 12.x libraries; drop it to run on
Vulkan, `CMF_GPU=0` runs on the CPU.

Listen to this file on CUDA, the release's sampling: [English](https://huggingface.co/infosave/OmniVoice-cmf/resolve/main/samples/en.wav) ·
[Russian, a designed voice](https://huggingface.co/infosave/OmniVoice-cmf/resolve/main/samples/ru_design.wav) · [Chinese](https://huggingface.co/infosave/OmniVoice-cmf/resolve/main/samples/zh.wav) · [Sichuanese](https://huggingface.co/infosave/OmniVoice-cmf/resolve/main/samples/zh_sichuan.wav) ·
[a laugh tag](https://huggingface.co/infosave/OmniVoice-cmf/resolve/main/samples/en_laughter.wav) · [a cloned voice](https://huggingface.co/infosave/OmniVoice-cmf/resolve/main/samples/clone.wav) (its reference: [ref](https://huggingface.co/infosave/OmniVoice-cmf/resolve/main/samples/clone_reference.wav)) ·
[a 34 s text in three pieces](https://huggingface.co/infosave/OmniVoice-cmf/resolve/main/samples/long.wav).

## Three ways to pick a voice

```bash
# the model picks a voice
cortiq tts omnivoice-q8_2f.cmf "Привет! Это проверка синтеза речи." --lang ru --out auto.wav
# voice design: gender, age, pitch, whisper, an accent (English) or a dialect (Chinese)
cortiq tts omnivoice-q8_2f.cmf "I can't believe it [laughter] that is so funny!" --lang en \
  --instruct "male, british accent" --out design.wav
# voice cloning: 3–10 s of one speaker and its transcript
cortiq tts omnivoice-q8_2f.cmf "This sentence is spoken in the cloned voice." --lang en \
  --ref-audio speaker.wav --ref-text "What the speaker says in the clip." --out clone.wav
```

| option | what it does |
|---|---|
| `--lang` | an ISO 639 code or an English name (`en`, `ru`, `zh`, `Swahili`, …); without it the model guesses |
| `--instruct` | `male` / `female`; `child`, `teenager`, `young adult`, `middle-aged`, `elderly`; `very low pitch` … `very high pitch`; `whisper`; `american accent`, `british accent`, …; Chinese: `男，四川话` |
| `--ref-audio`, `--ref-text` | a reference clip (WAV) and its transcript |
| `--steps` | decoding passes (32; 16 is faster and a little rougher) |
| `--duration`, `--speed` | seconds of speech, or a speaking rate (above 1 is faster) |
| `--temperature`, `--guidance`, `--seed` | code sampling (0 = greedy), classifier-free guidance (2), the seed |

Tags inside the text: `[laughter]`, `[sigh]`, `[question-en]`, `[surprise-oh]` and the others of the release. Text over
30 s of speech is cut at punctuation into ~15 s pieces and joined.

## Files

| file | size | what it is |
|---|---:|---|
| `omnivoice-q8_2f.cmf` | 1.17 GB | the default: the Qwen3 stack in 8 bits, the audio embeddings, heads and tokenizer in f16 |

The audio embeddings and heads stay f16: the guidance step multiplies their error, and 8-bit ones lost a third
of the first pass's choices against the release.

## Speed

### CUDA · NVIDIA A40

Stand: one A40 (46 GB), `cortiq tts` as above, 32 steps.

| text | speech | generation | decoding | total |
|---|---:|---:|---:|---:|
| English, 13 words | 4.5 s | 2.6 s | 1.8 s | 4.4 s |
| Russian, voice design | 3.3 s | 2.0 s | 1.3 s | 3.3 s |
| Chinese | 2.8 s | 1.8 s | 1.0 s | 2.8 s |
| a voice clone (a 4 s reference) | 3.2 s | 2.4 s | 1.5 s | 3.9 s, the reference encoded in 3.1 s more |
| English, 90 words (3 pieces) | 33.8 s | 17.9 s | 14.2 s | 32.1 s |

### Vulkan · NVIDIA A40

| text | speech | generation | decoding | total |
|---|---:|---:|---:|---:|
| English, 13 words | 4.5 s | 8.6 s | 2.0 s | 10.6 s |
| Russian, voice design | 3.3 s | 8.0 s | 1.3 s | 9.3 s |

### Metal · MacBook Air M4 (24 GB)

Stand: the fanless Air with the base M4's memory bus; an M4 Pro or Max was not measured.

| text | speech | generation | decoding | total |
|---|---:|---:|---:|---:|
| English, 13 words | 4.5 s | 13.1 s | 1.2 s | 14.3 s |
| Russian, voice design | 3.3 s | 9.5 s | 0.7 s | 10.2 s |
| Chinese | 2.8 s | 8.1 s | 0.6 s | 8.7 s |

## Verify the download

```bash
sha256sum omnivoice-q8_2f.cmf
# 696792de99feefe3757ec7626ca5b6bc9c7179f146d5fca39c16ec5791726ddb  (1 173 250 290 bytes)
```

## License

The OmniVoice weights are the release's under CC-BY-NC-4.0 (non-commercial, from its training data); see
[k2-fsa/OmniVoice](https://huggingface.co/k2-fsa/OmniVoice). The audio tokenizer is Boson AI's Higgs Audio 2
tokenizer under the Boson Higgs Audio 2 Community License ([LICENSE-HIGGS-AUDIO-2](LICENSE-HIGGS-AUDIO-2), notices in
[NOTICE](NOTICE)): Built with Higgs Materials licensed from Boson AI USA, Inc., Copyright Boson AI USA, Inc., All Rights
Reserved and Meta Llama 3 licensed under the Meta Llama 3 Community License, Copyright Meta Platforms, Inc., All Rights
Reserved. Do not clone a voice without its owner's consent. The CMF format and cortiq are Apache-2.0.

## Документация на русском

[OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) от k2-fsa говорит на 600+ языках, клонирует голос по нескольким
секундам записи и собирает голос по признакам. Здесь модель и её аудиотокенизатор Higgs в одном файле
[CMF](https://github.com/infosave2007/cmf) для `cortiq` — движка на Rust без Python.

- NVIDIA A40, CUDA: 4,5 с речи за 4,4 с, 34 с длинного текста за 32 с (32 шага) — реальное время.
- Тот же файл работает на Vulkan (карты любого производителя), на Apple silicon (Metal) и на CPU.
- Проверено против float32-релиза: тот же промпт и та же длина, Whisper распознаёт наш звук так же, как звук
  релиза — английский, русский, китайский, голос по признакам и клон.

```bash
cargo install cortiq-cli --features cuda   # NVIDIA; без фичи — Vulkan / CPU
hf download infosave/OmniVoice-cmf omnivoice-q8_2f.cmf --local-dir .
CMF_GPU=cuda cortiq tts omnivoice-q8_2f.cmf "Привет! Этот голос из одного файла." --lang ru --out hello.wav
# голос по признакам
cortiq tts omnivoice-q8_2f.cmf "Добрый вечер." --lang ru --instruct "female, low pitch" --out design.wav
# клон: 3–10 с одного говорящего и расшифровка записи
cortiq tts omnivoice-q8_2f.cmf "Эта фраза звучит клонированным голосом." --lang ru \
  --ref-audio speaker.wav --ref-text "Что говорит человек в записи." --out clone.wav
```

Нужен cortiq 0.8.99 или новее. Опции: `--lang` (код ISO 639 или английское название), `--instruct` (пол, возраст,
высота, `whisper`, акцент или китайский диалект), `--ref-audio` + `--ref-text`, `--steps` (32), `--duration`,
`--speed`, `--temperature`, `--guidance`, `--seed`. Теги в тексте: `[laughter]`, `[sigh]` и другие. Текст длиннее 30 с
режется по пунктуации на куски по ~15 с.

| файл | размер | что это |
|---|---:|---|
| `omnivoice-q8_2f.cmf` | 1,17 ГБ | по умолчанию: стек Qwen3 в 8 битах, аудиоэмбеддинги, головы и токенизатор в f16 |

Скорость, CUDA, NVIDIA A40: английская фраза 4,5 с речи — генерация 2,6 с + декодирование 1,8 с; русская с голосом
по признакам 3,3 с — 2,0 + 1,3 с; 90 слов (3 куска) 33,8 с — 17,9 + 14,2 с. Vulkan на той же карте: 4,5 с речи — 8,6 + 2,0 с.
Metal, MacBook Air M4 24 ГБ (без вентилятора, базовая шина памяти): 4,5 с речи — 13,1 + 1,2 с.

Проверка: `sha256sum omnivoice-q8_2f.cmf` → `696792de99feefe3757ec7626ca5b6bc9c7179f146d5fca39c16ec5791726ddb`.

Лицензия весов — CC-BY-NC-4.0 релиза (некоммерческая); аудиотокенизатор — Boson Higgs Audio 2 Community License
([LICENSE-HIGGS-AUDIO-2](LICENSE-HIGGS-AUDIO-2), [NOTICE](NOTICE)). Не клонируйте голос без согласия его владельца.

## 中文说明

[OmniVoice](https://huggingface.co/k2-fsa/OmniVoice)（k2-fsa）支持 600 多种语言，可用几秒录音克隆音色，也可按属性设计音色。本仓库把模型和
Higgs 音频分词器打包为一个 [CMF](https://github.com/infosave2007/cmf) 文件，由 Rust 推理引擎 `cortiq` 运行，无需 Python。

- NVIDIA A40（CUDA）：4.5 秒语音用时 4.4 秒，长文本 34 秒语音用时 32 秒（32 步），达到实时。
- 同一文件也可在 Vulkan（各厂商显卡）、Apple 芯片（Metal）和 CPU 上运行。
- 与 float32 原版对照：提示与输出长度一致，Whisper 对英语、俄语、中文以及设计音色和克隆音色的识别结果与原版相同。

```bash
cargo install cortiq-cli --features cuda   # NVIDIA；不加该特性则为 Vulkan / CPU
hf download infosave/OmniVoice-cmf omnivoice-q8_2f.cmf --local-dir .
CMF_GPU=cuda cortiq tts omnivoice-q8_2f.cmf "你好，这是一个语音合成测试。" --lang zh --out hello.wav
# 音色设计（中文方言）
cortiq tts omnivoice-q8_2f.cmf "今天天气真好。" --lang zh --instruct "男，四川话" --out design.wav
# 音色克隆：3–10 秒单人录音及其文本
cortiq tts omnivoice-q8_2f.cmf "这句话用克隆的声音说出。" --lang zh \
  --ref-audio speaker.wav --ref-text "录音中说的话。" --out clone.wav
```

需要 cortiq 0.8.99 或更高版本。选项：`--lang`（ISO 639 代码或英文名称）、`--instruct`、`--ref-audio` 与 `--ref-text`、`--steps`
（32）、`--duration`、`--speed`、`--temperature`、`--guidance`、`--seed`。文本中可用 `[laughter]`、`[sigh]` 等标签。超过 30 秒的文本按标点切成约
15 秒的片段。

| 文件 | 大小 | 说明 |
|---|---:|---|
| `omnivoice-q8_2f.cmf` | 1.17 GB | 默认：Qwen3 主干为 8 位，音频嵌入、输出头和分词器为 f16 |

速度（CUDA，NVIDIA A40）：英文 4.5 秒语音生成 2.6 秒、解码 1.8 秒；中文 2.8 秒语音生成 1.8 秒、解码 1.0 秒；90 个英文单词（3 段）33.8
秒语音生成 17.9 秒、解码 14.2 秒。同卡 Vulkan：4.5 秒语音生成 8.6 秒、解码 2.0 秒。Metal（MacBook Air M4 24 GB，无风扇、基础内存带宽）：4.5
秒语音生成 13.1 秒、解码 1.2 秒。

校验：`sha256sum omnivoice-q8_2f.cmf` → `696792de99feefe3757ec7626ca5b6bc9c7179f146d5fca39c16ec5791726ddb`。

权重沿用原版的 CC-BY-NC-4.0 许可（非商业）；音频分词器采用 Boson Higgs Audio 2 Community License（见 [LICENSE-HIGGS-AUDIO-2](LICENSE-HIGGS-AUDIO-2) 与
[NOTICE](NOTICE)）。未经本人同意请勿克隆其声音。
