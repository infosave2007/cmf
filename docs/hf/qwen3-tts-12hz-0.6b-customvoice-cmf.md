---
library_name: cortiq
license: apache-2.0
base_model:
- Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice
base_model_relation: quantized
pipeline_tag: text-to-speech
language:
- zh
- en
- de
- it
- pt
- es
- ja
- ko
- fr
- ru
tags:
- cmf
- cortiq
- text-to-speech
- qwen3-tts
- multilingual
- russian
- cuda
- vulkan
- metal
---

# Qwen3-TTS 12Hz 0.6B CustomVoice — speech in one file, no Python

[Qwen3-TTS-12Hz-0.6B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice) speaks ten
languages with nine built-in voices. This repository packages it as one [CMF](https://github.com/infosave2007/cmf)
file for `cortiq`, a Rust engine with no Python and no ML framework: the talker, its code predictor and the
12 Hz speech tokenizer's decoder together.

- **Apple M4, Metal: 1.7–1.9× faster than real time** (q4tp). RTX A5000, CUDA: 1.1–1.4×.
- Greedy decoding matches the reference `qwen_tts` frame for frame; whisper transcribes the speech back to the
  input text.

**Listen:** [Russian](https://huggingface.co/infosave/qwen3-tts-12hz-0.6b-customvoice-cmf/resolve/main/samples/q8_2f_cuda_ru.mp3) ·
[English](https://huggingface.co/infosave/qwen3-tts-12hz-0.6b-customvoice-cmf/resolve/main/samples/q4tp_cuda_en.mp3) — speaker `serena`.

## Quick start

```bash
cargo install cortiq-cli                    # add --features cuda for the CUDA backend on NVIDIA
hf download infosave/qwen3-tts-12hz-0.6b-customvoice-cmf qwen3-tts-12hz-0.6b-customvoice-q8_2f.cmf --local-dir .
cortiq tts qwen3-tts-12hz-0.6b-customvoice-q8_2f.cmf "Привет! Это пример синтеза речи." --lang ru --speaker serena --out speech.wav
```

`--lang`: `zh`, `en`, `de`, `it`, `pt`, `es`, `ja`, `ko`, `fr`, `ru` (or the full name), or `auto`. `--speaker`:
`serena`, `vivian`, `uncle_fu`, `ryan`, `aiden`, `ono_anna`, `sohee`, `eric`, `dylan`. The output is a 24 kHz
WAV. `--seed` picks another take; `--temperature 0` decodes greedily (default 0.9).

## Files

| file | size | |
|---|---:|---|
| `qwen3-tts-12hz-0.6b-customvoice-q8_2f.cmf` | 1.55 GB | talker and code predictor in 8-bit (q8_2f) |
| `qwen3-tts-12hz-0.6b-customvoice-q4tp.cmf` | 1.29 GB | talker and code predictor in 4-bit (q4tp) |

The text embedding, the code predictor's embeddings and heads and the speech decoder are f16 in both.
With sampling, q4tp garbles some Russian takes (3 of 8 seeds, on Metal and on the CPU alike); q8_2f read back
exactly in every take checked, so it is the file for Russian.

## Speed

**Apple M4 (24 GB), Metal**, q4tp: 8.56 s of English in 4.7 s (1.85× real time), 4.40 s of Russian in 2.5 s
(1.78×).

**RTX A5000 (24 GB), CUDA**:

| file | Russian sentence | English sentences |
|---|---:|---:|
| q4tp | 1.11× real time | 1.42× |
| q8_2f | 0.92× | 1.34× |

The repository also keeps the earlier `qwen3-tts-12hz-0.6b-customvoice.cmf`, a container of the original files for
the Python bridge in `run_qwen3_tts_cmf.py`.

## License

Apache-2.0, as the original. The weights are [Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice),
converted to CMF and quantized. This repository is not made or endorsed by Qwen.

---

## На русском

[Qwen3-TTS-12Hz-0.6B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice) говорит на десяти
языках девятью встроенными голосами. Здесь модель упакована в один файл [CMF](https://github.com/infosave2007/cmf)
для `cortiq` — движка на Rust без Python: talker, предсказатель кодов и декодер речевого токенизатора 12 Гц вместе.

- **Apple M4, Metal: в 1,7–1,9 раза быстрее реального времени** (q4tp). RTX A5000, CUDA: 1,1–1,4×.
- Жадное декодирование совпадает с эталонным `qwen_tts` кадр в кадр; whisper распознаёт в речи исходный текст.

```bash
cargo install cortiq-cli
hf download infosave/qwen3-tts-12hz-0.6b-customvoice-cmf qwen3-tts-12hz-0.6b-customvoice-q8_2f.cmf --local-dir .
cortiq tts qwen3-tts-12hz-0.6b-customvoice-q8_2f.cmf "Привет! Это пример синтеза речи." --lang ru --speaker serena --out speech.wav
```

Файлы: q8_2f (1,55 ГБ) и q4tp (1,29 ГБ); для русского берите q8_2f — q4tp при выборке портит часть дублей
(3 из 8 сидов на Metal и на CPU). Языки: `zh`, `en`, `de`, `it`, `pt`, `es`, `ja`, `ko`, `fr`, `ru` или
`auto`; голоса: `serena`, `vivian`, `uncle_fu`, `ryan`, `aiden`, `ono_anna`, `sohee`, `eric`, `dylan`. Лицензия
Apache-2.0, как у оригинала; репозиторий не сделан и не одобрен Qwen.

## 中文说明

[Qwen3-TTS-12Hz-0.6B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice) 支持十种语言和九个内置
音色。本仓库把它打包为一个 [CMF](https://github.com/infosave2007/cmf) 文件，由 `cortiq`（不依赖 Python 的 Rust 引擎）
运行：talker、码本预测器和 12 Hz 语音分词器解码器合为一体。

- **Apple M4，Metal：比实时快 1.7–1.9 倍**（q4tp）。RTX A5000，CUDA：1.1–1.4 倍。
- 贪心解码与参考实现 `qwen_tts` 逐帧一致；whisper 能从合成语音中识别出原文。

```bash
cargo install cortiq-cli
hf download infosave/qwen3-tts-12hz-0.6b-customvoice-cmf qwen3-tts-12hz-0.6b-customvoice-q8_2f.cmf --local-dir .
cortiq tts qwen3-tts-12hz-0.6b-customvoice-q8_2f.cmf "你好！这是语音合成的示例。" --lang zh --speaker vivian --out speech.wav
```

文件：q8_2f（1.55 GB）与 q4tp（1.29 GB）；俄语请用 q8_2f——采样时 q4tp 会弄坏部分俄语结果（Metal 与 CPU 上 8 个种子中 3 个）。语言：`zh`、`en`、`de`、`it`、`pt`、`es`、`ja`、`ko`、`fr`、`ru` 或 `auto`；
音色：`serena`、`vivian`、`uncle_fu`、`ryan`、`aiden`、`ono_anna`、`sohee`、`eric`、`dylan`。许可证为 Apache-2.0，与原模型
相同；本仓库并非由 Qwen 制作或认可。
