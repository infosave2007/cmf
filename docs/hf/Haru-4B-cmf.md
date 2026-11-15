---
library_name: cortiq
license: apache-2.0
base_model:
- higurashi-ai/Haru-4B
base_model_relation: quantized
pipeline_tag: audio-to-audio
language:
- en
- es
- pt
- de
- fr
- zh
- ja
- ru
- ko
tags:
- cmf
- cortiq
- speech
- full-duplex
- speech-to-speech
- haru
- cuda
- vulkan
- metal
---

# Haru-4B — full-duplex speech in one file, no Python

[Haru-4B](https://huggingface.co/higurashi-ai/Haru-4B) by higurashi-ai listens and speaks at the same time,
like a person on a phone call: every 133 ms it takes one frame of the user's audio and produces one frame of
its own audio and text. This repository packages it as one [CMF](https://github.com/infosave2007/cmf) file
for `cortiq`, a Rust engine with no Python and no ML framework — the Qwen3.5-4B language model, the audio
projectors, the MeanFlow head and the VibeVoice codec together.

- **RTX A5000, CUDA: 23 ms per 133 ms frame** — 5.8× faster than real time. Vulkan 53 ms, Apple M4 (Metal) 84 ms.
- The codec matches the PyTorch reference within 1.3·10⁻³.

**Listen:** [a conversation](https://huggingface.co/infosave/Haru-4B-cmf/resolve/main/samples/conversation_stereo.mp3)
— the user on the left channel, Haru on the right. The user asks *"Hi! How are you doing today? Could you tell me
something interesting about the ocean?"*; Haru, speaking with the
[reference voice](https://huggingface.co/infosave/Haru-4B-cmf/resolve/main/samples/voice_reference.mp3), answers
*"Hey, welcome to our daily talk show… Today I'm gonna be talking about science, but not just any science, the
science of the ocean."* Both input voices were made with cortiq's Qwen3-TTS.

## Quick start

```bash
cargo install cortiq-cli --features cuda   # NVIDIA: the CUDA backend; without the feature: Vulkan, or Metal on a Mac
hf download infosave/Haru-4B-cmf haru-4b-q8_2f.cmf --local-dir .
CMF_GPU=cuda cortiq duplex haru-4b-q8_2f.cmf user.wav reply.wav --reference voice.wav --stereo --transcript reply.jsonl
```

`user.wav` is played to the model frame by frame, exactly as in a live call, and `reply.wav` is what the model
says over the same span (`--stereo`: the user left, the model right). Leave silence after a question so the
model has time to answer. `--reference` sets the voice: 3 s to 3 min of one speaker, clean speech — use only
voices you have the right to use. Other options: `--system-prompt` (a character card, as the Haru repository
describes it), `--speak-first`, `--temperature` (0.7), `--top-k` (25), `--guidance-scale` (1.5), `--seed`.

## File

| file | size | |
|---|---:|---|
| `haru-4b-q8_2f.cmf` | 6.37 GB | the language model in 8-bit (q8_2f), the head and the codec in f16 |

Four-bit weights break this model: with q4tp, even with only the feed-forward layers at four bits, the text
stream turns to noise. There is no smaller file.

## Speed

Per 133 ms frame, after the prompt and the voice reference, which run once (4–13 s):

- **RTX A5000 (24 GB), CUDA:** 23 ms.
- **RTX A5000, Vulkan:** 53 ms.
- **Apple M4 (24 GB), Metal:** 84 ms.

## Notes

Haru-4B is a pretrained base model released for research: it holds a natural conversation, but it is not a
source of facts and does not follow instructions like an assistant. It can imitate the voice of a reference
clip; only use voices you have the right to use. The live, microphone-in-the-loop demo is in the
[Haru repository](https://github.com/higurashi-ai/Haru); `cortiq duplex` runs the same loop over a recording.

## License

Apache-2.0, as the original. The weights are [higurashi-ai/Haru-4B](https://huggingface.co/higurashi-ai/Haru-4B),
converted to CMF with the language model re-quantized to 8 bits. This repository is not made or endorsed by
higurashi-ai.

---

## На русском

[Haru-4B](https://huggingface.co/higurashi-ai/Haru-4B) от higurashi-ai слушает и говорит одновременно, как
человек в телефонном разговоре: каждые 133 мс берёт кадр звука пользователя и выдаёт кадр своего звука и
текста. Здесь модель упакована в один файл [CMF](https://github.com/infosave2007/cmf) для `cortiq` — движка на
Rust без Python: языковая модель Qwen3.5-4B, проекторы звука, голова MeanFlow и кодек VibeVoice вместе.

- **RTX A5000, CUDA: 23 мс на кадр 133 мс** — в 5,8 раза быстрее реального времени. Vulkan — 53 мс, Apple M4
  (Metal) — 84 мс.
- Кодек совпадает с эталоном PyTorch с точностью 1,3·10⁻³.

```bash
cargo install cortiq-cli --features cuda
hf download infosave/Haru-4B-cmf haru-4b-q8_2f.cmf --local-dir .
CMF_GPU=cuda cortiq duplex haru-4b-q8_2f.cmf user.wav reply.wav --reference voice.wav --stereo
```

`user.wav` проигрывается модели кадр за кадром, как в живом звонке; `reply.wav` — то, что она говорит за то же
время. После вопроса оставьте тишину, чтобы модель успела ответить. `--reference` задаёт голос (3 с – 3 мин
одного говорящего); используйте только голоса, на которые у вас есть права. Файл один — `haru-4b-q8_2f.cmf`
(6,37 ГБ): 4-битные веса эту модель ломают. Это базовая исследовательская модель: она ведёт разговор, но не
источник фактов. Лицензия Apache-2.0, как у оригинала; репозиторий не сделан и не одобрен higurashi-ai.

## 中文说明

higurashi-ai 的 [Haru-4B](https://huggingface.co/higurashi-ai/Haru-4B) 能像打电话一样同时听和说：每 133 毫秒接收一帧
用户音频，并输出一帧自己的音频和文本。本仓库把它打包为一个 [CMF](https://github.com/infosave2007/cmf) 文件，由
`cortiq`（不依赖 Python 的 Rust 引擎）运行：Qwen3.5-4B 语言模型、音频投影器、MeanFlow 头和 VibeVoice 编解码器合为一体。

- **RTX A5000，CUDA：每 133 毫秒帧 23 毫秒**——比实时快 5.8 倍。Vulkan 53 毫秒，Apple M4（Metal）84 毫秒。
- 编解码器与 PyTorch 参考实现误差在 1.3·10⁻³ 以内。

```bash
cargo install cortiq-cli --features cuda
hf download infosave/Haru-4B-cmf haru-4b-q8_2f.cmf --local-dir .
CMF_GPU=cuda cortiq duplex haru-4b-q8_2f.cmf user.wav reply.wav --reference voice.wav --stereo
```

`user.wav` 会像实时通话一样逐帧播放给模型，`reply.wav` 是模型在同一时间段内说的话。提问后请留出静音，让模型有时间
回答。`--reference` 设定声音（同一说话人 3 秒至 3 分钟）；只使用你有权使用的声音。只有一个文件 `haru-4b-q8_2f.cmf`
（6.37 GB）：4 位权重会破坏该模型。这是用于研究的预训练基础模型，能自然对话，但不是事实来源。许可证为 Apache-2.0，
与原模型相同；本仓库并非由 higurashi-ai 制作或认可。
