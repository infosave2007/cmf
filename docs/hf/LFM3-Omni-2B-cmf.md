---
license: other
license_name: lfm1.0
license_link: LICENSE
library_name: cortiq
base_model: LiquidAI/LFM3-Omni-2B
base_model_relation: quantized
pipeline_tag: any-to-any
tags:
  - cmf
  - cortiq
  - quantized
  - q4tp
  - liquid
  - lfm3
  - omni
  - vision
  - speech
  - asr
  - tts
  - vulkan
  - metal
language:
  - en
  - ar
  - zh
  - fr
  - de
  - it
  - ja
  - ko
  - pt
  - es
  - vi
  - th
  - id
  - hi
  - ru
  - pl
---

# LFM3-Omni-2B — CMF

[Liquid AI LFM3-Omni-2B](https://huggingface.co/LiquidAI/LFM3-Omni-2B) — a 2.2B
omni-modal model on LFM3: it reads text, images, video and 16 kHz speech and
answers in text, in 24 kHz speech, or in both interleaved — one model in place
of a speech recognizer, a vision-language model, an LLM and a TTS — packaged as
single [CMF](https://github.com/infosave2007/cmf) files with the speech codec
inside. They run with `cortiq`, a Rust inference engine with no Python and no
ML framework: NVIDIA, AMD and Intel GPUs through Vulkan/DX12, Apple silicon
through Metal, and the CPU.

[English](#quick-start) · [Русский](#документация-на-русском) · [中文](#中文说明)

## Quick start

```bash
cargo install cortiq-cli
hf download infosave/LFM3-Omni-2B-cmf lfm3-omni-2b-q8_2f.cmf --local-dir .

cortiq run lfm3-omni-2b-q8_2f.cmf --image photo.jpg --prompt "Describe this image."
cortiq run lfm3-omni-2b-q8_2f.cmf --system "Perform ASR in English." --audio clip.wav --prompt ""
cortiq run lfm3-omni-2b-q8_2f.cmf --system "Perform TTS in American English using a female voice." \
  --prompt "The quick brown fox jumps over the lazy dog." --speech-out tts.wav
cortiq run lfm3-omni-2b-q8_2f.cmf --image photo.jpg --audio question.wav --prompt "" \
  --system "Respond with interleaved text and audio." --speech-out reply.wav --speech-mode mixed
```

Use cortiq 0.8.50 or later; prebuilt binaries for Linux, macOS and Windows are
on the [releases page](https://github.com/infosave2007/cmf/releases). `--image`,
`--video` (a Y4M file or a frame directory) and `--audio` (WAV, other formats
through `ffmpeg`) may be repeated and come before the text. LFM3-Omni does not
reason before it answers.

## Files

| file | language model | vision, audio encoder, depth decoder, codec | size | recommended for |
|---|---|---|---:|---|
| `lfm3-omni-2b-q8_2f.cmf` | 8-bit | 16-bit | 2,780,603,290 B (2.78 GB) | **default** |
| `lfm3-omni-2b-q4tp.cmf` | 4-bit | 16-bit | 1,950,888,858 B (1.95 GB) | the smallest memory footprint |

Against the bf16 checkpoint run in float32 by Transformers (greedy):

| check | f16 conversion | `q8_2f` | `q4tp` |
|---|---|---|---|
| one-sentence description, 640×480 photo | identical | identical | same content, other words |
| description, 1296×864 image | — | identical for its first 28 words | same content, other words |
| transcripts of two English clips | identical | identical | identical |
| TTS frames, fixed codebook order (46 frames) | identical (every code) | other greedy codes after frame 1 | other greedy codes after frame 1 |
| TTS speech transcribed back | the input sentence | the input sentence | the input sentence |

The vision encoder, the audio encoder, the projectors, the depth decoder and the
codec are stored 16-bit in both files and run in float32.

## Performance

PERF

## Usage

Speech recognition (`Perform ASR in <language>.`), speech synthesis (`Perform
TTS in <language> using a female|male voice.`, 19 locales), questions about
pictures, video frames and recordings, and voice chat (`Respond with
interleaved text and audio.` with `--speech-mode mixed`). Speech is sampled as
the release recommends (temperature 0.4, repetition penalty 1.2 over 64
frames); `--greedy` takes the likeliest codes.

### OpenAI-compatible server

```bash
cortiq serve lfm3-omni-2b-q8_2f.cmf --port 8080
```

`/v1/chat/completions` takes `image_url`, `video` and `input_audio` parts in one
message and answers in text. Speech output (`--speech-out`), a video's
soundtrack and live (streaming) recognition are not supported yet.

## Hardware

**NVIDIA / AMD / Intel (Vulkan, DX12).** Both files fit any card with 3.5 GB of
free memory; on headless Linux install `libvulkan1 libglvnd0 libegl1 libgl1
libglx0` and set `XDG_RUNTIME_DIR=/tmp`. **Apple silicon:** the Metal backend
is the default. **CPU only:** `CMF_GPU=0`.

## Verify the download

```bash
sha256sum -c lfm3-omni-2b-q8_2f.cmf.sha256
cortiq verify lfm3-omni-2b-q8_2f.cmf
```

SHA_TABLE

## Reproduce

```bash
cortiq convert --model LiquidAI/LFM3-Omni-2B --quant q8_2f \
  --output lfm3-omni-2b-q8_2f.cmf --also-output q4tp=lfm3-omni-2b-q4tp.cmf
```

The converter reads the codec from the checkpoint's `audio_tokenizer/` folder
into the same file. Scripts (Transformers reference, image, speech and TTS
parity): [scripts/lfm3](https://github.com/infosave2007/cmf/tree/master/scripts/lfm3).

## License

These files are a modified form of Liquid AI's LFM3-Omni-2B: the weights were
converted to CMF and the language model quantized (8-bit `q8_2f`, 4-bit `q4tp`)
by cortiq. They remain under the LFM Open License v1.0, a copy of which is in
this repository ([LICENSE](LICENSE)); among its terms, commercial use by an
entity with annual revenue of US$10 million or more is not licensed. See the
[upstream card](https://huggingface.co/LiquidAI/LFM3-Omni-2B).

---

## Документация на русском

[LFM3-Omni-2B](https://huggingface.co/LiquidAI/LFM3-Omni-2B) от Liquid AI —
омни-модальная модель на 2.2 млрд параметров поверх LFM3: читает текст,
изображения, видео и речь 16 кГц и отвечает текстом, речью 24 кГц или тем и
другим вперемешку. Файлы [CMF](https://github.com/infosave2007/cmf) включают
кодек речи и запускаются движком `cortiq` на Rust без Python.

```bash
cargo install cortiq-cli
hf download infosave/LFM3-Omni-2B-cmf lfm3-omni-2b-q8_2f.cmf --local-dir .
cortiq run lfm3-omni-2b-q8_2f.cmf --image photo.jpg --prompt "Опиши изображение."
cortiq run lfm3-omni-2b-q8_2f.cmf --system "Perform ASR in Russian." --audio clip.wav --prompt ""
cortiq run lfm3-omni-2b-q8_2f.cmf --system "Perform TTS in American English using a female voice." \
  --prompt "Hello! How are you?" --speech-out tts.wav
```

| файл | языковая модель | энкодеры, depth-декодер, кодек | размер | назначение |
|---|---|---|---:|---|
| `lfm3-omni-2b-q8_2f.cmf` | 8 бит | 16 бит | 2.78 ГБ | **по умолчанию** |
| `lfm3-omni-2b-q4tp.cmf` | 4 бита | 16 бит | 1.95 ГБ | минимум памяти |

Нужен cortiq 0.8.50 или новее. Против Transformers float32: описание фото и
распознавание речи совпадают дословно (`q8_2f`, f16), синтез в f16 воспроизводит
все кадры речи, речь квантованных файлов распознаётся обратно в исходную фразу.
Изображения, видео (Y4M или каталог кадров) и аудио можно давать в одном запросе;
`--speech-mode mixed` — голосовой чат. `cortiq serve` принимает изображения,
видео и аудио в формате OpenAI и отвечает текстом. Звуковая дорожка видео и
потоковое распознавание пока не поддерживаются. Файлы — изменённая форма
LFM3-Omni-2B от Liquid AI по LFM Open License v1.0 (копия — [LICENSE](LICENSE));
коммерческое использование организацией с годовой выручкой от 10 млн долларов
США лицензией не покрывается.

## 中文说明

[LFM3-Omni-2B](https://huggingface.co/LiquidAI/LFM3-Omni-2B) 是 Liquid AI 基于 LFM3
的 2.2B 全模态模型：可读取文本、图像、视频和 16 kHz 语音，并以文本、24 kHz 语音或
二者交错的方式回答。本仓库的 CMF 文件内含语音编解码器，可由 Rust 推理引擎 `cortiq`
（0.8.50 及以上）加载。

```bash
cargo install cortiq-cli
hf download infosave/LFM3-Omni-2B-cmf lfm3-omni-2b-q8_2f.cmf --local-dir .
cortiq run lfm3-omni-2b-q8_2f.cmf --image photo.jpg --prompt "描述这张图片。"
cortiq run lfm3-omni-2b-q8_2f.cmf --system "Perform ASR in Chinese." --audio clip.wav --prompt ""
cortiq run lfm3-omni-2b-q8_2f.cmf --system "Perform TTS in Chinese using a female voice." \
  --prompt "你好，今天天气很好。" --speech-out tts.wav
```

| 文件 | 语言模型 | 编码器、depth 解码器、编解码器 | 大小 | 适用 |
|---|---|---|---:|---|
| `lfm3-omni-2b-q8_2f.cmf` | 8 位 | 16 位 | 2.78 GB | **默认** |
| `lfm3-omni-2b-q4tp.cmf` | 4 位 | 16 位 | 1.95 GB | 内存最小 |

与 Transformers float32 相比：图片描述与语音识别逐字一致（`q8_2f`、f16），f16 的语音帧
完全一致，量化文件合成的语音可被逐字识别回原句。图像、视频（Y4M 或帧目录）与音频可在
同一请求中给出；`--speech-mode mixed` 为语音对话。暂不支持视频音轨与流式识别。这些文件是
Liquid AI LFM3-Omni-2B 的修改形式，仍受 LFM Open License v1.0 约束（副本见
[LICENSE](LICENSE)）；年收入 1000 万美元及以上的实体进行商业使用不在许可范围内。
