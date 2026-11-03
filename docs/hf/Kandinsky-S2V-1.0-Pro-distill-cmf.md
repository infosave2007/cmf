---
license: mit
library_name: cortiq
base_model: kandinskylab/Kandinsky-S2V-1.0-Pro-distill-Diffusers
base_model_relation: quantized
pipeline_tag: image-to-video
tags:
  - cmf
  - cortiq
  - quantized
  - q4tp
  - q8_2f
  - speech-to-video
  - avatar
  - video-generation
  - kandinsky
  - vulkan
  - metal
language:
  - en
  - ru
  - zh
---

# Kandinsky S2V 1.0 Pro-distill — CMF

[Kandinsky S2V 1.0 Pro-distill](https://huggingface.co/kandinskylab/Kandinsky-S2V-1.0-Pro-distill-Diffusers)
turns one reference frame and a speech track into an avatar video: the whole figure
moves, not only the head. The DiT, the speech and text encoders, CLIP and the VAE are
packaged as one [CMF](https://github.com/infosave2007/cmf) file for `cortiq`, a Rust
inference engine with no Python and no ML framework.

![Frames from a clip made with q4tp](samples/sample.png)

[English](#quick-start) · [Русский](#документация-на-русском) · [中文](#中文说明)

## Quick start

```bash
cargo install cortiq-cli
hf download infosave/Kandinsky-S2V-1.0-Pro-distill-cmf kandinsky-s2v-pro-q4tp.cmf --local-dir .
cortiq s2v --model kandinsky-s2v-pro-q4tp.cmf --image reference.png --audio speech.wav --out avatar.avi
```

Use cortiq 0.8.55 or later; prebuilt binaries for Linux, macOS and Windows are on the
[releases page](https://github.com/infosave2007/cmf/releases). The output is MJPEG with
your speech track in an AVI at 25 fps. To get an mp4:

```bash
ffmpeg -i avatar.avi -c:v libx264 -crf 18 -c:a aac avatar.mp4
```

## Files

| file | DiT | size | peak VRAM at 512² | final latents against the original |
|---|---|---:|---:|---|
| `kandinsky-s2v-pro-q4tp.cmf` | 4-bit | 24.0 GB | 16.2 GB | cosine 0.9996; **default** |
| `kandinsky-s2v-pro-q8.cmf` | 8-bit | 32.3 GB | 23.7 GB | cosine 0.9999 |

The original is the vendor's bf16 pipeline on the same noise. Decoded frames of the
first chunk from q4tp match it at 40.9 dB PSNR. The speech and prompt blocks of the DiT,
the Thinker and the audio tower are 8-bit in both files, and the VAE is 16-bit. The
default prompt's text conditions are stored in the file, so the 7B text encoder is read
only for a custom `--prompt`.

## Performance

Measured on one 5 s chunk of 125 frames (16 steps), L40S with Vulkan:

| size | tokens | a step | VAE per chunk | 5 s of video | peak VRAM (q4tp) |
|---|---:|---:|---:|---:|---:|
| 512×512 | 36736 | 19.7 s | 8.3 s | 5.6 min | 16.2 GB |
| 768×512, 512×768 | 55104 | 36.9 s | 12.6 s | 10 min | 20.2 GB |
| 256×256, draft | 9184 | 3.3 s | 2.9 s | 64 s | — |

The Diffusers original with one block on the same card at a time takes 19.5 s a step. An
RTX 5090 ran q8 at 16.8 s a step with 6.8 s of VAE per chunk.
256×256 is outside the model's trained sizes: it keeps the speech and the motion and
loses detail in the face and the hands.

## Options

| option | default | meaning |
|---|---|---|
| `--image` | required | reference frame (PNG, JPEG) |
| `--audio` | required | speech track: WAV directly, other formats through `ffmpeg` |
| `--out` | `avatar.avi` | output file |
| `--prompt` | the reference prompt | describes the motion; loads the 7B text encoder |
| `--seed` | 5 | noise seed |
| `--steps` | 16 | pi-Flow segments per chunk |
| `--chunks` | as many as cover the track | chunks of 5 s; the video is cut to the track |
| `--height`, `--width` | the 512 bucket closest to the image | 512×512, 768×512, 512×768 |
| `--quality` | 92 | JPEG quality of the frames |

Long speech runs chunk by chunk, each conditioned on the one before it. The speech encoder
stops at 300 s.

## Hardware

**NVIDIA with 24 GB or more (Vulkan).** q4tp at 512² peaks at 16.2 GB and needs the card's
cooperative matrix support. **Apple silicon (Metal).** Everything runs on the GPU from
the file mapping. **CPU only.** `CMF_GPU=0` works, at hours per step.

## Responsible use

The model animates a real person from a single photograph, driven by a voice that need not
be theirs. Use a reference frame you have the subject's consent to animate, and do not
present the result as a genuine recording of anyone.

## License

MIT, as the [original](https://huggingface.co/kandinskylab/Kandinsky-S2V-1.0-Pro-distill-Diffusers).
The Qwen2.5-Omni Thinker and the HunyuanVideo VAE inside the file come under their own
licenses; see their upstream pages. Sample frames: Leonardo da Vinci's *Mona Lisa*, public
domain.

---

## Документация на русском

[Kandinsky S2V 1.0 Pro-distill](https://huggingface.co/kandinskylab/Kandinsky-S2V-1.0-Pro-distill-Diffusers)
превращает один опорный кадр и дорожку речи в видео с аватаром: движется вся фигура, а не только голова.
DiT, энкодеры речи и текста, CLIP и VAE лежат в одном файле [CMF](https://github.com/infosave2007/cmf)
для `cortiq`, движка на Rust без Python.

```bash
cargo install cortiq-cli
hf download infosave/Kandinsky-S2V-1.0-Pro-distill-cmf kandinsky-s2v-pro-q4tp.cmf --local-dir .
cortiq s2v --model kandinsky-s2v-pro-q4tp.cmf --image reference.png --audio speech.wav --out avatar.avi
```

Нужен cortiq 0.8.55 или новее. На выходе AVI (MJPEG и ваша дорожка, 25 кадров/с); mp4 получается через
`ffmpeg -i avatar.avi -c:v libx264 -crf 18 -c:a aac avatar.mp4`.

| файл | DiT | размер | пик видеопамяти на 512² |
|---|---|---:|---:|
| `kandinsky-s2v-pro-q4tp.cmf` | 4 бита, по умолчанию | 24.0 ГБ | 16.2 ГБ |
| `kandinsky-s2v-pro-q8.cmf` | 8 бит | 32.3 ГБ | 23.7 ГБ |

На L40S (Vulkan) шаг на 512×512 занимает 19.7 с, а 5 с видео — 5.6 мин, на 768×512 шаг занимает 36.9 с,
а 5 с видео — 10 мин. Оригинал на Diffusers на той же карте тратит 19.5 с на шаг. Черновик 256×256 готов за
64 с, но этот размер вне обученных: речь и движение сохраняются, детали лица и рук теряются.

Косинус конечных латентов с оригиналом на том же шуме 0.9996, PSNR кадров первого чанка 40.9 дБ. Параметры:
`--prompt`, `--seed`, `--steps`, `--chunks`, `--height`, `--width`, `--quality`. Подробнее в
[руководстве](https://github.com/infosave2007/cmf/blob/master/docs/KANDINSKY_S2V.ru.md).

**Ответственное использование.** Модель оживляет реального человека по одной фотографии голосом, который может
быть не его. Используйте опорный кадр только с согласия изображённого и не выдавайте результат за настоящую запись.

Лицензия MIT, как у оригинала; у Thinker Qwen2.5-Omni и VAE HunyuanVideo собственные лицензии, см. их страницы.

---

## 中文说明

[Kandinsky S2V 1.0 Pro-distill](https://huggingface.co/kandinskylab/Kandinsky-S2V-1.0-Pro-distill-Diffusers)
根据一张参考帧和一段语音生成数字人视频：动起来的是整个人物，而不只是头部。DiT、语音和文本编码器、CLIP 以及 VAE
都在一个 [CMF](https://github.com/infosave2007/cmf) 文件中，供 `cortiq`（无 Python 的 Rust 推理引擎）使用。

```bash
cargo install cortiq-cli
hf download infosave/Kandinsky-S2V-1.0-Pro-distill-cmf kandinsky-s2v-pro-q4tp.cmf --local-dir .
cortiq s2v --model kandinsky-s2v-pro-q4tp.cmf --image reference.png --audio speech.wav --out avatar.avi
```

需要 cortiq 0.8.55 及以上。输出为 AVI（MJPEG 加你的音轨，25 fps）；mp4 可用
`ffmpeg -i avatar.avi -c:v libx264 -crf 18 -c:a aac avatar.mp4` 转换。

| 文件 | DiT | 大小 | 512² 显存峰值 |
|---|---|---:|---:|
| `kandinsky-s2v-pro-q4tp.cmf` | 4 位，默认 | 24.0 GB | 16.2 GB |
| `kandinsky-s2v-pro-q8.cmf` | 8 位 | 32.3 GB | 23.7 GB |

在 L40S（Vulkan）上，512×512 每步 19.7 秒，5 秒视频需 5.6 分钟；768×512 每步 36.9 秒，5 秒视频需 10 分钟。
Diffusers 原版在同一张卡上每步 19.5 秒。256×256 草稿只需 64 秒，但这不是训练尺寸：语音和动作保留，脸和手的细节会损失。

在相同噪声下，最终潜变量与原模型的余弦相似度为 0.9996，第一个分块的帧 PSNR 为 40.9 dB。选项：`--prompt`、`--seed`、
`--steps`、`--chunks`、`--height`、`--width`、`--quality`。详见[使用指南](https://github.com/infosave2007/cmf/blob/master/docs/KANDINSKY_S2V.zh.md)。

**负责任地使用。** 该模型可以用一张照片和不一定属于本人的声音让真人动起来。请仅在获得本人同意时使用其参考帧，不要把结果当作真实录像。

许可证 MIT，与原模型相同；Qwen2.5-Omni Thinker 和 HunyuanVideo VAE 适用各自的许可证，请见其页面。
