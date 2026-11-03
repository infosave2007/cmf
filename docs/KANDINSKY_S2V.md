# Kandinsky S2V on cortiq

*Languages: **English** · [Русский](KANDINSKY_S2V.ru.md) · [中文](KANDINSKY_S2V.zh.md)*

[Kandinsky S2V 1.0 Pro-distill](https://huggingface.co/kandinskylab/Kandinsky-S2V-1.0-Pro-distill-Diffusers)
turns one reference frame and a speech track into an avatar video. The whole figure
moves, not only the head. The audio is a condition: the model does not generate
sound, and `cortiq` writes your track into the output file.

## Run

```bash
cargo install cortiq-cli
hf download infosave/Kandinsky-S2V-1.0-Pro-distill-cmf kandinsky-s2v-pro-q4tp.cmf --local-dir .
cortiq s2v --model kandinsky-s2v-pro-q4tp.cmf --image reference.png --audio speech.wav --out avatar.avi
```

The output is MJPEG with PCM audio in an AVI, 25 fps, and needs nothing installed.
To get an mp4:

```bash
ffmpeg -i avatar.avi -c:v libx264 -crf 18 -c:a aac avatar.mp4
```

| option | default | meaning |
|---|---|---|
| `--image` | required | reference frame (PNG, JPEG) |
| `--audio` | required | speech track: WAV here, other formats through `ffmpeg` |
| `--out` | `avatar.avi` | output file |
| `--prompt` | the reference prompt | describes the motion; a custom prompt loads the 7B text encoder |
| `--seed` | 5 | noise seed |
| `--steps` | 16 | pi-Flow segments per chunk |
| `--chunks` | as many as cover the track | chunks of 5 s; the video is cut to the track |
| `--height`, `--width` | the 512 bucket closest to the image | 512×512, 768×512 or 512×768 |
| `--quality` | 92 | JPEG quality of the frames |

## Pipeline

```
speech → mel → audio tower → 25 tokens/s ─┐
prompt → Thinker + CLIP (stored for the   ├─ DiT, 16 segments a chunk ─→ VAE decoder ─→ frames
         default prompt)                  │     ↑ FramePack memory of the chunk before
reference frame → VAE encoder ────────────┘
```

- **DiT.** 60 visual blocks of 4096 dimensions, 4 prompt blocks and 4 speech blocks,
  21B parameters. A chunk of the 512² bucket is 36736 tokens: FramePack memory at
  three temporal scales, the reference frame, and 32 latent frames.
- **pi-Flow.** One forward predicts 5 x0 values across a segment of raw time and the
  ODE is integrated over them with no further network evaluations. Guidance is
  distilled into the weights.
- **VAE.** HunyuanVideo's causal 3-D VAE, tiled in time as the class does by default:
  5 latent frames every 3, 4 pixel frames blended across each seam.

## How it runs

The weights do not fit a 32 GB card as f16 planes (42 GB). The container's quantized
bytes stay on the card as they are and one block's planes are expanded right before
that block runs, so q4tp needs about 14 GB of VRAM for the DiT. On Metal the weights
stay in the file mapping and are dequantized as a GEMM reads them. The VAE runs
resident on both. `cortiq s2v` falls back to the CPU when the device declines, which
at 36736 tokens is hours per segment.

## Parity

`scripts/s2v/oracle.py` dumps the reference stage by stage (the whole pipeline does
not fit one process on a 32 GB card) and `cortiq s2v-check` compares each stage:

```bash
python scripts/s2v/oracle.py enc --image i.png --audio a.wav --chunks 1 --out enc.safetensors
cortiq s2v-check --model kandinsky-s2v-pro-q8.cmf --stage enc --oracle enc.safetensors
```

The run log with every measurement is in `scripts/s2v/JOURNAL_S2V_2026-11-03.md`.

## Responsible use

The model animates a real person from a single photograph, driven by a voice that
need not be theirs. Use a reference frame you have the subject's consent to animate,
and do not present the result as a genuine recording of anyone.
