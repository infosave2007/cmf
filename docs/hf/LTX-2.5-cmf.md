---
library_name: cortiq
license: other
license_name: ltx-2-community-license-agreement
license_link: https://github.com/Lightricks/LTX-2/blob/main/LICENSE.md
base_model:
- Lightricks/LTX-2.5
base_model_relation: quantized
pipeline_tag: image-to-video
tags:
- cmf
- cortiq
- video
- audio
- ltx-video
- ltx-2.5
- 4-bit
---

# LTX-2.5 — the whole pipeline in one 21 GB file

[LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5) renders video **and its
soundtrack** from one prompt: a 21 B audio-video diffusion transformer that
denoises picture and sound in the same 48 blocks, a Gemma-4 12 B prompt
encoder, a 3-D video VAE, an audio VAE, two latent upscalers and a duration
head. The reference checkout is **71.35 GB across six safetensors** plus a
ComfyUI install.

Here it is **one memory-mapped [CMF](https://github.com/infosave2007/cmf) file
of 20.99 GB** — every component, the Gemma-4 tokenizer and every config
inside it — packed by `cortiq`, a Rust binary with no ML framework
underneath.

| | reference | this file |
|---|---|---|
| files | 6 safetensors + configs + tokenizer | **1** |
| bytes | 71.35 GB | **20.99 GB** (3.4× smaller) |
| weights | 35.65 B | 35.65 B — all of them |
| loader | diffusers / ComfyUI + PyTorch | `mmap` |

```
Arch:      ltx-2.5-av
Layers:    48 (48 full)      Hidden: 4096      Heads: 32 (KV: 32)
Vocab:     262144            Params: 35.65B    Tensors: 6693
Tokenizer: embedded
✓ envelope, sections, tensor directory (6693 tensors)
✓ all tensor hashes match
```

## What is inside

| component | weights | in the file | codec |
|---|---|---|---|
| `dit.*` — LTX-2.5 22B audio-video DiT (distilled) | 21.004 B | 10.28 GiB | q4tp |
| `te.*` — Gemma-4 12B prompt encoder + aggregate projections + vision tower | 13.116 B | 6.37 GiB | q4tp |
| `vvae.*` — video VAE (3-D conv, encoder + decoder) | 0.726 B | 1.35 GiB | f16 |
| `avae.*` — audio VAE | 0.182 B | 0.34 GiB | f16 |
| `ups.*` — latent spatial upscaler ×2 | 0.498 B | 0.93 GiB | f16 |
| `upt.*` — latent temporal upscaler ×2 | 0.131 B | 0.24 GiB | f16 |
| `dhead.*` — duration head | 1.9 M | 3.8 MB | f16 |
| `ltx.config_json`, `te.asset.*` | — | 55 KB | raw |
| tokenizer (`tokenizer_json`, 32 MB) | — | VOCAB section | raw |

The DiT is the **distilled** transformer — the short-schedule one. The `dev`
transformer is the same shape and packs the same way (`--dit …dev…`); the
LoRAs, the int8/nvfp4 builds and the IC-LoRA upscaler are not in this file.

## What the codec does per tensor

Four bits is not applied by fiat. `cortiq ltx-pack` decides per tensor:

* **2-D planes of at least 2²⁰ weights → q4tp**, 4.16 bits with a per-row
  scale ladder. That is every projection in the DiT and in the encoder —
  20.9 B of the 21.0 B, and 13.1 B of the 13.1 B.
* **The adaLN tables stay exact.** `scale_shift_table`, the per-block
  prompt/audio tables, the connector's learnable registers and the VAE's
  `per_channel_statistics` ship as F32 in the release, are read once a step
  and modulate everything downstream. 19 MB in total — quantizing them
  would move every block's normalization to save nothing.
* **Convolutions stay f16.** Both VAEs are 3-D/2-D convolutions; the decoder
  is what the eye sees. `--vae-quant q4tp` folds kernels to
  `[out, in·k·k·k]` and quantizes them anyway — offered, not the default.
* **Small 2-D planes, norms and biases**: exact when the source is F32,
  f16 otherwise. The attention gates (`to_gate`, `[32, 4096]`), the patchify
  projection and `proj_out` are tiny and read at full sequence length.

## The architecture it carries

`AVTransformer3DModel`, from the release's own config (kept verbatim in the
file as `ltx.config_json`):

* **48 blocks**, video stream 4096 (32 heads × 128), **audio stream 2048**
  (32 × 64), joint audio↔video cross-attention with adaLN-gated fusion
  (`av_ca_a2v_gate`, `av_ca_v2a_gate`).
* Per block: self-attention + cross-attention to the prompt, **RMS q/k-norm**,
  gated attention output (`to_gate`), gelu-approximate feed-forward without
  bias, and adaLN modulation from per-block `[9, 4096]` / `[9, 2048]` tables.
* **3-D RoPE** over (frames, height, width), θ = 10000, max positions
  `[20, 2048, 2048]`, causal temporal positioning.
* **Embeddings connectors** — 8 transformer layers each for video and audio
  with **128 learnable registers**, sitting between the encoder and the DiT.
* The prompt encoder projects the **concatenation of all 49 Gemma-4 layer
  outputs** (`[4096, 188160]` for video, `[2048, 188160]` for audio), not the
  last hidden state — the file carries both aggregates.

## Packing it yourself

Three passes, and each one deletes its source before the next lands — the
stand this was packed on has a 50 GB disk quota and the sources are 71 GB:

```bash
# 1 — the 22B transformer (42.0 GB → 11.0 GB), then delete it
cortiq ltx-pack --out p1.cmf \
  --dit  ltx-2.5-22b-distilled-transformer-bf16.safetensors

# 2 — the Gemma-4 12B encoder on top (--in carries pass 1 byte for byte)
cortiq ltx-pack --out p2.cmf --in p1.cmf \
  --te   gemma4-12b-with-proj-ltx-2.5-bf16.safetensors

# 3 — the VAEs, the upscalers and the duration head
cortiq ltx-pack --out ltx25-q4tp.cmf --in p2.cmf \
  --video-vae ltx-2.5-video-vae-conv-bf16.safetensors \
  --audio-vae ltx-2.5-audio-vae-bf16.safetensors \
  --spatial-upscaler  ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \
  --temporal-upscaler ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors \
  --duration-head     ltx-2.5-duration-head-bf16.safetensors

cortiq verify ltx25-q4tp.cmf && cortiq info ltx25-q4tp.cmf
```

Measured on a 32-core pod: 154 s for the transformer, 123 s for the encoder,
24 s for the rest — **five minutes** for 71 GB of bf16, single machine, no
Python, no GPU.

## Status: the container is done, the renderer is not

Be precise about what you are downloading:

* ✅ **The file is complete and verified** — all 35.65 B weights, the
  tokenizer, every config, `cortiq verify` clean, and it opens through the
  same `mmap` path as every other CMF model.
* ✅ **`cortiq info` / `verify` / `dequant` work on it** — any tensor can be
  read back to raw f32 for a reference comparison, which is how the port
  will be gated.
* ⏳ **`cortiq` cannot render LTX-2.5 yet.** The runtime speaks
  [MiniMax-H3](https://huggingface.co/infosave/MiniMax-H3-Turbo-cmf) and
  MiniMax-Music-3 today; LTX-2.5's `AVTransformer3DModel` — the audio↔video
  gated fusion, the connectors with their learnable registers, the 3-D RoPE
  and the LTX VAEs — is a separate port, in progress against this file.

Until that lands this is a **format artifact**: the pipeline in one
verifiable container, 3.4× smaller, for anyone who wants to read LTX-2.5's
weights without a PyTorch stack — or to watch the port land against it.

## Provenance

The weights are Lightricks' LTX-2.5 release and remain under the
[LTX-2.x Community License](https://github.com/Lightricks/LTX-2/blob/main/LICENSE.md)
— the licence text ships inside the source checkpoints and applies to this
repack unchanged. The CMF container format and the `cortiq` runtime are
Apache-2.0 ([repository](https://github.com/infosave2007/cmf), `PATENTS.md`).

No weight was altered: the pack is a codec change and a container change.
Every tensor's bytes are hashed in the directory, so `cortiq verify` proves
the file is the one that was written.
