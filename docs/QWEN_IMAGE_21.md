# Qwen-Image-2.1

Native Rust path for [Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1):
text-to-image with native transparency (RGBA) and generation from one or more
condition images, from ONE `.cmf` file.

```sh
cortiq imagine qwen-image-2.1.cmf --prompt "A red fox sitting in fresh snow, golden hour"
cortiq imagine qwen-image-2.1.cmf --out sticker.png \
  --prompt "This is an RGBA image with transparency. A cute cartoon dragon sticker. The image has alpha channel and the background is transparent."
cortiq imagine qwen-image-2.1.cmf --image photo.png --prompt "Change the background to a sunset beach"
```

The file carries the recipe (1024×1024, 40 steps, no CFG). PNG output keeps the
alpha channel; JPEG/PPM are composited over white. The ready container is
[infosave/Image-2.1-cmf](https://huggingface.co/infosave/Image-2.1-cmf)
(`qwen-image-2.1.cmf`, 12.7 GB, sha256 `b94309e4…`; the same bytes pack on
aarch64 and x86_64).

| option | meaning |
|---|---|
| `--width`, `--height` | floored to multiples of 32; with `--image` they default to the last image's aspect at `--reference-size`² |
| `--steps`, `--seed`, `--num-images N` | image i uses seed + i |
| `--image PATH` | condition image, repeatable (edit / reference generation) |
| `--reference-size S` | condition images are resized to S² area (default 1024) |
| `--cfg G`, `--negative-prompt` | true CFG, only when G > 1 and a negative prompt is given (the model is sampled without guidance by default) |
| `--out` | `.png` (default `out.png`, RGBA), `.jpg`, `.ppm` |

## Pack

```sh
hf download Qwen/Qwen-Image-2.1 --local-dir Qwen-Image-2.1
cortiq imagine-pack Qwen-Image-2.1 --out qwen-image-2.1.cmf
```

| part | tensors | default codec |
|---|---|---|
| `dit.*` | `QwenImage21Transformer2DModel`, diffusers names | block projections q4tp (`--quant`; `--dit-keep` lifts chosen ones to q8_2f); `txt_in`, modulation, time MLP, `norm_out`, `img_in`, `proj_out` bf16; norms f32 |
| `te.*` | Qwen3-VL-8B language model, 36 layers, no final norm | projections q8_2f (`--te-quant`, `--te-keep` for bf16), `embed_tokens` q8_row |
| `vis.*` | Qwen3-VL vision tower + deepstack mergers | q8_2f (`--vis-quant`); `--no-vision` = text-to-image only |
| `vae.*` | `AutoencoderKLQwenImage21` decoder + encoder | f16 (`--vae-quant`); the temporal convs are not packed |
| `qi21.config_json`, `qi21.scheduler_json` | defaults, scheduler | |

Size: 12.75 GB (DiT 3.90 GB, text encoder 7.58 GB, vision tower 0.59 GB, VAE 0.66 GB).

## Semantics

Mirrors diffusers `QwenImage21Pipeline` (0.41):

- prompt: the raw template `<|im_start|>system\n{sys}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n`,
  Qwen3-VL's last layer before its final norm, the 14 system rows dropped; a condition image
  goes to the vision tower composited over white, and its VAE latent replaces the vision slots
  (one slot = 2×2 latent tokens);
- DiT: one joint sequence `[text/condition prefix, target]`, one shared modulation for all
  32 blocks, affine-free LayerNorm, tanh gates, SwiGLU, per-head RMSNorm + 3-axis RoPE
  (θ 10000, axes 16/56/56; image blocks centred on zero, frame axis frozen at the running
  text position); block-causal attention (text causal, image blocks bidirectional inside);
  text and condition rows modulated from t = 0;
- because the prefix never sees the timestep nor the target, its keys/values are computed
  once per prompt and every step recomputes only the target rows (the pipeline's KV cache);
- scheduler: flow matching, σ = linspace(1, 1/N, N) through the exponential dynamic shift
  μ(target tokens) (base 256 → 0.5, 8192 → 0.9) and the terminal stretch to 0.02;
- VAE: the single-frame specialisation of the residual Wan-2.2 VAE (2-D convs, RMS channel
  norm, DupUp/AvgDown shortcuts), 64 latent channels, 16× spatial, 4 output channels.

## Accuracy (against the diffusers fp32 reference)

256², 4 steps, the reference's prompt features and noise (`CMF_QI21_EMBEDS`, `CMF_INIT_LATENT`):

| container | text encoder | v₀ | v₃ | final latent |
|---|---:|---:|---:|---:|
| bf16 (`--quant raw --te-quant raw`) | 2e-6 | < 1e-5 | 1e-5 | < 1e-5 |
| DiT q8_2f, encoder q8_2f | 0.113 | 0.0055 | 0.030 | 0.011 |
| DiT q4tp, encoder q4tp | 0.40 | 0.044 | 0.19 | 0.073 |

The bf16 row is the semantic check (exact). The encoder's last hidden state is a difference
of large terms in its last two layers (they cancel the massive channel 2276 written by
layers 6 and 16), so a small upstream error grows there: keeping layers 34–35 in bf16 only
moves q8_2f from 0.113 to 0.100, and q4tp stays unusable (0.40) — the encoder ships q8_2f.
The DiT error of q4tp is spread over every projection: lifting any one kind to q8_2f
(q/k, v, out, gate/up, down, the first/last blocks) changes v₀ by at most 0.4 points
(gate/up: 4.4 → 4.1 %), so the DiT ships plain q4tp. The VAE (f16) decodes the reference
latent to 67 dB PSNR against the reference image.

## Backends

- **Metal** (`gpu_metal/qi21.rs`): the prefix and every step run as one resident chain;
  weights are read in place from the file mapping (q4tp dequantized to half while the GEMM
  stages its tile, q8 int8 staged times its column field). Device vs host: v₀ 5e-4.
  Mac mini M4 (10-core GPU, 24 GB): a step is 5.3–5.9 s at 512² and 25–30 s at 1024²
  (it rises as the machine heats over a run; 1024²/40 steps take 20 min)
  (GEMM 91 % at ≈ 2.9 TF/s = 83 % of the half MMA peak, attention the rest).
- **Vulkan** (`gpu_wgpu/qi21.rs`, `gpu_wgpu/qi21_vae.rs`): f16 weight planes built
  once per model (q4tp and q8 alike, 1.3 s), tensor-core GEMMs, the masked prefix
  flash, a resident VAE decoder. Device vs host: v₀ 2–5e-4. RTX PRO 4000 Blackwell:
  a step 0.24 s at 512², 1.04 s at 1024², 6.0 s at 2048²; 1024²/40 steps in 48 s,
  2048² in 246 s, peak VRAM 17.5 GB at 1024². Frames whose VAE activations pass the
  2 GB binding limit decode in horizontal bands with exact halos (bit-identical to a
  whole-frame decode; 2048² 1.8 s, 4096² 8.3 s). `CMF_QI21_WGPU=0`, `CMF_QI21_VAE_CHAIN=0` turn the paths off.
- **CPU**: the reference DiT path (`CMF_QI21_GPU=0`); `CMF_QI21_VAE_GPU=0` keeps the
  VAE on the host too, and `CMF_GPU=0` runs everything on the host.
  (`CMF_QI21_VAE_CHAIN=0` only drops the resident wgpu VAE for the per-conv
  device convolutions.)

A condition image makes the prompt ~1k tokens longer (one vision slot per 32×32
pixels); the Qwen3-VL encoder and its vision tower run on the host, about a minute
for a 1024² image on a 28-core machine.

Knobs: `CMF_QI21_GPU=0` (host DiT), `CMF_QI21_VAE_GPU=0` (host VAE), `CMF_QI21_METAL=0`, `CMF_QI21_METAL_PROF=1`,
`CMF_QI21_AMAX=1`, `CMF_QI21_PROF=1` (stage times), `CMF_QI21_TRACE=<dir>`,
`CMF_INIT_LATENT=<f32 [h·w,64]>`, `CMF_QI21_EMBEDS=<dir>`, `CMF_QI21_DUMP=<dir>`,
`CMF_QI21_LATENT_IN=<f32>`.
