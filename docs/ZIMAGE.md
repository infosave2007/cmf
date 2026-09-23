# Z-Image and Z-Image-Turbo

Tongyi-MAI's text-to-image models. Both use the same 6.15B single-stream
`ZImageTransformer2DModel`, the same Qwen3-4B text encoder and the same Flux
VAE. cortiq ships each one as a single `.cmf` file:

| model | recipe stored in the file | source DiT |
|---|---|---|
| Z-Image-Turbo | 1024², 8 steps, guidance 0 (one DiT forward per step), shift 3 | fp32 |
| Z-Image (base) | 1024², 28 steps, guidance 4 with CFG, negative prompt "", shift 6 | bf16 |

## Run

```sh
cortiq imagine z-image-turbo.cmf --prompt "A cat sitting on a windowsill at sunset"
cortiq imagine z-image.cmf --prompt "…" --negative-prompt "blurry, low quality"
```

Every option falls back to the recipe stored in the file. Available options:

| option | meaning |
|---|---|
| `--height`, `--width` | image size in pixels; each must be a multiple of 16 |
| `--steps` | number of DiT forwards per image |
| `--cfg` (alias `--guidance`) | 0 turns CFG off. Above 0 the prediction is `pos + g·(pos − neg)`, the diffusers Z-Image formula, which is not Lumina's |
| `--negative-prompt` | the CFG negative prompt (default ""); it goes through the same chat template as the prompt |
| `--cfg-normalization [C]` | clips ‖pred‖ to C·‖pos‖, where the norm is taken over the whole tensor. A bare flag means 1.0, which is diffusers `True` |
| `--cfg-truncation T` | turns CFG off on every step whose `t_norm = 1 − σ` is greater than T |
| `--shift` | scheduler shift override |
| `--max-sequence-length` | token cap applied after the chat template (default 512) |
| `--num-images N` | image i uses seed + i. The prompt is encoded once, and each latent is finished before the VAE loads. The files are named `<stem>_<i>.<ext>` |
| `--seed` | random seed |
| `--out` | output path: `.png` (default `out.png`), `.jpg`, or `.ppm` |

Environment variables:

| variable | effect |
|---|---|
| `CMF_ZIMAGE_PROF=1` | prints stage times and the median step time |
| `CMF_INIT_LATENT=<raw f32 [1,16,H/8,W/8]>` | injects a noise latent, for example one of the oracle `noise_*.f32` files |
| `CMF_ZIMAGE_TRACE=<dir>` | dumps `v_i`, `lat_i`, `vpos_i` and `vneg_i` for every step |
| `CMF_ZIMAGE_DIT_DIR=<diffusers transformer dir>` | runs the DiT straight from the source weights (parity work only) |

## Pack

```sh
cortiq imagine-pack /path/Z-Image-Turbo --out z-image-turbo.cmf      # DiT q8, TE q8 (defaults)
cortiq imagine-pack /path/Z-Image --out z-image.cmf
```

The packer finds the model through `model_index.json` (`ZImagePipeline`) and
streams the shards one tensor at a time. It writes `<out>.sha256` next to the
file and records the sha256 of every source shard in the provenance.

The file holds four parts:

- `dit.*` uses the diffusers names. The block projections and `cap_embedder.1` use `--quant`. adaLN is kept at 16 bit. The embedders, the final layer, the pad tokens and every norm and bias are stored as f32.
- `te.*` holds Qwen3 layers 0..34, because `hidden_states[-2]` is the raw output of layer 34. The layer-35 weights and the final norm are dropped. The projections use `--te-quant`, `embed_tokens` is q8_row, and the `--te-keep` projections stay bf16.
- `vae.*` holds the decoder only, in f32.
- `zimage.config_json` holds the recipe, and the tokenizer is in VOCAB.

Dev flags:

| flag | effect |
|---|---|
| `--quant raw --te-quant raw` | keeps the source dtype, which gives an exact container |
| `--dit-layers N` | builds a truncated DiT for kernel tests |
| `--variant turbo\|base` | overrides the recipe written into the file |

## Correctness

Both models were checked against diffusers 0.40 with fp32 CPU oracles
(`python/zimage_oracle.py`, which uses a CPU `torch.Generator(42)` noise that
is then injected). The engine side ran through `tests/zimage_parity.rs`
(`CMF_GPU=0`) and through the CLI, with `python/zimage_compare.py` comparing
the outputs.

| stage | result |
|---|---|
| tokenizer | ids exact on 12 prompts, including one truncated to 512 tokens |
| text encoder `h_m2` | rel 2.1e-7 to 3.0e-7 against fp32 |
| DiT forward (Turbo r512 i0/i5, r400x592 i3 with image pad rows; base r512 c3 i0/i2) | `v` rel 6e-7 to 2.3e-6. RoPE tables are bit-exact. For comparison, diffusers bf16 is 1.2e-2 to 2.8e-2 away from fp32 |
| VAE decoder (r512, r400x592) | rel 6.6e-7 |
| full CLI run, Turbo 512² and 400×592, 8 steps | `lat_8` rel 5.2e-5 and 5.7e-5; PNG PSNR 78.3 and 78.4 dB |
| full CLI run, base 512², CFG 4 plus negative prompt, 3 steps | PSNR 78.4 dB |
| the same with `--cfg-normalization --cfg-truncation 0.1` | PSNR 84.3 dB (the last step correctly runs without CFG) |

## Diffusers baseline (RTX 3090, bf16, all resident, SDPA)

| model | 512² image | 512² DiT step | 1024² image | 1024² DiT step |
|---|---|---|---|---|
| Turbo (8 steps) | 2.27 s | 0.261 s | 8.37 s | 0.997 s |
| base (28 steps, CFG, batch-2 forward) | 14.3 s | 0.501 s | 53.7 s | 1.90 s |

Peak VRAM is 19.9 GB at 512² and 21.7 GB at 1024².

## Oracles

On a CUDA machine, `python/zimage_oracle.py` holds the full recipe: noise,
tokenizer, TE taps, fp32 CPU runs, DiT taps, the VAE, bf16 runs, the real
pipeline image and the timing bench. Run the fp32 phases with
`CUDA_VISIBLE_DEVICES=` and the GPU phases under the stand lock.
