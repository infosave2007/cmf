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
| `CMF_GPU=0` | the whole pipeline on the CPU (the bit-level reference; minutes per image) |
| `CMF_ZIMAGE_GPU=0` | the DiT on the CPU, everything else unchanged |
| `CMF_ZI_VAE=0` | the older per-conv VAE instead of the resident one |
| `CMF_ZIMAGE_TE_EXACT=0` | the text encoder's q8 projections through the default int8-activation kernel (3.3× less accurate, see below) |
| `CMF_ZIMAGE_OVERLAP=0` | uploads the device weights after the text encoder instead of beside it |
| `CMF_ZI_AMAX=1` | prints the per-block f16 maxima of the device chain |
| `CMF_ZI_ACC16=N` | the DiT GEMMs accumulate in f16, flushed into f32 every N K slices (`qkv=N,o=N,w13=N,w2=N` per site; `CMF_ZI_TILE16` sets its tile). Off by default: slower and less precise, see "f16 accumulation" below |
| `CMF_ZIMAGE_TE_DEV=all\|q,k,v,o,gate,up,down` | those text-encoder projections through the device GEMM of their codec, the rest exact on the host (experiment; see "Text encoder on the device") |
| `CMF_TE_TAPS=<dir>` | dumps the text encoder's residual after every layer and each layer's intermediates |

## Where it runs

On Linux and Windows with a Vulkan GPU (tensor cores via cooperative
matrices, f16, 32-wide subgroups — NVIDIA), everything but the text encoder
runs on the device with no flags:

- **DiT**: every block's weights are expanded once per process to f16
  planes (11.6 GB for either model), and a step is one resident chain —
  tensor-core GEMMs, flash attention, fused row kernels, one 1 MB upload and
  one 1 MB readback. CFG runs the cond/uncond pair as one batch-2 forward;
  steps past `--cfg-truncation` switch to a batch-1 program.
- **VAE**: resident decoder — implicit-GEMM 3×3 convs on the tensor cores
  (the 2× upsample folded into the gather), device GroupNorm, the mid-block
  attention in query chunks; one upload, one readback.
- **Text encoder**: on the CPU, with weight-only exact q8 projections on
  x86-64 with AVX2 (other CPUs use the int8-activation kernel, 3.3× less
  accurate), while a helper thread uploads the DiT weights; the device context and the kernel
  compiles start on another helper at the beginning of the run, and the VAE
  weights upload while the steps run.

On Apple silicon (Metal), the DiT and the VAE run on the GPU with no flags:

- **DiT**: the int8 weights are read in place from the mapped file (no f16
  planes: 11.6 GB would not fit beside the text encoder in the shared
  memory, and planes measured no faster). A step is one resident chain —
  simdgroup-matrix GEMMs that stage the int8 tile as half, flash attention,
  fused row kernels — split into command buffers of two blocks each. CFG
  runs the pair as one batch-2 program; the context refiner runs on the GPU.
  On the M4 the CPU's matrix unit (Accelerate) computes a fixed share of
  every large GEMM's output features beside the GPU (a quarter at 512²,
  a fifth above 1600 rows), ordered with the GPU chain by a shared event;
  `CMF_ZI_CPU_FRAC=0` runs the GPU alone, `=auto` adapts the share (then
  the image depends on timing and is not bit-stable run to run).
- **VAE**: resident decoder — NHWC half activations, implicit-GEMM convs
  (the 2× upsample folded into the gather), two-pass GroupNorm, the mid
  attention as GEMMs; its weights upload while the steps run.
- **Text encoder**: on the CPU (Accelerate), beside the device setup.

Other devices and backends fall back to the CPU path piece by piece.

## Pack

```sh
cortiq imagine-pack /path/Z-Image-Turbo --out z-image-turbo.cmf      # DiT q8_2f, TE q8_2f (defaults)
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

Why the defaults are q8 (measured, Turbo unless noted, device path, injected
oracle noise, PSNR in dB of the u8 image against the fp32 diffusers image):

| codec | size | text encoder `h_m2` rel vs fp32 | images |
|---|---|---|---|
| DiT q8_2f, TE q8_2f (shipped) | 10.46 GB | 4.1e-3 (p0), 5.2e-3 (p1) | 6-seed sweep 512² p0: median 28.6, min 27.1 |
| DiT q8_2f, TE q4tp (layer-6 down_proj kept bf16) | 8.78 GB | 5.7e-2, 6.6e-2 | sweep median 22.2, min 16.4; 6 to 10 dB below the q8 TE on every case below |
| diffusers bf16 (reference) | 20.5 GB | 8.7e-3 | sweep median 32.9, min 16.0 |

| case (PSNR vs fp32; vs bf16 where no fp32 exists) | TE q8 | TE q4tp |
|---|---|---|
| Turbo 512² p0 / p1 | 26.1 / 26.1 | 19.4 / 17.0 |
| Turbo 512² p2 (vs bf16) | 32.9 | 23.1 |
| Turbo 1024² p0 / p1 (p1 vs bf16) | 33.3 / 23.6 | 23.2 / 21.5 |
| base 512², 3 steps, CFG + negative | 35.4 | 24.2 |
| base 1024², 28 steps, CFG (vs bf16) | 26.6 | 18.4 |

The q4tp text encoder keeps the composition and legible text but moves
details and colours visibly; it does not hold quality, so the files stay at
10.46 GB. (The DiT codec was settled earlier on the CPU path: q4tp costs 3 to
15 dB against q8.)

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

## Device parity (RTX 3090, Vulkan)

| check | result |
|---|---|
| one DiT forward on the oracle inputs, device vs the CPU path on the same container (Turbo r512 i0 / i5, r1024 i0) | `v` rel 8.8e-4 / 5.7e-4 / 1.7e-3 |
| the same, base (r512 c3 i0 / i2) | 3.0e-4 / 1.8e-4 |
| CFG pair (one batch-2 forward) against the two items stepped one by one (base r512 i0, either order) | both 0 (bit-identical, every block); single vs CPU 1.7e-4. Earlier builds put the first item 2.2e-4 off (see "CFG pair" below) |
| VAE on the oracle latent (r512, r400x592, r1024) | `img` rel 1.4e-4 to 1.6e-4, u8 PSNR 69.3 to 69.9 dB vs the fp32 decoder |
| Turbo 512², whole CLI run, device vs the CPU pipeline (same file, same exact text encoder), p0 / p1 | `v_0` 8.9e-4 / 1.1e-3, `lat_8` 8.8e-3 / 1.9e-2, PNG PSNR 53.4 / 45.5 dB |
| Turbo, 6 seeds 512², device vs the CPU pipeline, both with the older int8-activation text encoder | PSNR median 52.8 dB, min 49.1 dB |
| Turbo images vs fp32 (512² p0, p1, 400×592 p0, 1024² p0) | 26.1, 26.1, 23.6, 33.3 dB; diffusers bf16 vs fp32: 24.2, 30.0, 33.3, 35.2 |
| base, 3 steps CFG + negative (512² p0, 400×592 p1) vs fp32 | 35.4, 29.5 dB; diffusers bf16: 25.5 |
| base with `--cfg-normalization --cfg-truncation 0.1` vs fp32 | 37.9 dB; diffusers bf16: 33.1 |

Turbo's trajectory is chaotic in the first steps: a 1.4e-4 change of
`lat_1` moves `v_1` by 3.5e-2 on the same program, and diffusers bf16 itself
ends at `lat_8` rel 0.23 from fp32 on p0. Per-step error is therefore
checked on identical inputs (first rows above), and whole images over seeds.

### CFG pair

The flash kernel rescales its running softmax lazily, and the decision is
taken for the whole workgroup (64 queries). When a segment's length is not a
multiple of 64, its last query block also holds rows past its end: the next
CFG item's rows in a batch-2 program, pad rows in a batch-1 one. Those rows
used to vote, so the first item of a pair was re-anchored differently from
its own single forward. That is exact in real arithmetic but rounds P to f16
differently, and it put the first item 2.2e-4 off (in either order it was
always the first item). Now only rows inside the segment vote. The pair
matches both singles bit for bit, block by block. Turbo outputs are
unchanged bit for bit, because its pad query rows are zero and never
outvote. Base images move by rounding only. Over 28 CFG steps at 1024² that
is 33 to 47 dB between the two builds on the four card prompts. Quality is
unchanged: base 1024² p0 (negative prompt, oracle noise) scores 26.58 dB
before and 26.64 dB after, both against diffusers bf16.

### f16 accumulation (measured, off)

The DiT GEMMs can accumulate in f16 (`CMF_ZI_ACC16`), which runs at twice
the matrix-unit rate, flushing into the f32 accumulators every N K slices.
WGSL has no conversion between cooperative-matrix types, so each flush
stores the f16 fragment, reloads it as an A operand and multiplies it by
the identity into the f32 accumulator. That step is exact, but it costs one
f32 MMA and 1 KB of shared traffic per fragment per flush. On the 3090 this
arm loses on both counts:

| | f32 accumulate (default) | f16, flushed every slice |
|---|---|---|
| GEMM, the four sites, M 1056–8448 | 51–61 TF | 41–48 TF (best tile 128×64×64); 25–40 TF flushing every 2–4 slices |
| GEMM rel vs f64, f32 epilogue (K 3840 / 10240) | 6.2e-6 / 1.6e-5 | 3.3e-4 to 8.5e-4 (grows with the flush period) |
| GEMM rel vs f64, f16 epilogues | 2.1e-4 (the output rounding) | 3.9e-4 to 1.2e-3 |
| median step, Turbo 512² / 1024² | 0.256 / 1.039 s | 0.308 / 1.224 s |
| median step, base 512² / 1024² | 0.486 / 2.030 s | 0.585 / 2.417 s |
| Turbo 512² p0 / p1 vs the CPU pipeline | 53.4 / 45.5 dB, `v_0` 8.9e-4 / 1.1e-3 | 46.3 / 45.6 dB, `v_0` 1.3e-3 / 1.5e-3 |
| Turbo 512² p0 / p1 vs fp32 | 26.11 / 26.08 dB | 26.10 / 25.97 dB |

The tensor core rounds the f16 accumulator after every 16-deep MMA. The
error therefore has a floor of one rounding per MMA, and a model of that
rounding in numpy predicts the measured numbers. No flush schedule meets
the f16-epilogue gate (2e-4), and every schedule is slower than f32.

### Text encoder on the device

The text encoder runs on the CPU. Before this was measured, the per-op
device arm (`CMF_ZIMAGE_TE_GPU=1`) was said to move the caption and `v_0`
by 3–7 %. That movement does not come from the device. With per-layer taps
(`CMF_TE_TAPS`, `zimage_techeck` with `ZC_TE_DEV`) and a whole Turbo 512²
run per arm on the oracle noise (`v_0` against the CPU pipeline), three
causes show up:

| cause | effect | status |
|---|---|---|
| the host int8-activation (a8w8) kernel, which the old arm used whenever the probe picked the CPU and for every projection below the device size gate (k and v at 22 tokens) | residual 1.9e-2 at layer 0; `v_0` 1.0e-1 / 4.4e-2 (p0 / p1), PSNR 20.2 / 29.0 dB | the pipeline uses the exact weight-only kernel (the default since B2) |
| layer 6's `down_proj`, kept bf16 (F32 in memory), sent through the device's cooperative f32 GEMM, which is tf32-class | residual 2.2e-4 from layer 6 on, the massive-activation layer; `v_0` 1.4e-3 against 8.9e-4 (p0) | fixed: the exact mode keeps F32 projections on the host f32 GEMM |
| the q8 matrix-unit arm (at 64 tokens and more: f16 activations, scaled down when their largest magnitude exceeds 1000, times an f16 plane) | residual 2.8e-4 at layer 0 (p1, 67 tokens) | documented; `CMF_Q8_COOP=0` selects the f32-activation scalar arm |

RMSNorm eps and RoPE are not causes. Both run on the host in every arm,
and the taps (q after RoPE, attention output) are bit-identical between
arms until the first device GEMM. With the F32 fix and the scalar arms,
device projections plus exact host fallback (`CMF_ZIMAGE_TE_DEV=all
CMF_Q8_COOP=0`) put the residual within 3.1e-6 of the exact CPU encoder
at every layer. `v_0` is then 8.3e-4 / 1.3e-3 against the CPU pipeline,
where the exact CPU encoder gives 8.9e-4 / 1.1e-3, so both sit at the
DiT's own device-vs-CPU floor. Even so, it takes 3.6–5.1 s against
0.5–1.7 s on the host, because each projection is a synchronous upload,
dispatch and readback. A future device encoder therefore needs a resident
chain like the DiT's, not the per-op path.

## Speed (RTX 3090, driver 610.43, in-process timers)

`cortiq imagine <file> --prompt P [--height/--width]`, nothing else, one
image per process (the cold command line: model open, device upload and
kernel compile included). Medians of 3 alternating runs; peak = device
memory (nvidia-smi).

| model, size | total | text encoder (beside the upload) | upload + prepare | steps (median step) | VAE | peak |
|---|---|---|---|---|---|---|
| Turbo 512² | 4.55 s | 0.66 s | 1.79 + 0.11 s | 2.05 s (0.255) | 0.13 s | 13.8 GB |
| Turbo 1024² | 11.37 s | 0.67 s | 1.77 + 0.12 s | 8.33 s (1.039) | 0.50 s | 14.3 GB |
| base 512² | 16.20 s | 0.85 s | 1.75 + 0.30 s | 13.60 s (0.486) | 0.12 s | 13.8 GB |
| base 1024² | 59.96 s | 0.84 s | 1.81 + 0.33 s | 56.83 s (2.029) | 0.49 s | 15.3 GB |

diffusers 0.40 bf16 on the same card (all resident, SDPA, `torch 2.8`),
measured the same day, medians of 3:

| model | warm image 512² / 1024² | DiT step 512² / 1024² | cold process → first 512² image | peak |
|---|---|---|---|---|
| Turbo | 2.25 s / 8.33 s | 0.261 / 0.997 s | 17.3 s | 19.9 / 21.7 GB |
| base | 14.26 s / 53.57 s | 0.501 / 1.90 s | 24.0 s | 19.9 / 21.7 GB |

Ratios (cortiq / diffusers):

- cold command line against a cold diffusers process: Turbo 512² 0.26×,
  base 512² 0.68×;
- warm, per image in one process (4 images of one prompt; steps + VAE, plus
  the 0.64–0.85 s CPU text encoding once per prompt): Turbo 2.15 s at 512²
  (0.95×; 1.24× if the prompt is re-encoded every image), 8.81 s at 1024²
  (1.06×; 1.13×); base 13.8 s at 512² (0.97×; 1.03×), 57.4 s at 1024²
  (1.07×; 1.09×);
- DiT step: 0.98× / 1.04× (Turbo 512² / 1024²), 0.97× / 1.07× (base).

## Device parity (Mac mini M4, Metal)

Default = with the CPU share; "GPU only" = `CMF_ZI_CPU_FRAC=0`.

| check | default | GPU only |
|---|---|---|
| one DiT forward on the oracle inputs, device vs the CPU path on the same file: Turbo r512 i0 / r1024 i0, base r512 c3 i0 | `v` rel 7.0e-4 / 9.4e-4 / 2.2e-4 | 1.15e-3 / 8.0e-4 / 2.7e-4 |
| the same against the fp32 oracle (the CPU path: 2.90e-2 / 4.26e-2 / 8.6e-3) | 2.86e-2 / 4.25e-2 / 8.6e-3 | 2.89e-2 / 4.28e-2 / 8.5e-3 |
| CFG pair (one batch-2 forward) against the two single forwards | 2.6e-4 (the pair and the singles get different shares) | bit-identical |
| VAE on the oracle latent (r512) | `img` rel 1.6e-4, u8 PSNR 69.3 dB vs the fp32 decoder | same |
| Turbo 512² p0, 6 seeds (the oracle's sweep noise), PNG PSNR vs fp32, median / min | 28.58 / 27.03 dB | 28.55 / 26.83 dB |
| the same, the CPU pipeline and diffusers bf16 vs fp32 | CPU 28.54 / 26.80; bf16 32.89 / 16.03 | |
| the same 6 seeds, PSNR vs the CPU pipeline, median / min | 52.9 / 46.5 dB | 57.4 / 45.2 dB |
| Turbo whole run vs the CPU pipeline (s42 oracle noise): 512² p0 / 400×592 / 1024² p0 | 32.3 / 45.3 / 54.6 dB | 48.6 / 53.2 / 53.9 dB |
| Turbo s42 images vs fp32 (512² p0, 1024² p0; the CPU path 26.12 / 33.31) | 28.01 / 33.24 dB | 26.25 / 33.29 dB |
| base, 3 steps CFG + negative, 512²: vs the CPU pipeline; vs fp32 (CPU 35.4) | 45.3 dB (`lat_3` 1.5e-2); 35.6 dB | 53.8 dB (4.6e-3); 35.6 dB |

Per step on identical inputs the share is closer to the CPU path (a fifth
to a quarter of the features are f32); whole Turbo runs then diverge
chaotically in either arm (`lat_1` 5e-5 → `lat_8` 8e-2 for the s42 case),
which is why single images vs the CPU pipeline scatter from 32 to 60 dB
while every arm stays as close to fp32 as the CPU path itself.

## Speed (Mac mini M4, 10-core GPU, 24 GB, in-process timers)

`cortiq imagine <file> --prompt P --width W --height W`, one image per
process (model open and kernel compile included), cool-down between runs.
The M4 slows as it heats: a 1024² step goes from ~17 s (share) / 19.6 s
(GPU only) to ~20–23 s over a run, and a base 1024² CFG step from 33 s to
45 s. "Before" = the same binary with the Metal DiT/VAE declined (the DiT
on the CPU through Accelerate, the per-conv VAE). Default = with the CPU
share; GPU only = `CMF_ZI_CPU_FRAC=0`.

| model, size | total, default | GPU only | before | text encoder | prepare | steps (median step), default | VAE | peak RSS / footprint |
|---|---|---|---|---|---|---|---|---|
| Turbo 512² | 30.1–30.5 s | 34.5–35.6 s | 81.6 s | 0.63 s | 0.20 s | 28.2–28.6 s (3.61–3.69 s; GPU only 4.10–4.21, CPU 9.69) | 1.03 s (before 3.28) | 7.5 / 2.6 GB (GPU only 5.4 / 2.2) |
| Turbo 1024² | 160.1–160.9 s | 165–173 s | 371.5 s | 0.63 s | 0.20 s | 154.3–155.0 s (19.5–19.6 s; GPU only 20.2–20.9, CPU 44.0) | 4.8 s (before 13.5) | 7.6 / 4.6 GB (before 13.3 / 17.5) |
| base 512² | 231.8 s | 251.2 s | ≈ 548 s, derived | 1.19 s | 0.38 s | 228.8 s (8.42 s a CFG pair; GPU only 8.85, CPU 19.4) | 1.30 s | 7.5 / 2.6 GB |
| base 1024² | 1234.3 s | 1276.7 s | ≈ 2490 s, derived | 1.19 s | 0.37 s | 1227.6 s (44.8 s; GPU only 45.9, CPU ≈ 88) | 5.0 s | 8.6 / 5.2 GB |

"Derived" = the measured CPU step × steps plus the measured stages. The
first run after switching between the two files pays the page-in of the
new file (text encoder ~3.7 s and prepare ~3 s instead of 1.2 and 0.4).
The CPU share is worth −13 % per Turbo 512² image and −6 % at 1024²
(alternating A/B), but only −3 % on a 20-minute base 1024² image: once
the package is hot, the CPU's power comes out of the GPU's.

diffusers 0.36 bf16 on MPS (`torch 2.8`), the transformer alone on the
same Mac, same day: one forward at 512² takes 4.80 s (the CFG pair 9.77 s;
the first call 23 s, loading 18 s, peak footprint 14.5 GB) — cortiq's step
is 0.75× with the CPU share, 0.88× on the GPU alone (the pair 0.86× /
0.91×). At 1024² diffusers needs 30 GB and swaps: 48–66 s a forward, and
the CFG pair was killed after its first 154 s forward; cortiq takes 19.5 s
(0.30–0.41×) in 4.6 GB.

The DiT step is 3.5 TF/s effective at 512² (3.0 on the GPU alone) and
2.8 TF/s at 1024² (12.5 / 55.1 TFLOP per forward). The GEMMs run at
3.2–3.35 TF/s on the GPU, 91–94 % of what the chip's simdgroup matrix
units issue in a pure multiply loop (3.55 TF/s), plus ~1.1–1.5 TF/s on the
CPU's matrix unit; attention runs at 1.5–1.7 TF/s and is a quarter of a
1024² step. The text encoder is 2 % of a Turbo 512² image.

## Oracles

On a CUDA machine, `python/zimage_oracle.py` holds the full recipe: noise,
tokenizer, TE taps, fp32 CPU runs, DiT taps, the VAE, bf16 runs, the real
pipeline image and the timing bench. Run the fp32 phases with
`CUDA_VISIBLE_DEVICES=` and the GPU phases under the stand lock.
