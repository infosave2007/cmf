# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

The eight-bit build stops paying a readback per projection, and the DiT
learns to render a clip in chunks.

### Added
- **The chunk-causal path (`--stream-chunk`).** The DiT renders a clip in
  chunks instead of one packed sequence: each chunk attends to a sink of the
  first frames and a window of the last, with the already-clean frames entering
  at timestep 0. `--stream-sink` and `--stream-window` set the two spans. Built
  for the RAVEN streaming adapter, which trains exactly this rollout.
- **`ffn_packed`, `fused_panel_keep`, `fused_gemm_from_device`.** The fused DiT
  submissions used to ask for a four-bit weight by name; a container packed any
  other way fell back to per-op GEMMs with a readback between every one, and a
  readback drains the queue. The qkv panel, the output projection and the FFN
  pair now pick a kernel by the weight's dtype. For the two-field int8 codec
  the column field folds into the host activation on the way in, and meets a
  device-side panel through a `colscale` kernel on the way out.
- **`CMF_FUSED_ANY=0`** puts a non-four-bit container back on the per-op path,
  so the fusion can be measured against itself on one machine.

### Changed
- **The streaming sink and window count latent frames, not chunks.** The
  reference card's `sink 2 / window 2` is the attention-sink trick as everyone
  else writes it — two frames pinned at the start, a couple trailing the
  current chunk — and reading them as chunks made every chunk attend to four
  chunks of context, more rows than the bidirectional path it replaces.
  `CMF_STREAM_UNIT=chunks` keeps the old reading.
- **The two-field int8 encoder uses the whole machine.** `encode_q8_2f` walked
  the tensor twice on one core while `encode_q4tp` beside it split rows across
  the box — an order of magnitude on the codec a 20 B DiT gets packed in, and
  packing a container was the wait. The column field is a reduction across
  rows, so it is blocked at a fixed 256 rows and summed in index order: the
  bytes must not depend on the core count, and the byte-identity test now
  covers this codec too.

### Fixed
- The parity harness did not know the two context segment kinds the streaming
  layout added, which stopped the whole workspace compiling under
  `--all-targets`.


## [0.5.94] - 2026-08-19

The eight-bit build runs on the card: two gates, not a missing kernel.

### Fixed
- **The MiniMax-H3 startup probe matched on `Q4TiledP`.** It looked for a
  four-bit qkv weight by name *and* dtype and, finding none, declared the host
  path for the whole render — for a codec that has a device GEMM of its own.
  `QTensor::device_matmat` is now the one place that knows which entry point
  each codec has, and the probe asks it. The two-field int8 folds its column
  field into the activation, which leaves the per-row int8 kernel both
  backends already ship.
- **The wgpu weight-residency budget was the whole heap minus a gigabyte.**
  Every container published before this one had weights far under it; 24 GB of
  eight-bit weights took the card and the first scratch allocation died with
  `wgpu error: Out of Memory`. A quarter of the heap is held back now (1–8 GB):
  a 32 GB card budgets 24 GB of weights, a 24 GB card 18, a 16 GB card 12.

  Measured on an RTX 5090, `mmh3-turbo-clipproj4b-fl2va-v2-q8_2f.cmf`,
  512×288, 22 frames, four steps: **357.3 s → 171.5 s**, GPU 0% → 58%,
  2 MiB → 29.7 GB resident. The probe agrees with the CPU arm to 5.77e-3.

  Still 2.8× the four-bit file's 60.2 s, and that part is kernels: `q4tp` has
  the fused qkv → attention → output submission and the packed FFN, `q8_2f`
  goes through the generic per-op GEMM.

## [0.5.93] - 2026-08-19

The latent upscaler, an eight-bit build, and reference clips.

### Added
- **The MiniMax-H3 latent upscaler is ported** (`crates/cortiq-engine/src/mmh3ups.rs`),
  and `cortiq animate --upscale <file.safetensors> --upscale-by 2.0` runs it:
  render small, resize the *latent* with the learned net, decode at the
  larger size. The 5 B-parameter VAE never does a decode → pixel resize →
  encode round trip, and none of the ghosting a bilinear latent resize
  leaves is there.

  345 M parameters: twelve residual blocks and six temporal convolutions on
  each side of a trilinear resize, with a scalar scale embedding modulating
  every block. **Parity against the node's own torch module: worst 6.68e-6,
  relative rms 3.92e-7** (`examples/mmh3_ups_parity.rs`, oracle written by
  `tools/mmh3_ups_oracle.py`). Loaded from the published `.safetensors`
  beside the container, the way an adapter is — it is a third-party model
  with its own licence.
- **`animate-pack --quant q8_2f`** — the two-field int8, `w = q·row[o]·col[i]`:
  eight bits with a second scale field along the *input* axis, which is
  where an activation-outlier channel shows up from the weight side. The
  step up from q4tp for a machine with the memory, asked for in discussion #5.
  Packed and published as `mmh3-turbo-clipproj4b-fl2va-v2-q8_2f.cmf`, 26.90 GB,
  26.21 B parameters, `cortiq verify` clean. **It renders on the host**: every
  device path here is q4tp-shaped and there is no `q8_2f` matmat on either
  backend yet, only a matvec — the probe refuses with `no q4tp qkv tensor —
  host path` and the card sits idle. Measured on a 5090, 512×288, 22 frames:
  **357.3 s** against 60.2 s for the q4tp file. Said on the model card rather
  than discovered by a user.
- **`animate --video <dir> --video-stride n`** — every n-th frame of a
  reference clip becomes a condition pinned to its own moment of the render.
  This is the `fl2va` keyframe path with more than two frames; the source
  clip is mapped onto the render's length by position. Not a port of the
  release's own v2v node, and the card says so.
- The audio VAE's **encoder half is packed** from now on. The DiT already
  takes a reference soundtrack — the layout carries the segment kind and its
  condition timestep — and the container had nothing to turn a `.wav` into
  latents. The runtime path is still to write; the weights are in place so
  nobody re-downloads a 14 GB file for them.

### Verified
- The spatial-physics adapter's behaviour was a prediction from its header
  and is now measured: `lora: rank 16, 208/258 branches bound; not applied:
  blocks…linear ×50`, and the render completes — rank 16 is not a multiple
  of 32, so it takes the split path rather than the fused kernel.

## [0.5.92] - 2026-08-19

Community adapters run against MiniMax-H3, and a router that turns off the
branches that do not matter.

### Added
- **`cortiq animate --lora <file.safetensors> --lora-strength <s>`** — runtime
  LoRA for MiniMax-H3, the way `ltx-video` already had it. The container's
  weights are q4tp and a rank-32 update cannot be folded into a four-bit
  ladder, so the branch `s·(x·Aᵀ)·Bᵀ` is evaluated beside the base projection
  (`crates/cortiq-engine/src/mmh3.rs`, `BlockLora`). It binds
  `attn.qkv_proj`, `attn.out_proj`, `mlp.fc1` and `mlp.fc2` on all fifty
  blocks and both token-refiner blocks, and prints what bound:
  `lora: rank 32, 104/104 branches bound`.

  Branches that find no projection are **named**, not dropped in silence —
  `adaln_proj.linear` is the real gap, because this container carries the
  modulation as a rank-24 curve and folding an adaLN update into it needs the
  time embedding only the packer has (`animate-pack --lora --time-embedder`
  does that, and is how the Turbo LoRA got in).
- **`CMF_LORA_ROUTE=<r>` — the branch router.** Each branch's contribution
  `‖s·ΔY‖/‖Y‖` is measured against the base panel on the first step it runs;
  every branch below `r` is switched off for the rest of the render, and the
  projection it sat on gets its fused device path back. That last part is the
  point: an adapter's cost is not its arithmetic (0.5% of the projection) but
  the fusion it stands down — a branch has to read the panel the fused kernels
  keep on the card.
- **`CMF_LORA_PROBE=1`** prints every branch by measured contribution,
  loudest first, with the router's verdict beside it.

  Measured on an M4 (24 GB), 512×288, 22 frames, 4 steps, against
  `fal/MiniMax-H3-Realism-People-LoRA` — 104 branches at rank 32, all of
  which bind:

  ```
  lora branches by contribution ‖sΔY‖/‖Y‖ (41 of 104 live):
      0.1633  on   blocks.13.attn.qkv_proj
      0.0946  on   blocks.9.attn.qkv_proj
      …
      0.0011  off  blocks.1.attn.qkv_proj
  ```

  The loudest branch is 150× the quietest; blocks 0–2 contribute nothing this
  adapter would miss, and what matters sits in the middle of the stack. At
  `r = 0.02`, 41 of 104 branches survive, and the render still looks like the
  adapter rather than like the base: 13.34 dB PSNR against the base render
  where the full adapter is 13.76, and 16.75 dB against the full adapter.

  What it costs, measured back to back with the machine's memory freed:
  **denoise 133.6 s with the adapter against 126.1 s without — 1.06×.** The
  video VAE, which no adapter touches, moved 53.4 → 56.9 s between those two
  runs, so the adapter sits at or below this machine's own noise. Earlier in
  the same session, with 6 GB of swap in use, the same pair read 1.13× and
  1.37×, because the base render alone drifted from 22.2 to 32.5 s a step.
  On a memory-bound Mac an adapter is effectively free; where the DiT is fully
  resident the fusion it stands down should show, and that number is owed.

  The branch's own GEMMs are not the cost — on Metal they ride inside the base
  GEMM's submission (`q4tp_matmat_lora`) and pay no transfer. What an adapter
  costs is the *attention* fusion standing down: `dit_qkv_attn_out` keeps qkv,
  the attention and the output projection on the card with nothing in between,
  and a branch on either projection needs exactly those panels.

  `--lora-strength 0` reproduces the base render **byte for byte** — the gate
  that says the adapter path perturbs nothing it should not.
- [`docs/LORA.md`](docs/LORA.md) — adapters on both video models: which names
  bind, what an adapter costs and why, the router, reference conditioning.

### Changed
- The adapter loader accepts the three naming conventions in the wild rather
  than one: `diffusion_model.…` (ComfyUI single-file), `base_model.model.…`
  (PEFT) and the bare module path, and normalizes the container's own `dit.`
  prefix on both sides. A PEFT-trained adapter used to bind nothing at all.

### Fixed
- The MiniMax-H3 card claimed "there is no reference-audio input — H3
  conditions on text and keyframes only". That was wrong: the release is
  tagged `audio-to-audio-video` and `video-to-audio-video`, and the packed
  layout here already carries a reference-audio segment kind with its own
  condition timestep. What is missing is the audio VAE's **encoder half**,
  which `pack_audio_vae` skips as unused. The card now says so, and lists
  what each unported conditioning path actually needs.

## [0.5.91] - 2026-08-18

An LTX adapter stops costing 2.6× a step.

### Changed
- **The LoRA branch is fused into the base GEMM's submission on Metal**
  (`q4tp_matmat_lora`). It reads the activation buffer that GEMM already
  uploaded, runs its two products as `mul_mm_f32nt` encoders in the same
  command buffer, and accumulates into the output buffer through `axpy`
  before the single download — so an adapter now costs **no transfer at all**.
  The adapter's own matrices upload once per render and are keyed by a stable
  branch id.

  Measured on an M4, 384 tokens, same render throughout: **182.7 s** a step
  with the branch as scalar loops, **39.7** through `gemm_nt` under the
  device probe, **22.2** pinned to the host, **10.1** fused — against **8.6**
  with no adapter at all. An adapter is now **1.17×** a step, not 2.6×.

  Correctness is gated by `examples/lora_fused_parity.rs`: fused against a
  host reference on a real container tensor, worst relative 5.96e-4, where
  the base GEMM alone already differs from the host by 6.40e-4. The branch
  adds nothing beyond the codec's own half-precision staging.

  One trap the oracle exists for: `activation_boost` divides the activation
  buffer by a power of two when a row would overflow half and folds the
  factor into the *weight* side. The branch's weights get no such fold, so it
  has to put the factor back — and `wboost` is 1.0 unless something
  overflows, so the wrong direction would have been invisible until the one
  prompt that does.
- The batched q/k/v entry declines when an adapter is loaded: one submission
  carrying three weights cannot carry three branches, and fusing each
  projection with its own branch is the cheaper of the two.

## [0.5.90] - 2026-08-18

LoRA adapters for LTX-2.5, and the multi-subject reference conditioning the
community adapters ship with them.

### Added
- **`--lora <file.safetensors>` / `--lora-strength` on `ltx-video`.** The
  container's weights are q4tp, so an adapter cannot be folded into them
  without dequantizing the whole DiT and requantizing it — the branch is
  evaluated beside them instead, `y = x·Wᵀ + s·(x·Aᵀ)·Bᵀ`, on every path
  including the fused Metal q/k/v submission. Both halves go through the same
  blocked `gemm_nt` the rest of the engine uses, which is not a detail: the
  first version wrote them as scalar loops and cost 182.7 s a step against
  68.1 through Accelerate's AMX, on the same render. They are also pinned to
  the host: the generic `GemmNt` probe sends them to the device, where they
  queue behind the q4tp projection they are standing beside — 39.7 s a step
  through the probe against 22.2 pinned, at 384 tokens on an M4.

  What an adapter costs is placement, not arithmetic. The branch's own GEMMs
  are 1.1 s of a step measured directly (250–1300 GFLOP/s on those shapes),
  about 4% of the base model's flops; the step still goes 8.6 s → ~22 s,
  because host-side f32 work standing beside a device-side 4-bit GEMM does not
  overlap it. A fused device-side branch is the way to close that, and is not
  in this release.
- **`--ref <still.ppm>` (1 to 5) and `--ref-frames`.** Adapters carrying a
  `reference_slot_embedding` condition on reference images. Each still is held
  for 25 or 33 pixel frames, encoded by the same video VAE the render uses,
  given its slot's learned per-channel bias on the latent, and placed at a
  negative frame offset — slot 1 furthest back, the last reference nearest the
  clip. The tokens ride in the same sequence, frozen, and are cropped off the
  result. Verified against LiconStudio's MSR V1: 480 branches at rank 128 bind
  to the container's projections, and three references at 384×256 add 1152
  tokens beside 384 of clip at frames −3, −2, −1. The render carries the
  references' subjects: the man's jacket and haircut, the woman's coat, and
  the club interior all arrive from their own slots. Eight steps at that size
  with three references: 509.5 s denoise, 576.2 s end to end.
- `Conditioning::with_references` and `Geometry::guide_positions` — the
  sequence-extension and the negative-time RoPE coordinates, separately
  testable from the CLI.

### Notes
- A file that does not match is refused rather than approximated: a lone
  `lora_A`, a slot embedding whose width is not the latent's channel count, or
  metadata asking for a token order this build does not implement, all stop
  with a sentence. Reference stills must already be the render's size — this
  build does not guess an aspect fit.

## [0.5.89] - 2026-08-18

A finished stage gives its memory back.

### Changed
- **LTX-2.5 render: the page cache of a stage that has run is released.** The
  container is 20.5 GiB and a Mac with 24 GB of unified memory cannot hold it
  and the render at once, so the machine answers with the compressor and the
  denoise loop slows down as it goes. The prompt encoder is 6.8 GiB that runs
  once; the DiT is 10.8 GiB that is finished before either VAE opens. Both are
  now madvised away at the moment they stop being read — clean file-backed
  pages, so anything that wants them again just refaults. Measured on an M4
  (24 GB), 384x256x25 with sound: the eight denoising steps used to climb
  12.4 -> 13.1 s with a 26 s spike and now hold 8.5-8.7 s flat, and the stage
  goes from 117.5 s to 72.8 s. Nothing about the arithmetic changes.
- `ltx-video` prints the `CMF_MM_AB=1` table when that diagnostic is on. Both
  arms of every eligible q4tp GEMM run back to back on the same data inside one
  call, which is the only device-vs-host comparison a laptop that drifts
  between runs can be trusted to give. On the M4 the Metal kernel is 2.01x the
  host over a whole render (4096x16384 2.17x, 16384x4096 2.21x, 4096x4096
  1.81x) and the two arms disagree by at most 9e-4 relative.

## [0.5.88] - 2026-08-18

The prompt is encoded once.

`ltx-video` caches the prompt encoder's output under
`~/.cache/cortiq/ltx-context/`, keyed by the token ids and by the container's
path, size and mtime. A 12 B forward that depends on nothing else has no
reason to run twice for the same prompt:

```
prompt: 24 tokens → 1024-token context in 29.1s          # first render
prompt: 24 tokens → 1024-token context from cache in 0.03s
```

Twenty-five megabytes a prompt, and `--no-context-cache` turns it off. On an
M4 a small clip went 76 s → 41 s; on a 384×256 49-frame render it is half a
minute off a three-and-a-half-minute total.


## [0.5.87] - 2026-08-18

Metal, a quarter faster again — by profiling instead of guessing.

`CMF_LTX_PROF=1` now reports the q4tp GEMM's own split (upload, kernel,
download) and attention's internals. Three things fell out of it:

* **The feed-forward's gelu ran in f64 on one thread** — half a billion
  values a step, 672 tokens × 16384 wide × 48 blocks, while nine cores
  waited. In f32 across the pool: that phase 11.6 s → 6.1 s.
* **Every activation buffer was scanned scalar-with-a-branch** to find
  whether anything exceeds half's range: 2.7 billion floats a step, one
  thread. Eight accumulators and a comparison instead of `is_finite()` lets
  it vectorize, and past a megabyte it goes to the pool — which is idle at
  that moment, because it is the thread that called in. The value is
  identical, exactly.
* **Independent projections now share a command buffer.** A completion costs
  ~1.3 ms whatever it holds and a step submitted 1344 of them; self-attention's
  q, k and v read the same buffer, and every attention's k and v do.

M4 with 24 GB, 384×256, 49 frames, back to back on the same machine:
**22-23 s → 16.2 s a denoising step**, and the optimized run holds 16.1-16.3 s
where the old one drifted upward under load.

Quality is the same picture: the same prompt and seed before and after match
at 42.6 dB with the same composition, lighting and detail — the last-bit
difference between an f32 and an f64 gelu, amplified by eight sampling steps.

The scalar paths are unchanged, so this is a Metal and CPU-arithmetic
improvement that costs nothing anywhere else.


## [0.5.86] - 2026-08-18

The conditioned modes, actually run.

### Fixed

* **A frozen stream is clean, and has to say so.** The audio↔video fusion gate
  reads the *other* stream's sigma and closes on noise, so a picture handed to
  the model intact was being discounted because it still carried the
  schedule's sigma. The reference forces zero for a frozen modality; so do we
  now. Measured on `--video-to-audio`: the soundtrack came out three times
  louder, and it follows the prompt — "a dog panting softly, a quiet kitchen"
  gives an even 0.007 RMS, "loud sizzling and clattering pans, a dog barking
  sharply twice" gives 0.035 with transients at 0.046, 0.056 and 0.040.

### Added

* **`--video-strength`**, and with it a real video-to-video. A clip that
  covers the whole render is now *re-noised* to the requested level and
  denoised from there — freezing it, which is what the previous code did,
  handed back exactly what was given. A clip shorter than the render is still
  frozen and continued, which is the other useful thing.
* The schedule for a strength starts at exactly that level. Filtering the
  distilled ladder alone silently started lower: at 0.72 the nearest rung
  below is 0.42, a different edit than the one asked for.


## [0.5.85] - 2026-08-18

Metal on a 24 GB Mac: the device is used, and it stays used.

A 22 GB container does not fit in one Metal buffer, so it is mapped as two
overlapping windows — and the driver accounts its working set by buffer
length. Materializing both put more on its books than it will keep wired, so
it evicted and re-wired between commits and a 190 ms matmul took 2.7 s. Three
strikes and the kill switch put the whole process on the CPU. Three changes:

* **windows are built on first use**, so a phase that lives in one of them
  never causes the other to exist;
* **a phase can park the device process-wide** (`gpu::pause_gpu`) — the LTX
  prompt encoder does, because its weights are in the window the denoising
  loop never touches, and `cpu_scope` is thread-local so the pool's workers
  ignore it;
* **a phase can take the probe out of the loop** (`gpu::trust_gpu`). The
  per-op probe times ops in isolation and alternates arms to do it, which
  reads a sustained diffusion step as slower on the device than it is:
  it measured 1.25 ms against the CPU's 0.88 ms and picked the CPU, whose
  loop then ran 23.9 s a step against the device's 19.7 s.

Measured on an M4 with 24 GB, 384×256, 49 frames, no environment variables:

| | before | after |
|---|---|---|
| prompt encode | 45 s | 33 s |
| denoising step | 23.2 s | 19.9 s |
| GPU stalls | 3, then the CPU for the rest of the process | none |


## [0.5.84] - 2026-08-18

Every conditioning mode LTX-2.5 has, and the numbers to justify two of them.

### Added

* **The video VAE encoder** (`ltxenc.rs`) — the decoder run backwards:
  `patchify(4)`, a causal 3-D convolution, then the checkpoint's own
  `encoder_blocks` ladder of `res_x` stacks and `SpaceToDepthDownsample`
  (a stride-1 convolution folded into the channel axis, plus a skip that
  folds the input the same way and averages each group down to width).
* **The audio VAE encoder and a log-mel front end** (`ltxaudio.rs`) — a
  slaney filterbank and a centered Hann STFT, rebuilt because the reference's
  preprocessing computes them rather than storing them.
* **Conditioning in the sampler**: `mask[t] = 0` freezes token `t` at its
  clean value — before the noise, in the velocity, and after every step — and
  hands the transformer a timestep of zero for it. That single mechanism is
  every mode: `--image` (image-to-video), `--video` (video-to-video),
  `--video-to-audio`, `--audio-in` (audio-to-video, audio-to-audio) and the
  image+audio pairs.
* A phase profiler behind `CMF_LTX_PROF=1`: where a denoising step actually
  goes, because guessing at that is how ports stay slow. On an RTX 5090 at
  672 video tokens it is feed-forward 45-50 %, prompt cross-attention 25-28 %,
  self-attention 15 %, audio↔video fusion 10 %, and the adaLN and modulation
  arithmetic together under 1 %.

### Changed

* The per-token work inside a block (ada-zero, the post-SA fold, the gated
  residuals, the output head) runs on the pool instead of one thread.
  Together with attention as GEMMs, a step at 384×256 went from 30 s to 19 s
  on an RTX 5090 and the three-step 768×512 refinement from 580 s to 211 s.
* Metal windows over the file mapping are built on first use, and a phase can
  park the device process-wide (`gpu::pause_gpu`). On a 24 GB Mac the prompt
  encoder uses it: its weights are in a part of the container the denoising
  loop never touches, and putting both on the driver's books made it evict
  between commits while the hot loop paid for it.

### Fixed

* `patchify` is `b c (f p) (h q) (w r) -> b (c p r q) f h w` — the *width*
  offset varies slower than the height one. Getting it backwards does not
  fail; it encodes a picture as a grid of shuffled tiles.


## [0.5.83] - 2026-08-18

The LTX-2.5 release: text to video, end to end, on the Rust engine.

### Added

* **`cortiq ltx-video`** — prompt in, frames out, one process and one 22 GB
  CMF file. It runs the Gemma-4 12 B prompt encoder, the aggregate
  projections and the connectors; the 48-block audio-video diffusion
  transformer; the distilled sampler; and the 3-D convolutional video VAE.
  `--two-stage` adds the learned latent upscaler and the three-step
  refinement the distilled model was trained for.
* **`crates/cortiq-engine/src/ltxdit.rs`** — the `AVTransformer3DModel`
  forward. Per block: video and audio self-attention under split 3-D/1-D
  RoPE evaluated at patch midpoints, prompt cross-attention with its own
  adaLN pair on the query and on the prompt's keys and values, both
  directions of audio↔video cross-attention taken off the pre-fusion state,
  and a gelu-approximate feed-forward — every one of them modulated by adaLN
  values computed once per *distinct* timestep rather than once per token.
* **`ltxte.rs`** — Gemma-4 12 B over a 1024-token window: forty
  sliding-window layers at head 256 and eight full-attention layers at head
  512 whose value projection is the key projection, unscaled attention,
  per-layer scalars. Then the per-token per-layer RMS over all forty-nine
  hidden states, the two aggregate projections, and the eight-block
  connectors with their 128 learnable registers.
* **`ltxups.rs`** — the ×2 spatial latent upscaler (1024-channel 3-D
  convolutions, GroupNorm, pixel shuffle), run in the VAE's own units.
* **`ltxpipe.rs`** — latent geometry, the distilled sigma schedules and the
  ancestral / deterministic Euler steps.
* **`ltx-encode`, `ltx-render`, `ltx-decode`, `ltx-dit`** — the stages
  separately, each with a `--gate` that walks every captured activation
  against a dump of the reference implementation and reports the first place
  it diverges.

* **The audio path** (`ltxaudio.rs`): the spectrogram VAE — 2-D convolutions
  over (time, mel bin) with PixelNorm and height-causal padding so a frame
  never sees the future — and BigVGAN v2 with bandwidth extension: six
  transposed convolutions each followed by three multi-receptive-field blocks
  whose outputs are averaged, every activation a SnakeBeta sandwiched between
  a ×2 sinc upsample and a ×2 sinc downsample, then a second generator
  predicting a 48 kHz residual from the mel of the first one's output.
  `ltx-video --out-audio out.wav` writes 48 kHz stereo; `ltx-audio` decodes a
  saved audio latent on its own, with per-stage statistics and a gate.
* **The duration head** (`ltxdur.rs`): how long a shot the prompt implies,
  read off the connector outputs before a single denoising step runs.

### Changed

* Attention is a pair of GEMMs per head instead of a scalar loop, in both the
  transformer and the prompt encoder — a third off every denoising step.
* `ltx-pack` keeps two more things exact, both found by that gate rather than
  by taste: the **adaLN-single stacks** (their output is the scale and shift
  applied to every token in every block — quantized, they put 3.6·10⁻² of
  relative error into the first normalization of block 0; exact, 5.9·10⁻³)
  and the **token embedding table** at 8 bits (it *is* the residual stream at
  layer zero — q4tp put 11 % into every hidden state). One gigabyte on a
  22 GB file.
* Large projections are chunked before the GPU dispatch: the prompt encoder's
  aggregate reads 188160 numbers per token, which is past what a single
  binding may address.
* The video VAE picks whichever packed config actually describes a VAE, so a
  container packed from the transformer first no longer fails on
  `vae.decoder_blocks`.


## [0.5.82] - 2026-08-16

The Apple-silicon release. Two silent numerics bugs in the Metal backend
are fixed — every q4tp batched prefill on a Mac was noise, and the
device attend of every Qwen3.5-family model ran 15–20% off the CPU on
every token — and speculative decode is now native Metal: a b-row
whole-model graph verifies the MTP head's chain in one submit, the same
graph runs the prompt in 256-token chunks, and the GDN recurrent state
no longer crosses the CPU/GPU boundary at all. Qwen3.8-27B q4tp on an
M4 (24 GB): plain **5.7 → 6.7 tok/s**, code **12 tok/s** greedy with
speculation, a 447-token prompt **42 → 11.6 s** to first token.

### Fixed

- **Metal q4tp GEMM read an unbound `wboost` constant.** The half-range
  activation boost added to `q4tp_mul_mm` in the animate fix was bound
  only by `q4tp_matmat`; the chunk graph (`enc_mul_mm`), the fused
  prefill FFN (`q4tp_ffn`) and the DiT block dispatched the same kernel
  with slot 6 unbound — an undefined weight scale. Every q4tp batched
  prefill on Metal produced noise since then: Qwen3.5-0.8B `ppl` read
  1.3e6 (18.27 now, equal to the CPU); Qwen3.8-27B answered
  `<parameter=1>` to any prompt longer than a chunk. Bound explicitly
  everywhere (the host boost where the activations are host-visible,
  1.0 for device-resident inputs).
- **Metal `attn_rope_qkn` skipped the K heads' norm and RoPE** on models
  whose `nh + nkv` is not a multiple of 8. The kernel derived its head
  from `threadgroup_position × simdgroups_per_threadgroup + simdgroup`
  under a `dispatch_threads` launch, and the partial last threadgroup
  reports its own smaller simdgroup count — heads 8,9 of Qwen3.5-0.8B
  and 24..27 of Qwen3.8-27B (and their MTP blocks) came out as heads
  0..3, so the raw K went into the cache and the device attend ran
  15–20% off the CPU's on every token (`CMF_GPU_ATTEND=0` was exact).
  The head is `thread_position_in_grid / 32` now; the attend oracle
  (`CMF_ATTN_ORACLE=1`) reads max|Δao| = 0 against the CPU, and the
  decode hidden equals the strict CPU path to 1e-6.

### Added

- **Native Metal speculative decode** (on by default for greedy q4tp on
  macOS, the same gate as the wgpu graph): `q4tp_mul_mm_n8`, a 64-row ×
  ≤8-batch simdgroup-matrix GEMM (magic-mantissa half unpack, permuted
  K order, per-row power-of-two activation pre-scale) — 0.64 ms per 46 MB
  gate call, flat in the batch; `VerifyGraph`, a b-row whole-model
  graph: batched norms and FFN, the GDN recurrence over the b positions
  in registers WITHOUT writing state (the commit replays the accepted
  prefix from the same initial state), attention appends b rows to the
  mirror and attends each row over its own prefix, the head folded in;
  the MTP block draft as one token-graph submit (`mtp_step_metal`, its
  input projection folded in), the round's warm-ups as one b-row run
  of the block, a draft-head vocabulary shortlist (`CMF_DRAFT_VOCAB`,
  default 65536: 662 → 170 MB a step, the verify keeps the full head so
  a token past the cut is only a rejected draft), k = 7 on Metal (the
  8-wide tile is free). Oracles: `CMF_METAL_VERIFY_CHECK=1` (every
  verify row against the plain path) and `=2` (the replayed states and
  appended K/V rows against the plain path after commit).
- **Rows-graph prefill on Metal** for q4tp GDN hybrids: the prompt in
  chunks of 256 positions (`CMF_METAL_PREFILL_CHUNK`, `=0` off via
  `CMF_METAL_PREFILL=0`) through the same graph — wide GEMMs, the device
  recurrence writing its state in the same pass, batched RoPE
  (`attn_rope_qkn_b`), the chunk attend — and the MTP warm-up rows as
  one batched run of the block per chunk. Qwen3.8-27B, 447 tokens: 42 s
  (per-position graph) / 24 s (CPU chunked) → **11.6 s**; 0.8B, 910
  tokens: 3.3 s → 0.9 s. Numerics equal the CPU-batched path (8e-4 rel
  against the strict CPU).
- **Zero-copy GDN states on Metal** (`host_state_buffer`): the CPU
  owner's Vec is wrapped as a shared buffer (large mallocs are page-
  aligned on macOS), so the token graph and the verify read and write it
  in place — 300 MB a token / 400 MB a round of memcpy gone; plain 6.0 →
  6.7 tok/s on the 27B. `CMF_METAL_STATE_ZEROCOPY=0` restores the copy.
- Diagnostics: `CMF_LOGIT_DUMP=path` (+`CMF_LOGIT_DUMP_STEP=n`) writes
  one decode step's hidden and logits as raw f32 for cross-backend
  diffing (the strict CPU reference is `CMF_SDOT=0 CMF_GPU=0`);
  `CMF_SPEC_DBG=1` prints drafts against verified ids per round;
  `CMF_VERIFY_SKIP=sag` attributes the verify's cost; `CMF_MM_AB=1` now
  reports any GPU/CPU matmat disagreement above 1%; `CMF_PREFILL_GRAPH=0`
  takes the CPU chunked prefill on Metal.

### Measured (Qwen3.8-27B q4tp, M4 mini 24 GB)

- Plain decode 6.7 tok/s (was 5.7); speculation: code body 12.2 tok/s
  (avg 3 of 7 drafts accepted, round ≈ 295 ms = draft 33 + verify 242 +
  commit 18), prose 7 (the monitor mostly sits at plain); TTFT 38 tok/s.
- The verify's 242 ms is 199 ms of GEMMs at 68 GB/s (the n8 kernel; the
  one-vector matvec streams at 95) + 14 GDN recurrence + 8 attend + ~20
  of small kernels; the memory path alone measures 0.54 ms per gate call
  against the matvec's 0.48 — five layouts tried, none faster.
- The MTP chain probe on the fox text reads d1..d4 = 91/87/84/81 for both
  MTP arms now (the pod's "94%" was that degenerate bench prompt at k=4);
  the 0.8B's head is weak (42% first draft) — acceptance work belongs on
  the 27B.
- `iogpu.wired_limit_mb` makes no difference (the weights are not
  evicted); the M4's 120 GB/s is the only wall for the plain token.

## [0.5.81] - 2026-08-16

### Fixed
- Speculation does not default on over **wgpu/Metal**: the batched
  verify graph there returned 0 accepted drafts and garbage text on
  Qwen3.5-0.8B (the plain graph is fine, Vulkan is bit-exact). The Mac's
  default backend is native Metal, which has no batch graph, so this
  only touched `CMF_GPU=wgpu` on macOS; `CMF_GRAPH_SPEC=1` still forces
  it, with a warning, for the investigation.

## [0.5.80] - 2026-08-16

The Qwen3.8-27B q4tp release, converted straight from the bf16
checkpoint (streamed shard by shard from the hub — never re-coded
from q4t: a second quantization is a quality cut). 14.27 GB against
q4t's 15.43, wikitext-2 perplexity 8.793 against 8.857 on the same
windows. Published to `infosave/Qwen3.8-27B-cmf` with its own
aquarium example. And the decode it gets: speculation off the MTP head
is on by default for greedy — **76 tok/s against a plain 48.7** on the
RTX 5090 (`bench --core`; a code prompt 56.5 against 45.7, prose at
the plain rate by the monitor's choice) — with the draft on the token
graph, an int8-activation batched verify, a sparse sampler chain and a
speculation monitor in place of the one-shot trial. Beside it, the
tools that made the week's measurements: `dequant`/`patch-tensor`, the
FCD tail heal (q4tp 8.79 → 8.49), the NVG fold to a 19.8B and its
KD-LoRA heal, and two field-report fixes for the MiniMax video path.

### Added

- **The int8-activation batched verify** (default; `CMF_VERIFY_I8=0`
  restores the f32 one; `q4tp_matvec4_bk8`): the speculative verify's rows on int8
  activations (per-32 symmetric, the Q8_1 grid) and `dot4I8Packed` —
  a weight word's eight nibbles become two packed int8x4 shared across
  the batch, each element costs two dp4a per word against 32 FMAs and
  a share of 32 unpacks before; one pipeline per batch size (constant
  `NB8`) so the element loop unrolls. Measured on the RTX 5090,
  Qwen3.8-27B q4tp, `bench --core`: plain 48.5, speculative f32 66.4,
  **speculative int8 76.3 tok/s (k=5)**; a code prompt at 2.3k
  context 45.7 → 51.7 (f32) → **56.4** (int8); acceptance unchanged.
  Not the plain path's bits (a near-tie can resolve differently — the
  essay diverged, the aquarium did not); `CMF_VERIFY_I8=0` gives the
  bit-exact f32 verify back; the parity test bounds the drift at 4.7e-3
  rel rms. Speculation itself is on by default for greedy decoding of
  q4tp files (the monitor below stops it where it does not pay);
  speculative SAMPLING stays opt-in (`CMF_GRAPH_SPEC_SAMPLE=1`: 60%
  acceptance at the instruct row, break-even).
- **The speculation monitor** replaces the one-shot trial: an
  exponential average of tokens-per-round and round time against the
  measured plain token decides on EVERY round (stop after four losing
  rounds, retry 128 tokens later). The trial mis-called prose (the
  formulaic first rounds accept well, the body does not: an essay
  measured 39 against a plain 44.8 while "speculating"); with the
  monitor prose sits at the plain rate (46.5 vs 45.3–47.7) and code
  keeps its gain.
- **FCD tail heal** of a quantized file (`tools/fcd_tail_heal.py`):
  layers 0..61 as the file has them (dequantized), the last two layers
  + final norm trained against the bf16 teacher (0.3·CE + 0.7·KL on
  the teacher's top-64), 200 AdamW steps at 2e-5 on 32k calibration
  tokens, best checkpoint by held-out score, exported through
  `patch-tensor`. Measured, wikitext-2 12×512 (`cortiq ppl`): **q4tp
  8.793 → 8.467** (tail at f16, +1.1 GB), **q2tp 14.333 → 12.580**;
  the teacher itself scores 9.14 against the q4tp student's 9.51 in
  the same torch harness, so the heal closes most of the 4-bit gap on
  this eval — with the caveat that wikitext-train was in the
  calibration mix (the held-out code/chat rows improved too: CE 2.208
  → 2.174, KL to teacher 0.099 → 0.094). A tail at f16 has no token-
  graph arm and decodes at 6.8 tok/s; the q8_2f tail is the shippable
  form (below).
- **NVG holographic fold** of Qwen3.8-27B to the per-layer Qwen3.5-9B
  configuration (FFN 12288, 16 attention heads, 32 GDN value heads;
  `tools/nvg_fold_qwen35.py`, math in docs/NVG_FOLD_ATTN_GDN.ru.md):
  19.84B parameters, q4tp 10.37 GB, closed form from activation Grams
  on 32k tokens, no training. Measured: ppl 11.97 (×1.36 the 27B's
  8.79), decode 58.7 tok/s (+20%); the essay is coherent with dated
  facts, the aquarium prompt gets a plan and an early stop — a heal
  (KD from the 27B) is what makes this a model, not the fold alone.
- **Video: the contention kill is disarmed during the prompt encode**
  (`gpu::mm_kill_arm`). The encoder is a one-shot pass over 12 GB; on a
  24 GB Mac with the 25.7 GB fl2va file it streams from disk and its
  GEMMs run over budget for reasons that are not contention — users had
  to gut `mm_kill` in the source to keep the denoise loop on the GPU
  (HF discussion #4). It arms when the denoise starts.
- **macOS: the pool's workers ask for the performance cores**
  (`QOS_CLASS_USER_INITIATED`). A keyframe's video-VAE encode ran ~140 s
  on an M4 with the E-cores at 100% and the P-cores asleep (`asitop`,
  same report); threads without a QoS class are the scheduler's to
  place.
- The MiniMax card's field notes carry the 24 GB frame budgets from that
  sweep (25.7 GB file: 512×256 to ~70–90 frames, 768×448 caps at 40–50;
  the 14.5 GB v2 file: 448×768 × 50 frames at 136 s/step, 90 frames is
  the paging edge) and the v2 face-consistency confirmation.
- **The sparse sampler chain** for configs with a top-k (Qwen's own
  rows: 0.7/0.8/20 and 1.0/0.95/20): the dense chain built six or
  seven passes over 248k floats per token — copy, exp, select,
  filter, sort, renormalise — and every one past the selection
  touched only the k survivors. Now: penalties on a copy only when
  there are penalties, one pooled pass for the top-k penalized
  logits (ties at the k-th place kept, as the dense chain keeps
  them), one for the vocab-wide softmax denominator (top-p is defined
  against the FULL normalisation), the rest over k entries. Same
  distribution — the test checks the survivor set, the probabilities
  to fp, and that a seed lands on the same token; the draw walks ids
  in order like the dense inverse-CDF, so seeded runs reproduce.
  Speculative SAMPLING uses it for its nine distributions a round.
- **The speculation trial is a measurement, not a guess**: on
  ambiguous inputs the round pays or does not depending on the text.
  The loop now times rounds 2–5 (round 1 pays the batch scratch and
  the draft mirror), then 8 plain tokens, keeps whichever is faster
  by 3%, re-checks every 256 tokens; a batch graph that keeps
  DECLINING the verify counts as a round that produced one token (a
  q2tp file spun forever: 792 drafts, 0 accepted, 35 against 46
  tok/s). Off by default under penalties (repetition 1.1 lets 2 of
  16 drafts through). Measured on the 5090 pod: greedy core 60.5
  against a plain 48.7 tok/s; a penalized full loop 46.4 (was 43.4
  when the trial still tried).
- The batch graph admits the 2-bit plane (kind 9) to its GEMM path —
  a q2tp file's verify declined every round.
- **Magic-mantissa nibble unpack** in every q4tp/q2tp decode kernel:
  a nibble ORed into the mantissa of 2^23 minus 2^23 + 8 is exactly
  n − 8, without an integer-to-float conversion (I2F issues at 1/8 of
  the FMA rate on NVIDIA and there were eight per weight word). Same
  bits, greedy output identical; `CMF_MAGIC_UNPACK=0` restores the
  conversions. Measured neutral on the 5090's token (the one-vector
  kernel is not ALU-bound), kept for the batched kernels.
- `cortiq dequant` (one tensor, or a prefix with `--all`, → raw
  f32/bf16) and `cortiq patch-tensor` (raw f32 → the tensor's own
  quantization in place, or another dtype into a new file with
  everything else copied verbatim) — the bridge that lets offline
  tools (the FCD tail heal, the NVG fold) work on a quantized file's
  exact numerics and write their result back.
- `tools/nvg_fold_qwen35.py` — the holographic fold of a Qwen3.5/3.8
  hybrid in closed form (FFN width, attention q-heads, GDN value
  heads) from activation Grams, layer-streamed and grouped for a
  memory-capped box; `tools/fcd_tail_heal.py` — the last layers of a
  quantized file trained against the bf16 teacher (0.3·CE + 0.7·KL
  on the teacher's top-k), layer-streamed on one GPU. Both are
  experiments in flight; see docs/NVG_FOLD_ATTN_GDN.ru.md.
- **Speculative sampling on the token graph** (`CMF_GRAPH_SPEC=1
  CMF_GRAPH_SPEC_SAMPLE=1`): draft from the MTP head's own post-chain
  distribution, accept with min(1, p/q), correct from max(0, p − q) —
  the emitted stream is distributed exactly as the plain sampler's (a
  400k-trial test holds the empirical law within L1 0.01 of the
  target). Opt-in: on the RTX 5090 pod at the Qwen instruct row it
  decoded 19–22 tok/s against a plain 40 (nine post-chain
  distributions a round, lower acceptance than greedy, a verify that
  costs ~2.7 single tokens). Greedy — now with penalties too, by
  argmax equality of penalized rows — stays bit-exact and pays:
  q4tp k=3 51.2 / k=4 52.3 tok/s against a plain 47.1–48.2.
- **The MTP draft block runs on the token graph**: one submit for
  block + fused head with device attention over the block's own
  mirror, hidden and logits back in one map; the round's accepted
  pairs warm the block as one batched graph run. `CMF_MTP_GRAPH=0`
  keeps the per-op arm. Measured, q4tp on the 5090 pod: speculative
  greedy k=4 **58.7 tok/s** against 52.5 with the per-op draft and a
  plain 48.1.
- **Pass merging** on the two graph encoders: a compute-pass boundary
  costs ~9 µs on this Vulkan stack and a token issued 147 of them, a
  batched verify ~500; dispatches inside a pass are serialized with
  memory visibility, so a graph needs a boundary only where the
  encoder itself is used (copy, timestamp, swap, finish) — those flush
  it. `CMF_PASS_MERGE=0` reverts.
- **The batched verify's rows land on the plain token's bits**: the
  batched matvec now sums a group's four scalar chains first and scales
  once, the one-vector kernel's expression — 0 of 3000 outputs apart on
  the RTX 5090 (Metal's fast math keeps a rounding between them, which
  the test reports rather than asserts). A speculative greedy decode is
  therefore identical to the plain one on Vulkan.
- The batch graph's per-row attention (verify, batched prefill) takes
  the decode/split kernels — it was walking every cached position with
  the 32-lane per-head kernel; at a 2.3k-token prompt a k=4 verify spent
  more there than in its matvecs (26 tok/s against a plain 44).
- **GQA-shared split-K decode attention** (`gqa_attend_gpart`): one
  workgroup per (kv head, 256-position chunk) serves every query head
  of the group, so a K/V row is read once instead of once per query
  head — Qwen3.8 has six per kv head, and at 12k context the old
  kernel streamed ~600 MB a layer against a 100 MB cache. Parity with
  the CPU reference 2.7e-7 on the 5090. `CMF_ATTEND_GQA=0` reverts.
- `run` prints first-token latency and decode tok/s separately — the
  single number over the whole window read "6 tok/s" on a short
  answer from a model that decodes at 45.
- `CMF_MV16W=0`: the wide-row q4tp decode matvec on the 8-row pair
  kernel (q4t's shape) for like-for-like kernel A/B.
- **`q4tp_matvec16w_x2`** — two wide projections of one input in ONE
  dispatch (FFN gate+up, the GDN's qkv+z, attention's k+v), body
  generated from the 16w kernel so per-row arithmetic is bit-identical
  (tested). Motive: 700 tiny dispatches issued in one submit cost
  ~10-20 µs each on the RTX 5090 pod — a third of a token that the
  kernels themselves never see. `CMF_MV_X2=0` splits them again.
- Batched prefill can route its wide GEMMs through the tensor-core
  kernel with a device-computed activation scale (`CMF_BATCH_COOP=1`,
  opt-in until measured).
- MiniMax-H3 card: field notes from the 24 GB Mac and 20 GB card
  reports — what fits, for how many frames, and the draft → final
  workflow.

### Changed

- **The sampler runs its whole-vocab passes over the CPU pool** and
  finds top-k's threshold with a streaming k-slot heap instead of a
  second vocab-sized copy and `select_nth`. Bit-identical token for
  the same seed (tested); at Qwen3.8's 248k vocab the serial chain
  had cost the production loop ~8% of the token.

### Fixed

- **The f16 tensor-core GEMM applied its device-computed activation
  scale at one end only**: the operand went in unscaled and the result
  came out divided by 1000/max|x| — off by max|x|/1000 on exactly the
  inputs large enough to need the scale (fused-FFN fc2, the DiT block's
  device-scaled path). A test now drives it with activations at 3000:
  relative rms 2.0 before, 2.8e-4 after.
- **The contention kill needed one slow op; it needs three now.** A
  24 GB Mac running the 25.7 GB MiniMax file (field report, HF
  discussion #2) pages the first post-encode weights from the SSD, and a
  single seconds-long op sent the whole run to the CPU. Consecutive
  strikes, page-ins exempt, `CMF_MM_KILL=0` to disable.
- `bench --json weight_gb_per_token` divided the PROCESS total (warmup,
  the timed prefill, the pair micro-bench) by the steady token count —
  21.9 GB/token on a 15.4 GB file, a 1.5× "amplification" that was the
  arithmetic. It is the steady-window delta now.

## [0.5.79] - 2026-08-15

The Mac 27B release. A windowed weight arena — a model beyond Metal's
`maxBufferLength` runs on the GPU at all — with the windows overlapping
by one tensor rather than half (27B decode 2.2 → 5.7 tok/s), the device
attend by default for GDN hybrids up to head dim 256, and the O(1)
Nyström attention ported to Metal (`CMF_O1_METAL=1`). Qwen3.8-27B q4t
on an M4 24 GB: 5.8 tok/s decode, 20.8 tok/s prefill at 2k, o1 decode
4.7 tok/s at 2k and flat beyond. The diagnostics that closed the "2×
amplification" question (weight bytes per token across backends; the
factor was a looped architecture, not the kernels) rode along.

## [0.5.78] - 2026-08-15

O(1) decode rides the whole-token graph. Two design requirements had
collided silently on GDN hybrids — the graph seeds its recurrent
mirrors during prefill, Nyström records its sealing q-trace only in
the CPU prefill — and Qwen3.8, the first model with both, ran o1 at
0.7 tok/s on the per-op path with nothing saying why. Prefill now
stays on the CPU when o1 is active and the graph's first decode seeds
the GDN state from the host: ~25 tok/s measured, 35× the per-op path,
output verified coherent. Four silent gates on that trail now speak
(the o1 view export, the seal counter, the geometry limits, the
exact-only seal), because the whole day was spent listening to their
silence.

Also here: an in-kernel elimination toolkit for the q4tp matvec
(CMF_MV_PROBE bits for codes, activations, reduction and ladder; a
dual gate+up dispatch behind CMF_MV_DUAL; a VRAM bandwidth test), and
the honest verdict it produced — every in-kernel suspect measures
null on a virtualized 5090 whose clean streams reach 1.6 TB/s, so the
next decode speedup needs bare metal or a whole-layer fusion, not
another shave.

## [0.5.77] - 2026-08-14

The Qwen3.8-27B bring-up release. The model's recommended sampling
needs six knobs and `run` exposed two; the missing four — `--top-p`,
`--top-k`, `--min-p` and a true additive `--presence-penalty` — are the
difference between a 27B that writes a working Three.js aquarium in one
shot and one that drifts into a Turkish essay at the same depth every
time. And the depth itself had a reason: the KV cache's silent 8192
default. At that boundary the token graph declines, the cache evicts
half, and a GDN hybrid's device-resident state goes stale — fluent
output, vanished mind. The default is 32768 now, the eviction warns
loudly, and `CMF_MAX_SEQ` overrides both ways.

## [0.5.76] - 2026-08-14

`cortiq music` got its GPU. A 3090's denoise dropped 1.75x and both
vocoders dropped ~3-4.7x, and not one of those seconds came from a
faster kernel — the q4tp GEMM was measured at ~5 TFLOP/s all along.
The engine was spending its time on the bus: column matrices built,
transposed and shipped when the source was `k` times smaller, 45 MB
readbacks between projections that ran back to back on the same card.
This release deletes the traffic instead of racing it, and ships the
instruments that told the truth after four wall-clock A/Bs had lied —
per-call arm comparison, kernel-vs-readback split, per-stage render
timers.

### Fixed

- **The DiT block now runs resident on the card, both halves.** The
  attention half goes through the engine's fused qkv → RoPE →
  attention → out-projection chain with ONE readback at the end;
  Music-3 could not call it before because that path unconditionally
  applied a qk-norm this model does not have, and `eps < 0` is now the
  rope-only sentinel. The FFN half keeps `ff_in`'s result on the
  device, runs the GLU where that result lives, and lets `ff_out` read
  it in place — 68 MB of traffic per block-step becomes 11. Measured in
  one clock-stable window against the full host chain, same seed:
  denoise 50.1 s → 28.6 s, parity 0.105% RMS, worst sample 26 of 32767.
  `CMF_MUSIC3_DEVATT=0` / `CMF_MUSIC3_DEVFFN=0` restore the host arms
  independently.

- **The DiT's FFN stopped shipping its activations across the bus.**
  A split timer (`CMF_MM_SPLIT=1`) put the q4tp GEMM at ~5 TFLOP/s and
  the readback around it at 82% of the device arm's time — every
  earlier "the GPU barely wins" number was nine parts PCIe to one part
  math. The FFN chain now runs resident: ff_in keeps its result on the
  card, the GLU runs there, ff_out reads it in place, and 68 MB of
  traffic per block-step becomes 11. Measured on a 3090: ffn
  29.1 s → 17.3 s, denoise 50.0 s → 35.2 s, parity 0.10% RMS.
  `CMF_MUSIC3_DEVFFN=0` restores the host chain.

- **The same convolution on Metal, where it was 42% of a render.** The
  wgpu backend stopped shipping the column matrix across the bus; Metal
  kept building it, and on a Mac that made the vocoder the most
  expensive thing in a song — 2152 s of the 5090 s a 95-second render
  took. `conv1d_mul_mm` goes one better than the wgpu arm: the columns
  are not materialized on either side, the gather walking one axis with
  a dilation straight out of the implicit-GEMM tile loop. On an M4 the
  vocoder goes 84.8 s → 18.2 s and a whole render 141.8 s → 72.4 s, at
  0.26% relative RMS against the host arm.

  The kernel ships with a unit test against a reference loop, and that
  is not ceremony: the first cut was fast and wrong — one digit in the
  B-tile stride — and an implicit GEMM fails by making plausible audio
  rather than by crashing. A whole-render parity check said "245% off"
  without saying where; five shapes against a reference said which line.

- **The vocoder stopped losing to its own CPU fallback.** It was the
  slowest stage of a `cortiq music` render and the one the GPU made
  worse — 54.0 s with the device on against 30.2 s with it off. The
  convolution was never the problem. `Conv1d::apply` built a column
  buffer on the host, built a second buffer of equal size to transpose
  it, and uploaded that: up to 2.37 GB of bus traffic for a 20-second
  song, to send data already on the machine `k` times smaller. The new
  `conv1d_gemm` uploads the input and expands the columns in a kernel,
  writing the layout the GEMM wants directly. On a 3090 the vocoder
  goes 54.0 s → 18.2 s and a whole render 151.4 s → 103.5 s, which is
  the first time the device path beats the host one end to end.
  Parity against the host arm on the same latent is 0.086% relative
  RMS — f16 accumulation in the cooperative-matrix GEMM.

### Changed

- Every render now prints where its time went (`stages: denoise …s,
  vocoder …s`) alongside the AR timer. Three optimizations in a row
  had been argued about without this.
- The AR's whole-token graph is opt-in behind `CMF_MUSIC3_GRAPH=1`. It
  had been gated on a GPU check that the AR runs too early to see as
  true, so it was never once attempted; with that removed and the graph
  genuinely taking the token, it costs 136.7 s of AR against 47.8 s on
  the host. A batch-1 matvec is too small an errand to amortise a
  submit when the fallback has 256 cores to spread it over.
- The f32 cooperative GEMM stages its K tiles double-buffered — one
  barrier per K step instead of two — and reads its operands as vec4.
  Parity-clean at every shape including ragged tiles; measured within
  noise on the stand it was written on BECAUSE transfers dominated
  there, kept for the machines where they do not.

### Diagnostics

The reason this release's numbers can be believed, and the previous
attempts' could not. `CMF_MM_AB=1` runs both matmat arms back to back
on the same activations inside one call and reports their ratio per
shape with the worst output disagreement alongside — wall-clock A/B on
a shared stand had read 2.4x, 1.0x and 1.0x for the same change.
`CMF_MM_SPLIT=1` times the kernel and its readback separately (that is
what convicted the bus), `CMF_RB_TRACE=1` splits a readback into
DMA+fence vs mapped-memory copy, `CMF_MUSIC3_PROF=1` splits a denoise
step four ways, and a roofline test reports the coop GEMM in GFLOP/s
on the DiT's own shapes. One stand-level finding worth repeating: a
virtualized card can sit at P8 / 210 MHz in the middle of a render —
bursty submits never wake the driver's DVFS, a process lands in a fast
or slow clock state and stays there, and every whole-process wall-clock
number taken without pinning the clocks carries up to 7x of that.

## [0.5.75] - 2026-08-14

Two fixes to `cortiq music`, one that halves a render and one that
changes how it sounds. Both came out of a session where measurement
also killed three optimizations, which is the more useful half of the
story.

### Fixed
- **The sigma schedule was uniform, and it should not be.** I walked
  σ from 1 to 0 in equal steps; ComfyUI's `normal_scheduler` evaluates
  at `linspace(σ_max, σ_min, steps)` and only THEN appends zero, and
  this model's `ModelSamplingDiscreteFlow` has σ_min = 1/1000. So the
  reference measures a velocity essentially at the end of the
  trajectory while mine stopped at 1/steps — 0.125 at eight steps —
  and integrated the whole remaining tail from a velocity sampled well
  before it. That is a smeared final approach, and it is audible.
  Corrected to the reference's schedule. Measured on the same prompt
  and seed: rms 0.046 → 0.075 and the spectral centroid 3792 → 2410 Hz
  at eight steps, 0.094 and 2099 Hz at sixteen. A high centroid over a
  low rms is hiss carrying the signal; the correction moves energy back
  into the body.

### Performance
- **The DiT's attention is parallel over heads**, which it was not.
  It is quadratic in the sequence and runs 36 times a step, and at 431
  latent frames it WAS the denoise — on one core, while every GEMM
  around it was already threaded. Heads are independent and write
  disjoint columns, so they split without a lock. On an RTX 3090 pod,
  8 steps over 430 latent frames: denoise **192.1 s → 84.1 s** with the
  device, **203.8 s → 93.9 s** without; a 5-second render 308 s → 201 s.
  Verified bit-identical across runs.

### Measured, and reverted
- **Lowering the `b >= 32` floor on GPU matmats made it slower.** The
  reasoning was sound — an autoregressive decode runs at b=2 and reads
  25 M weights to produce two rows, which is bandwidth rather than
  arithmetic, the regime the matvec arm already admits on `rows * cols`
  alone. The measurement disagreed: 336 s against 308 s with the device
  off, and the AR itself went 63.9 s → 75.5 s. Reverted.
- **On that pod the CPU path beats the GPU path for this workload**
  (201 s against 228 s). The device wins the denoise (84 s against 94)
  and loses more than that to initialization and shader compilation.


## [0.5.74] - 2026-08-14

**`cortiq music`: a caption and lyrics become 44.1 kHz stereo, out of one
5.55 GB file.** MiniMax-Music-3 is ported — all five pieces — and the
converted weights are published at
[infosave/MiniMax-Music-3-cmf](https://huggingface.co/infosave/MiniMax-Music-3-cmf)
with a sample this build generated.

| | source | packed |
|---|---|---|
| AR stack (Qwen3-8B + RVQ depth decoder) | 15.56 GiB bf16 | `q4tp` |
| flow-matching DiT | 4.58 GiB fp16 | `q4tp` |
| DAV vocoder | 207 MiB | exact |
| **total** | **20.34 GiB, three files** | **5.55 GiB, one** |

### Added
- **`animate-pack --music-te / --music-dit / --music-vae`** and
  **`cortiq music`**. The AR stack does not encode the prompt, it
  GENERATES the conditioning: a Qwen3-8B backbone prefilled at batch two
  (the words, and a copy whose middle is `<|audio_cfg|>`), then sampled a
  frame at a time at 25 fps — `c0` from the pruned head under
  classifier-free guidance at 1.5 with the top-k mask taken from the
  CONDITIONED logits, then seven codebooks through the depth decoder,
  each fed back through its own embedding table. Those eight hidden
  states, softmax-mixed by `cond_layer_logits`, are what the DiT sees.
  Then an Euler flow walk (σ 1→0, the DiT asked at `1−σ`, windowed
  689/344 with the overlap averaged) and the vocoder at 512 samples a
  latent frame.
- Its own KV cache: `multi_head_attention` takes raw f32 weights, and
  dequantizing a `q4tp` 8B to reach it would be 3.6 GB of qkv alone.

### Seven conventions that a tensor name cannot tell you
Each was a wrong guess first, each corrected by reading ComfyUI's
`comfy/ldm/minimax_music/`, and each fails SILENTLY — the model keeps
producing plausible sound:
- the vocoder's residual dilations are **1, 3, 9**, not BigVGAN's 1, 3, 5
- its Snake reads α **verbatim** where this engine's H3 vocoder keeps α
  and β in log scale and exponentiates on load
- the 128 latent channels are a **stereo pair of 64**, not evidence of a
  second VAE between `latent_channels: 128` and `dec_in_proj [1024,64,1]`
- the DiT's input is `[x | zeros_like(x) | condition]` on the channel
  axis, and both its 1×1 convs are **residual**
- its timestep embedding is a prepended **token**, dropped before
  `project_out`, and its output is **negated**
- `cond_layer_logits[8]` mixes RVQ **codebook levels**, not transformer
  layers — the condition encoder's `num_condition_layers: 8` invites the
  wrong one of two eights
- the sampler is NOT the `FlowMatchEulerDiscreteScheduler` named in
  MiniMax's own config. With `num_train_timesteps: 1` that scheduler's
  schedule degenerates to a constant; ComfyUI registers this model as a
  plain `ModelType.FLOW` with `process_timestep = 1 − t`

### Deliberately not the reference, and marked in the source
- The top-k sampler is a plain xorshift, not torch's seeded `Generator`.
  Nothing downstream needs one seed to mean the same song across
  implementations, only that it means one song here.
- The lyrics normaliser skips the reference's markdown scrubbing, which
  only ever removes characters a caption should not carry.

### Fixed
- **`cortiq animate` panicked on every run on macOS** —
  `gpu::dit_attention_packed` was compiled out under
  `not(target_os = "macos")`, so it was a silent `false`, and the caller
  had already skipped the host qk-norm on the strength of it. The
  decision now asks `dit_attention_packed_available()`, which consults
  the live wgpu context, and the wgpu arm compiles on macOS.
- **The Metal `q4tp` GEMM overflowed `half` on outlier activations**, so
  Apple Silicon rendered uniform grey above 256×160. One row of 982 in a
  MiniMax-H3 frame reaches 3.0e6 by block 44 against `half`'s 65504. The
  host now scales by a power of two ONLY when the panel would overflow
  and folds the reciprocal into the weight side; at 1.0 the path is
  bit-identical, which a text model over Metal confirms by never
  triggering it. 512×288 on an M4: **298.5 s, correct**, against 747.7 s
  over wgpu.

### Measured
- `CMF_METAL_MMPROF=1` splits the Metal GEMM: over 1071 calls, upload
  0.7 s, submit+wait 41.6 s, readback 1.4 s. The copies are 5% — on
  unified memory the round trip is not the cost, so fusing a block into
  one command buffer buys almost nothing.
- The GEMMs are 86% of an H3 denoise (fc1 32.0%, qkv 25.9%, fc2 20.8%),
  attention 11%. The Metal flash-attention kernel is a **2× regression**
  on that workload, not a win.
- Music-3 on an M4: 5 s at 8 steps is 206 s — AR 0.45 s/frame, denoise
  6.5 s/step at 430 latent frames, vocoder 19 s. The vocoder was 75 s
  until it was handed the thread pool.

### Fixed
- **`cortiq animate` panicked on every run on macOS.**
  `gpu::dit_attention_packed` was compiled out under
  `not(target_os = "macos")`, so on that platform it was a silent
  `false` — and `mmh3::attention` turns that refusal into
  `assert!(nr.is_none())`, because the block had already SKIPPED the
  host qk-norm on the strength of the device taking it. The decision to
  skip asked only whether a GPU existed (`enabled_here() && n >= 256`),
  never whether this backend had the kernel. It now asks
  `dit_attention_packed_available()`, which consults the live wgpu
  context and its `dit_qkv_split` pipeline, and the wgpu arm compiles on
  macOS too — `CMF_GPU=wgpu` runs it over Metal like anywhere else. A
  device that refuses is a fallback again, not a crash.

  Measured on an M4 after the fix, 256×160×22 at four steps: native
  Metal 46.1 s, wgpu 117.3 s. Both correct, and neither ran at all
  before.

### Measured, and it says where Metal time is NOT
- **`CMF_METAL_MMPROF=1`** splits the q4tp GEMM into upload, submit-and-wait
  and readback. On an M4 over 1071 calls of a MiniMax-H3 render: **upload
  0.7 s, submit+wait 41.6 s, readback 1.4 s.** The host↔device copies are
  5% — on unified memory the round trip is not the cost, the kernel is.
  That retires the obvious optimization (fuse a block into one command
  buffer to save round trips) before anyone spends a week on it.
- **`CMF_MMH3_PROF=1` on Metal, 512×288:** fc1 32.0%, qkv 25.9%, fc2
  20.8%, out 10.7% — **the GEMMs are 86% of the denoise and attention is
  11%.** Optimization belongs in `q4tp_mul_mm`'s tiling, not around it.
- **The Metal flash-attention kernel (`CMF_DIT_FLASH=1`) is a 2×
  regression** on this workload, not a win: denoise 346.9 s against
  174.1 s for the scalar path at 512×288. It stays off by default, now
  with a number attached.

### Fixed
- **The Metal `q4tp` GEMM overflowed `half` on outlier activations,**
  which is why `cortiq animate` came out uniform grey above 256×160 on
  Apple Silicon while wgpu rendered the same file correctly. The kernel
  stages activations into threadgroup memory as `half`, so anything past
  65504 is `inf` and then NaN. It was not hypothetical and not diffuse:
  on MiniMax-H3 exactly ONE row of 982 — row 49, inside the audio
  segment, the same row every run — grows from 1.7e3 through the middle
  blocks to 4.3e5 at block 36 and 3.0e6 by block 44, then takes the
  audio stream NaN on the second sampling step and the picture with it.

  The host now measures the panel's absmax on the way in and, only when
  it would overflow, scales the activations by a POWER OF TWO — exact,
  no mantissa lost — handing the kernel the reciprocal to fold into the
  weight side, where quantized values have room to spare. The product is
  unchanged. **When nothing is out of range the factor is 1.0 and every
  bit is what it was**, which is the point: this kernel serves every
  q4tp model on Metal. Verified: a text model over Metal triggers the
  scale zero times and its output is unchanged.

  512×288 on an M4 now renders correctly in **298.5 s against wgpu's
  747.7 s** — the backend that was broken is also the fast one.

  Found with two probes kept behind env vars because they are what made
  it findable: `CMF_MMH3_NANPROBE=1` reports which side of the output
  head a NaN is on and which rows of the residual stream carry it, and
  `CMF_MMH3_WATCHROW=<r>` traces one row's magnitude block by block —
  a growing number is an overflow, a sudden NaN is a kernel.

## [0.5.73] - 2026-08-13

**A video model fits a 20 GB card by swapping its prompt encoder, not by
spending fewer bits.** Half of MiniMax-H3's file is a Qwen3-VL-32B truncated
at layer 50 — 12.2 GB that runs once per generation and then sits in the way
of the DiT that actually draws. The `q2tp` build squeezed it and the model
stopped following the prompt. Replacing it instead keeps four bits everywhere:

| `mmh3-turbo-*-q4tp.cmf` | encoder | file | prompt encode |
|---|---|---|---|
| stock | 32B @ 50 | 23.47 GB | 49.2 s |
| ClipProj | 4B @ 24 | **13.16 GB** | **1.9 s** |

The conditioning is a tap, so the layers above it never execute and packing
them is pure file size.

### Added
- **`animate-pack --clip-proj <file>` and `--te-layers <n>`.** A
  [ClipProj](https://github.com/nicolab28/ComfyUI-ClipProj) file carries a
  ridge fit from a small Qwen3-VL's hidden state into the DiT's conditioning
  space: `cond = ((h - mean_in)/std_in) @ W [+ GELU residual] * std_out +
  mean_out`, with token 0 — the attention sink, an outlier no regression fits
  — overwritten by a stored `sink_out`. All ten tensors are packed EXACT
  (f32): 304 MB deciding whether a 10 GB saving still reads as the same
  prompt is not where to spend a quantizer.
- **`CMF_TE_DUMP=<path>`** writes the conditioning the DiT receives as
  `[u64 n][u64 width]` then f32 rows. This is what makes the substitution
  checkable without rendering a frame — cosine against the 32B on one prompt:
  **0.9198** mean at tap 24 (worst token 0.7982), against a floor of 0.5914
  between two different tokens of the teacher itself, and **0.9999** on the
  sink. At tap 25 the worst token sits ON that floor, so the off-by-one in a
  0-based tap index is a real failure mode and now has a cheap detector.
- **`CMF_TE_TAP=<n>`** runs fewer encoder layers than the file carries, so a
  tap can be calibrated against the teacher without repacking.

### Fixed
- **`animate-pack --in` carried components by TENSOR NAME.** A 24-layer
  encoder packed over a carried 50-layer one left `te.layers.24..49` of the
  old stack in the output — six gigabytes that `num_hidden_layers` then
  excludes from the forward, so disk and VRAM budget pay for weight that
  never runs, and the only symptom is a file that is inexplicably too big.
  Re-running a component now replaces it whole.
- **The prompt-encoder packer only knew `model.layers.N`.** A full Qwen3-VL
  checkpoint — which is what the ComfyUI single-file text encoders are — puts
  the LM under `model.language_model.` beside `model.visual.`, and packing
  one failed with "no layers". Both roots are detected.
- **q8_2f addressed its column-scale plane in whole words.** That plane starts
  `rows` half-words after the int8 body, so on an ODD row count the old
  address landed half a word early and every column scale was off by one f16.
  No shipped file trips it (`down_proj` rows are even), which is exactly why
  it needed a test rather than a rendering: `wgpu_q8_2f_odd_rows_matches_cpu_reference`.

### Changed
- **Adreno runs the q1 matvec as 16 rows / 256 threads** instead of the
  desktop 8 / 128, reusing each staged activation tile twice as far; steady
  decode 0.589 → 0.856 tok/s on an Adreno 642L. `CMF_Q1_RPG=8|16` pins it.
  Read that number with its caveat: it is wall-clock, and the 8/128 baseline
  wandered 0.589 / 0.602 / 0.758 across sessions on that phone, so only the
  sign is safe. Other GPUs keep the measured desktop default.
- **`bench --json` reports `tensor_dtypes`,** a histogram over every dtype in
  the file. A whole Adreno tuning round went into a q8_2f kernel that the
  measured artifact has zero tensors of, while wall-clock noise obligingly
  drew an "effect" — two files named `bonsai` with the same 310 tensors and
  different composition. A histogram, not three hand-picked counters: the q1
  family alone is `q1`/`q1s`/`q1t`.
- **`bench --json` GPU counters are steady-state deltas.** `wgpu_submits_per_token`
  and `wgpu_passes_per_token` divided PROCESS-GLOBAL totals — warmup, prefill
  and a microbenchmark included — by the tokens measured. They are now taken
  between the first and last token stamp, the same window as
  `decode_tok_s_steady`, and the counters themselves were made complete
  (every pass and submit is counted, not most of them). The honest numbers
  for a token graph on Adreno are **1 submit and 87 passes per token**, not
  43 and 319 — which retracts "token time = pass count × 4.4 ms" from 0.5.72's
  notes, a law derived entirely from the broken denominator.

### Not fixed, stated plainly
The ClipProj file renders the right clip with plainer set dressing: across the
39 frames the pancake still leaves the surface and comes back, but the 32B's
griddle and patterned wallpaper become a white plate and a flat ground. A 0.92
cosine is close, not equal, and where it is not equal is the furniture. If you
have the VRAM for the 32B encoder, use it.

## [0.5.72] - 2026-08-13

**Turning the GPU on stopped costing three minutes.** On a Snapdragon
778G the phone app spent 256.8 s producing its first answer with the GPU
enabled, against 10.5 s on the CPU path, and paid it again on every
launch. It was the driver's shader compiler, and nothing was keeping
what it produced.

| Adreno 642L / Vulkan | before | now |
|---|---|---|
| CLI bench, whole run | 176 s | **10 s** |
| app, first answer in a fresh process | 256.8 s | **49 s** |
| app, a later turn | 3.6–7.0 s | 2.7–4.6 s |
| steady decode with GPU enabled | 12.40 tok/s | **14.19–14.42** (CPU path: 14.31) |

### Added
- **A `wgpu::PipelineCache` on disk.** Loaded before the first pipeline
  is built and passed to every one of them; 1.73 MB on this device.
  Three things guard a file that goes straight to a driver — which is
  why wgpu makes `create_pipeline_cache` unsafe: the name is keyed by
  driver string, device and engine version, so another build or another
  GPU does not find it; `fallback: true` makes a rejected blob cost a
  recompile rather than the run; and the write is write-then-rename,
  because a half-written blob handed over next boot is exactly the crash
  being avoided. `CMF_PIPELINE_CACHE=0` opts out, `=path` relocates it.
- **Probe verdicts survive the process** (`CMF_PROBE_CACHE`), keyed the
  same way. It does not fix the compile — measured, the wall clock did
  not move — but a verdict reached from cold, unramped samples should
  not be re-rolled every launch: the CPU arm of one probe measured 7.42
  ms in one run and 55.23 in the next, which is the governor, not the
  kernel.

### Fixed
- **Both caches had nowhere to write on Android.** `std::env::temp_dir()`
  answers `/tmp` with no `TMPDIR`, which does not exist in an app
  sandbox, so every save failed silently — and it looked like it worked
  because the shell binary has `TMPDIR=/data/local/tmp`. They now write
  beside the model, a directory the caller already writes to.

### Measured, not changed
- **The GPU still loses on this phone, and the arbitration is right.**
  Forced onto it, decode is 0.883 tok/s against 14.31 on the CPU. What
  changed is that the losing option no longer costs minutes to discover.
  The earlier "−13% for enabling the GPU" is retracted: that was the
  compile bleeding into the measurement.
- **Flushing the cache more than once buys nothing.** An exponential
  backoff (generations 2, 4, 8 …) was tried on the theory that a chat
  turn compiles shapes the first did not: a fresh process then spent
  49.0, 58.7 and 61.3 s on its first answer across three passes. One
  flush, at the start of the second generation.
- **Unexplained:** the app's first answer is 50–60 s where the CLI on
  the same phone and model is 10. Not the pipeline cache — the backoff
  would have caught late pipelines and did not.
- **The GPU path is correct here, just slow.** Greedy text with
  `CMF_GPU=1` is byte-identical to the CPU path on the same prompt, so
  nothing about this is a wrong-answer bug.
- **And the hardware is not what limits it.** 0.883 tok/s on a 591 MB
  model is ~0.5 GB/s, on a bus the CPU itself drives at ~8.5. Routing
  the token through the whole-token graph instead of per-op dispatches
  makes it worse, not better (0.224 tok/s), so it is not submit
  overhead either. These kernels are written and tuned for desktop
  GPUs; an Adreno wants its own pass — subgroup width (64/128 against
  the 32 the cross-lane reductions assume) and the byte-load pattern of
  the quantized unpack are where to start. The verdict "the GPU loses"
  belongs to THESE kernels on THIS chip, not to the silicon.
- **The first Adreno arm was tried and it is a null result.** The matvec
  re-reads the activation vector out of global memory for every row —
  2048 rows × 8 KB on a 2048-wide layer — so `CMF_Q8MV=tiled` stages it
  into workgroup memory once and every row reads it from there. Decode
  went 0.905 → 0.913 tok/s. Kept behind the flag with the number,
  because the useful part is what it rules out: activation traffic is
  not what this GPU is spending its time on. At ~112 dispatches a token
  and 1.1 s a token, that time is ~10 ms per dispatch — a submit and a
  full readback each — which points at the whole-token graph, not at
  the kernel. That path exists and today measures WORSE (0.224 tok/s),
  which makes it the next thing to fix rather than the answer.

## [0.5.71] - 2026-08-12

**The wire now does its work and leaves.** Every split mode so far kept
the network in the loop forever; this release adds the one that does
not. A phone hands its prompt to a laptop, takes the state home, and the
cable can be pulled — measured end to end, with the peer killed the
moment the state landed.

| Xiaomi 12 Lite ↔ M4 | before | now |
|---|---|---|
| 1800-token prompt to first token (USB) | 86.9 s | **16.3 s** |
| thin client over Wi-Fi, bonsai 1.7B | 13.3 tok/s | **23.6** |
| thin client over Wi-Fi, 34.7B MoE | 6.5 tok/s | **12.4** |
| finding a peer | type the address | `cortiq peers`, 5 s |

### Added
- **Wire v7 / `--peer-prefill`: prefill on the peer, pull the state
  home, decode locally.** Prefill and decode have opposite shapes — a
  prompt ships in whole chunks while decode pays one round trip per
  token — so offloading the prompt is the one thing a split does that a
  local run cannot match: 1799 positions cost 86.9 s on the phone and
  10.3 s on the laptop, and the state follows in 5.9 s over USB (206.5
  MB in f16 at the 35 MB/s a cable actually sustains). Verified as
  stated: the worker AND the tunnel were killed the moment the state
  arrived, and the phone generated its tokens anyway.
- **`LayerKvCache::{export_wire, import_wire}`.** The state is 224 KiB a
  position (measured), so `--net-dtype f16` decides whether the trade
  pays. It REFUSES rather than travelling half-described — q8 storage, a
  Nyström overlay and frozen columns each hold state this format does
  not carry. Born-rule importance does travel: every attention call
  accumulates it and eviction reads it, so leaving it behind would make
  the far side forget the wrong positions later, under pressure, long
  after the transfer looked fine. The oracle is what the layer ANSWERS —
  export, import into a fresh layer, `attend()` must match bit for bit.
- **Wire v6 / `--peer-run-ahead K`: K tokens for one round trip.** In
  head mode over the whole stack the coordinator does nothing between
  tokens, so asking permission each time costs a round trip and buys
  nothing. Not speculation: no draft model, no verification, the same
  sequential decode with the handshakes removed, and output identical
  token for token. `StepId` carries the caller's remaining budget so a
  batch never advances the worker's KV past what will be emitted.
- **`cortiq peers`: discovery by asking.** The obvious design — a worker
  announcing itself — was refuted before it shipped: broadcast from the
  laptop never reaches an Android phone on the same Wi-Fi, while unicast
  to that phone always does. So the seeker broadcasts one query and each
  worker answers by unicast. The answer carries the model file name,
  dir_hash, geometry, wire version and whether a token is needed — never
  the token, never the path. `CMF_NET_BEACON=0` opts out.

### Measured, not changed
- **A worker running ON a phone is not discoverable**, and it is not
  broken: it binds the port, serves fine when its address is typed in,
  and is invisible to a scan because the phone never receives the query.
  Two phones therefore cannot find each other this way at all — the
  phone-to-phone section of `docs/MOBILE_SPLIT.ru.md` records what a
  pair would need instead, and the arithmetic saying a pair is only ever
  worth it for a model that fits neither.
- **Run-ahead cannot help a capacity split.** It is legal only when the
  coordinator idles between tokens; two phones split precisely because
  each must compute. That topology is where a real speculative batch —
  and a draft head — would be needed.

## [0.5.70] - 2026-08-12

**A phone can now drive a desktop, which is the opposite of what the
roadmap planned for it — and the measurements are why.** The mobile
extender was specified as a phone adding memory and compute to a host.
It fails that test: a layer split does not parallelise a token, so every
configuration where both sides compute is slower than the faster side
alone (bonsai 1.7B: 28.6 tok/s on an M4, 14.3 on the phone, 14.0 split
in half over USB). Inverted, it pays enormously — a Xiaomi 12 Lite with
2 GB of free RAM generates from a 34.7B MoE at 16.3 tok/s.

| Android coordinator + Mac worker, all layers remote, USB | head local | head on peer |
|---|---|---|
| bonsai 1.7B | 12.6–13.0 tok/s | **26.0–27.3** |
| Qwen3.6 35B-A3B q2tp (34.7B) | 9.1–9.4 | **15.7–16.3** |
| per-token wire | 4 KB | **16 bytes** |
| prefill, 1800 positions, phone alone → offloaded | 86.9 s | **11.3 s** |

### Added
- **Wire v5: `Assign.head` moves the final norm, `lm_head` and the
  sampler to the worker, which answers a token ID** (`--peer-head`).
  The head does not shrink as layers move away — at `--peer-split 0` a
  phone computing no layers at all still spent 29 ms of a 73 ms token on
  it, because 151669 × 2048 sat on the weakest machine. Moving only the
  head would have lost: those logits are 300 KB a token in f16, 20 ms at
  the measured 14.6 MB/s of Wi-Fi. The sampler goes with it, so `Ids`
  ships the prompt history the repetition penalty reads, and the sampler
  config travels as JSON in `Assign` — a greedy run here against a
  default temperature there would have been a different answer with no
  error anywhere. With the worker on layer 0, `StepId` replaces the
  hidden and the token costs 16 bytes of wire.
- **`Stats` and `nodestat.rs`: what a node is worth right now.** Thermal
  zone, mains power, the fastest core's current and peak clock, free
  memory, pool size. Every field is `Option` — on the test device
  `/sys/class/power_supply` is unreadable to a shell process and the
  field stays empty rather than claiming false. Unknown and zero must
  never be the same value to a scheduler. `cortiq run --peer` prints the
  line on connect.
- **`cortiq_worker_start(json)`, `cortiq_set_peer(json)`,
  `cortiq_peer_stats()` in the C ABI.** Phase 4b required the worker to
  be a library, and the ABI had no networking at all — which blocked iOS
  outright. The port is bound on the caller's thread so a busy port is a
  message, not a dead background thread; the peer segment is cached
  across generates for cross-turn KV reuse and dropped on a broken wire
  so the next call redials.
- **`docs/MOBILE_SPLIT.ru.md`** — the integration guide: four roles with
  the numbers that rank them, the ABI, Dart bindings, and six traps.

### Fixed
- Nothing behavioural. `serve --peer` keeps the head local (the sampler
  config is per-request there and `Assign` carries it once).
- **The GPU tests raced each other**, and CI caught it on this release:
  `gpu_q4tp` failed with "Buffer with 'q4tpmm-stage' label is still
  mapped" plus two poisoned-mutex panics behind it. The tests set
  `CMF_GPU` — process-global — from parallel threads and share the wgpu
  scratch slots, whose lock is dropped before the buffers it hands out
  are used. The engine is single-stream by contract; the harness broke
  that contract, so every test in that binary now takes one mutex and
  recovers from poisoning, which is what turned one failure into three.
  Test-only; no runtime path changed. Five other GPU test binaries hold
  the same exposure and are named in the fix rather than changed blind.

### Measured, not changed
- **The transport is ranked by its tail, not its bandwidth.** One 4 KB
  round trip — exactly one hidden state: USB p50 1.89 ms / p99 2.94;
  Wi-Fi 5 GHz ax at −53 dBm with a 680 Mbit/s link, p50 8.95 / **p99
  94.81**, and p90 85 when the coordinator pauses 100 ms between tokens.
  One round trip per token means the tail is what the user sees.
- **The phone's DVFS is worth 2×, and it is not thermal.** The same
  worker served the same span at 22.6 ms and then 42.4 ms at 36 °C with
  no throttling: `schedutil` never raises the clock for a task that
  computes for a few milliseconds and then blocks on a socket (cpu4 sat
  at 691 of 2400 MHz). The fix is ADPF, for which `cortiq_worker_tids()`
  already exists.
- **The worker pool's big-core sizing is already optimal on big.LITTLE**
  — pinning adds 0.3%, and letting the pool grow to all 8 cores costs
  28% (14.3 → 10.3 tok/s), because an A55 becomes the straggler on every
  layer barrier. An earlier +16% for pinning was an artifact of timing
  `run` instead of `bench --json`.

## [0.5.69] - 2026-08-12

**Speculative decode stopped costing money and started making it, and the
27B's decode is now known to be bus-bound — proven three ways, so nobody
spends another day tuning that matvec.** The headline speed number is
small and said so plainly: on an RTX 5090 the production loop goes 43.6
→ 44.3 tok/s and greedy 48.2 → 49.0. Greedy with speculation goes 43.6 →
**51.1**, which is the real result: that path was an 11% LOSS and is now
a win.

| Qwen3.6-27B q4tp, RTX 5090, medians of three | before | now |
|---|---|---|
| production loop (full sampler) | 43.6 | 44.3 |
| greedy | 48.2 | 49.0 |
| greedy + `CMF_GRAPH_SPEC=1` | 43.6 | **51.1** |
| batched FFN in a k=2 verify | 15.05 ms | **11.15 ms** |
| speculative verify round | 53.3 ms | **45.9 ms** |

### Added
- **`CMF_MV_PROBE` — the quad-row matvec with its arithmetic taken
  out, and the answer it gives: the decode is the weight stream and
  nothing else.** Bit 1 drops the nibble unpack and the per-weight FMA,
  +2 the activation loads, +4 the code-plane loads, +8 the cross-lane
  reduction. Stripped to nothing but weight loads the token goes 17.68
  → **16.95 ms** — four percent. The answers are garbage on purpose;
  the time is the point.

- **`q4tp_matvec4_bku` (`CMF_MV_BK`, default 2): a batched matvec that
  unpacks each packed u32 ONCE and multiplies it by every activation
  vector.** At b=1 a dense FFN layer needs 8.9 ms of weight stream
  against 6.3 ms of arithmetic, so the arithmetic hides; at b=3 the
  stream still needs 6.5 while the arithmetic needs 13.8, dead linear in
  batch, because the fused unpack-and-multiply redid the unpack per
  element. Sharing it: batched FFN 15.05 → 11.15 ms, verify round 53.3
  → 45.9. It also carries batches past four now, in chunks of four — a
  k=4 draft verifies five positions, fell off the kernel entirely and
  paid five weight reads, and measured slower than k=3 for that reason
  alone (43.8 → 50.0).

- **The speculative round's third phase is timed, and the submissions
  each phase pays are counted** (`CMF_GRAPH_SPEC_TIME`). Draft and
  verify did not add up to the round; the commit — the MTP block re-run
  once per accepted token — is 5.4 ms of 68. It earns it:
  `CMF_SPEC_WARM=0` skips it and acceptance falls 89% → 81% at k=3.

### Changed
- **`CMF_GRAPH_SPLIT` defaults to 16, i.e. four-layer submit chunks**
  (49.5 vs 48.4 tok/s, medians of three). Every setting above 16 is the
  same configuration once the four-layer floor binds — and five such
  settings measured 44.8/44.7/44.7/44.7/49.4, which is this stand's
  honest spread on a single end-to-end run and the reason every number
  here is a median. GPU timestamps are steady to 0.3%; tok/s is not.
- **`CMF_GRAPH_SPEC_K` defaults to 3**, measured: k=2 46.1, k=3 51.1,
  k=4 50.0, k=5 47.4, k=6 45.2, with acceptance 89-91% throughout. What
  turns the curve over is the verify at ~7.4 ms per extra position, not
  the draft quality. Speculation itself stays opt-in — the greedy
  continuation is byte-identical to the plain path on this model, but
  one architecture is not grounds for changing what every greedy decode
  does.
- The batched graph shares the token graph's content-addressed constant
  buffer cache instead of minting a fresh device buffer per norm weight
  per call. Measured null on time (52.5-53.6 ms verify against
  52.5-53.9) and kept for being strictly less work, not for being
  faster.

### Fixed
- **The sampler allocated a second whole-vocab copy every token.**
  `apply_top_k` needs its own buffer because `select_nth_unstable`
  permutes, but it took a fresh one each call — a megabyte at Qwen3.6's
  248320 vocab, on the decode hot path, next to the `probs` copy that
  had already been moved into the scratch struct for exactly this
  reason. Production loop 44.3 → 45.1 tok/s in isolation.
- **`--no-default-features` did not compile.** The manifest documents it
  as the CPU-only build; `Commands::Gpu` called into `gpu_wgpu`
  unconditionally and that module is behind the `gpu` feature.

### Measured and rejected
Recorded with their numbers in `GPU_KERNEL_RECIPES.md` so they are not
re-walked:
- Persistent grids for the matvecs (`CMF_MV_GRID`): 17.66 → 17.67 ms a
  token; one workgroup per SM is worse at 20.81.
- Halving the batch kernel's live registers by unpacking row pairs: 52%
  SLOWER (FFN 13.40 → 20.39 ms). Registers were never the wall — the
  activation loads are, which is the opposite of the hypothesis.
- Unpacking into f32 registers before the batch loop: fewer ALU ops and
  still a loss, because 32 registers of dequantized weight crowd out the
  occupancy that hides load latency.
- Lowering `par_copy`'s threshold so the per-token logit readback splits
  across threads: 44.4 → 42.5 tok/s. `thread::scope` spawns fresh OS
  threads per call.
- `CMF_MULTISTEP` (k frames a submit, on-device argmax) on this model:
  29 tok/s against 49.

### Retracted
- "The matvec runs at 63% of the card's peak, so there is headroom" —
  there is not. Two decode processes on one 5090 aggregate 52.8 tok/s
  against a single process's 48.8: one stream already saturates ~92% of
  what two can pull, and 1056 GB/s is the bus for this access pattern.
- "Half of a speculative verify is host command encoding" — the batch
  graph's encode timer includes queue backpressure from its own chunked
  submissions, so encode-vs-k came out 16.2/29.0/17.1 ms with no shape
  at all. Against GPU timestamps the verify is ~86% device.
- The bench's `Pair: 2 singles 38.56 ms vs fused 1641.00 ms` is not a
  bug in serve. `measure_pair_fusion` compares device singles against
  the HOST pair walk; all three production callers of `forward_pair` are
  already behind graph guards.

## [0.5.68] - 2026-08-11

**A render is 7.5× faster than it was this morning: 716.6 s → 91.6 s at
8 steps, 42.8 s at 2.** Every stage of the video pipeline now runs on
the card and stays there — the DiT block, both VAE decoders, and the
audio vocoder's convolutions — and the two bugs that were quietly
corrupting the fused paths are fixed. Renders are bit-reproducible for a
given seed, which they were not.

| stage | before | now |
|---|---|---|
| denoise (8 steps) | 8×11.23 s | 8×7.92 s |
| video VAE | 38.3 s | 16.6 s |
| audio VAE | 9.2 s | 5.0 s |
| **whole render, 8 steps** | **143.2 s** | **91.6 s** |

(The 716.6 s figure is where this pipeline started the session; 143.2 s
is where 0.5.67 left it.)

### Added
- **A phase profiler for the audio decoder (`CMF_AVAE_TIME=1`), and
  what it says: 95.4% of that stage is the resblocks' dilated
  convolutions.** Tiling those over time as well as output channels
  took the stage 8.6 s → 8.2 s, bit-identical — small, and that IS the
  result: they are not starved of parallelism, they are ~8 s of
  convolution arithmetic on the CPU while the card is idle. The audio
  decoder is now the only host-bound stage left in a render.

- **The audio decoder's FIR runs across the pool: that stage 9.2 s →
  8.1 s, output bit-identical.** Every channel is independent and the
  whole filter ran on one thread, sharing a single scratch buffer;
  each worker owns one now. The win is smaller than the shape
  suggested, which is itself the finding: the stage's time is in the
  convolutions, and those split over output channels — a dimension
  this decoder narrows to a handful of near the output, exactly where
  the samples are longest.

- **The VAE decoder's resident chain is the default: the stage runs
  26.1 s → 16.6 s and an 8-step render 107.3 s → 95.6 s.** qkv GEMM,
  bias, the weightless q/k norm, RoPE, attention and the output
  projection all run without a panel crossing the bus; only the
  projection's result comes home. 2.33e-3 rel rms from the host chain
  (max 8/255 on 3% of pixels), four times inside the gate the DiT's own
  device path is held to, and bit-reproducible run to run.
  `CMF_VAE3D_FUSE=0` restores the host chain.

  It measured 0.296 rel rms and 97% of pixels for most of a day, and
  none of that was this code — see the split-encoder fix below.

- **The VAE split kernel is exonerated: it reproduces the host repack
  exactly.** Run alone into private buffers (`CMF_VAE3D_CHECK=1`), the
  head-interleaved split matches the host element for element —
  maxdiff 0.0000e0. The zeros the device attention returns are made
  further down the chain, in qk-norm, QK, softmax, PV or the unstack,
  every one of which reads shared grow-only scratch the DiT touched
  first at larger dimensions. The probe itself hit that trap: a
  grow-only slot keeps the usage flags it was FIRST created with, so
  asking an existing plane for COPY_SRC is silently not granted.

- **The VAE chain's defect located: the device attention writes zeros.**
  With a harness that proves the device arm wrote (sentinel fill) and
  cannot recurse into itself, the verdict is unambiguous: every element
  is written, and every element is zero — the reported distance to the
  host equals `max|host|` exactly. So the split's addressing, its bias
  and its layout were all downstream of a stage already producing
  nothing, which is why every layout experiment agreed with every
  other. Prime suspect is shape: the DiT drives these same kernels at
  nh=56 hd=128 and is correct, the VAE asks for hd=64.

- **On the VAE chain: a measurement that measured itself.** A check
  harness compared the device attention against a host reference and
  reported "both layouts byte-identical, 7.2765 off the host". Both
  halves were wrong: its reference arm re-entered the very function it
  was checking, and its device arm was never asserted to have written
  anything — 7.2765 is simply max|host| against two buffers of zeros.
  A tripwire settled it: making the split kernel write ZEROS for this
  layout changed the verdict by nothing, which no working comparison
  can report. Harness removed rather than left to be believed. The
  replacement must assert the device arm wrote before comparing, and
  call the host repack directly instead of through the dispatcher.

- **The VAE decoder's resident chain, opt-in (`CMF_VAE3D_FUSE=1`), NOT
  yet correct.** The split and qk-norm kernels learned this decoder's
  head-interleaved panel and its qkv bias, and the chain runs: 26.1 s →
  16.1 s on the stage, 57.6 s → 47.2 s on a 2-step render. But it is
  0.296 rel rms off the host chain on 97% of pixels, so the default is
  unchanged and byte-identical to before (same md5, 57.2 s).

  Bisected: the error survives with both GEMMs on the host
  (`CMF_VAE3D_SPLIT=1` reproduces it byte for byte), so it lives in the
  two kernels, not the resident hand-off. The open contradiction:
  teaching BOTH kernels the layout and the bias changed the output by
  NOTHING — impossible at nh=32, where block and head-interleaved
  addressing differ for every head above the first. The one-shot probe
  fires on the DiT's first call and never reaches the VAE's; instrument
  per layout.

- **The VAE decoder's host loops meet the thread pool: the stage drops
  38.3 s → 26.1 s and a 2-step render 69.8 s → 57.5 s (−17.6%).** Its
  norms, biases and gated residuals were sequential `chunks_exact`
  walks — the DiT has run the same work across the pool since the
  start. Norm rows alone: 3.3 s → 0.5 s. Every loop is per-token and
  independent, so the arithmetic is unchanged, operand for operand.
- **A phase profiler for the VAE decoder (`CMF_VAE3D_PROF=1`) and a
  permanent stage line for every render.** The DiT had a profiler for
  months; the VAE — the LARGER half of a short render — had none, and
  that is the whole reason it went untuned this long. The stage line
  (`prepare · text encode · denoise · video vae · audio vae`) is what
  showed it: at 2 steps the denoiser is 19.2 s of 68.9 and the two VAEs
  are 47.1.

### Fixed
- **The qkv split kernel never ran — for anyone.** (Swept afterwards:
  of the 92 command encoders in the wgpu backend this was the only one
  that was never consumed — every other is either finished in place or
  handed to `readback`/`submit`, which finish it.) Its command encoder
  was created, filled with the dispatch, and dropped; only the qk-norm
  encoder that follows reached the queue. q and k hid it, because
  qk-norm reads the packed panel itself and rewrites both planes. Only
  `v` had no second writer, so it held whatever the previous block left
  in the slot.

  The DiT's fused attention has been shipping with that stale `v`. With
  the encoder submitted, the fused path now equals the host chain
  EXACTLY — identical md5, zero differing pixels — where before it did
  not, and it is 15% faster than that chain (57.1 s against 67.2 s).

  It also explains the VAE's all-zero device attention: there `v` was
  never anything but zeros, and P·V of a zero V is zero — which is why
  every layout experiment agreed with every other. With the split
  running, the layout finally matters: 0.61 from the host at this
  decoder's layout against 15.29 at the DiT's.

- **Two GEMMs on one device could compute on each other's operand.**
  The activation, result and staging slots are ONE buffer each per
  context: written under the scratch lock, then read at submit time
  after that lock is released. Two threads in `tp_matmat` would upload
  into the same slot and one would run against the other's data. The
  batch parity test caught it the moment an unrelated edit moved the
  parameter write a few lines later — but the window had been open all
  along, and the parameters were a shared slot too. Parameters are now
  built per call, and a per-device gate holds the whole GEMM (upload,
  encode, submit, readback). The scratch lock could not do that job:
  `readback` takes it itself and std mutexes do not re-enter. Verified
  four consecutive clean runs against one failure in three before.

- **A seeded render is reproducible again.** It was not: three runs of
  one binary with one seed produced two different files. While a probe
  class is undecided the arbitration ALTERNATES arms on real user data
  (`flip % 2`), and whether it ever decides depends on how many samples
  the cold-discard throws away — so which ops ran on which arm, and the
  frames that came out, turned on a race. `animate` now pins the arms
  (an explicit `CMF_GPU_PROBE` still wins). Measured cost: none —
  57.4 s and 57.8 s against 57.5 s with probing.

  This also invalidated every md5 comparison made against a render from
  a previous binary: the 6.7e-4 and 7.8e-4 "deltas" attributed to two
  optimizations today were this race, not the changes, both of which are
  arithmetic-for-arithmetic identical.

- **The output projection joins the resident chain: a DiT step is 29%
  faster (11.23 s → 7.92 s), a full render 143.2 s → 116.4 s at 8 steps.**
  With qkv, qk-norm, RoPE and attention already on the card, the output
  projection was the last thing pulling a panel home — and a readback is
  never just its own bytes, it drains the queue, so the card idled once
  per block waiting for the host to take delivery. `dit_qkv_attn_out`
  now runs qkv → attention → projection with nothing crossing the bus;
  only the projection's own result (n×hidden) comes home.

  Two guards had to learn about a resident operand, and both refused
  silently until instrumented: `dit_attention_inner` size-checks a host
  slice that keep-mode does not have, and `tp_matmat_impl` rejects an
  empty `xs`. The second one refused AFTER attention had already run, so
  the caller redid the whole chain — the fused arm measured 3 s SLOWER
  than the one it replaced until that was found. Refuse at the door or
  not at all.

  The activation scale is now computed by the card (`act_absmax` over
  the resident panel) because the host never sees the numbers; without
  a scale the f16 operands overflow, which is how this kernel first
  produced NaN. When that reduction is unavailable the scalar arm runs
  instead — it reads f32 and cannot overflow. Frames: max 6/255 on
  0.26% of pixels, rel rms 7.8e-4, an order below the parity gate; the
  difference is f16 rounding around a scale computed in a different
  place, not a different result. `CMF_MMH3_FUSEOUT=0` restores the
  previous chain.

  Measured in both orders (cold-first and warm-first) after a first
  attempt compared a cold run against a warm one and read the probe's
  cold arbitration as a frame difference.
- **The DiT profiler stops lying by 59%.** The fused attention path
  never stamped its slots, so a step the card spent 13.4 s in reported
  as 5.5 s and named the FFN as 92% of the work. It is 38%. Every early
  return now stamps the same slots the long path does.
- **The DiT's qkv panel stops crossing the bus: 716.6 s → 98.1 s (7.3×).**
  Three pieces had to land in order. qk-norm and RoPE became a device
  kernel (one workgroup per token-head for the reduction; q and k are
  two dispatches of the same kernel, differing only in output plane,
  weights and offset — WGSL cannot take a storage binding as a
  parameter, which is what sank the first draft). Then the FFN half of
  a block became its own function, so the attention half can return
  early. Then the projection writes into a device buffer that the split
  reads in place. 160 MB down and the same back up, per block, per
  step — gone. Frames bit-identical at 1, 2 and 4 steps.
  `CMF_MMH3_QKNORM=cpu` / `CMF_MMH3_FUSEQKV=0` restore the host path.

### Added
- **`CMF_DIT_ATTN_PROF=1` splits the attention phase into its three
  walls, and it named the culprit in one run**: QK 1.65 s, softmax
  0.97 s, **PV 4.76 s** per step at 512×288. PV is 3× QK at identical
  FLOPs, so it is not arithmetic — its right operand `v` is stored
  [n][hd] and the GEMM reads it down columns. The fix is n·hd of
  transposing against n²·hd of work.
- **PV now rides the matrix units too, and it is the default.** Its
  operand is transposed first (`dit_v_transpose`, n·hd of copying
  against n²·hd of work), which turns PV into the same NT product QK
  already was. Two traps on the way, both recorded: an auto bind-group
  layout is built PER ENTRY POINT and renumbers, so the transpose needed
  its own module; and the NT form took its reduction width as
  `cols4 * 4`, which truncated PV's k of 1859 to 1856 and dropped three
  columns of every score row (frames 59% off) — the spare uniform word
  now carries the true k. Measured: **PV 4.76 → 2.91 s** a step, frames
  within 0.1% of the scalar path, full render **111.0 → 106.9 s**.
  `CMF_DIT_ATTN_COOP=0` / `CMF_DIT_PV_COOP=0` opt out.

### Added
- **The qkv split moved to the device, and it was the attention
  bottleneck all along.** Timing the phase's halves separately showed
  4.4 s of a 7.3 s attention phase was the HOST building head-major
  q/k/v out of the interleaved panel — more than the device work it fed,
  and the reason last round's tensor-core QK (correct, and worth
  nothing) optimized the wrong half. `dit_qkv_split` sends the panel up
  in one piece and scatters it on the card. Measured at render size:
  repack **4.4 → 1.5 s**, full render **130.0 → 111.8 s**, frames
  **bit-identical** (delta 0.000). `CMF_MMH3_ATTN=repack` forces the
  host form. Against where this started, a render is now 716.6 →
  **111.8 s (6.4×)**.

### Fixed
- **A rejected shader had silently disabled the dequantize-once path.**
  The device-side activation scale added a `pmm_s` binding to the f16
  cooperative GEMM but declared it in the wrong source, so wgpu rejected
  the module — with a warning nobody was reading — and every call fell
  back to the in-kernel unpacker for several commits. Declaration fixed;
  more usefully, the parity test now ASSERTS instead of skipping when a
  device that advertises cooperative matrices has no f16 pipeline, which
  is exactly the shape this bug had. A skipped test is not a passing one.
- **And the binding it declared then failed to bind.** With the shader
  compiling again, every dispatch hit "4 bindings against a layout of
  5": the code decided whether to add the scale entry by comparing
  pipeline ADDRESSES, and the handle passed in is not the object held
  in the context. wgpu 30 exposes no identity on either handle, so the
  caller now says so directly — it passes a scale buffer exactly when
  the f16 twin is the pipeline. Render restored end to end: 130.0 s,
  frames within 2.6% of the reference.

### Fixed
- **A cold call now takes the device arm while the probe is deciding.**
  Every GPU sample from a cold weight is discarded (the upload is not
  steady state), and in a diffusion stack every layer is touched once
  per step — so on step 1 the GPU arm never accumulates and the class
  keeps alternating. With exactly two arbitrated GEMMs per block and a
  shared alternation counter, one projection drew the CPU arm for the
  whole first step. Handing it to the host bought nothing: the upload
  was needed for step 2 regardless. MiniMax DiT, first step: `out`
  **9.8 s → 3.0 s**, whole step **27.5 s → 18.9 s**; full render
  136.4 s → **127.8 s**, frames unchanged. LLM decode/prefill unmoved
  (W2 122.6/155.7, nanbeige 81.2/58.7).

### Added
- **The video FFN runs end to end on the device — a render is now
  716.6 s → 137.4 s (5.2×).** MiniMax's DiT and the 3D VAE both use a
  SwiGLU whose fc1 emits gate and up PACKED in one row, so neither could
  reach the existing fused path; each block shipped its intermediate
  panel across the bus twice (at render size ~660 MB down and ~330 MB
  back up, per block, per step). `q4tp_ffn_packed` keeps fc1 → SwiGLU →
  fc2 in device buffers: one upload, one readback. DiT step at 256×160:
  **16.0 s → 6.4 s**; full render **251.9 s → 136.4 s**, frames within
  2.6% of the host reference — and 716.6 s → 136.4 s (**5.3×**) against
  where the host path started. `CMF_MMH3_FFN=cpu` / `CMF_VAE3D_FFN=cpu`
  force the host chain.

### Added (measured neutral, kept behind a flag)
- **`CMF_FFN_FC2_COOP=1`** puts the fused FFN's second GEMM on the
  matrix units, with the activation scale computed ON the device
  (`act_absmax` reduces max|x| of a panel the host never sees, and the
  cooperative kernel reads its scale from that buffer instead of a
  uniform). Correct — frames match the default path exactly — but a
  wash: 139.8 s against 137.4 s. The suspected cause — a single
  workgroup walking 330 MB — was then removed (a two-stage reduction:
  512 workgroups of partials, then a fold) and it made no difference
  either: 139.6 s, frames bit-identical. So the cost is the SECOND plane
  unpack, 77 M elements per call, and the scalar arm stays the default
  on evidence rather than on suspicion.

### Fixed
- **`.min(MAX_WG)` on a 1-D dispatch is silent corruption, not a
  guard.** wgpu caps each grid dimension at 65 535; clamping x means the
  tail of the data keeps whatever the buffer held. It bit twice today —
  the q4tp plane dequantized a fifth of its weights, and the SwiGLU left
  most activations untouched (frames 92% smaller, i.e. nearly blank, at
  512×288 while 256×160 looked perfect). Both now dispatch 2-D and index
  `gid.y·(65535·256) + gid.x`.
- **A fused chain must carry the activation scale to EVERY cooperative
  GEMM in it.** The second GEMM of the packed FFN read its input from a
  device panel the host never sees, so no scale could be computed for it
  and its f16 operands overflowed. It runs the scalar arm — which reads
  f32 directly — until a device-side max reduction can feed the scale.

### Added
- **A real gate under the tensor-core GEMM.** The neighbouring coop test
  only PRINTED its worst relative error, so the f16 operands, the
  activation scale and the unpacked plane could all drift unwatched.
  `wgpu_q4tp_mm_coop_matches_scalar` now runs both arms over the same
  weights with activations spanning ±3000 — the range that used to
  overflow f16 — and asserts they agree: measured 3.19e-4 relative rms
  on an RTX 5090, threshold 1e-2. It skips cleanly on machines without
  a cooperative-matrix device.
- **Kernel-level number for the dequantize-once work**: the same q4tp
  GEMM (9216×2304, n=2085) runs **25.0 → 51.9 TFLOP/s** with the tensor
  cores, measured by `wgpu_q4tp_mm_throughput` on both arms.

## [0.5.67] - 2026-08-11

### Added
- **The tensor cores stop waiting on a nibble unpacker: q4tp GEMM
  dequantizes ONCE per call.** The cooperative-matrix kernel unpacked
  4-bit weights inside its inner loop — about as many scalar ops per
  weight tile as the matrix units spend MACs on it, repeated for every
  64-row tile of activations, so the units idled through a
  dequantizer. Now a separate pass unpacks the plane into f16 in a
  reused scratch buffer and a pure f16 GEMM runs over it. Measured on
  an RTX 5090 (MiniMax DiT step): **22.1 s → 14.7 s**, fc1 8.3 → 4.3,
  qkv 4.9 → 2.1. Caching planes per tensor was tried first and is the
  wrong idea — this model would want 38 GB of them.
- **`cortiq animate` runs the tensor cores by default now.** The hold
  that pinned it to the scalar arm is lifted: the kernel carries an
  activation scale, dequantizes once, and the engine's parity probe
  validates that exact path on the file's own weights before a frame is
  drawn. `CMF_COOP=0` still forces the scalar arm.
- **The f16 operands carry an activation scale.** DiT activations run
  past f16's 65504 and that overflow was the NaN this kernel was
  benched for; the host now brings max|x| to ~1000 and the store
  multiplies it back (f16's relative precision is scale-free, so
  nothing is lost). Parity probe on the real weights: 4.65e-3 relative
  rms, the same as the scalar arm.
- **`--peer-split` now moves the boundary of the in-process split too**
  (`run/bench --gpus 2 --peer-split K`), not just the network one —
  the help had promised it for both.
- **The whole video pipeline, end to end: 716.6 s → 251.9 s (2.8×) on
  one RTX 5090.** Denoise step 105 s (host) → 37.6 s, video VAE 201 s →
  91 s, and the frames land within 0.1% of the previous render's — the
  attention move is not a quality trade.
- **MiniMax-H3 attention moved to the device — 41.5% of a denoise step
  became 16.6%.** The kernels were already there (`dit_qk` →
  `dit_softmax` → `dit_pv`, scores staying in device buffers) and
  Lumina's DiT already rode them; MiniMax kept its own host loop, which
  materialized an n×n score plane — 144 MB at render size — and walked
  it across the bus twice per head. Now it repacks q/k/v head-major and
  calls the same path. Measured (512×288, RTX 5090, CMF_MMH3_PROF):
  attention **33.3 s → 8.5 s**, whole DiT step **80.3 s → 51.4 s**.
  `CMF_MMH3_ATTN=cpu` forces the host loop back.
- **The local layer split runs IN-PROCESS now — and it is finally
  free.** `run/bench --gpus N` splits the stack across cards inside one
  process: segment i executes pinned to card i, and the only thing
  crossing a boundary is a hidden vector that never leaves the address
  space. No second process, no socket, no serialization, no dir_hash
  handshake. Measured on 2×RTX 5090 (honest bench, steady decode):
  nanbeige 4.2 **80.9 → 82.4 tok/s (1.02×)**, W2 34.7B **115.4 → 115.7
  (1.00×)** — where the TCP split cost 0.77–0.85×. Output stays
  byte-identical to the single-card run on both. `--peer` remains the
  answer for a model bigger than one HOST.
- **Per-token confidence is computed when it will be shown, not always.**
  It is a softmax over the whole vocabulary (151936 rows on this model)
  on every token; the OpenAI surface never returns it and `run` only
  shows it under `--confidence`/`--trace`. Serving one request went
  109.2 → 114.0 tok/s, and four concurrent 185 → 222.9.
- **`CMF_SLOTS_PER_GPU`: slots per card, default 1 — because the
  measurement said so.** The frame profiler (`CMF_GPU_TS=1`) shows a
  decode dispatch leaves an RTX 5090 mostly idle — 30-70 µs of work per
  layer where the card wants thousands of workgroups — so a second slot
  per card looks like free throughput. It is not: four concurrent
  requests over four slots ran 185 tok/s against 210 over two. They
  contend for the queue and the CPU pool rather than interleaving. The
  knob ships for hardware that disagrees; the default does not guess.
- **`serve --gpus N` picks the right mode by itself.** Replicas need the
  model to fit one card; when it does not, the honest answer is the
  layer split, and the server now says which mode it took and why
  (weights vs the card's measured budget) instead of refusing or
  thrashing N copies.
- **`cortiq serve --gpus N`: N GPU replicas in ONE process, and the
  multi-GPU mode that actually scales.** Each slot holds the whole
  model on its own card and serves whole requests, so N requests decode
  at once. Measured on 2×RTX 5090 (W2 34.7B, 200-token completions):
  one request 115.3 tok/s, two concurrent **218.5 tok/s aggregate
  (1.9×)**, both cards resident at 14.4 GB and busy. Refuses loudly
  when asked for more replicas than there are adapters, or together
  with --peer (replicas and layer split are different answers to
  different problems: throughput vs capacity).
- **The engine addresses GPUs by device, not by process.** wgpu
  contexts are now a registry keyed by adapter index — and since weight
  buffers, KV mirrors and scratch live INSIDE a context, per-device
  contexts give per-device caches for free. A thread-local pin says
  which card the current thread talks to (`gpu::set_current_device` /
  `with_device`), the worker pool carries that pin into its threads
  (a dispatch begun on card 1 no longer finishes on card 0), and the
  server re-pins inside its blocking task — without that last piece two
  "replicas" quietly shared card 0 and measured exactly one card's
  throughput.

### Fixed
- **Prefill was 15× slower than it had to be: the CPU pair walk was
  eating the GPU graph's positions.** `forward_ids` — the path `bench`
  times as "prefill" and `ppl` scores through — used neither of
  generation's two routing guards. It took the batched CPU prefill on
  models whose recurrent state is GPU-resident (a correctness hazard:
  decode then reads buffers the prefill never wrote), and then let the
  CPU pair walk consume every remaining position, 89 ms of host forward
  where the resident token graph needs 7 ms. Both guards now live in
  one predicate used by both entry points. W2 34.7B on an RTX 5090,
  ctx 512: prefill **8.7 → 137.4 tok/s**, decode unchanged at ~105.
  Attention-only models keep the chunked CPU prefill they measured
  faster on (nanbeige 101.6, bonsai-8b 51.6 — unchanged).
- **The same fix on the split's span prefill made capacity mode FREE.**
  `prefill_span_ids`/`prefill_span_hidden` took the batched CPU span on
  models whose state is device-resident, so both sides of a two-GPU
  split paid host prefill and desynced their device state. With the
  predicate applied there too, W2 on 2×RTX 5090: split decode
  **100.6 → 119.96 tok/s — dead level with a single card's 119.8** (was
  0.85×), and TTFT **4569 → 268 ms**. Split output stays byte-identical
  to the single-GPU run.

### Added
- **`cortiq animate` decides host-vs-GPU by a parity probe, not a
  hardcoded verdict.** The wgpu wide-GEMM arm was measured wrong on one
  driver stack (RTX PRO 6000: step-1 rms off, step-2 NaN) and byte-
  healthy on another (2×RTX 5090: the plain f32 arm within 3.5% of the
  host render — legal accumulation drift; the coop arm stays off for
  animate, its f16-operand overflow at packed-sequence scale is
  bisected and unfixed). So
  the pipeline now probes ITS OWN first qkv weight at DiT-scale
  activations (±2000 mixed with ±2) against the host path and takes
  the device only under 1e-2 relative rms — the measured failure was
  ~24%, honest drift ~1e-5, a decade of margin each side.
  CMF_MMH3_GPU=1/0 still forces. On the 5090 the device render cuts a
  denoise step from 105 s to ~60 s with attention STILL on the host —
  the device-resident DiT block is the recorded next lever.
- **An honest multi-GPU benchmark: `cortiq bench --gpus 2` (also
  --peer/--peer-split/--net-dtype).** One untimed warmup generation
  (shader compile, weight upload, cold prefill land there), then three
  measured repeats of ≥256 tokens, median steady decode from
  inter-token stamps, TTFT, cold-prefill line, wire share. The contract
  has teeth: a run where the local segment refused the token graph or
  where weights re-uploaded inside the steady window EXITS NON-ZERO —
  a CPU number can no longer wear a GPU label. Engine grew the
  counters that enforce it (GRAPH_TOK_OK/MISS, gpu::upload_bytes()).
  First honest numbers (W2 34.7B, 2×RTX 5090): single 119.8 tok/s,
  split 101.4 — capacity mode costs 15%, not the 2-10× the
  window-tainted `run` numbers suggested.

### Fixed
- **`--gpus` device pinning could put both processes on one card.** An
  externally set CMF_GPU_ADAPTER was inherited by the worker verbatim;
  now the coordinator keeps its pin and the worker takes a DIFFERENT
  index, both logged. `--peer-split` no longer requires `--peer` (it
  works with --gpus); `--gpus 0/1` is a loud error instead of a silent
  single-GPU run.

- **A span ending ON a loop boundary dropped the boundary norm (wgpu
  graph).** The executor fuses each layer's residual with the NEXT
  layer's input norm, and the last layer of a span took the plain-
  residual branch without consulting `loop_norm_at` — so a network/
  multi-GPU split cut exactly at a Looped-Transformer iteration edge
  (nanbeige's default half: 22 of 44 virtual layers) handed the peer a
  raw, un-normed hidden. The tail then spoke template noise ("М user").
  The builder always promised the boundary norm stays with the span
  ("keeps its boundary norm even when it is the span's own last
  layer"); the executor now honours it: residual + final_norm before
  readback. Found on 2×RTX 5090, reproduced on wgpu-Metal, fixed with
  the split producing the CPU reference's wording.

### Added
- **Local multi-GPU in one flag: `cortiq run model.cmf --gpus 2`.** The
  layer stack splits across two cards via the proven `--peer` path — a
  worker process is spawned on 127.0.0.1 pinned to the second adapter,
  handshakes by dir_hash, and is killed when the run ends (guard on
  drop, no zombie model-sized processes). The wire is loopback, whose
  cost the Thunderbolt-stand numbers already showed is amortised by the
  two processes overlapping their submit bubbles. Exactly two cards in
  v1: the coordinator speaks to ONE worker; N-card chaining is the
  recorded next step, and pretending otherwise would run N−2 cards idle.
- **`CMF_GPU_ADAPTER` pins the wgpu card** — an index into the
  `cortiq gpu` listing (now numbered) or a case-insensitive name
  substring. Without the pin every process on a host gets the same
  "best" adapter, which is precisely wrong for a two-GPU split. Unknown
  values fail loudly with the adapter count instead of silently taking
  the default card.

## [0.5.66] - 2026-08-10

### Added
- **Metal-MoE in the shared command buffer.** MoE layers now live inside
  the whole-token Metal graph: a device-side `moe_topk_select` kernel
  (softmax router, top-k with lower-index tie-break, norm/scale weight
  semantics, gated shared expert — unit oracle against `moe_route`
  including a deliberate bit-equal tie) fills the weight vector and the
  jobs base tables, so routing never leaves the GPU and the per-layer
  commit+wait round trip is gone. Layer types carry `MetalFfn::{Dense,
  Moe}`; `encode_post_moe_ffn` chains addnorm → f32 router → select →
  gate/up jobs (q4tp or the mixed q2tp profile) → fused SiLU → down
  jobs → weighted reduce → residual in ONE encoder. The plan builder
  maps softmax-router MoE with a gated shared expert and refuses
  sigmoid/bias/τ routers, masks, per-expert scales, router-input norm
  and non-SiLU experts to the CPU path unchanged.

### Changed
- **Device-attend policy is decided after the plan scan.** The hd>128
  default-off was measured on dense models; a MoE plan inverts it —
  every CPU-attend sandwich costs a commit+wait. W2 q2tp-MoE 34.7B on
  M4 (steady decode): sandwiched graph 14.7 tok/s, device-attend graph
  **24.8–27.1 tok/s**, pure CPU 18.8 — the graph now beats CPU on MoE,
  which was the point of the spec. Dense models keep the measured
  hd≤128 default; `CMF_GPU_ATTEND=0/off/force/256` still override.
  Parity: with the CPU-attend sandwich the graph's texts match the
  exact-CPU reference 48/48 with confidence drift 0.098 — inside the
  CPU's own i8↔exact envelope (0.11); device-attend drift is the
  pre-existing lever's class, not the MoE branch's.
- **Known limit, measured:** the 17.8 GB 35B-A3B q4tp file exceeds this
  M4's Metal `maxBufferLength` (13.6 GB), so NO Metal weight path (per
  op or graph) can run it single-device; the 12.9 GB W2 fits. Follow-up
  recorded: window the no-copy file buffer to the layer span (the same
  windowing the network split wants).

### Fixed
- **Streaming swallowed short no-think replies whole.** With
  `enable_thinking=false` the chat template prefills an EMPTY think
  block, so the reply never emits `</think>` — and the SSE think-filter
  buffered forever waiting for one; any answer under its 100-char
  escape hatch shipped zero content chunks (usage and finish arrived,
  the words did not). The filter buffer is now shared with the
  post-generation path and flushed by the filter's own rules: a buffer
  that never opened `<think>` IS the answer, a closed block ships its
  tail, an unterminated block stays private. Found while validating
  `serve --peer` streaming; applies to every serve, split or not.

### Added
- **Network pipeline-split (v1): a second machine's compute joins a
  generation** — `cortiq worker model.cmf --listen IP:9911` holds a layer
  span of the model; `cortiq run … --peer IP:9911 [--peer-split N]` runs
  layers `[0..N)` + embed/head/sampler locally and ships one boundary
  hidden per token (prefill batches positions per frame, one round trip
  per chunk). New crate `cortiq-net`: length-prefixed bincode frames,
  wire version checked at handshake, worker as a LIBRARY entry
  (`worker_serve`) so app-packaged platforms (iOS) can host one. The
  worker proves it holds the same model by `dir_hash` — a mismatch is
  refused with both hashes named; beyond loopback `--token` is required.
  The `.cmf` container is untouched: the split is a property of the run,
  never of the file. Engine side: `forward_layers_upto` generalized to a
  span `[from..=upto]` (delegation, zero behavior change at `from=0`),
  pub building blocks `embed_id` / `forward_span` / `logits_from_hidden`
  / `sample_next` / `reset_session`, and `split_supported()` refusing
  the un-cuttable stacks (DSV4, Gemma-3n, O(1) attention) loudly.
  Measured (bonsai-1.7b-q1, greedy, 48 tok, byte-identical output in
  EVERY config): loopback split 82.5 vs 83.1 tok/s single-process; over
  a real Thunderbolt bridge to an M1 worker — 34.5 tok/s at half the
  stack remote (15.96 ms/token remote+wire), 44.1 tok/s at one layer
  remote. A split of a model that already fits locally onto a slower
  peer is a net LOSS, as the roadmap predicts — pipeline split buys
  capacity, not single-stream speed. And the capacity claim is now
  MEASURED at scale: Qwen3.6-27B q4tp (13.3 GB, 64 layers) on a
  24 GB + 16 GB Mac pair answers correctly at ~1.3–1.6 tok/s warm
  (split 40, CPU worker) where the 24 GB machine alone thrashes at
  0.4–0.9 tok/s and the 16 GB one cannot run it at all. Two pinned
  operational facts: a memory-tight worker must run `CMF_GPU=0`
  (Metal's buffer footprint on top of a near-full span collapses it —
  11 s/token at 6.4 GB span), and the M1's practical resident span
  ceiling is ~4.8 GB regardless of nominal free RAM — plan spans to
  the measured ceiling, not the spec sheet.
- **Wire v2: the protocol stopped costing more than the wire.** v1 shipped
  a frame as TWO write syscalls — under TCP_NODELAY the 4-byte length
  prefix went out as its own segment, four packets per round trip. v2:
  one buffer, one write; per-token frames (Step/Hidden/Prefill) are raw
  little-endian instead of bincode; float payloads reuse scratch (the
  hot path does not allocate); optional f16 payload negotiated
  EXPLICITLY in Assign (`run --net-dtype f16`, default stays bit-exact
  f32); a bounded busy-poll before blocking reads (`CMF_NET_SPIN` µs,
  default 3000, 0 disables) — the wakeup from a cold blocking read on a
  power-managed core costs more than the Thunderbolt wire, and the
  no-spin A/B drops 77.7 → 50.8 tok/s at the same RTT. Measured on the
  same stand, same prompt, output identical in every config INCLUDING
  f16: Thunderbolt split-14 34.5 → 53.6 tok/s (+55%; the last chunk
  from `CMF_GPU=0` on the worker — per-op Metal submits on a partial
  stack lose to the NEON q1 path on M1, 13.0 vs 9.0 ms for 14 layers —
  the honest fix is the per-segment graph, roadmap Phase 2), split-27
  44.1 → 77.7 tok/s (+76%), wire cost per token 4.33 → 3.24 ms;
  loopback split-14 82.5 → 99.8 tok/s — ABOVE the 83.1 single-process
  baseline (two processes overlap their submit/readback bubbles).
- **Wire v3: pipelined prefill.** Prefill frames are fire-and-forget and
  a `Sync` barrier fetches the last boundary hidden — the coordinator
  computes chunk k+1 while the worker chews chunk k (prefill wall ≈ max
  of the sides, not their sum); decode round-trip stats no longer mix
  prefill in. Validated on the LOOPED Nanbeige-4.2 (22 phys × 2 loops =
  44 virtual layers): split at the loop boundary, mid-loop-1 and
  mid-loop-2 all match the single-process output token-for-token —
  bit-exact when both sides run the same device kind (CPU⇔CPU); a
  mixed CPU/Metal split diverges within normal GPU-vs-CPU numerics,
  same as any `CMF_GPU` toggle (ppl's CPU-reference rule applies).
  Worker device recipe is PER-QUANT on the M1: q1 runs faster on NEON
  (`CMF_GPU=0`, 9.0 vs 13.0 ms), q4t runs faster on Metal (24.9 vs
  37.9 ms per token for its span). And the honest headline: a 4.17B
  that FITS locally decodes 16 tok/s local vs 6.1 tok/s best-split to
  a 2× slower peer — pipeline split buys capacity, not single-stream
  speed, exactly as the roadmap says.
- **Batched span prefill.** `prefill_batch_masked` generalized to a layer
  span `[from..upto)` over ids OR ready boundary hiddens (pure
  delegation at full range — local behavior bit-identical, baselines
  re-verified); the macOS chunk graph takes a hard cap so a device run
  never crosses the split boundary. Both split sides now prefill through
  the same layer-major GEMM/graph machinery as a local run: Nanbeige
  42-position prefill on the stand 2.3 s → 0.47 s, 138 positions →
  1.07 s (~5× vs the per-position walk), best-split decode 6.1 →
  11.1 tok/s. Found and pinned along the way: PREFILL PANEL WIDTH IS
  PART OF THE GENERATION — a different `CMF_PREFILL_CHUNK` reorders
  GEMM accumulation and can legitimately flip a greedy argmax (measured
  on the same CPU with chunk 64 vs 512). `pipeline::prefill_chunk()` is
  now pub and the network split chunks EXACTLY like the local path, so
  a split run reproduces the local generation bit-for-bit when both
  sides run the same device kind.
- **Cross-turn KV reuse over the wire.** `generate_split` mirrors
  `generate_from_ids`: when a chat turn strictly EXTENDS the forwarded
  history, only the tail is prefilled and the worker's KV rides along
  untouched (no Reset frame) — turn latency stays proportional to the
  new text, not the session. `CMF_KV_REUSE=0` disables, a cancelled
  prefill poisons the history, and the net line now reports it
  honestly: `prefill 1224 ms (37 of 197 pos, 160 reused)` — measured
  on the stand, a full 197-position re-prefill would have cost ~3.5 s.
  Same trigger as local reuse: the re-rendered history must retokenize
  to the exact forwarded ids — a reply ending in a truncated multibyte
  sequence blocks it (observed on Nanbeige's byte-token tails).
  Pinned, not yet chased: a short TAIL prefill runs at ~2× the
  per-position cost of a long fresh one (small-batch attend arms below
  their GEMM thresholds + deeper KV) — a constant factor the linear
  history saving dwarfs; belongs to the per-segment-graph phase.
- **Layer-skip speculation measured DEAD on present models** — recorded
  so nobody builds it. New tool `spec-probe` (cortiq-net bin):
  teacher-forced acceptance of argmax(final-norm + lm_head over the
  hidden after L layers) against the full model's own greedy token.
  Nanbeige-4.2 (looped, 44 virtual): 0% at every depth except exactly
  the loop-1 boundary (9.4% at layer 22) — mid-loop hiddens live in a
  different representation space than the head reads. Bonsai (plain
  qwen3, 28L): ≤1.6% until layer 23, and only 32.8% at 82% depth,
  where a draft saves nothing. Network speculative decode therefore
  needs an MTP-carrying arch or a companion draft model — not an
  early-exit head. Ten minutes of measurement, days of machinery not
  built.
- **First MoE over the wire.** Qwen3.6-35B-A3B q4tp (18.7 GB) across
  the Thunderbolt pair: split 20/20, f16, CPU worker — 12.2–12.4 tok/s
  warm, remote 32 ms/token, byte-identical to the local CPU run. That
  MATCHES the box's own best (11.6–12.1 tok/s local CPU) while halving
  each side's hot set — MoE is the split's natural cargo: per-token
  remote compute is the routed experts only, so the wire's fixed cost
  stops dominating. Same file's local numbers double as the honest
  llama.cpp comparison base for the "20 tok/s" claim.
- **Model open no longer blocks on a whole-file readahead.** `open`
  issued one `madvise(WillNeed)` over the entire mapping; macOS runs
  that SYNCHRONOUSLY, so a 12.9 GB MoE file held open() for ~6.5 s
  while prefetching experts a routed decode never touches. The advise
  is now capped at 4 GiB: dense small/mid files keep the readahead
  (every weight gets touched anyway), big files — the MoE class above
  all — rely on demand paging. `CMF_MMAP_ADVISE=1` forces the old
  blanket advise, `=0` still disables. Measured: W2 open 6.5 s →
  0.49 s cold / 0.01 s warm, `run -n 2` 7.6 → 3.4 s, chat turns
  4.0/5.0 → 4.8/5.5 tok/s; nanbeige (2.2 GB, under the cap) keeps its
  readahead and its 15-16 tok/s.
- **MoE prefill: parallelism inverted over experts.** The batched
  prefill already grouped positions by expert, but each tiny panel
  (a few positions of a 512-inter expert) went through `dense_ffn_batch`
  with the POOL — thousands of barrier-priced dispatches per chunk.
  Workers now take WHOLE experts (serial math inside, zero inner
  barriers), results land in per-expert panels, and one deterministic
  scatter in expert order reproduces the serial accumulation exactly;
  small batches keep the old path. W2 sustained decode+prefill rose
  8.1–10.1 tok/s at n=64 (from 7.7–7.9), chat turns 4.0/5.0/3.1
  (from 3.5/4.7/2.9), output byte-identical. Engine suite 152/152.
- **Metal grew its 2-bit MoE kernel** — the gap behind "GPU no faster
  than CPU on the W2": the experts (the bulk of a 2-bit MoE's compute)
  never reached Metal at all. New MSL `q2tp_matvec_jobs` (twin of the
  q4tp jobs kernel: same ladder planes and 5-bit rung codes; 8-byte
  weight chunks where a uint carries SIXTEEN 2-bit fields, and rung 0
  of the ladder is an EXACT ZERO so pruned groups stay silent), the
  jobs pipeline flips only stage 1 (gate/up) — silu, the q4tp down and
  the weighted sum are dtype-blind. The mixed trio now flows end to
  end: `MoeJob.gu_q2`, a Q2TiledP arm in `moe_parts`, trio validation
  demanding a plain q4tp down under 2-bit gate/up, and an honest
  refusal in the wgpu per-op moe_block (its 2-bit lanes live in the
  graphs). Measured on the M4: the CPU still wins (5.9 vs 2.8 tok/s
  forced-GPU — 30 synchronous per-layer submits) and the Batch/Ffn
  probes keep it, exactly as designed; the kernel is for GPU-strong
  devices, and folding the MoE block into the token command buffer is
  the recorded follow-up, together with a device-vs-dequant unit
  oracle for strict bit-level kernel verification.
- **q2tp CPU decode: the scalar holdout joined the i8 family.**
  `dot_q2tp_row_i8` — the half-integer grid (c − 1.5) becomes exact
  integer math through group sums, NEON unpacks the four 2-bit fields
  by shift+mask against a vld4-deinterleaved activation vector
  (widening MACs, exact i32); the a8w8 branch landed in `q2tp_matvec`,
  the fused `matvec_silu_mul` arm covers 2-bit gate/up pairs, and
  `moe_gate_up_many` accepts the q2tp expert profile (coverage gap #12
  for this class) — the per-expert pool-barrier storm of a 2-bit MoE
  collapses into two dispatches per layer. Grid-exact unit test pins
  i8 == exact scalar; W2 output byte-identical. Profile after: the
  2-bit dot fell from 58% of decode compute to a co-equal share with
  the q4tp ladder decode; measured sustained W2 decode on the M4 rose
  to 7.7–7.9 tok/s at n=64 (3 runs), with per-turn arithmetic putting
  steady decode near ~15-20 tok/s and the remaining wall in PREFILL —
  the MoE prefill still walks experts per position, that is the next
  slice, together with the promised Metal q2tp expert kernel (MSL twin
  of `q4tp_matvec`, same ladder planes, 2-bit unpack) + the mixed-trio
  `MoeJob` so Metal finally accelerates 2-bit MoE.
- **Coverage gap #6 CLOSED: the q2tp-MoE mixed profile reaches the wgpu
  batch/prefill graph** — validated against a real parity oracle
  (Qwen3.6-35B-A3B-Escha-W2, 34.7B, 2-bit experts): the builder gained
  the token graph's q2tp ladder (q4t → q2tp gate/up over q4tp down →
  q4tp, uniform per layer), `gu_q2` now flows through `BFfn::Moe` into
  both executor lanes; the per-position lane dispatches
  `moe_gate_up_q2tp` + the EXACT `moe_down_q4tp` (the DSV4-chain pair),
  the GU uniform stride comes from `expected_nbytes(Q2TiledP)`, and the
  whole-chunk q4tp fast lane now EXCLUDES `gu_q2` explicitly — the
  mixed profile sets `q4tp=true`, so without `!gu_q2` that lane would
  have ground q2tp bytes through the q4tp kernel into silent garbage.
  CPU⇔wgpu long-prompt output is byte-identical, zero batch-graph
  refusals under RUST_LOG, engine suite 151/151, bonsai/nanbeige
  baselines untouched. Deliberately NOT wired: `bt_moe_gate_up_q2tp`
  (whole-chunk 2-bit lane stays a future optimization — the
  per-position lane lives in the same submit) and `moe_down_q2tp_b`
  (a DSpark draft-grade kernel, excluded from the exact path by its
  own contract).
- **Coverage gap #6 (q2tp MoE batch) rediagnosed, wiring deliberately
  NOT done (superseded above the same day).** The batch q2tp kernels (`bt_moe_gate_up_q2tp(_r4)`,
  `moe_down_q2tp_b`) are registered but dispatched NOWHERE: the batch
  executor's whole-chunk lane hardcodes the q4tp pipeline and the
  per-position lane branches on `q4tp` only — q2tp bytes under a q4tp
  kernel would be silent garbage, which is exactly why the builder's
  `gu_q2: false` stays. The wire needs the builder plus BOTH executor
  lanes with their own layouts/strides, and it cannot be validated
  blind: the stand has no MoE checkpoint at all. Prerequisite recorded
  in QUANT_COVERAGE.ru.md — convert an MoE model (Qwen3-30B-A3B also
  settles the llama.cpp comparison), then wire with parity.
- **Coverage gaps #5 and #8-q8 closed.** `gpu_batch_job` grew Q4Tiled /
  Q4TiledP arms — the attention QKV batch can now reach the Metal batch
  kernels the GDN projection path already uses; the existing
  OpClass::Batch probe arbitrates, and on both stand machines it
  honestly keeps the CPU (measured: no regression at 15.3–15.9 and
  8.6–8.7 tok/s) — the route matters where the batch amortizes its
  submit. The dense-FFN fuse grew a Q8Row arm on a new portable
  `q8_row_dot` (SDOT/AVX2 fast arms + an exact scalar oracle, unit test
  pins fast==scalar; a q8 checkpoint bench is pending — none on the
  stand). Q8_2f is excluded from the fuse ON PURPOSE: its column field
  prescales activations per tensor, breaking the shared split_act
  contract. Engine lib suite: 151/151.
- **Coverage gaps #2 and #8 closed** (from QUANT_COVERAGE.ru.md).
  Q4Tiled decode matvec got its missing GPU dispatcher — `gpu::q4t_matvec`
  facade + the probe arm in `QTensor::matvec`, same shape as the q4tp
  twin (Metal enters via the existing kernel; wgpu stays an honest
  refusal until a discrete-GPU q4t model reaches the bench; on M4 the
  probe keeps the CPU head, as it does for q4tp — the route matters
  where the GPU wins). And the q1 dense FFN got its fused
  `matvec_silu_mul` arm — one row pass over both sign streams with
  shared activation group-sums instead of two dispatches + a combine
  loop: the bonsai-recipe M1 worker span dropped 9.03 → 7.5–7.9
  ms/token (−14%), CPU output byte-identical to the reference.
- **Quant-coverage audit: every dtype × every execution path**, written
  down as `docs/QUANT_COVERAGE.ru.md` — 17 dtypes against 10 paths with
  file:line evidence, 13 open "kernel exists, caller filters it out"
  gaps and the honest no-kernel list. Two findings overturned this
  session's own working theory: the Metal block graph is NOT q1-only —
  `proj_abs` already encodes Q1/Q1T/Q4Block/Q4Tiled/Q4TiledP/Q8Row/
  Q8_2f and the graph executes the whole GDN-hybrid 27B (the `q1_*`
  names are historical); and the decode brake on hd=256 archs is the
  device-attend gate, not the kernels — `CMF_GPU_ATTEND=256` cut the
  27B coordinator span 159 → 108 ms/token (−35%) on the stand, taking
  the 24+16 GB pair to ~3.5 tok/s steady decode (from 0.4 at the start
  of the investigation). Fixed along the way: `q1_parts` advertised
  Q2TiledP to a backend with no q2tp kernel, silently truncating the
  block plan at the first q2tp layer; the graph's break reasons now
  print under the existing `CMF_GRAPH_DBG`.
- **q4tp prefill panels reach the Metal GPU.** `dense_ffn_batch` wired
  the fused on-device SwiGLU only for q4_tiled; a q4tp model's prefill
  FFN silently stayed on the CPU. The q4tp twin now rides
  `gpu::q4tp_ffn` — the exact kernel the image DiT has run in
  production. (On the GDN-hybrid 27B the prefill floor is the
  sequential GDN recurrence, so the win there is small; full-attention
  q4tp models get the full panel offload.) Pinned for the next Phase-2
  slice, with the decode floor measured: a GDN layer's q4tp
  projections run on CPU (~4-5 ms/layer incl. the scales_into ladder
  decode) because the Metal token graph's weight extraction is
  q1-only (`q1_parts()` throughout `q1_graph_gpu`), while the
  ProjKind::Q4tp ENCODER already exists — the extension is typed
  weights in GdnGpuLayer/AttnGpuLayer, not new shaders.
- **Per-segment wgpu token graph (roadmap Phase 2, first cut).** The
  whole-token wgpu graph — the discrete-GPU fast path (4090 class:
  76 → 137 tok/s) — generalized from all-or-nothing to a layer span
  `[from..=upto]`: a split coordinator and worker each run their whole
  SEGMENT in one submit per token instead of falling to the per-op
  path. `forward_token_graph` takes a `layer_base` so span and
  full-stack runs address the same per-layer KV/GDN mirrors; lm_head
  folds in only when the span reaches the last layer; looped-model
  boundary norms go span-relative (a span ending mid-stack keeps its
  boundary norm). Span runs skip the graph RACE (its state is global
  and calibrated on full stacks) and engage only where the graph is
  trusted by default. Verified on loopback with `CMF_GPU=wgpu`
  (bonsai): both sides build exactly their segment (`graph L1..L13` /
  `graph L14..L27` under CMF_GRAPH_DEBUG), output byte-identical to
  the full-graph baseline; the default Metal path is untouched
  (`wgpu_graph_default` already requires the wgpu backend). The
  Metal-native q4t decode graph remains the next Phase-2 slice.
- **Skills, task masks and O(1) attention ride the network split**
  (wire v4). `Assign` carries the session spec — skill overlay id,
  task-mask name, O(1) config — and the worker mirrors it over its own
  layers: reloads its pipeline with the same skill (same file, verified
  by dir_hash, so the overlay resolves to identical bytes), resolves
  the mask from ITS catalog and masks the layers IT runs, applies the
  same `O1Cfg` and runs the o1 lifecycle over its span (begin at
  Reset, collect through prefill, seal at the Sync barrier — exactly
  where the local path seals). Unknown skill / unknown task / bad o1
  spec are loud Ack refusals; `--blend`, `--route-dynamic` and
  `--state` stay refused (they desync by construction). Validated with
  a new parity oracle — `mkskill-test` writes a `skill.test` whose
  tensors are byte copies of the layer-0 FFN, so `--skill test` MUST
  equal the backbone: local backbone == local skill == split skill,
  byte-for-byte (CPU⇔CPU), and o1 local == o1 split with the
  approximation active. Measured on the Thunderbolt stand at a
  ~1000-token context: `--o1 all` cuts the worker's per-token cost
  14.44 → 9.73 ms (−33%), the O(1)-vs-O(n) gap growing with context.
- **`serve --peer`: the OpenAI API rides the network split.** Same
  flags as `run --peer` (`--peer-split`, `--net-token`, `--net-dtype`);
  the worker is dialed and verified (dir_hash, geometry, wire version)
  BEFORE the listener starts, so a bad peer fails the whole serve
  loudly. Peer mode forces one slot — one worker holds one KV session —
  and a request carrying a task mask is refused with a clear message
  rather than half-masked. SSE streaming, usage accounting, session
  turnover and the cross-turn KV reuse all flow through unchanged;
  per-generation net stats land in the server log via tracing.
  Speculation,
  task masks, skills and multi-worker plans stay refused-not-degraded —
  they return per MULTI_DEVICE_ROADMAP.ru.md.

## [0.5.65] - 2026-08-10

### Fixed
- **Masked GENERATION on a looped model produced garbage** — the first
  decode with an active per-visit mask ever run. Three layers to it:
  `layer_alive()` defaulted an ABSENT gate to dead, the writer recorded
  gates per physical layer, and the decode loop asks by the VIRTUAL
  index — so the entire second pass was silently skipped as "dead
  layers" (batched scoring never consults gates, which is why every
  scorer said the file was fine). Absent gates now read alive, the
  codec replicates gates to every visit exactly like the FFN rows, and
  the writer records them per visit. Pinned in `loop_masks.rs`.
- **Masked decode rides the same activation-zeroing arm as the batched
  sweep** (b=1) — the per-dtype zoo of sparse decode arms diverged on
  q8_row trained masters; one contract, one implementation. Masked
  generation is coherent at normal decode speed now.
- **The scalar dx arm accepts only matvec-sized grids** (≤2^16
  workgroups without cooperative matrices): it has no probe arbitrating
  it, and on macOS it shadowed the Accelerate BLAS arm below — a
  phase-B dx ground a bake to 12% CPU for an hour.

## [0.5.64] - 2026-08-09

### Added
- **Masked inference is a fast path.** A task mask used to eject
  scoring and generation into a per-position path that dequantized
  three FFN matrices per call (a masked gate ran hours). The mask now
  lands on the ACTIVATIONS between the fused halves — arithmetically
  the pruned network, zero quantized bytes touched — and rides the
  batched sweep: scoring 75 s where it took an hour, generation with an
  active skill mask batched too, the bake gate scores specialists
  masked (the way they serve). Faithfulness proven by a decomposition
  probe (`skillbake::replica_score_file_mask`): runtime masked within
  0.8% of the replica's f32 math on the same written file.
- **The bake's attention forward is one device chain** (tensor-core
  qkv/wo around split+qk-norm+RoPE, causal softmax·V and the output
  gate on device; RoPE from a host-precomputed f64 table). Parity vs
  the host reference 7e-4; Nyström layers and strict-f32 phase A stay
  host by design.

### Fixed
- **Trained FCD masters write as 8-bit rows.** Requantizing freshly
  trained masters to q4 erased the fine co-adaptation with the mask
  (+7.9% held-PPL, enough to turn the mask into a net loss). With
  Q8Row masters the coding specialist's masked runtime gate went from
  +7.7% to **−1.1% vs its backbone** (replica 9.515, runtime 9.633 —
  the requant now costs 1.2%).
- **Matvec-style GPU kernels refuse billion-thread grids.** Phase B's
  dw shapes on a device without cooperative matrices dispatched
  3072×21504 workgroups — minutes per call, a bake that looked hung.
  Grid products past 2^22 go to the CPU outright.
- The Metal context banner logs once, not per operation (a bake left
  47k copies in its log).

## [0.5.63] - 2026-08-09

### Added
- **Standalone skill files: bake once, share the delta.**
  `cortiq skill export <specialist> --base <base> -o x.skill.cmf` cuts
  a baked specialist against its base by the directory's per-tensor
  hashes into a `.cmf` carrying only the changed tensors, the mask
  catalog, and IDENTITY KEYS (`base_dir_hash` — the exact bytes the
  delta was cut against, `base_arch`, `task`, `provenance`).
  `cortiq skill apply <base> <skill> -o out.cmf` verifies the key,
  refuses a stranger base (`--force` overrides), and reproduces the
  specialist byte-equivalently. New feature bit `SKILL_FILE` (6): old
  readers refuse the file loudly; the runtime refuses to RUN one and
  says to apply it. Measured: the Nanbeige graphics specialist
  (2364 MB) travels as a 557 MB skill; applied PPL identical to the
  last digit. Spec §9.1; docs/SKILLS.md.

### Fixed
- **Phase A trains in strict f32 — f16 rounding was mis-selecting
  neurons.** The mask SELECTS by the ordering of tiny gradient values,
  and f16 operand rounding on the tensor-core arms reordered that tail:
  hard-PPL 5.207 vs the reference 4.293 at the SAME 2.56% sparsity,
  while an f32 run retraces the reference to the third decimal. The
  bake's training steps now decline the device arms (forward/eval
  sweeps keep tensor cores — their PPL matches f32 at print
  precision). Verified end-to-end: phase A bottom 4.225@4.40% (ref
  4.223), FCD 3.929 on the replica, runtime gate −1.5% vs backbone.

### Changed
- **The bake's GEMMs run on tensor cores** (`gemm_nt_coop` /
  `gemm_nn_coop`, wgpu cooperative matrices — f16 operands, f32
  accumulator, parity vs CPU reference ~1e-3). The vocabulary head
  (166k rows) reaches the card for the first time; the probe judges
  warm calls only (weight uploads are one-off, not steady state);
  activation/staging buffers are pooled; mapped readbacks above 4 MB
  copy in parallel; the weight cache's content check is sampled.
  Measured on a rented RTX 5090, 3-step phase A: 455 s (CPU fallback)
  → 200 s, GEMM 233 → 44 ms/call, backward 52.5 → 12.8 s — with
  baseline/mask/FCD pinned at 4.187 and the runtime gate at +0.0%
  at every step. Devices without cooperative matrices (Apple via
  wgpu) keep the scalar arm bit-for-bit.
- **The bake runs as device chains** — frozen FFN forward (gate+up →
  silu·mask → down, one submit) and its backward riding per-layer
  planes parked on the card; a finished bake releases its VRAM before
  the runtime gate. The weight cache admits only matrices that RECUR
  (fingerprint probation), so phase B's per-step activation planes can
  no longer grow it past the card — the failure mode that silently
  killed a 90-step phase B at step ~40.

### Fixed
- **A GPU cache hit by host address now proves its bytes.** Both
  backends parked constant vectors (norms, row scales, sinks, bake
  weights) on the card keyed by `(pointer, length)` — but an address is
  not an identity: a reloaded model's mmap lands where the dropped
  one's was, and the streaming replica re-dequantizes evicted layers
  into recycled Vecs. Scoring model B after model A in one process
  served A's norms to B wherever the mappings overlapped — an intact
  specialist file measured PPL in the millions while a fresh process
  measured 5.1. Every pointer-keyed entry now carries a sampled content
  fingerprint (~µs even on a 126 MB matrix) and is refreshed **in
  place** on mismatch, so cached bind groups stay coherent. Found by
  the bake gate the moment it was made fast enough to finish.
- **`skill bake`'s runtime gate scores the specialist bare, and says
  so.** The per-position masked scorer was an hours-long no-op on
  quantized storage (the FFN mask application doesn't reach fused
  quantized kernels yet); the gate now takes the fast batched path and
  prints that masked quality is the replica's mask+FCD figure.

## [0.5.62] - 2026-08-08

### Added
- **Per-visit FFN masks — a Looped Transformer can finally be masked**
  (format feature bit `LOOP_MASKS`). The runtime indexes task masks by
  the VIRTUAL layer, exactly as it indexes KV, but the mask area stored
  one row per physical layer: on a looped model every row past the
  first pass was missing and the sparse path silently zeroed the entire
  second pass's FFN. Rows are now stored per virtual layer
  (pass-major); legacy masks replicate to every pass at decode;
  unlooped files are byte-identical. Measured on Nanbeige 4.2: a shared
  mask closing 24 of 236 544 neurons cost ×32 perplexity; per-visit
  masks close ~21 000 visit-neurons at the denoising bottom for 0.9%,
  and FCD on top lands the specialist BELOW the untouched baseline
  (4.069 vs 4.187 held-out).
- **Tool calling in `cortiq serve`** — the OpenAI protocol end to end:
  `tools`/`tool_choice` in, the FILE's own chat template renders them
  (its `{%- if tools %}` branch had been waiting since the first
  convert), `<tool_call>` blocks parsed out — JSON and Nanbeige's XML
  grammar both — plus the strict bare-JSON fallback small models need,
  streaming with marker holdback and an indexed `tool_calls` delta,
  `finish_reason: "tool_calls"`, nullable `content`, `role: "tool"`
  history. Proven live on converted Qwen3-0.6B and Nanbeige 4.2: call,
  tool-result round trip, stream. Both real templates are test
  fixtures now.
- **`CMF_TASK=off`** starts a mask-carrying file bare — full-speed
  paths, no default task, honest `Sparsity: 0%`.
- **`--calib-chunks`** on `skill bake`: the corpus cap (112 chunks,
  silently truncating in file order) becomes a knob.

### Fixed
- **The bake now works on Looped Transformers.** Three compounding
  defects: the mask init (σ(2.0) per VISIT squares to 0.776 over two
  passes — baseline 4.187 read 278.4 at step 30 with nothing pruned),
  the unnormalised per-visit gradient accumulation (one step meant
  `loops` steps), and `pruned={:.0}%` printing a flat 0% for anything
  under half a percent. Init is solved per loop depth, steps are
  normalised per phase, five closed-form invariants pin all of it.
- **A mask-carrying file no longer pays for masks it is not using**:
  the loader forced whole-model f32 (16.7 GB for a 2.4 GB file, swap on
  a laptop) for ANY mask; f32 is now forced only by actual head
  restrictions. The specialist writer emitted all-zero head rows ("no
  active heads"); `examples/fix_head_masks` repairs written files.
  Measured: 0.2 → 32 tok/s.
- **Chat templates stopped failing silently**: minijinja gains the
  `json` feature (`tojson` — without it every tools branch errored into
  a TOOLLESS ChatML fallback), transformers' `visible_text()` helper is
  provided, `tool_call_format` is pinned to the JSON grammar in every
  render arm, and string `arguments` normalise to objects for templates
  that iterate them.

### Performance
- **Skill-bake GEMM submits cut ×4** (~4200 → 1040 on the 5-step
  probe), each an exact-equivalence change: the held set scores in one
  batched pass instead of twelve; QKV and gate+up fuse to one submit in
  the forward; their gradients fuse in the backward; the lm-head scores
  in 64-position blocks. In-process phase and per-shape profilers keep
  the accounting honest — the dispatch tax (~20 ms per submit on the
  stand's Vulkan stack, 5-10× the multiply it carries) is now a named,
  measured quantity. Projected on the profiled stand: ~26 s/step → ~10.

## [0.5.61] - 2026-08-07

### Fixed
- **A software rasteriser was being accepted as a GPU.** Mesa ships
  lavapipe/llvmpipe in most container images — Hugging Face Spaces, CI
  runners, cloud VMs — and with no real card present wgpu hands it to
  `request_adapter` as the best available. `force_fallback_adapter: false`
  means "do not PREFER a fallback", not "refuse one", so the engine took
  it, logged `wgpu GPU path: on (llvmpipe (LLVM 19.1.7) / Vulkan)` and ran
  every shader through an LLVM rasteriser on the same cores its native
  kernels were already using: the same silicon plus an emulation layer,
  and slower than doing nothing.

  Caught on a CPU Space, where a 256x256 Lumina render announced a GPU
  that did not exist. `init()` now declines a `DeviceType::Cpu` adapter
  and returns Err, which the caller already treats as a clean CPU
  fallback. `CMF_GPU_SOFTWARE=1` takes it anyway — checking a shader
  against a reference rasteriser is the one job it is good at.

  The same trap sits on ZeroGPU: a probe there found eight NVIDIA devices
  present but the ICD directory holding only Mesa drivers, no
  `nvidia_icd.json` and `NVIDIA_DRIVER_CAPABILITIES` unset. A card cortiq
  cannot enumerate through Vulkan is a card it cannot use, whatever CUDA
  reports.

## [0.5.60] - 2026-08-07

Keyframe-to-video on the published MiniMax-H3 file, and a two-bit build
measured beside it.

### Added
- **The video VAE's encoder, one temporal tap of it.** `fl2va`
  conditions on single FRAMES, and a causal 3-D convolution fed one
  frame pads its front with zeros — the reference's own `autopad`
  therefore trims the kernel to `weight[:, :, -T:]`, which at T = 1 is
  the last tap and nothing else. `animate-pack` stores that tap alone
  and the runtime runs plain 2-D convolutions over it: a third of the
  encoder's bytes for the same numbers, matched to max 2.4e-6 on a
  signal of rms 0.67. Encoding real video (`ref2va`) needs the other
  two taps back.

- **Qwen3-VL's vision tower** — the other half of what a keyframe needs,
  since the presentation puts the picture in the TEXT stream
  (`"<Picture 1>: "` and a vision block) alongside the latent the DiT
  conditions on. 27 ViT blocks at hidden 1152, a merger down to the
  LM's 5120, three deepstack mergers.

  Three of its conventions pass any single-patch check and fail on a
  real image, so the fixture uses a grid that exercises all three: the
  48×48 position table is read BILINEARLY and then permuted into 2×2
  merge blocks; the patches arrive in that same block order rather than
  row-major (the reference's `permute(0, 3, 6, 4, 7, …)`); and the
  blocks' MLP wants the tanh GELU while both mergers want the exact
  one. Preprocessing matches bit for bit, the merged tokens to 6.2e-6
  on a signal of rms 1.79, both deepstack outputs the same.

- **`fl2va`'s conditioning, and the DiT agrees with the reference on
  it.** Keyframe condition rows sit between the text and the audio,
  share the TARGET spatial grid, and are pinned to the time coordinate
  of the frame they stand for — the first at the text's end, the last a
  whole clip further on minus one span. They never advance the cursor,
  so the audio and video streams still start where they would have.
  They also get a timestep of their own near 1: they are conditions,
  not noise being removed. Measured against the reference at
  7.1e-5 (video) and 4.9e-5 (audio), beside t2va's 8.7e-5.

  That timestep is the noise-augmentation figure and not a separate
  constant — the reference blends `aug` of the latent with `1 − aug` of
  noise and then tells the block the row sits at `aug` — so it is a
  field here rather than a literal. The blend's 0.1% comes from a
  torch-seeded stream this port does not reproduce; ours is its own,
  and the parity gate sets `aug = 1` so everything else compares
  exactly.

- **`cortiq animate --first-frame / --last-frame`** — the pipeline
  around all of the above. A picture conditions the run twice, from one
  read: the VAE encoder turns it into the latent the DiT holds in a
  condition row, and the vision tower turns it into the tokens the
  prompt carries as `"<Picture 1>: "` and a vision block. The first
  frame is a geometry anchor and is stretched to the canvas; the last
  one follows and is cover-cropped, which is what the reference node
  does with each.

  Pictures come in as binary P6 PPM. A PNG or JPEG decoder is a few
  hundred lines of table-driven code for something every tool on the
  machine can already write, and this binary earns its "nothing to
  install" by not carrying code it does not need.

  `tools/mmh3_fl2va_pack.sh` adds the two stacks to an existing file
  rather than repacking it: the tower comes out of the 51.5 GB prompt
  encoder by ranged read — 351 tensors, 1.19 GB — and the encoder out of
  the video VAE, so the whole upgrade is ~7 GB of transfer and about
  half a gigabyte of growth.

- **`ref2va`'s layout**, which is the part of it that is arithmetic. A
  reference block ADVANCES the cursor where a keyframe does not — an
  image by one, standalone audio by its length, a clip by the greater
  of its soundtrack and its frame spans — so every later stream's time
  coordinate moves with it. A clip's soundtrack packs immediately
  before its frames from the same origin and takes its stereo w
  extremes from the block's own grid rather than the target's. Compared
  against the reference's `PackedLayout` row by row: 164 rows, all
  three coordinates, worst difference 0.000e0, and the same eight
  segments at the same bounds. Reference rows also get their own
  timestep — visual conditions at 0.999, audio at 1.0, which is not
  noised at all.

  What `ref2va` still needs to RUN is two encoders: the video one's
  other two temporal taps, which a single frame cannot reach, and the
  audio VAE's encoder, which is not packed at all.

  The parity harness for all of this now runs on an ordinary host, with
  no GPU and no stand. The DiT's reference needs a CUDA-only fused
  rms+rope kernel, so `tools/mk_mmh3_toy.py` carries a torch stand-in
  for it — checked rather than trusted: the t2va golden it produces
  lands the port at 8.733e-5 where the CUDA-built one landed 8.784e-5,
  so a stand-in with the convention wrong would have moved that.

- **A device GEMM for the two-bit plane.** `q2tp` had no GPU arm at all,
  so a two-bit file ran its WIDEST projections on the host while the
  four-bit one had the card — a codec paying for its size twice. The
  kernel is the q4tp one with three changes and no others: the weight
  plane is 8 bytes a group rather than 16, a byte holds four 2-bit
  codes instead of two nibbles, and the ladder is shifted down a rung
  because rung 0 is spent naming the exact zero the ±0.5/±1.5 grid
  cannot reach. Params and codes are byte-identical, so the dispatch,
  the buffers and the bind group are shared — `mm_pipeline` picks by
  width.

  No cooperative variant: the tensor-core kernel is written against the
  4-bit plane, and a shader compiled against a layout the weights do
  not have is a silently wrong answer — which this release already has
  three of. Against the host on the shapes a q2tp file actually uses:
  max |Δ| 1.3e-5 on a signal of 3.5, no non-finite values, at both
  batches (`tests/gpu_q4tp_batch.rs`).

- **A two-bit build, measured and not shipped.** `--quant q2tp` puts the
  gate/up planes at two bits and leaves the rest at four, which is the
  policy DeepSeek-V4 measured at 1.3x the perplexity for 0.71x the file.
  Here it builds (23.94 GB → 18.74, `verify` clean) and, with the device
  kernel above, renders FASTER than the four-bit file — 217.3 s against
  258.4, there being less weight to move.

  It also stops following the prompt. Same seed, same four steps: the
  four-bit render puts the corgi behind a pan with batter in it, drawn
  flat and clean; the two-bit one is a different animal in a different
  style with no pan and no pancake at all. That is not detail traded for
  size. The likely cause is where the two bits landed — half the file is
  the PROMPT ENCODER, so the policy cut the part that decides what the
  clip is about rather than the part that draws it. A build confined to
  the DiT would save ~2.9 GB instead of 5.2 and is what to measure next.
  Both clips are on the Hub beside each other.

## [0.5.59] - 2026-08-07

### Added
- **MiniMax-H3 with the 4-step Turbo LoRA: video AND synchronized stereo
  audio, from one 23.5 GB file.** `cortiq animate-pack` packs the DiT, the
  Qwen3-VL-32B prompt encoder, the ViT3D video decoder and the BigVGAN
  vocoder into one mmap; `cortiq animate` renders an MJPEG+PCM AVI and a
  .wav with no ffmpeg and nothing to install. Against the reference — four
  files plus a ComfyUI checkout — that is 124.4 GB down to 23.5.

  Most of the reduction is not quantization. **Forty per cent of the
  released 33 B DiT is `adaln_proj.linear`**: `[96768, 2688]` per block,
  13 B parameters, for a map whose input is one number. Its output over
  the schedule is a 1-D curve, and Comfy-Org's pruned checkpoints already
  ship it as a rank-8 one. The Turbo LoRA is written against the full
  matrix, which is why the ComfyUI node re-injects the time conditioning
  at run time; the packer folds both into one rank-24 basis,
  `[W_p | B] · [u(t) ; A·silu(e(t))]`, 4.6 MB a block instead of 520.
  `tools/mmh3_fetch.py check` measures what the collapse costs by
  range-reading 520 MB out of the 66 GB file: rms 8.7e-5 against a signal
  of rms 0.464, with the time curve's 9th singular value already 3.7e-5
  of its first.

  Parity is per stack, against ComfyUI's own modules on toy checkpoints
  carrying the release's real tensor names and real schedules
  (`tools/mmh3_toy_gate.sh`): DiT video velocity 8.8e-5 worst on a signal
  of 0.515, audio velocity 5.2e-5 on 0.409, prompt encoder 1.1e-6, video
  decoder 4.2e-7 including its 256-pixel tiling — which is part of the
  output, not a memory strategy, because the decoder is global attention —
  and the vocoder 1.7e-9.

### Fixed
- **`gemm_nt_f32` cached the device-side weight buffer BY POINTER
  ADDRESS.** Every batched attention in this engine allocates one k/v
  scratch pair per call and refills it per head, so the address is
  constant across heads while the matrix is not — head 0's keys came
  back for every head, on the GPU only, silently. It is the same
  mistake `CmfModel::uid` documents on the model mapping, one file
  over. The cache is keyed on a content fingerprint now: one sequential
  read against a PCIe upload, so a genuinely stable weight still
  uploads once. `tests/gpu_gemm_scratch.rs` refills a buffer between
  two calls and would have caught it.
- **`gemm_nt` took every job over 4 M MACs on sight, with no CPU arm to
  lose to.** It carries attention's per-head QKᵀ and AV and the video
  decoder's projections, where the round trip costs more than the work
  — on the MiniMax-H3 decoder the device was THREE TIMES SLOWER than
  the host it displaced. It goes through the same probe as every other
  op class now (`OpClass::GemmNt`), which on that stack measures
  0.24 ms against the host's 0.13 and hands the work back.

  That probe is shared by every `gemm_nt` caller, and one verdict
  deciding for two populations is the hazard `MatmatWide` was split to
  avoid — so Lumina-Image, the other model on this path, was measured
  either side of the change: 512×512 at 30 steps, 14.4 s on 0.5.58
  against 14.6 s on 0.5.59, and the two renders agree to 55 dB (mean
  |Δ| 0.16 of 255 — the summation order of a GEMM that moved off the
  device, not a change in behaviour). Both of its populations land on
  the host too, so the shared verdict costs nothing today; it is worth
  splitting the moment one of them wants the card.
- **The cooperative-matrix GEMM runs THIS model out of f16 range.** At
  256×160 the render is correct; at 512×288 the audio stream goes NaN
  on the second sampling step and the video follows. Bisected against
  the alternatives: `CMF_BAKE_GPU=0` does not help, `CMF_COOP=0` does.
  The hold is model-scoped on purpose — Lumina-Image on the same card
  and the same kernel renders 20.5 s without it against 14.8 with, and
  the two agree to 42.6 dB, which is the f16 cost the kernel documents
  and not a fault. So `cortiq animate` pins `CMF_COOP=0` and nothing
  else changes; what the kernel needs is a scale on its operands
  before it can carry activations this large.

  Separately and NOT from this work: the native Vulkan lane's own
  `tests/vk_coop.rs` fails identically on 0.5.58 and 0.5.59 — "256x512
  b=48: cooperative GEMM off by 8.760e3 of the row's magnitude", same
  cell either side — which is what master's CI has been red on. Two
  cooperative-matrix implementations, two separate problems.
- **`cortiq animate` decides its own device policy before the backend
  comes up.** `gpu::cpu_scope` cannot do it: it sets a thread-local the
  op probe's own CPU arm reads, while the probe still executes the
  device arm for real while timing it.

### Changed
- **MiniMax-H3 on one GPU: 371.3 → 172.0 s** for a 512×288, 39-frame
  render, against the same host baseline and the same four steps. The
  weight GEMMs go to the device (25.8 ms against the host's 92.0),
  attention and the small per-head products stay on it. What was left
  on the host after that was the DiT's elementwise glue — four RMS
  norms, two modulations, two gated residuals and a SwiGLU per block,
  all running on one thread while forty-seven sat idle — plus four full
  copies of `n·heads·head_dim` per block to normalize q and k through a
  scratch buffer. The norms, the modulation, the residuals and the
  activation are across the pool now, and the rotation happens where
  the values already lie: 10 G element copies a render, gone.

- **DeepSeek-V4 decode: 10.6 → 27.2 tok/s** on an RTX PRO 6000, measured at
  each step against the same 192-token bench and gated on five toy stands
  plus golden parity. In order: the chain's bind groups cached; the hash
  layers joined the chain; the whole token became ONE submission; the head
  reached the card through the matvec kernel rather than the batched GEMM
  at b=1 (0.32 ms against the host's 9.43); compute passes went 23 a layer
  to 4; and then the strategy changed.

  What changed it was measuring what a dispatch actually costs instead of
  dividing a frame by its count: `CMF_GPU_DISPATCH_BENCH=1` on `cortiq gpu`
  says 2.67 µs on Vulkan, with a barrier between dependent dispatches
  essentially free. So the chain was never made of overhead — it was
  kernels using a rounding error of the card. `f32_matvec` gave a row 64
  threads and one workgroup, which for the hyper-connection mix is 24
  workgroups of 64: fifteen hundred threads on a machine that holds
  hundreds of thousands. Width by shape — 256 threads where rows are many,
  1024 where they are few, 1024 for the reducing kernels that are one
  workgroup by construction, four rows to a workgroup where the columns
  leave no width to take — is most of the second half of that number.

- **`CMF_GPU_UPLOAD=staged`** maps a staging buffer and issues the weight
  copy itself instead of handing the bytes to `queue.write_buffer`: 48 MB/s
  to 280+ on a cold page cache, which is half an hour of first-token wait
  turned into minutes. The profile reports the rate.

### Fixed
- **DeepSeek-V4 on the device now matches the host EXACTLY** — perplexity
  3.282 against the CPU's 3.282, where the chained path had read 3.301 and
  that 0.6% had been written down as a device-summation contract. It was
  not arithmetic: the indexer's head weights were left at zero by the
  `axpy` bug below, so every compressed position scored the same and the
  top-k degenerated to the first k. The model's long-range memory was
  attending to the wrong positions.
- **`axpy` did nothing.** Its uniform was written `[n, 0, 0, w]` against a
  shader declaring `{ w: f32, n: u32, … }`, so the kernel read `n` as zero
  and every invocation returned at the bounds check. Two callers: the
  DeepSeek-V4 indexer's score scaling (head weights left at zero, so every
  compressed position scored the same and top-k fell back to the first k)
  and the overlapping compressor's position bias. It had no test; it has
  one now.
- **A WGSL reserved word took the whole device with it.** Naming a field
  `set` made the q8 module fail to parse — which does not disable one
  kernel, it stops wgpu initialising, and every op falls to the host behind
  a single WARN. `tools/dsv4_toy_gate.sh` proves the device is up before it
  compares anything.
- **`cortiq bench` could not bench DeepSeek-V4 at all** — the paired prefill
  inside `forward_ids` and `measure_pair_fusion` walk `weights.layers`, which
  an architecture that loads its own layers leaves empty. Both ask
  `pair_supported()` now. 0.5.45 shipped with this because the numbers were
  taken with `run`, which never takes that path.
- **`matvec` refuses a short `out`** instead of writing past it: the Mapped
  kernels write `rows` entries through a raw pointer, and the `debug_assert`
  two of them carried is compiled out of the release.
- **The budget message no longer blames a mask that isn't there** when a
  layer has no VRAM left for a single expert.

### Changed
- **The experts' host pages are released once they reach the card**
  (`madvise(DONTNEED)` + `posix_fadvise`; discrete GPUs on Linux,
  `CMF_UPLOAD_EVICT=0` opts out). Uploading reads ~94 GB through the mapping
  and the page cache kept every byte — a second copy of weights that now live
  in VRAM, and on a 112 GB model that copy is the machine's RAM. Alternated
  off/on/off/on from a warmed file, 256 tokens each:

  | | resident | decode |
  |---|---|---|
  | off | 94 GB, 91 GB | 1.0, 0.5 tok/s |
  | on | 5 GB, 5 GB | 6.9, 6.1 tok/s |

  It is not a trade: holding the second copy is what was costing the speed.
  Opening no longer asks for `WillNeed` over the whole file when the pages
  are headed for a device that will drop them — reading 104 GB ahead only to
  evict it behind the uploader had the kernel fetching the same bytes twice.
  A CPU run, a UMA device and every non-Linux target keep the readahead.

### Changed
- **The kernels borrow their scale row** instead of allocating one per
  worker per dispatch: 22474 allocations a token become 15021. Decode is
  unchanged, which is the point worth recording — five separate reductions
  of overhead (dispatch count twice, the harness loop, submissions, and now
  allocations) have all left the token where it was, so the cost is in the
  kernels, not around them.

### Added
- **`CMF_DSV4_CHAIN=1`** puts a run of consecutive device-capable layers in
  ONE submission: 59 pool dispatches a token against 285. The compressor,
  the indexer and the window append all moved to the card to make it
  possible. Exact on the toy (perplexity 136.750 either way over 200
  tokens); **wrong on the release checkpoint (50.280 against the CPU's
  3.282) and slower (12.8 tok/s against 16.2)**, so it is off by default.

### Known limitations
- **`CMF_DSV4_CHAIN` is correct now and not yet fast.** The in-run clobber
  that produced perplexity 50.280 is found and fixed — queue writes land
  before a run's single submit, so every layer routed with the LAST layer's
  noaux_tc bias; the bias now lives in the pack with a process-stable
  address, and the hash layers run as runs of one. Full-run parity on the
  release is 3.301 against the CPU's 3.282, the same device-summation
  contract as every configuration of this path. Speed is 12.5 tok/s against
  the two-frame path's 14.9-16.3: the saved fences are spent encoding ~30
  passes a layer, ~8 of them building fresh bind groups per call. Caching
  those — with generation guards on the regrowable buffers — is the sized,
  single next step.
- **The process can abort at exit, after printing a correct answer**
  (`double free or corruption`, `corrupted double-linked list`). Located
  with AddressSanitizer, and it is not ours: a 48-byte block allocated by
  `libEGL` while the NVIDIA driver compiles a shader is freed through the
  Rust allocator on the `init → create_compute_pipeline → wgpu_hal::vulkan
  → naga` path, and freed a second time by `libGLX_nvidia` at process exit.
  Nothing in this crate writes out of bounds — results are unaffected, the
  abort lands after the work is done. Seen on wgpu 30 with driver 580 on a
  hand-registered GLVND EGL vendor; not reproduced elsewhere yet. Earlier
  notes tied it to the cold split, which was coincidence: it aborts with the
  split off, on both 60 GB and 94 GB budgets, in `ppl` and in `bench`.

## [0.5.45] — 2026-08-02

The cold-expert split works: hot experts run on the card, cold ones are
finished on the host, and no layer has to leave the device because its
experts do not all fit. This closes the first known limitation of 0.5.44 and
unblocks the whole-layer frame's economics.

**Correction to this entry, written after release:** the split is opt-in
(`CMF_DSV4_COLD_CPU=1`), not on by default — the code always required the
variable to be set. It stays opt-in for now: it is exact (perplexity 3.282
against the CPU's 3.282 on the release checkpoint, at every budget), but it
is also the arm the abort above follows.

### Fixed
- **The router's bias went in short.** The bias was uploaded `n_pack` long
  while the kernel ranked over `n_all`. WGSL CLAMPS an out-of-bounds read
  instead of faulting, so every expert past the packing boundary got the
  LAST packed expert's bias — a plausible number, and one that made the
  router prefer the resident experts. It looked exactly like a router bug.
  Scores, bias and the routing uniform now take ONE width, computed once;
  toy perplexity is the CPU's number at packings of 64, 32, 8 and 2 of 64.
  `CMF_DSV4_COLD_CPU=0` restores all-or-nothing.
- **The kernel reports every winner** in the second half of `rt_cold`, so
  "the router chose no cold experts" and "the readback is broken" are no
  longer the same empty list. Nothing reads it but a human.
- **`x86_gemm`'s CPU-against-CPU comparison silently became CPU-against-GPU**
  whenever `CMF_GPU` was set — off by the 5e-3 that summation order costs,
  and blaming the blocked kernel for it. It refuses the changed question now.
- **A machine nobody pointed at wgpu and a machine where wgpu came up dead
  read the same in tests.** `selected_and_up()` tells them apart: not
  selected is a legitimate skip (what CI runners look like), selected with
  no context fails — that is the case that once hid a reserved-word shader
  error behind a wall of green skips.

### Added
- **The Hub model cards live in the tree** (`docs/hf/`), so the copy that
  says what the engine can do is versioned with the engine that does it.

## [0.5.44] — 2026-08-01

DeepSeek-V4-Flash: the architecture, the converter that fits it on a laptop's
disk, and a GPU path that decodes it **2.7 → 12.7 tok/s** on one card without
moving a digit of the answer. Also a thread-pool bug that had been quietly
single-threading every short job in the engine.

### Added
- **`deepseek_v4` architecture** — hyper-connections with a Sinkhorn mix,
  double-LoRA attention, a 512-wide shared KV, the learned attention sink, an
  overlapping KV compressor, the sparse indexer, grouped low-rank output and
  hash-routed MoE layers. Numerical parity against a NumPy transcription of
  the reference forward (1.6e-3), then against the CPU at every step.
  Published: `infosave/DeepSeek-V4-Flash-0731-cmf`, q4tp 158 GB and q2tp
  112 GB.
- **`CmfStreamWriter`** — conversion writes each payload ONCE, appending and
  patching the head afterwards, so a 158 GB model no longer needs 276 GB of
  disk to be born. `convert --resume` continues from a per-shard checkpoint,
  the download included.
- **GPU frames for DeepSeek-V4** (`CMF_DSV4_GPU_ATTN=1`,
  `CMF_DSV4_GPU_MOE2=1`): the attention block and the MoE block each in ONE
  submission, with the experts resident. Perplexity 5.211 against the CPU's
  5.211 on the release checkpoint.
- **Whole-layer frame** (`CMF_DSV4_GPU_LAYER=1`) — one submission per layer,
  hyper-connections and the router on the device. Correct and, on a card that
  cannot hold every expert, slower than the two frames; see below.
- **Routing field tools**: `CMF_MOE_STATS` records it, `CMF_MOE_MASK` /
  `CMF_MOE_MASK_COVER` preview a restriction, `cortiq gpu` says what the
  backend can see, `CMF_DSV4_PROFILE` splits a token into attention, experts,
  hyper-connections and head.

### Changed
- **Thread pool grain** — `grain = (rows / (workers*8)).max(32)` handed any
  job with fewer than 32 rows to ONE worker while waking all the others. The
  grain is now also capped at one chunk per worker; wide jobs keep the stride
  they had, bit for bit. The hyper-connection projection (24 rows, twice a
  layer) was the visible victim: 42 → 4 ms a token.
- **Expert upload** goes straight from the mapping to the queue instead of
  through a per-layer gather buffer — 94 GB of copying that bought nothing.
  Short generations gained 40%.
- **Quantizer**: the group scale is chosen by error rather than by absmax —
  25% less noise at 2 bits, 6% at 4, same bytes. `q2tp` rung 0 is an exact
  zero, because a pruned group must not come back as dither.

### Fixed
- **RoPE pairs ADJACENT coordinates**, not halves. The two agree exactly at
  position 0 and nowhere else, which is why every short test passed.
- **The tokenizer applies every `Split` of a `Sequence`**, not just the first.
- **A declared MTP head is not a present one** — the loader now checks.
- **`const_buf` is keyed on a host address**, which is sound for model weights
  and wrong for anything built per call: the MoE bias, assembled per layer,
  had every layer routing with layer zero's bias.
- **`shared` is a reserved word in WGSL** — naming a local that compiled and
  then failed pipeline creation, which takes the whole context down. Every
  GPU test "passed" by skipping; they now panic instead.

### Known limitations
- The per-expert cold split (`CMF_DSV4_COLD_CPU=1`, hot experts resident and
  the rest finished on the host) is off by default: on a partial packing the
  router still ranges over only the packed set on some layers. Five suspects
  are eliminated by measurement — the weights, the readback, both caches, the
  full-packing path — and `CMF_DSV4_PACK_MAX=N` reproduces it on a toy in
  seconds.
- The whole-layer frame is correct but slower where the experts do not all
  fit: a layer that misses runs on the host entirely. It pays once the cold
  split above works.


## [0.5.43] — 2026-07-31

A GPU-kernel release. Same weights, same answers, roughly half the frame:
decode on a discrete card went **67.8 → 99.6 tok/s** on Qwen3.6-35B-A3B,
**32.3 → 39.1** on the dense 27B, **37.5 → 77.6** on a q4t 3B and
**28.8 → 37.4** on a q1t 8B. Nothing here changes a file format.

### Added
- **`q2tp`, a 2-bit tile layout** (dtype 16): the `q4tp` predicted per-row
  scale ladder over a 4-level ±0.5/±1.5 grid, 8 bytes per 32-weight group.
  Rung 0 of the ladder is an EXACT ZERO — the grid cannot otherwise spell
  one, and a pruned group would come back as noise. `--quant q2tp` writes
  the mixed profile the 2-bit-class checkpoints want: 2-bit expert
  `gate`/`up` (routed AND shared), 4-bit `down` and skeleton.
- **GPU frame profiler**: `CMF_GPU_TS=1` stamps pass boundaries,
  `CMF_GPU_TS=2` stamps every dispatch inside the first layer of each kind.
  Every optimization below was found with it; see
  `docs/GPU_KERNEL_RECIPES.md` for the recipes AND the measured failures.

### Changed
- **Attention decode kernel**: 256 threads per head (lanes are positions
  for the scores, output dims for the values) instead of one warp with a
  257-stride accumulator — 137 → 45 µs a layer. `CMF_ATTEND_DEC=0` reverts.
- **GDN state access follows the layout's grain**: four dv-contiguous
  columns per workgroup as one 16-byte access instead of a column per
  workgroup striding the row. The single largest step of the release.
- **vec4 loads across the kernels**: q4tp matvec (8-row and a 16-row twin
  for narrow matrices), the q4t matvec, the MoE gate/up and down kernels,
  and the batched GEMM trio. Add order is preserved everywhere, so greedy
  output is unchanged — gated on real models, not just unit tests.
- **q1t** reads each base-3 code byte once for its five codes instead of
  re-reading it per weight.
- **Fewer compute passes**: the layer's norms ride the FFN pass, and a
  short-context attention layer runs rope + kv-append + attend + gate +
  O-projection in ONE pass. `CMF_PASSFUSE=0` reverts.
- **Subgroup MoE select** where the adapter carries the feature
  (`Features::SUBGROUP`), in its own shader module. `CMF_SELECT_SG=0` off.

### Notes
- Binary `q1` was measured and left alone: it reads 4 bytes per 32 weights
  and the recipe buys nothing there (152.8 vs 152.3 tok/s).
- Multi-step greedy decode (k frames per submit, on-device argmax) is
  implemented behind `CMF_MULTISTEP` and defaults OFF — at equal
  generation length it measured at or below the plain path.

## [0.5.42] — 2026-07-31

The Qwen3.6 release. The hybrid MoE flagship (35B-A3B) and the dense 27B run
whole-token on discrete GPUs — 64.5 tok/s decode on an RTX 5090, 99 tok/s
batched prompt ingest — keep their multi-token-prediction heads through
conversion, and O(1) Nyström attention now rides the GPU graph instead of
dragging the model back to the CPU.

### Added
- **Qwen3.6 MTP heads survive conversion.** The converter keeps `mtp.*`
  (renamed to the loader layout) instead of dropping it, and the loader
  builds MoE FFNs inside the MTP block — previously it only knew dense
  heads, so an MTP-bearing MoE file failed to load.
- **O(1) attention on the wgpu graph** (`CMF_O1_GPU=1`): the sealed
  landmark skeleton, far-field flash accumulators and FM rectifier run as
  graph kernels. Qwen3.6-35B-A3B holds a flat 53.8 tok/s at ctx 16 384
  where exact attention has fallen to 37.8.
- **Batched prompt ingest through the graph** (`CMF_BATCH_K`): token-axis
  MoE kernels route each position individually while the surrounding
  skeleton stays batched — 99 tok/s ingest against 33 per-position.
- Metal: the whole MoE block in four dispatches (M4 35B-A3B decode
  10.6 → 18.2 tok/s), with a loud `maxBufferLength` refusal.
- Measurement probes for the decode frame: `CMF_TOPK_PROBE`,
  `CMF_SKIP_PROBE`, `CMF_LAYERS_PROBE` — measure before optimizing.

### Fixed
- **q4t/q4tp files never entered the whole-token graph** — a hole in the
  layout prep silently sent every quantized-tile model to the CPU.
- **The graph's GDN device state started zeroed after a CPU prefill**
  (o1 calibration, `--state` resume): now seeded from the CPU recurrence.
- The OpenAI-compatible server accepts `content` as a block array as well
  as a string (Cline/Zoo-style clients got 422).
- `bench --ctx` decoded without the synthetic context it had just built.
- CPU MoE: every routed expert now runs under one pool dispatch instead
  of one dispatch each.

### Changed
- Graph refusals are loud: the graph logs ACTIVE/declined with the
  reason, and the VRAM budget prints the numbers it compared
  (`RUST_LOG=info`).

## [0.5.41] — 2026-07-30

A hotfix: **0.5.40 shipped a Metal backend that does not initialize.** Update.

### Fixed
- **The q4tp GEMMs referenced a variable that 0.5.40 deleted.** Hoisting the
  per-row scale out of the K loop removed `uint wr`, and two lines still used
  it. The MSL then fails to compile, `ctx()` returns None, and every model on
  macOS silently falls back to the CPU — not just q4tp ones.

  It reached a release because the parity test returned early when
  `gpu_metal::enabled()` was false, which is exactly what a broken shader
  causes. A test that skips on the condition it is meant to detect is worse
  than no test; it now asserts the backend came up.

- **Packed 3-D expert tensors doubled the `mlp.` prefix for the qwen
  lineage.** gemma-4 names the tensor `layers.N.experts.gate_up_proj` and
  qwen3.5/3.6 `layers.N.mlp.experts.gate_up_proj`; the splitter stripped the
  suffix and appended `.mlp.experts`, so the qwen form came out as
  `layers.N.mlp.mlp.experts.0.*`. Conversion succeeded — valid file, hashes
  matched — and the model failed at load with "router present but no expert
  tensors", which points nowhere near the naming. Qwen3.6-35B-A3B converts
  and runs now.


## [0.5.40] — 2026-07-29

q4tp reaches parity on the image path, plus the AWNP tooling and the
measurements that closed it. Measured on a fanless MacBook Air M4.

### Added
- **`cortiq awnp`** — activation-weighted nullspace projection: drop the
  weakest input channels of every MoE expert and refit the survivors through
  `C[:,S]·C[S,S]⁻¹` so they absorb what was removed. `--drop`, `--ridge`,
  `--rescale`. Output keeps the original shapes, so the result runs on
  today's runtime and the quality question is settled before any format
  change.
- **`CMF_RMS_TRACE`** (per-channel activation energy) and **`CMF_ACT_DUMP`**
  (raw FFN-input rows) — the calibration AWNP needs. Nothing in the runtime
  produced activation statistics before, so "is the flat weight-energy curve
  a property of the weights or of the criterion" could only be asserted.
- **q4tp in the Russian and Chinese README and spec.** The English pair had
  the new dtype and the localized ones did not.

### Fixed
- **The q4tp GEMMs read their per-row scale inside the K loop.** `lo`/`step`
  are loop-invariant — a thread's row is fixed across the whole loop — so
  that was a dependent chain (load → exp2 → scale) paid per staged tile for
  nothing. On Lumina's DiT the runtime probe measured the q4tp GEMM as the
  slower arm and routed the whole render to the CPU: 18 s against q4t's 13.
  Hoisted, interleaved at 256px/8 steps: 26 and 27 s for q4tp against 27 and
  26 for q4t. The image is unchanged at 43.6 dB PSNR.
- **The fused DiT SwiGLU chain had no q4tp twin**, so a q4tp image model
  shipped its `[b, inter]` intermediates across the CPU boundary twice per
  layer — 28 s before the twin, 18 s after, parity after the hoist above.
- **`sgemm_public` did not compile off macOS/aarch64.** `sgemm_rm` exists
  only where there is an f32 GEMM to call; x86 has quantized kernels and no
  such path. A portable triple loop now backs it — only the offline AWNP
  pass reaches it, where correctness matters and throughput does not.

### Notes
- **AWNP measured and not productized.** On KAT-Coder-V2.5 (qwen3_5_moe),
  25000 calibration rows over 2048 channels from hundreds of distinct
  dialogues, evaluated on a held-out half sharing no record with the
  calibration: baseline PPL 23.4, −12.5% channels 50.1, −25% channels 194.8.
  Widening the calibration from 2.9 to 12.2 samples per channel and adding a
  real ridge halved the damage, and the patent's variance-preserving rescale
  does not help — a least-squares refit already returns the minimum-error
  weights, so scaling them back up moves away from that optimum. The
  projection alone, without a healing pass, does not transfer out of sample.
  A skill overlay is the better lever for the same goal.


## [0.5.39] — 2026-07-29

Same code as 0.5.38 plus the documentation it should have shipped with.
0.5.38 was tagged but never published to crates.io: its CI was red because
`examples/q4tp_gpu_ab.rs` referenced the Metal module unconditionally and so
failed `--all-targets` on Linux, and the README and spec said nothing about
the new dtype. Both fixed here; 0.5.39 is the release to use.

### Added
- **`q4tp` in the README** — what it costs, what it saves, and `requant` for
  files already published.
- **`q4tp` in `docs/CMF_V2_SPEC.md`** as dtype 15, with its byte layout and
  the two encoder requirements (round `lo`/`step` to f16 before choosing
  codes; quantize nibbles against the reconstructed scale) that keep a
  writer and a reader on the same scale.

### Fixed
- `examples/q4tp_gpu_ab.rs` is a stub outside macOS. It times Metal encoders
  directly; the wgpu kernels are covered by `tests/gpu_q4tp.rs`, which runs
  on every backend.


## [0.5.38] — 2026-07-29

A new weight layout and the kernels to run it, plus a negative result that
is the reason the layout looks the way it does. Measured on a fanless
MacBook Air M4 (Metal and, via `CMF_GPU=wgpu`, the WGSL path).

### Added
- **`q4tp` (dtype 15) — q4_tiled with predicted scales.** A q4t tile spends
  16 bits on a standalone f16 scale for 32 weights: 0.5 of its 4.5
  bits/weight, 11% of the file. q4tp keeps the nibbles byte-identical and
  makes that scale a 5-bit rung on a per-row geometric ladder
  (`[nibbles][row (f16 lo, f16 step)][5-bit codes]`), for 4.17 bits/weight.

  Quantizing the SAME fp32 weights both ways, q4tp's error against the
  source is 9.71% where q4t's is 9.71% — +0.1% relative at the median
  within-row scale spread measured on KAT-Coder-V2.5 (1.27 in log2), +0.3%
  at its 90th percentile, +0.8% past the tail. A coarser scale costs almost
  nothing because the nibbles simply re-round against it; the 4-bit grid
  dominates either way. Nanbeige-3B keeps its top-5 next tokens in the same
  order with logits within 0.7%.

  Design points, each settled by measurement rather than taste: the row's
  `lo`/`step` are rounded to f16 FIRST and the codes chosen against those
  values, so `lo`'s rounding is absorbed by the code instead of stacking on
  it; codes round to nearest, not up (1.19% vs 1.60% RMS — letting the top
  tile clip one nibble beats coarsening the whole row); a per-tensor Lloyd
  codebook was tried and rejected (0.0576% vs 0.0591% NMSE at 4 bits, since
  within a row the log-scales are near-uniform); and `lo`/`step` come from
  the row's exact min/max, so no code is ever out of range and the format
  needs no escape hatch.

- **`cortiq requant <model> --output <out> --quant q4tp`** — recodes a
  published `.cmf` in place, without the original checkpoint. The
  checkpoints behind published CMF files are tens of gigabytes;
  re-converting them is the expensive path this avoids. KAT-Coder-V2.5:
  12.65 → 11.80 GB (19254 tensors) in 2 min. Nanbeige-3B: 2.36 → 2.19 GB,
  peak RSS 2.42 → 2.19 GB. `convert --quant q4tp` also works from HF.

- **q4tp kernels on every path.** CPU: scalar, sdot/avx2/vnni int8, blocked
  1×4, Accelerate sgemm, and the fused SwiGLU pair. Metal: `q4tp_matvec`,
  `q4tp_mul_mm`, `q4tp_mul_mm_silu`, plus the chunk-graph plumbing. wgpu:
  `q4tp_matvec` and `q4tp_mul_mm` under codec kind 6 — deliberately not
  sharing q4t's 5, since sharing a kind is exactly how q4t once got fed to
  the q4_block kernel and produced garbage on real Vulkan.

  The layout suits a GPU better than q4t's: a 16 B nibble stride is
  4-aligned, so Metal loads four uints where q4t needs nine unaligned
  ushorts and WGSL reads words straight off the storage array. Decode is at
  parity model-level (Metal 12.4 vs 12.4 tok/s) and the kernels themselves
  are faster — 0.100 vs 0.117 ms on o_proj, 23.8 vs 26.2 ms for a cold
  sweep over all 156 weight tensors. The batched GEMM is faster on both
  backends at b=64: Metal 9.51 vs 10.13 ms (2096 vs 1968 GFLOP/s), wgpu
  9.83 vs 10.76 (2027 vs 1853).

### Notes
- **MoE dictionary compression: gate failed, and the mechanism is known.**
  The hypothesis was that fine-grained MoE experts are one function with
  local variations — keep one as a base, code the rest as residuals against
  a shared dictionary. On KAT-Coder-V2.5 (~202 experts/layer) it does not
  hold: `sigma(W_e − W_base)/sigma(W_e)` is 1.34–1.38 against 1.414 for
  independent matrices, so the residual is WIDER than the weight and a base
  costs more bits than it saves. After optimal bipartite matching of neuron
  triplets (gate-row ⊕ up-row ⊕ down-column, 6144 dims) the aligned cosine
  is 0.037–0.047 against a chance level of 0.045 — there is no permutation
  bringing experts together, so tuning the metric, block shape or clustering
  has nothing to find. Load balancing decorrelates experts on purpose.

  What survived is the only redundancy that measured real: a shared
  per-channel scale profile (neuron-profile correlation 0.000 across
  experts, channel-profile +0.16…+0.62). That is what q4tp exploits — and
  per-row coding turned out to subsume the channel profile entirely, which
  is why the format has no profile plane.

  Structural pruning by raw magnitude is also empty here (dropping the
  weakest 12.5% of neurons costs 11% of Frobenius energy — proportional).
  That is NOT a verdict on AWNP, whose criterion is activation-weighted; no
  calibration traces were run.

## [0.5.37] — 2026-07-29

A batch of "the kernel was already there, nobody wired it up" and two
copies of the same defect in two languages. Measured on an RTX 3090 with
a 256-core EPYC beside it, and on a fanless MacBook Air M4.

### Added
- **DiT attention on wgpu** (`gpu::dit_attention` had a Metal arm and
  nothing else, so Vulkan ran the whole attention block on the CPU).
  Three WGSL kernels on the tile machinery the quantized GEMMs use —
  scores, row softmax, P·V — plus a panel unstack, all in one
  submission with the intermediates resident.
- **Causal chunk attention on wgpu** for LLM prefill. The CPU twin needs
  Accelerate or the NEON micro-GEMM, so x86 had no batched attend at all
  and fell back to a per-position scalar loop.
- **Fused QKV** for the batched attention path: one upload of the normed
  chunk, three GEMMs, one readback.

### Fixed
- **The tiled WGSL GEMMs kept their register block in memory.** All four
  (q8, q1, q1t, q4t) declared the 4×4 accumulator as
  `array<array<f32,4>,4>` and indexed it with loop variables; a
  dynamically indexed private array does not stay in registers on this
  backend. The GEMM ran at 373 GFLOP/s of an RTX 3090's ~35 TFLOP/s.
  Sixteen named scalars instead — bit-identical output, and the stage
  itself went ×4.1.
- **The same defect in AVX2**: the blocked q4t kernel folded each group's
  eight i32 lanes to a scalar inside the loop, 288 cross-lane reductions
  per weight row. Four f32 accumulator vectors and one reduction per row.
  Also more accurate — perplexity 9.777 → 9.572, because summing 72
  scalars in sequence loses more than accumulating in eight lanes.
  (`avx2` does not imply `fma`: without it declared, LLVM lowered
  `_mm256_fmadd_ps` to a libm call per lane and the "optimized" kernel
  ran 2× slower than the one it replaced.)
- **The batched dense FFN never used the fused GPU kernel**, so prefill
  paid three round trips per layer and shipped the 22 MB gate/up panels
  across the bus for nothing. The kernel had existed since the image DiT.
- **The default thread cap left big machines idle.** `min(available−1, 8)`
  meant eight threads on a 256-core box. Raised to 32 — a ceiling, not a
  target: decode is memory-bound and collapses past it (14.8 tok/s at 32,
  5.5 at 64, 1.6 at 256). Machines with nine cores or fewer, and
  heterogeneous ARM, are untouched.
- **A coin flip decided whether image generation ran at half speed**: the
  fused DiT paths gated themselves on a per-op probe that is a tie on
  Metal (2.62 vs 2.56 ms) and mere overhead on a discrete card. They now
  trust the device on native Metal and on discrete wgpu adapters;
  integrated and mobile keep probing.

### Changed
- Lumina's caption embedding and context refiner are hoisted out of the
  denoise loop — they depend on the prompt alone, so 30 CFG steps were
  evaluating 60 times a value with two distinct instances. Bit-identical,
  and honestly small (~0.25% of a step).
- `cortiq imagine`'s help caught up with reality: it still said "CPU f32
  (experimental; CMF packaging comes later)" long after `imagine-pack`
  shipped and the DiT moved onto the device.

### Measured
Lumina-Image 2.0 at 512², two steps, RTX 3090: **67.7 s → 24.75 s**.
Nanbeige 4.2 on the same box: Vulkan prefill **15.1 → 25.0 tok/s** at
ctx 512 and **15.0 → 23.0** at ctx 1024; CPU decode **7.8 → 14.3**, CPU
prefill **11.8 → 16.7**. Perplexity is unchanged at every step.

## [0.5.36] — 2026-07-28

### Fixed
- **A coin flip decided whether image generation ran at half speed.**
  The fused DiT paths (whole block, all-heads attention, SwiGLU FFN)
  gated themselves on the wide-GEMM probe, which asks "is ONE wide
  matmat faster on the GPU" — and for Lumina that is a tie: 2.62 ms GPU
  against 2.56 ms CPU, landing on either arm run to run. But a fused
  block's advantage is not per-op speed, it is that the hidden state,
  the packs and the attention panels never leave the device. Measured
  on a fanless MacBook Air M4, 512² Lumina-Image 2.0: a step takes
  ~4.5–5.8 s when the probe picks the GPU and **7.7–9.0 s when it picks
  the CPU**, and it picked the CPU in a third to two thirds of runs. On
  native Metal the fused paths now trust the device
  (`gpu::fused_block_trusted`); other backends keep probing, where the
  submit latency is real. Renders are bit-identical, just no longer
  randomly half-speed.

### Changed
- Lumina's caption embedding and its context-refiner blocks are hoisted
  out of the denoise loop. They depend on the prompt alone — not the
  timestep, not the latents — so 30 CFG steps evaluated 60 times what
  has two distinct values. Output is bit-identical; the saving is small
  (~0.25% of a step: the caption is ~60 tokens against 1024 image
  tokens), this is about not recomputing a constant.

## [0.5.35] — 2026-07-28

### Added
- **q4_tiled weights in the Metal chunk-prefill graph.** The graph
  required `q8_row` on all seven projections of a layer, so every q4t
  model — the whole Nanbeige/Bonsai class — silently fell back to the
  CPU prefill. An empty `row_scale` now marks q4t per weight, and the
  new `q4t_mul_mm_silu` kernel gives the fused down-GEMM its q4t twin.
  A non-`q8_row` embedding matrix no longer refuses the whole run
  either: the CPU fills the hidden and the graph starts from it.
  Nanbeige 4.2 on M4: 512-token prefill 6.1 s → 2.8 s (≈185 tok/s),
  1024-token 13.1 s → 6.6 s.
- `CMF_GQA_SPLIT` — how many simdgroups may split one Q-head's
  positions in the Metal decode attend (default 8).

### Fixed
- **Decode collapsed with context depth on Metal.** Two causes, both in
  `gqa_attend`: its Born-importance pass recomputed the QK dot with one
  position per lane, so each lane walked a whole K row and the reads
  never coalesced; and the kernel ran one simdgroup per Q-head, putting
  only `num_heads` simdgroups on the device — nowhere near enough to
  hide the per-position `simd_sum` latency. The importance pass now
  reuses the main loop's lane-sliced layout, and the attend is
  flash-decoding shaped (one threadgroup per head, its simdgroups
  splitting the positions, partials combined through threadgroup
  memory). Nanbeige 4.2 on M4, decode: 9.1 → 13.8 tok/s at ctx 512,
  6.0 → 11.6 at ctx 1024. `chunk_attend` got the same importance fix.
- **Looped Transformers prefilled one position at a time.**
  `graph_prefill_preferred()` sent them through the decode graph on the
  grounds that it beat the CPU *pair* path; the real competitor is the
  batched chunk-GEMM, which amortizes each weight over the whole chunk
  (512-token prompt: 85 tok/s chunked vs 14 through the graph). TTFT at
  ctx 1024 fell from 107 s to 7.9 s.
- **Metaspace `prepend_scheme` in the PRE-tokenizer was ignored.** The
  leading `▁` was read only off a `Prepend` normalizer (the llama
  shape), so tokenizers that carry Metaspace in the pre-tokenizer with
  a null normalizer — Nanbeige 4.2 — lost it: every raw prompt encoded
  `'Hello'` where HF encodes `'▁Hello'`, and `decode` kept the leading
  space the decoder's `Strip` removes. Chat prompts hid it (they open
  with an added token, so no section is at offset 0 and neither side
  prepends). Nanbeige perplexity over the same text: 10.13 → 9.81.
  `tokenizer_parity` is green on Nanbeige 4.2, gemma-3n-E4B(-it),
  gemma-4-26B and MiniCPM3.

### Added
- **q4_tiled weights in the Metal chunk-prefill graph.** The graph
  required `q8_row` on all seven projections of a layer, so every q4t
  model — the whole Nanbeige/Bonsai class — silently fell back to the
  CPU prefill. An empty `row_scale` now marks q4t per weight, and the
  new `q4t_mul_mm_silu` kernel gives the fused down-GEMM its q4t twin.
  A non-`q8_row` embedding matrix no longer refuses the whole run
  either: the CPU fills the hidden and the graph starts from it.
  Nanbeige 4.2 on M4: 512-token prefill 6.1 s → 2.8 s (≈181 tok/s),
  1024-token 13.1 s → 6.6 s.
- `CMF_GQA_SPLIT` — how many simdgroups may split one Q-head's
  positions in the Metal decode attend (default 8).

### Fixed
- **Decode collapsed with context depth on Metal.** Two causes, both in
  `gqa_attend`: its Born-importance pass recomputed the QK dot with one
  position per lane, so each lane walked a whole K row and the reads
  never coalesced; and the kernel ran one simdgroup per Q-head, putting
  only `num_heads` simdgroups on the device — nowhere near enough to
  hide the per-position `simd_sum` latency. The importance pass now
  reuses the main loop's lane-sliced layout, and the attend is
  flash-decoding shaped (one threadgroup per head, its simdgroups
  splitting the positions, partials combined through threadgroup
  memory). Nanbeige 4.2 on a fanless MacBook Air M4: decode 9.1 → 17.6
  tok/s at ctx 512, 6.0 → 14.2 at ctx 1024. `chunk_attend` got the same
  importance fix.
- **Looped Transformers prefilled one position at a time.**
  `graph_prefill_preferred()` sent them through the decode graph on the
  grounds that it beat the CPU *pair* path; the real competitor is the
  batched chunk-GEMM, which amortizes each weight over the whole chunk
  (512-token prompt: 85 tok/s chunked vs 14 through the graph). Time to
  first token at ctx 1024 fell from 107 s to 7.9 s.

### Changed
- One Metal compute encoder per attention layer instead of eleven.
  Dispatches inside an encoder are serial on Apple Silicon and already
  see the previous dispatch's writes, so the per-step encoder was a
  GPU pass with nothing to show for it (44 virtual layers × 11 = 484
  passes per token on Nanbeige 4.2).

## [0.5.34] — 2026-07-28

### Added
- **Gemma-3n E-series** (E4B/E2B): the dedicated stack — AltUp's four
  hidden replicas with tanh-router predict/correct, LAuReL low-rank
  residuals, per-layer embeddings, KV sharing across the last 15
  layers (shared layers attend over the source layer's cache and
  never append), and gaussian-top-k activation sparsity (population
  std + Acklam's inverse normal CDF). Formulas 1:1 from transformers'
  reference; the coefficient transpose is oracle-tested against a
  literal index port. Gated end-to-end on E4B-it (7.9 GB streamed →
  3.96 GB q4t): factual answer + clean stop, correct code generation
  at 11.7 tok/s on M4. v1 runs the sequential path (batched/pair
  prefill and GPU graphs stay off for this arch).

### Fixed
- Gemma-3n declares its multi-space ▁-runs as ADDED tokens — decode
  passed them through verbatim and leaked ▁ into generated code.
  Added non-special tokens now de-metaspace like vocabulary ones in
  both the full and the streaming decoder.

## [0.5.33] — 2026-07-28

### Added
- **`cortiq_execution_info()`**: one JSON line for status/About
  surfaces — `{"simd":"neon","threads":4,"gpu_backend":true}` — with
  the thread count from the pool's own resolution (forced >
  `CMF_THREADS` > big-core topology), valid before and after load.

### Fixed
- `cortiq_worker_tids` raced pool startup: tid registration ran on
  the worker threads while the embedder reads the registry right
  after `cortiq_load` — a phone saw "· 1 threads" with four workers
  alive. Pool construction now waits for every registration (thread
  start is milliseconds); the registry is complete the moment the
  pool exists.

## [0.5.32] — 2026-07-28

### Added
- **`cortiq_cancel(handle)`** in the C ABI: thread-safe cooperative
  cancellation — the engine honours it at every prefill chunk AND
  decode step (a 50-second mobile prefill is exactly where it
  matters), finishes with `finish_reason: "cancelled"`, clears the
  flag itself, and invalidates the KV-reuse history on a mid-prefill
  abort. A server can now honour a dropped connection.

### Fixed
- The forward-layers GPU-graph gate's unset branch read "is the GPU
  on" instead of the discrete-only `wgpu_graph_default()` — the one
  site of three that skipped the rule. `cortiq_set_gpu(true)` alone
  made the 0.2 tok/s whole-token graph race-eligible on mobile
  adapters (a measured 12–14× first-token cost on Adreno). Unset now
  means the same everywhere: discrete adapters only — the app-side
  `CMF_GPU_WGPU_GRAPH=0` workaround can be deleted.
- `execution_mode` reported a stub ("Avx2 · available_parallelism"
  even on an ARM phone with a 4-worker pool). It now reports the REAL
  pool size (the same forced > `CMF_THREADS` > big-core-topology
  resolution the pool itself uses, via `Pool::effective_threads`) and
  the target's actual SIMD.

## [0.5.31] — 2026-07-28

### Added
- **Cross-turn KV reuse**: chat apps resend the whole conversation
  every turn — the engine now remembers which ids the KV cache holds
  and, when the new turn strictly EXTENDS them, prefills only the new
  tail. Turn latency becomes proportional to the new text instead of
  the whole session (previously linear per turn, quadratic per
  conversation). Extension-only — no rollback — so it is exact for
  every layer kind including recurrent state (GDN/KDA/conv rings) and
  sliding windows; an edited turn, a new chat, a skill swap or any
  scorer run falls back to the fresh-sequence path. Gated off under
  MTP, O(1) attention and task masks; `CMF_KV_REUSE=0` disables;
  `CMF_PREFILL_PROF` logs the cache hit. Gate: a reused second turn
  is token-identical to a fresh prefill of the same ids.

### Fixed
- Release pipeline: every Android artifact links with 16 KB
  max-page-size (Android 15+ rejects 4 KB-aligned .so) and a readelf
  verify step FAILS the release on a regression; the iOS static
  library ships the wgpu-Metal backend so `cortiq_gpu_available()`
  can turn true on an iPhone.

## [0.5.30] — 2026-07-28

### Added
- **Mobile GPU support in the C ABI** (cmfmobile TUNING.md engine
  items): `cortiq_gpu_available()` — true when the build carries a
  GPU backend AND the device brings an adapter up (Vulkan on Android,
  Metal on iOS/macOS), so an app can tell "GPU off" from "GPU
  impossible"; `cortiq_set_threads(n)` — worker-pool size from the
  embedder instead of the process-wide `CMF_THREADS`;
  `cortiq_worker_tids(out, cap)` — kernel tids of the pool workers
  for ADPF work-duration reporting. Verified on real cross builds
  (NDK 28, 16 KB pages): arm64-v8a `--features gpu` imports dlopen
  and carries the wgpu Vulkan backend; armeabi-v7a and x86_64 build
  too; the aarch64-apple-ios staticlib carries wgpu-Metal.
- **Native MXFP4 decode** (OCP Microscaling FP4,
  compressed-tensors "mxfp4-pack-quantized" — Kimi-K3 experts): E2M1
  nibbles + per-32-group E8M0 scales decode straight into the normal
  quantize path; E8M0 NaN refuses loudly. Validated on a real Kimi-K3
  expert fetched by HTTP Range and by an exact synthetic roundtrip.
- **Big-core detection on EAS-less kernels**: when
  `cpu_capacity` is absent, the pool sizing AND the worker affinity
  pin fall back to `cpufreq/cpuinfo_max_freq` — same cluster
  ordering, so the 62.5% big-core rule keeps working.

### Fixed
- `moe-defrag` now slices the noaux_tc selection bias
  (`mlp.expert_bias`) with its experts and accepts F16 router rows —
  a renumbered expert used to read another expert's bias on
  Kimi/DeepSeek-V3/LFM2-class routers. Gated on Kimi-Linear-48B:
  27.7 → 17.7 GB (−36%), held-out code ppl +2.7%, decode ×2.3 on a
  24 GB MacBook (the specialist fits the page cache).

## [0.5.29] — 2026-07-28

### Added
- **DeepSeek-V2 MLA** (E12): latent attention executed as
  expand-to-MHA — per-token latent projection, K heads laid out
  [rope|nope] with the DeepSeek pair-interleave undone at convert, V
  zero-padded to the K head dim and sliced before O; YaRN mscale²
  softmax-scale correction. Gated on DeepSeek-V2-Lite (ppl 8.78).
- **gemma-4 MoE 26B-A4B** (E13b): dual-branch `DenseMoe` FFN — dense
  branch from the normed input plus expert branch from the RAW
  residual through its own norm sandwich; softmax→top-8→renorm router
  with per-expert scales, its scale-less input gain folded into the
  projection columns at convert; dual-geometry attention (sliding GQA
  hd=256 + every-6th global MQA hd=512). Gated by scorer/decoder
  parity: the scorer reproduces the model's own greedy tokens 40/40.
- **Kimi Delta Attention (KDA)** — Kimi Linear / Kimi-K3 linear mixer:
  delta rule with a PER-CHANNEL log decay (diagonal, vs GDN's
  per-head scalar), separate q/k/v projections each behind its own
  causal short conv, low-rank decay stage, sigmoid-gated output
  RMSNorm; both decay-gate formulas (standard softplus and the K3
  lower-bound sigmoid). Validated against a literal port of the FLA
  reference kernels; **Kimi-Linear-48B-A3B gated end-to-end** (98 GB
  streamed → 27.7 GB q4t on a MacBook, coherent generation).
- **Kimi-K3 machinery**: compressed-q MLA (q_a→rms→q_b), the `situ`
  activation, MLA NoPE (full-attention layers apply no rotary — the
  KDA layers carry position). K3-only pieces with no public modeling
  reference (mxfp4 packing, `attn_res_block_size` residual streams,
  latent MoE, MLA output gate) refuse with named errors.
- **tiktoken → tokenizer.json in Rust**: Kimi ships a tiktoken rank
  table instead of tokenizer.json; the converter synthesizes a
  standard one — byte-level BPE vocab plus the merge list RECOVERED
  from the ranks (transformers' algorithm), specials from
  `added_tokens_decoder`. 163k-entry table converts in ~320 ms.
- **MiniCPM3** — the compressed-q gate carrier: `scale_depth`/√L
  residual scaling folded into o_proj/down_proj at convert,
  `scale_emb` via the embedding multiplier, `dim_model_base` logit
  scale as a new `logit_multiplier` header field (the head is tied to
  the embedding — the divisor cannot be baked into the shared
  tensor), and longrope's per-dim short factors as
  `rope_freq_factors` dividing inv_freq at load (they are the TRAINED
  rope inside the native window — nothing like phi's ≈1 factors).
- **Streaming hub convert**: hub checkpoints convert one shard on
  disk at a time (fetch → process → delete) — a 98 GB release
  converts on a laptop whose free disk only fits the output plus one
  shard.

### Fixed
- Teacher-forced scorers double-applied the gemma final-logit softcap
  in the sequential branches (`lm_head_forward` already caps) —
  tanh∘tanh flattered every reported gemma-class ppl; branches now
  agree within int8-activation noise.
- The fused-pair prefill cached RAW V on v-norm architectures
  (gemma-4): q/k were normalized, V was not — pair prefill diverged
  from singles (argmax parity 28/40 → 40/40). Regression-tested
  against two singles.
- Pair prefill dispatched dual-branch FFNs into an `unreachable!`;
  models whose layers have no pair arm (MLA, KDA) now fall back to
  the single-position path instead of panicking.
- gemma tokenizer.json declares `<bos>` only as an added token —
  detect it there and set add_bos, or scoring ran bos-less.
- DeepSeek MLA rope pairs are interleaved in the checkpoint (view
  d/2,2 → transpose): the converter stores rope dims even-first so
  the runtime's half-split rotation reproduces their math; the
  interleave is correctly SKIPPED for half-split (MiniCPM3) and NoPE
  (Kimi) families.

## [0.5.28] — 2026-07-27

### Added
- **`cortiq moe-mask`** — the runtime-switchable twin of `moe-defrag`:
  bakes a task's expert restriction as a first-class task mask (new
  optional expert-bitfield area in the masks section, spec §5) instead
  of cutting the file. The full expert set stays on disk; `run --task
  <name>` narrows MoE routing to the mask's experts — one file, many
  specialists. Gated: decode through a baked mask is token-identical
  to the equivalent `CMF_MOE_MASK` runtime restriction.
- **`cortiq sign`** — detached model signing: Ed25519 over the file's
  SHA-256 written to `<model>.sig` (spec §8.2); `cortiq verify` checks
  it automatically when present. Authenticity on top of the format's
  integrity hash chain, no container rewrite.
- **`cortiq compact`** — native-Rust port of the container compactor:
  reclaims dead directory/header tails left by append-only skill
  growth, streaming payloads from the source mmap.
- `moe-defrag` writes an honest provenance block
  (`provenance.moe_defrag`: tool, cover, stats hash, kept_per_layer)
  and embeds the routing B-field remapped to the new expert numbering —
  `--stats` becomes optional when re-defragging a specialist tighter.

### Changed
- Spec synchronized with the implementation: dtypes 13 (`q1s`) / 14
  (`q1t`) fully specified, MoE expert defrag contract (§11.1), pipeline
  containers (§12), expert role-contiguity SHOULD (§2.2), sparse index
  marked deprecation-pending (§7); the RU/ZH specs — a whole section
  behind — brought level with EN.

## [0.5.27] — 2026-07-27

### Added
- **`cortiq moe-defrag` — carve a task specialist out of a MoE model.**
  MoE expert usage turns out to be strongly task-conditional (measured
  on KAT-Coder 34.7B-A3B: the top-64 expert sets for code vs prose
  overlap with Jaccard 0.25 — near-disjoint), so a model serving ONE
  task carries hundreds of experts it never routes to. The pipeline:
  run the task's representative corpus once with `CMF_MOE_STATS=f.json`
  (per-layer expert-selection counts), then
  `cortiq moe-defrag model.cmf --stats f.json --cover 0.95 --output
  specialist.cmf` keeps, per layer, the smallest top expert set
  reaching that fraction of the routing mass, renumbers the kept
  experts into a contiguous prefix, slices the router's rows to match
  and drops the rest from the file. Softmax renormalizes over the kept
  set. Measured on KAT-Coder, code-calibrated at 95% cover, on an M4
  24 GB: **19.6 → 12.7 GB (−35%)**, held-out code perplexity 5.058 →
  5.198 (+2.8%) — and because the specialist now fits the machine's
  memory where the full model paged, decode goes **7.6 → 13.7 tok/s
  (×1.8)** and prefill **6.1 → 20.0 tok/s (×3.3)**.
- **`CMF_MOE_MASK=<stats.json>`** (+ `CMF_MOE_MASK_COVER`, default
  0.9): the same expert restriction applied at RUNTIME, no file
  rewrite — selection happens over the allowed set only. Use it to
  ppl-gate a mask before committing to the physical defrag (the two
  are semantically identical; measured ppl matches to 3 digits).
- `CmfModel::write_ref` (+ `TensorSpecRef`): container rewrite with
  borrowed payloads sliced from the source mmap — a 19.6 GB model
  repacks in 94 s without materializing its tensors in RAM.

### Changed
- MoE expert lists now enumerate by tensor presence up to the header
  count (a defrag'd layer keeps fewer than `arch.moe.num_experts`);
  the router's row count must match the expert count, and `top_k`
  clamps to it. Files with the full expert set load exactly as before.

## [0.5.26] — 2026-07-27

### Changed
- **`cargo install cortiq-cli` now includes the GPU backend by
  default.** The `gpu` feature (wgpu → Vulkan / DX12) is on by default
  for the CLI crate — a bare install from crates.io previously built
  CPU-only and silently missed Vulkan, while the release binaries had
  always shipped with it. `--no-default-features` restores the
  CPU-only build; native Metal on macOS was and remains always in.

## [0.5.25] — 2026-07-27

### Added
- **Routed MoE inside the whole-token GPU graph** (wgpu / Vulkan &
  DX12). A MoE layer no longer breaks the one-submit-per-token decode
  graph: the router matvec, the shared-expert gate, the top-k
  selection, the fused gate+up+SiLU over every selected expert and the
  weighted down-projection all execute on-device, in one compute pass
  per layer. Expert weights live in per-layer concatenated buffers
  (the always-on shared expert rides as the trailing block, pinned by
  the select kernel with its sigmoid weight), counted against the
  usual VRAM budget. The select kernel is fully parallel — k rounds of
  an argmax reduction with lowest-index tie-breaking that matches the
  CPU scan on routing decisions; the serial one-thread top-k it
  replaces cost 22 ms/token at k=8 across 40 layers. Scope: softmax
  routers with a shared expert and q4t expert weights, ≤256 routed
  experts, top-k <16 (the KAT-Coder / Qwen3.6-MoE class);
  sigmoid/biased routers and `CMF_MOE_TAU` keep the CPU path.
  Measured on KAT-Coder-V2.5-Dev (34.7B-A3B) on an RTX 5090: decode
  **32.8 tok/s steady vs 14.4 on the host's 32-core CPU (2.3×)**,
  15.6 ms/token forward, output token-coherent. A step-by-step
  walkthrough from GGUF download to GPU decode on both Vulkan and
  Metal is in [docs/KAT_CODER.md](docs/KAT_CODER.md).
- **q4_tiled MoE expert blocks on both per-op GPU backends** (Metal +
  wgpu) — MoE models whose experts quantize to q4t now qualify for the
  per-op expert offload path (previously q1-only), with the runtime
  probe arbitrating as usual.
- `CMF_GRAPH_PROF=1` now also prints the caller-side total per token,
  so graph-build cost is visible next to encode and submit+readback.

### Fixed
- **GPU-vs-CPU probe verdicts now compare per-arm minima** instead of
  means. A first CPU op measured mmap-cold (page-fault storm, ~3×
  steady state) could poison the CPU arm's mean and lock a losing GPU
  path for the whole process; minima measure honest steady state.
- Uploading many large weight buffers before the first submit no
  longer transiently doubles memory (the staging belt is flushed per
  MoE layer) — fixes a device OOM on discrete cards when a 17 GB
  expert set was staged all at once.
- `import-gguf --quant` help text now lists the full codec set
  (`q4t`, `q1` were accepted but undocumented).

## [0.5.24] — 2026-07-27

### Added
- **qwen35moe GGUF import** — `cortiq import-gguf` now handles the
  Qwen3.6-MoE / KAT-Coder class (GDN linear-attention hybrid +
  256-expert MoE + shared expert) natively in Rust, instead of
  refusing all SSM hybrids. The layer schedule is derived from tensor
  presence; the 3-D routed-expert tensors split into per-expert
  matrices; and every llama.cpp storage convention is undone on
  import (the baked +1 in RMS norm weights — except the GDN gated
  norm, whose weights live near 1 naturally; `ssm_a` stored as
  −exp(A_log); the tiled V-head order on every V-indexed tensor,
  including the out_proj columns). GGUF files are memory-mapped (a
  21 GB MoE file imports on a 24 GB machine) and tied-embedding
  GGUFs work. Measured on KAT-Coder-V2.5-Dev (34.7B-A3B, Q4_K_M →
  q4t): a 32-core EPYC-class CPU decodes at **16.6 tok/s where
  llama.cpp does 4.7 on the same file — 3.5× faster**; `--o1 all`
  runs the whole model O(1)-in-context (30 of 40 layers are already
  linear) and cuts KV+state at 4K context from 238 MB to **83 MB**.
- **Adaptive MoE routing** (`CMF_MOE_TAU=0.x`, opt-in): keep the
  smallest prefix of the router's top-k whose renormalized mass
  reaches τ — confident tokens touch 1–2 experts, flat ones keep all
  k. MoE decode is memory-bound (every routed expert streams its
  three matrices per token), so skipped experts are skipped weight
  traffic. Preliminary numbers on KAT (M4): τ=0.9 decodes **12%
  faster at better perplexity than the model's own fixed top-8**
  (trailing near-zero experts contribute mostly noise); τ=0.8 is
  +58% at a modest quality cost and beats fixed k=4 on both axes.
  `CMF_MOE_TOPK=N` provides the fixed-k variant. Defaults untouched.
- **Bring-up diagnostics**: `CMF_DEBUG_LAYERS=1` prints per-layer
  hidden-state rms/max with layer kinds, and the `cmpcmf` example
  numerically diffs two `.cmf` files worst-cosine-first — together
  they turned a "healthy dynamics, garbage output" import bug into a
  one-tensor pinpoint.

## [0.5.23] — 2026-07-26

### Added
- **Text-to-image on Vulkan / DX12 (wgpu)** — the DiT's quantized
  GEMMs now run on every wgpu adapter (NVIDIA / AMD / Intel desktop
  cards, Adreno / Mali phone GPUs), not just Apple Metal. The WGSL
  `q4t_mul_mm` is the register-blocked cousin of the Metal kernel
  (18-byte tile decode in the W staging; parity on an RTX 4090:
  5.2e-6 vs an exact f64 reference), and the SwiGLU FFN runs fused —
  w1/w3/silu·u/w2 as four passes in ONE submission with one readback,
  because on discrete cards the per-op submits and the PCIe round
  trips of the intermediates dominate the GEMM itself. Weights stay
  cached in VRAM. The same per-process CPU-vs-GPU probe and
  contention kill-switch gate the path everywhere: a phone GPU that
  loses simply keeps the CPU arm — enabling the GPU never makes
  generation slower.
- **`cortiq_imagine` in the C FFI** (cortiq-ffi): text → interleaved
  RGB8 into a caller buffer, with a per-step progress callback — the
  entry point mobile apps need. `guidance ≤ 1` disables CFG and
  halves the work (the right default on phones); weights stream from
  the mmap, so peak RSS stays far below the model size.

### Performance
RunPod RTX 4090 (Vulkan) vs the same pod's 32-core CPU, 30 steps,
CFG 4, the one 3.2 GB q4t file:
- 512×512: **454 s** fused GPU / ~700 s per-op GPU / ~960 s CPU
- 256×256: **147 s** GPU / 222 s CPU
Renders are visually identical to the CPU path. The remaining wall on
discrete cards is CPU-side attention and the portable f32 GEMM — the
next port targets.

## [0.5.22] — 2026-07-26

### Added
- **VAE decoder on the Metal GPU**. conv2d runs as an implicit GEMM —
  the receptive field is gathered straight from the NCHW image inside
  the kernel's staging, where the CPU path materializes a ≥2 GB im2col
  patch matrix per high-resolution conv (that matrix, not the GEMM,
  was the VAE's wall). Whole resnet blocks execute in one command
  buffer (new GroupNorm reduce/apply kernels with SiLU fused, residual
  add on-device), and the nearest-2× upsample is fused with its
  following conv so only the small pre-upsample image is uploaded.
  Conv/norm weights upload once per process. VAE decode on an M4:
  512px **7.8 → 2.9 s**, 1024px **33.4 → 14.7 s**; renders visually
  identical (parity 9.3e-4 vs the CPU conv, pixel drift unchanged).
- **Flash attention for the DiT** — experimental, opt-in
  (`CMF_DIT_FLASH=1`): online softmax, no n×n scores in device
  memory, essentially exact (3.4e-8 vs an f64 reference). The V2
  kernel loads operand blocks straight from device memory and runs
  its KV loop with zero threadgroup barriers, but still trails the
  default GEMM attention chain on M4 (15.5 vs 12 ms at 512px shapes,
  270 vs 124 ms at 1024px) — the measured dead ends are documented in
  the gate comment, and the default path is unchanged.
- **GPU micro-bench harness** (`CMF_BENCH=1 cargo test --test
  gpu_q4t_bench`): the q4t GEMM at the Lumina FFN shape, the
  attention chain at 512px/1024px sequence lengths, and the VAE
  decode (`CMF_VAE_BENCH_HW` picks the resolution). Written for a
  noisy machine: alternate A/B runs, trust medians.

### Performance
- 256×256 / 30 steps / CFG 4 now renders in **~37 s** end to end on
  an M4 (0.5.20 CPU baseline: 164 s). The 512px render is VAE-bound
  no longer; the remaining wall is the q4t GEMM itself, measured at
  ~75% of the device's practical fp32 peak (a grid swizzle and a
  pre-transposed K both landed inside noise — the X panel already
  lives in the SLC).

## [0.5.21] — 2026-07-25

### Added
- **Text → image, end to end** (`cortiq imagine`): a native Rust
  pipeline for **Lumina-Image 2.0 (2.6B)** — Gemma-2-2B prompt encoder,
  Next-DiT flow-matching denoiser, FLUX VAE decoder — each with a numpy
  parity reference on the real weights (rel/abs max|Δ| 1.4e-4 / 2.1e-5 /
  6.6e-6). CFG with per-row norm rescale, σ′=6σ/(1+5σ) shift schedule,
  P6 PPM output.
- **`cortiq imagine-pack`**: folds a diffusers tree into ONE quantized
  `.cmf` (text encoder + DiT + VAE + tokenizer + configs) — 19 GB →
  **3.2 GB**. Projections q4t (default) or q8; AdaLN modulation and
  embeddings stay q8, VAE f16, norms f32. The runtime runs the
  quantized projections straight off the mmap.
- **DiT on the Metal GPU**: every modulated DiT block executes as one
  command buffer — norms/modulation, q4t GEMMs, qk-norm + 3-axis RoPE,
  all-heads attention (scores → row softmax → P·V), and the SwiGLU FFN
  all device-resident; only the hidden state crosses the CPU boundary
  (~10 MB/block instead of ~100 MB over ~7 roundtrips). The new
  `q4t_mul_mm` kernel decodes the 18-byte tiles inside the GEMM's
  K loop — the two-pass dequant variant was bandwidth-bound on ~2.8 GB
  of scratch re-reads per FFN-shaped op. The CPU/GPU probe gained a
  wide-batch class (prompt-encode and DiT batches have opposite
  winners in the same process), and a work-proportional contention
  kill-switch (cold PSO builds exempt) drops to CPU when another
  process owns the device.
- `CMF_DIT_PROF=1` — per-stage wall-time profile of the denoiser,
  dumped when the model unloads.

### Performance
M4, 30 steps, CFG 4, the one 3.2 GB q4t file; CPU/GPU renders visually
identical (pixel drift ≤ 10/255 at 512px):
- 256×256: **48 s** GPU / 79 s CPU-only
- 512×512: **161 s** GPU / 322 s CPU-only
- 1024×1024 is now practical: the DiT forward takes 13.6 s at 4136
  tokens on the block graph.
The CPU DiT path is pool-parallel end to end (bit-exact — same-seed
renders are byte-identical to the serial code).

### Fixed
- Gemma tokenizer picked `<s>` (id 204) as BOS instead of `<bos>`
  (id 2) — both live in added_tokens and the scan order chose the
  wrong one. BOS now comes from the post_processor template; golden
  test against HF tokenizers.
- A stale probe cold-flag could permanently disarm the GPU contention
  kill-switch on a thread, and a first-use pipeline compile counted as
  device contention — cold one-off costs are now exempt everywhere.

## [0.5.20] — 2026-07-25

### Fixed
- **Mobile GPU ~75× slower than the CPU**: 0.5.19 made the wgpu
  whole-token graph both correct and default-on everywhere, but tiled
  mobile GPUs (Adreno/Mali) drain their pipeline at every barrier and
  the ~300-dispatch graph collapsed to ~0.2 tok/s. The graph now **races
  the normal path** on integrated adapters at generation granularity:
  generations alternate arms, per-token wall times accumulate, and the
  faster side wins for the process — a fast phone GPU keeps the graph, a
  slow one costs a single discarded token (the first graph token bails
  immediately when it is already >4× the measured CPU pace). Discrete
  adapters and GDN hybrids trust the graph as before;
  `CMF_GPU_WGPU_GRAPH=1/0` still forces either way.

## [0.5.19] — 2026-07-25

### Fixed
- **Mobile GPU decoded garbage**: the wgpu attend kernels need 33 KB of
  workgroup memory, but phone GPUs (Adreno/Mali) and wgpu-on-Metal cap it
  at 32 KB — the invalid pipeline silently turned attention into a no-op
  (slow AND wrong). Stride-129 kernel twins (16.5 KB) now serve
  `head_dim ≤ 128` on every device; `head_dim > 128` models fall back to
  the CPU cleanly on 32 KB devices; and pipeline validation errors now
  fail GPU init into an honest CPU fallback instead of corrupting output.

## [0.5.18] — 2026-07-25

### Fixed
- **wgpu whole-token graph never engaged from the CLI**: the gate's default
  read only the FFI toggle (`cortiq_set_gpu`), so `CMF_GPU=1` on discrete
  Vulkan/DX12 cards silently ran the slow per-op path. The graph now
  defaults ON whenever the wgpu backend is active. RTX 4090 decode:
  Bonsai-27B q1 (GDN hybrid) **5.2 → 40.2 tok/s (×7.7)**, Bonsai-1.7B q1
  65 → 154 tok/s, Nanbeige4.2-3B (Looped Transformer) 10.6 → 30.8 tok/s —
  greedy output token-identical to the CPU path on all three.
- **q4_tiled on the wgpu graph produced garbage**: `Q4Tiled` and `Q4Block`
  shared graph codec kind 2, so 18-byte interleaved tiles were fed to the
  split-layout `q4b_matvec` shader. `Q4Tiled` is now kind 5 with its own
  `q4t_matvec` WGSL kernel (u16-assembled tile reads).
- clippy `erasing_op` in the wgpu GDN ring-size expression
  (compiled only under `--features gpu`).

### Added
- **wgpu attend depth work**: vec4 K/V/Q loads (+12% short-context decode)
  and split-K flash-decoding past 256 cached positions — one 128-position
  chunk per workgroup plus a log-sum-exp merge in the same compute pass.
  4090, 1.7B q1: ctx1024 47 → 99.5 tok/s (×2.1), ctx2048 94.9 vs CPU ~40.
- **Metal whole-token graph: Q4Tiled** — new `q4t_matvec` MSL kernel
  (ushort tile reads, parity with the q4b kernel); q4t models used to
  silently fall off the graph.
- **CPU kernels**: ARM 1×4 SDOT blocking for q4_tiled (prefill GEMM ×2.1
  on Apple Silicon, mirroring the ×2.1 EPYC result), fused ternary pair
  `q1t_matvec2` (×1.83 on the MTP verify path), fused `silu(gate)·up` for
  q4_tiled pairs (+7% on AVX2).
- **AVX-512 VNNI tile kernels** for q4t/q4b/q1/q1t (`vpdpbusd`, 256-bit VL,
  bit-identical sums), default ON where VNNI exists — measured on Zen 4
  (7950X): q4t +8%, q1 +6%, q4b +4%. `CMF_VNNI_TILES=0` opts out.
- **`skill bake`**: `--target-sparsity`, `--l1-aggression`, `--ffn-align`
  (default 32 — keeps the defragged FFN on grouped codecs and SIMD fast
  paths) and `--uniform-inter` (one FFN width across layers, required by
  the whole-token GPU graphs).

## [0.5.17] — 2026-07-23

### Added
- **Sampler-Level Token Suppression (`suppress_tokens`)**: Added `suppress_tokens: Vec<u32>` field to `SamplerConfig`. Suppressed tokens have their logits set to `-infinity` prior to sampling, preventing models from generating banned initial tokens (e.g. `<think>`) on CPU/GPU.

### Fixed
- **Reasoning / Thinking Suppression**:
  - Implemented token logit suppression for reasoning tokens (`<think>`) in `cortiq-server` when `enable_thinking=false` or `think_budget=0`, forcing immediate direct answer generation in 0.1s without wasting completion tokens on thinking.
  - Added stream and non-stream `<think>...</think>` block filtering fallback in `cortiq-server`.
  - Added `chatml_fallback_opts` thinking suppression support to `Tokenizer` when no custom Jinja template is embedded.

## [0.5.16] — 2026-07-23

### Fixed
- **Reasoning / Thinking Suppression**: Improved chat template `enable_thinking=false` prefill handling in `cortiq-engine` tokenizer to match `assistant` markers without requiring a trailing newline. Fixes reasoning mode suppression for ChatML models like Nanbeige 4.2 / Qwen 3.5.

## [0.5.12] — 2026-07-23

### Added
- **Looped Transformer GPU acceleration** (Metal, Apple Silicon): both loop
  iterations now execute in a single Metal graph submission with the
  `loop_final_norm` applied on-device via `encode_loop_norm` (RMS norm + blit).
  Eliminates the CPU round-trip at loop boundaries. Nanbeige4.2-3B Q4: 5.6 →
  18.6 tok/s steady decode (3.3×), TTFT 5.2 → 1.8 s (2.9×).
- **GPU graph prefill for looped models**: `graph_prefill_preferred()` returns
  true for `loop_final_norm` models on macOS — each prompt token goes through
  the same device-attend graph as decode, doubling prefill throughput.

### Fixed
- **GPU loop_final_norm insertion**: the `continue` after `q1_graph_gpu` /
  `chunk_run_gpu` returns no longer skips the per-loop norm — the norm is
  applied either on-device (fused graph) or on CPU (fallback) before the
  next loop iteration.
- **Clippy**: resolved all CI errors — `div_ceil`, `needless_borrow`,
  `needless_range_loop`, missing struct fields in tests.

## [0.5.11] — 2026-07-23

### Fixed
- **Looped Transformer prefill**: `loop_final_norm` was only applied in the decode
  path (`forward_layers_upto`) but missing from `prefill_batch` and `forward_pair`
  (MTP speculative). This corrupted the KV cache at loop boundaries during prompt
  processing, producing garbage output for Nanbeige4.2-3B. Now all forward paths
  apply the per-loop final norm correctly.
- **GPU graph guards**: `q1_graph_gpu`, `chunk_run_gpu`, and `try_token_graph_wgpu`
  now refuse looped models (`loop_final_norm=true`) — the flat layer graph cannot
  express mid-stack norm insertion. Falls through to the correct CPU path.

### Added
- **`enable_thinking` fallback**: Templates that ignore `enable_thinking` (e.g.
  Nanbeige/Qwen-legacy) get `</think>\n\n` injected after `assistant\n` when
  thinking is explicitly disabled — the model answers directly without reasoning.

## [0.5.10] — 2026-07-22

### Fixed
- **Metal GPU Q4Block matvec**: Fixed nibble extraction order in `q4_dot8_fast`
  and `q4_dot8_half` — the v0.5.7 ILP refactor swapped lo/hi nibble lanes,
  producing garbage output for all Q4Block models on the whole-token graph path.

## [0.5.9] — 2026-07-22

### Added
- **Looped Transformer support (Nanbeige 4.2)**: Native `num_loops` + `loop_final_norm`
  architecture fields. The 22-layer Nanbeige4.2-3B re-applies its layer stack twice
  (44 virtual layers) with per-loop final normalization — 4.17B effective parameters
  from 2.1B physical weights. Conversion, inference (CPU + Metal GPU), and O(1)
  Nyström attention all work with looped models.
- **Metal GPU whole-token graph for looped models**: `q1_graph_gpu` iterates
  `total_layers()` (num_layers × num_loops) with per-loop final-norm insertion,
  device-attend KV mirror handles the growing cache across loops.

### Performance (Nanbeige4.2-3B, Apple M4, CMF_GPU=1)
- Q8 decode: **13.2 tok/s** (92% of theoretical bandwidth limit)
- Q4 decode: **20.4 tok/s** (best throughput, 2.4 GB model)
- O(1) mode: **10.2 tok/s constant** at any context (vs 2.8 tok/s exact at ctx=2048 — ×3.7 speedup)
- Q8 GPU prefill: **211 tok/s** (chunk graph)

## [0.5.7] — 2026-07-21

### Added & Optimized
- **Metal GPU Shader Optimizations (Q4 & Q8)**: Implemented 4-way ILP unrolling and register activation vector caching (`float4 xv[8]`) in `q4b_matvec`, `q8_matvec`, and `q8_matmat`, achieving up to **+21.9% decode speedup** on Apple Silicon (M4).
- **Q1T CPU Performance**: Added zero-stack bitwise register unpacking (`q1t_unpack_reg_u64s`) and 2-way ILP unrolling along with macOS physical P-core auto-discovery via `sysctlbyname("hw.perflevel0.physicalcpu")`, yielding **14.85 tok/s** on Bonsai-8B (**8.25x speedup** over single-thread baseline).
- **Quantization Parity & Verification**: Verified 100% text generation accuracy and coherence across Q1T (1.58-bit), Q4 (4-bit), VBIT (4.25-bit), and Q8 (8-bit) models.

## [0.5.5] — 2026-07-21

### Fixed
- **GPU Metal/WGSL**: Fixed MSL compilation crash caused by invalid `packed_uint` memory access in the `q1t_matvec` kernel which was forcing a silent fallback to CPU.
- **WGSL Backend**: Replaced undefined `pow3t` usage with `Q1T_LUT` table in `q1t_matmat`.
- **CPU Path**: Resolved vector allocation bottlenecks by reusing thread-local buffers (`PRESCALE_BUF`), dramatically improving inference speed.

## [0.5.3] — 2026-07-21

### Fixed
- Fixed the global GPU toggle: `cortiq_set_gpu(bool)` now correctly enables/disables the `wgpu` and `metal` device initializers. Previously they ignored the toggle because the backend selection only checked `CMF_GPU`.

## [0.5.2] — 2026-07-21

### Added
- **Mobile GPU Toggle**: Added `cortiq_set_gpu(bool)` to the FFI C ABI to allow mobile apps (e.g. Flutter) to enable/disable the discrete Vulkan/Metal graph dynamically at runtime before loading a model.

## [0.5.1] — 2026-07-21

### Added
- **GPU Metal optimizations**: ported fused `add_rmsnorm` from Vulkan.
- **Metal `TokenGraph` Q8 support**: `q8_row` tensors are now supported natively in the Metal decode graph, eliminating CPU fallback.
- **CPU `add_rmsnorm` fusion**: integrated residual addition and RMSNorm into a single SIMD pass (`add_rmsnorm_fused_into`), reducing memory bandwidth overhead.

### Added

- **Whole-token wgpu decode graph on discrete GPUs (Vulkan / DX12 / Metal).**
  The entire layer stack for one decode token is encoded into a single command
  buffer with the hidden state resident in VRAM and exactly one readback per
  token — covering q1 / q8 / q4_tiled / q1t projections and Gated-DeltaNet
  attention hybrids (Bonsai-27B, Qwen3.5), with the final RMSNorm + lm_head
  folded into the same submit so the graph hands logits straight to the sampler.
  Opt-in via `CMF_GPU_WGPU_GRAPH=1`; token-identical to the CPU f32-activation
  path. Ships portable WGSL kernels (rmsnorm, RoPE + q/k-norm, flash-decode GQA
  attention, GDN conv + delta-rule step, 1-bit and int8 matvecs).

### Changed

- **Blob laid out in execution order.** The converter now writes tensors in the
  order the engine touches them (embed → per-layer: norm/attn/norm/router/experts/
  ffn → final norm → lm_head → mtp → tail), with each layer's — and each MoE
  expert's — tensors contiguous. The kernel's up-front `madvise(WILLNEED)` readahead
  now streams the file in the same order the forward pass consumes it, so page
  faults are hidden behind compute instead of thrashing. Byte-for-byte identical
  weights; only their on-disk position changed. Readers are unaffected.
- **Large tensors are page-aligned.** Tensors ≥ 16 KB are aligned to 4096 in the
  blob (was a uniform 64 B). Cold skill / MoE-expert / mask weights now sit on
  their own page(s), so "unused weights cost 0 RSS" holds at page granularity.
  Small tensors keep the 64 B packing (no size bloat). `4096 % 64 == 0`, so
  existing readers accept the files unchanged.

### Performance

- **~2× faster GPU decode on 1-bit models (RTX 4090, Vulkan).** A 16-way
  shared-memory bank conflict in the q1 matvec's activation tile was inflating
  the FFN kernel ~8× (all 16 lanes of a row hit the same 4 banks). Padding each
  32-column group to 33 slots spreads them across 16 distinct banks; identical
  math and accumulation order, so still token-identical. Bonsai-27B q1 decode
  ~18 → ~36 tok/s pure decode; the whole-token submit halved (51 → 25 ms).
- **Token-invariant graph state cached across decode tokens.** Per-layer norm
  weights, the GDN f32 in-projections (~63 MB re-uploaded every token), q8 row
  scales, and matvec param uniforms are now uploaded once and reused, cutting
  per-token host work from ~33 ms to ~1 ms and re-upload traffic to zero.
- **Independent projections share one compute pass.** QKV, the GDN in-projection
  (qkv/z/a/b), and FFN gate/up — all reading the same normed hidden — are issued
  in a single compute pass so the GPU overlaps them instead of draining between
  per-op barriers (+5–8%, token-identical).
- **Model open touches less memory.** The in-memory tensor directory is now indexed
  by a 64-bit hash of the tensor name (with a collision-safe overflow list) instead
  of a `String`-keyed map, so opening a model no longer allocates a copy of every
  tensor name. Lookups verify the full name on hit, so there are no false matches.

## [0.4.1] — 2026-07-19

### Added

- **`cortiq_set_options` gains `enable_thinking`** (C FFI) — a sticky per-handle
  flag for reasoning-model chat templates (Qwen3/3.5). `false` makes the model
  answer directly with no `<think>` block; `true` re-enables it; absent or
  `null` leaves the current value untouched. Lets embedders (the CMF Mobile app)
  expose a "disable thinking" toggle without a bespoke API. The `cortiq_chat` /
  `cortiq_chat_messages` render path now honors it.

## [0.4.0] — 2026-07-19

Training-free **q1t** ternary post-training quantization — take an ordinary
checkpoint to ~2.25–3.5 bits/weight (below `q4`) with no retraining — and full
GPU acceleration for it on **both** engine backends: native Metal and wgpu
(Vulkan / DX12 / Intel). On a 14.8B GDN-hybrid the q1t model is 6.27 GB (−25 %
vs `q4`) and, on the GPU, *faster* than the same model in `q4` on the CPU:
decode 3.9 tok/s, TTFT 6.0 s, PPL identical to the CPU path.

### Added

- **q1t codec** (`TensorDtype::Q1T`) — per 32-group ternary `{−s,0,+s}` packed
  base-3 (5 values/byte → ~2.25 bpw) with a sparse per-row outlier overlay
  `[u32 row_ptr[rows+1]][(u16 col, f16 val)]` (4 B/outlier, no binary search).
  Built on the holographic-transfer idea: preserve the layer output `W·x`.
- **`quantize-gptq` command** — a calibration-driven, training-free path
  (`CMF_GPTQ_TERNARY=1`): two-field outlier mask (`|W|·RMS(x)`), a closed-form
  per-row output-stabilising rescale (*докрутка*), and a keep-precise skip-list
  (`CMF_GPTQ_SKIP`, `CMF_GPTQ_DOWN_KEEP`). Streams one tensor per worker;
  diagonal Hessian capture fits a 12B.
- **q1t CPU kernels** — fused sign-LUT decode (no per-weight base-3 divide),
  int8 SDOT on **ARM dotprod and x86 AVX2**, a u64-store group unpack, and a
  `value·x` overlay correction. Decode + batched prefill both accelerated;
  `CMF_SDOT=0` keeps the exact f32 path.
- **q1t GPU (Metal)** — `q1t_matvec`/`q1t_overlay` (full-precision decode),
  `q1t_mul_mm` register-blocked prefill GEMM, and integration into the
  **whole-token GPU graph** so a q1t decode token runs entirely on-device.
- **q1t GPU (wgpu)** — WGSL `q1t_matvec` + `q1t_overlay`, `q4b_matvec`, and the
  `q1t_mul_mm` prefill GEMM + `q1t_overlay_mm`, with weights resident in VRAM.
  q1t/q4 now GPU-accelerate on NVIDIA / AMD / Intel via Vulkan / DX12.
- **`q4_block` GPU kernels** (Metal `q4b_matvec`, wgpu `q4b_matvec`) — a precise
  4-bit weight (e.g. `down_proj`, `lm_head`) stays on the GPU without
  ternarizing.

### Changed

- The whole-token GPU graph's projection dispatch (`proj_abs`/`encode_proj`)
  now accepts **Q1, Q1T or Q4-block** (was Q1-only). Consequence: **q4 models
  get the whole-token GPU decode path too** — 12B `q4` decode 3.0 → 5.6 tok/s
  on an M4, where before `q4` had no GPU kernel at all.
- `dequant_q1t` takes `(rows, cols)` (like `dequant_q8_row`/`vbit`) for the
  per-row overlay.

## [0.3.12] — 2026-07-18

LFM2-MoE support: the LiquidAI **LFM2.5-8B-A1B** hybrid — short-convolution
mixers, a sparse Mixture-of-Experts FFN, and a handful of full-attention
layers — converts and runs natively. Coherent generation verified
end-to-end (q4, `<think>` reasoning + correct answers).

### Added

- **LFM2 / LFM2-MoE architecture** (`lfm2_moe`). A new `ShortConv` token
  mixer (`AttnKind::ShortConv`, `LayerType::ShortConv`): the gated short
  convolution `out_proj(C ⊙ conv1d(B ⊙ x))` with a causal depthwise
  kernel and a per-channel ring state kept in the layer's linear state —
  decode and chunked prefill share one path, verified bit-identical. The
  full-attention layers reuse the existing per-head qk-norm → RoPE path
  with no new code.
- **Sigmoid MoE routing** (DeepSeek-V3 `noaux_tc` family): a shared
  `moe_route` scores each expert with a sigmoid, adds an optional
  per-expert selection bias (`mlp.expert_bias`) to the top-k *choice*
  only (the gathered weights stay unbiased), then renormalizes with a
  1e-6 floor and a routed scale. The Qwen softmax-over-all path is
  unchanged, bit-identical.
- **Converter** maps the LFM2 vendor tensor names onto CMF's canonical
  layout (`operator_norm`→`input_layernorm`, `conv.*`→`short_conv.*`,
  `feed_forward.wN`→`mlp.{gate,up,down}_proj`, `embedding_norm`→`norm`,
  `self_attn.out_proj`→`o_proj`, …) and reads the `lfm2_moe` config
  (`conv` → `ShortConv` layers, sigmoid routing, `conv_L_cache` kernel,
  `norm_eps`).

### Fixed

- **Chat template not bundled** when a checkpoint ships it as a sidecar
  `chat_template.jinja` (LFM2, newer Qwen3 releases) rather than
  embedding it in `tokenizer_config.json`: the downloader now fetches the
  file and the converter ignores an empty one. Without it, `run` fell
  back to a generic ChatML default that did not match the model and
  produced degenerate output.

### Changed

- `cortiq info` / `story` report conv-mixer layers distinctly (e.g.
  `24 (6 full / 18 conv)`) instead of lumping them under "linear".

## [0.3.4] — 2026-07-17

The whole token on the GPU, and the prefill on the AMX. Bonsai-27B (q1)
decode on an Apple M4 goes 5 → 10–11 tok/s with the first token 12.4 → 3.5 s;
Bonsai-1.7B goes 28 → ~75–79. On q8 (Qwen2.5-0.5B, same M4, interleaved
runs, both sides at their best measured configs) CMF now decodes faster
than llama.cpp's Metal backend and its default CPU config, within 5% of
its best CPU config, and prefills within 12% (pp512 377 → ~1030).

### Added

- **Whole-token Metal graph for q1** (macOS): full-attention layers join
  the GDN block graph. First as a sandwich — norm+QKV on the GPU, one
  sync, the CPU attends (it owns the KV cache), O+FFN encode into the
  next buffer with the following GDN run (~17 syncs/token instead of
  ~64) — and then all the way: new MSL kernels for per-head qk-norm +
  partial RoPE with gate split, KV append into per-layer shared-memory
  mirrors, grouped online-softmax attend with Born importance banked via
  `atomic_float`, and the sigmoid output gate. One wait per token. The
  CPU cache stays the owner of record — any divergence (eviction,
  rollback, a CPU-path append) re-uploads the mirror, and after each
  token the appended row replays through the normal `append` +
  importance bookkeeping. Guards fall back per-layer to the sandwich
  (`CMF_GPU_ATTEND=0` forces it; `CMF_GPU_BLOCK=0` disables the graph).
- **Early commit**: the graph submits each command buffer as soon as it
  is encoded, so the GPU crunches layer N while the CPU encodes N+1 —
  continuous submission also keeps the Metal clocks warm (measured 5.8 ms
  warm vs 8.8 ms mixed per block). The token's single `sync` waits only
  on the last buffer (queue order covers the rest).
- **Hybrid q1 prefill rides the token graph**: the chunked CPU
  prefill-GEMM is walled by the sequential scalar GDN recurrence on q1
  hybrids, so the prompt now runs position-by-position through the same
  graph as decode (Bonsai-27B TTFT 12.4 → 3.5 s). Pure-attention models
  keep the batched path, where chunk-GEMM amortization wins.
- **Prefill on the Apple matrix units** (macOS): big prefill batches
  route through Accelerate `cblas_sgemm` over dequantized f32 tiles
  (scale folded in, pool-parallel dequant, tiles stay in cache) — the
  same engine llama.cpp's `-ngl 0` prefill uses. The prefill chunk is
  platform-adaptive (512 on macOS; `CMF_PREFILL_CHUNK` overrides), and
  `CMF_ACCEL=0` opts out. Decode (M=1) never takes this path.
- **Batched causal attention for prefill**: the chunk preps and appends
  every position first, then attends per KV group in two fat GEMMs
  (scores `Q·Kᵀ` with the group's Q-heads stacked into one panel, and
  `P·V` after a causal masked softmax that zeroes the invisible tail),
  with Born importance from the masked column sums. Softmax `exp` is a
  NEON Cephes-style polynomial — scalar `expf` over a long prefill's
  ~10⁸ calls would have eaten the GEMM win. The quadratic wall is gone:
  pp1024 390 → 976 tok/s. Chunks under 32 positions and non-F32 KV
  modes keep the exact per-position order.
- **`cortiq bench --core`** — llama-bench-contract timing: greedy argmax
  without the sampler's full-vocab working copy, no repetition-penalty
  pass, no per-token confidence softmax (`Pipeline::set_confidence`).
  The default `bench` still measures the full production loop. The
  clone-free greedy argmax also lands in production for every
  greedy-with-no-penalty caller.
- q1 Metal matvec goes four rows per simdgroup with per-tile processing
  (halves the L1 activation traffic per weight byte; the earlier
  four-row attempt cached the whole x block and spilled).

### Changed

- **Numerics contract, stated plainly**: GPU-graph decode and GEMM-path
  prefill are distribution-equivalent to the CPU path (first-token
  probabilities match to ~0.3%, PPL matches) but not bit-identical on
  every prompt — floating-point reductions run in a different order.
  This was already true of every GPU offload since 0.3.3; now it is
  documented instead of implied. CPU paths remain bit-exact (21 suites +
  token-for-token golden parity).

### Measured, for the record

- llama.cpp head-to-head (Qwen2.5-0.5B q8, M4, interleaved, fresh
  processes): tg128 — theirs 165.5 tok/s at its best `-t 6`, 129.4 at
  its default `-t 4`, 150.9 on Metal; CMF `--core` 151–158. pp512 —
  theirs 1168, CMF 1017–1037; pp1024 CMF 976.
- Dead ends, measured and reverted: an XOR sign-flip in the q1 kernel
  lost 23% to the `select` chain the Metal compiler already emits
  optimally; double-buffering the prefill dequant against the sgemm
  lost ~6% (Accelerate's own threads starve); a hybrid CPU∥GPU lm_head
  split on UMA lost 15% (the runtime probe had it right all along).

## [0.3.3] — 2026-07-16

1-bit models get a real GPU: Bonsai-27B (q1) decode on an Apple M4 goes
from 2.2 to 5.0–5.8 tok/s.

### Added

- **q1 on the native Metal backend**: a two-rows-per-simdgroup matvec
  kernel over the 6-byte tiles (aligned u32 pair loads, activations hot
  in L1), q1 trios in the FFN chain, q1 jobs in the batched matvec
  (QKV / GDN mixers), and a single-matvec route for out_proj/lm_head —
  all no-copy over the mmap (UMA), GPU math in plain f32 (no A8
  activation quantization at all). wgpu refuses q1 jobs honestly until
  its WGSL kernel lands.
- **Whole-block GDN graph**: a run of consecutive GatedDeltaNet layers
  executes in ONE command buffer — rmsnorm (Qwen/Gemma), mixer, causal
  conv + silu, decay/β gates, per-head l2 norms, the delta-rule
  recurrence with gated RMSNorm, out_proj, residuals and the FFN chain —
  hidden state device-resident across the block, one sync per block of
  ~3 layers instead of ~12 per layer. Recurrent states round-trip
  through shared memory, so the CPU stays their owner and prefill / MTP /
  probe paths remain coherent by construction. Anything ineligible
  falls through to the per-layer path unchanged; `CMF_GPU_BLOCK=0`
  opts out.
- q1 ops skip the runtime probe on native Metal (`gpu::q1_force`): the
  CPU q1 kernel is load-port-bound and probe alternation itself cooled
  the device between samples. Other dtypes and backends keep probing.

### Measured, for the record

- A synchronous Metal command-buffer round trip costs ~1.3 ms while
  back-to-back submits pipeline at 0.022 ms — the wall is completion
  latency, which is why the block graph (fewer submissions) is the
  design, not faster waits. A shared-buffer "fast flag" completion
  trick was tried and reverted: flag visibility does not order other
  buffers' write-backs (parity tests passed, real decode corrupted).

## [0.3.2] — 2026-07-16

The 1-bit release: a 27B in 4.8 GB on a 24 GB MacBook.

### Added

- **`q1` (dtype 12)** — 1-bit binary weights for 1-bit-TRAINED models
  (Bonsai / BitNet class): 6-byte tiles `[f16 scale][4B sign bits]` per
  32-group, 1.5 bits/weight; the scale is the group's mean |v| — the
  L2-optimal binary level, which recovers a binary-trained checkpoint's
  stored levels exactly. Explicit opt-in (`--quant q1`): as PTQ of a
  normal model it destroys quality. Fused kernels on all paths; on ARM
  the vtst mask feeds `sdot` directly (0xFF = −1) via
  `dot = −(2·sdot(mask, x) + Σx_group)` — no ±1 expansion at all, with
  per-group activation sums shared across every row. Verified
  end-to-end on prism-ml Bonsai: 1.7B q1 = 334 MB (vs 1653 MB q8) with
  greedy output token-identical to q8; 27B = 4.75 GB, ~3.2 tok/s on an
  M4 with `CMF_THREADS=10`.
- **qwen3_5 hybrid runs from safetensors**: GatedDeltaNet linear layers
  + full attention every 4th (Bonsai-27B class), 248K vocab, MTP head —
  the native converter maps it 1:1; hybrid GGUFs stay refused by
  design (the mixer tensors would be lost).
- Q1 joins `matvec_many` multi-matrix jobs (QKV / gate+up fuse again on
  new-arch models) and the four GDN input projections run under one
  pool dispatch — hybrid 27B: 449 → 353 dispatches/token.

### Changed

- **GDN/linear-core state is f32** (the vendor operator's own dtype —
  `mamba_ssm_dtype: float32`): SIMD-able elementwise state passes
  (read×2/write×1 instead of ×2/×2), heads fan out across the worker
  pool, per-worker scratch from the shared freelists. State memory
  halves; the GDN oracle stays green at 1e-3. `vmf_phase` keeps f64
  math per cell at half the storage.
- **Worker pool defaults to `CMF_POOL_SPIN=4000`** (was 0): at ~39
  dispatches/token, park-immediately paid an unpark syscall per worker
  per dispatch. Measured on M4: q8 decode +14%, q4t +27%, the 50M bench
  model +74%. `CMF_POOL_SPIN=0` remains the share-the-box serving mode.
- q8 4-row interleaved repack ships opt-in (`CMF_REPACK=1`): the
  single-stream hypothesis lost on Apple Silicon (the prefetcher likes
  four adjacent row streams more); kept for x86 experiments,
  bit-identical either way.

## [0.3.1] — 2026-07-16

The GPU release. Field report that triggered it: a 35B model (70 GB bf16 →
35 GB CMF) decoding at 1.9 tok/s on an RTX 4090 — the weights were streaming
through DDR on every token because `CMF_GPU` offload was effectively
unreachable for layer-class matrices and the release binaries shipped
without the backend. Both are fixed; the design principle that emerged is
**measure, don't trust**: enabling the GPU must never make you slower.

### Added

- **Runtime GPU-vs-CPU probe**: per op class (FFN chain, large matvec,
  prefill GEMM, QKV batch) the first calls alternate between the GPU arm
  and the pure-CPU arm, both timed; cold GPU calls (weight upload, cache
  fill) are discarded, and after six clean samples per arm the faster arm
  wins for the rest of the process. Measured on a discrete Radeon Pro 560X,
  where per-op submit+poll costs ~3–4 ms: the old always-GPU path lost 4×
  on decode and 8× on prefill against CPU AVX2 — the probe settles on CPU
  and keeps full speed; on stacks with cheap submissions the same probe
  keeps the GPU. `CMF_GPU_PROBE=0` restores unconditional offload.
- **VRAM-budget weight residency** (`CMF_GPU_VRAM_MB`, default 8192 on
  discrete cards, unlimited on unified memory): tensors become resident in
  first-touch order — decode touches layers in order, so the budget behaves
  like llama.cpp's `-ngl` without a flag. Over budget → the honest CPU path.
- **Device-class thresholds**: discrete cards take FFN/QKV-class matrices
  (≥4096 rows), unified memory only lm_head-class (≥65536) —
  `CMF_GPU_MIN_ROWS` overrides. `WGPU_BACKEND=vulkan|dx12|metal|gl` pins
  the wgpu backend.
- **Fewer polls per token**: Q/K/V projections in one device submission
  (`matvec_batch`, one pooled staging buffer for all readbacks), the dense
  FFN chain gate→silu·up→down in one command buffer with device-resident
  intermediates (the MoE block path, now also covering `q8_row`), pooled
  per-op scratch buffers, and per-tensor scale/col buffers cached across
  tokens. Field path for a dense model: 7 → 3 submissions per layer.
- **Pipeline slot pool in `cortiq serve`** (`CMF_SERVE_SLOTS`): N pipelines
  over one shared mmap check out per request — concurrent requests no
  longer serialize on a mutex.
- **`vbit_ro` (dtype 10)**: v-bit with an in-file row-offset table — readers
  index any row in O(1) instead of scanning bit-lengths; the native
  converter writes it by default, legacy `vbit` (dtype 8) stays readable
  forever, the Python reference reader handles both.
- **`q4_tiled` (dtype 11, `--quant q4t`)**: 18-byte interleaved q4 tiles
  (`[f16 scale][16B nibbles]`) — scale and payload land in one cache line.
  Kernel A/B: ARM ×1.66, x86 ×1.13; end-to-end on Qwen2.5-0.5B: prefill
  +24–32%, decode at parity, bit-identical to `q4_block` (parity-tested).
  The `q4` default stays split-layout until the x86 end-to-end pass.

### Hardened

- `validate_payload` now checks exact payload lengths for every dtype
  (v-bit included: exact bit-length sum, offsets monotonic, bounds before
  slice), and duplicate tensor names are rejected at open and shard merge.

### Fixed

- **Correction of the 0.3.0 performance claim.** The published
  "+70% pp512 / +60% tg128 over llama.cpp" table had unknowingly
  benchmarked an x86-64 build of `llama.cpp` under Rosetta 2 emulation
  (no SIMD). Against native arm64 `llama.cpp` on the same machine, CMF is
  currently **behind**: −67% pp512 / −38% tg128 on CPU (and llama.cpp's
  Metal GPU path is ~9× ahead on prefill). The README table is corrected
  and the correction is kept visible; the file-size (−26%) and
  quant-quality (+0.38% PPL) rows were unaffected.
- Release binaries now build with `--features gpu` on every platform — the
  0.3.0 artifacts shipped CPU-only, so `CMF_GPU=1` did nothing for binary
  users (the root of the field report above).
- The v0.3.0 release was missing `cortiq-aarch64-apple-darwin.tar.gz.sha256`
  (upload interrupted); re-uploaded.
- CLI logs now go to stderr — `bench --json` and piped generation output
  stay machine-parseable under `RUST_LOG`.

## [0.3.0] — 2026-07-16

The performance release: ten waves of engine work guided by the internal
performance roadmap, verified on three machines (Apple Silicon, Intel AVX2,
Xeon Granite Rapids). First like-for-like run against `llama.cpp` (b9310,
Qwen2.5-0.5B, CPU-only, 8 threads, exact attention both): **pp512 +70%,
tg128 +60%, file −26%**, with quantization quality matched (CMF q8 vs own
f16: +0.38% PPL over 12×512 windows). One model on one machine — the full
matrix is still open; reproduce with `cortiq bench --json`.

> **Correction (0.3.1):** the +70%/+60% figures above are wrong — that run
> had benchmarked an x86-64 `llama.cpp` under Rosetta 2 emulation. See the
> 0.3.1 "Fixed" section; the engine-work speedups over CMF's own baseline
> and the file-size/quality rows stand.

### Added

- **x86 SIMD kernels** (the engine previously had explicit SIMD only on
  AArch64): AVX2/FMA i8×f32 and f32 dots, `maddubs` A8W8 int8 path for q8,
  register-level q4 nibble kernels, SIMD unpack for the dominant vbit width
  (B=4), and an AVX-512 VNNI q8 path (bias-trick `vpdpbusd`, four
  accumulators). Runtime-detected; `CMF_AVX2=0` / `CMF_AVX512=0` opt out,
  `CMF_SDOT=0` keeps exact kernels on every architecture.
- **Multi-matrix jobs**: Q/K/V and gate+up projections run under a single
  worker-pool dispatch (`Pool::run_many`, `QTensor::matvec_many` /
  `matvec2_many`) on all codecs, in both the single and the MTP-pair decode
  paths.
- **Fused multi-token q4/vbit kernels**: true `matvec2` (weights unpacked
  once per activation pair) and batched `matmat` (weight row decoded once
  per prefill microbatch) — bit-identical to the per-position kernels.
- **Chunk-GEMM prefill attention**: Q/K/V and O projections run as
  chunk-level GEMMs inside `prefill_batch`; generation now uses the same
  batched prefill as `bench`/`ppl` (the pair path remains for
  dynamic-routing prompts).
- **Grouped exact-GQA attention**: all Q-heads of a KV group stream the
  shared K/V storage once per position (bit-identical per head, covered by
  parity tests in both f32 and q8 KV modes).
- **`cortiq bench --json`** with steady-state counters: allocations/token
  and pool dispatches/token are sampled per token over the same inter-token
  window as the steady tok/s.

### Changed

- **Worker pool rewritten**: shared job slot + atomic epoch + park/unpark
  instead of an `Arc<Latch>` and an mpsc message per worker per matvec; the
  caller joins the work as an extra participant. Dispatch no longer
  heap-allocates (`CMF_POOL_SPIN` tunes the spin budget, default 0).
- **Steady-state allocations cut from hundreds to ~26 per token**: reusable
  norm/projection/FFN buffers, a crate-wide buffer freelist (attention and
  FFN outputs, vocab-sized lm_head logits), allocation-free activation
  splitting, vbit row offsets precomputed at load, `select_nth_unstable`
  sampler top-k with candidates-only top-p.
- Release profile now builds with thin LTO and a single codegen unit.

### Fixed

- Metal no-copy buffers on Macs without unified memory (Intel-era discrete
  GPUs) silently returned stale data — such devices are now refused at
  init with a CPU fallback.
- Batched q4 `matmat` on non-SDOT platforms rounded differently from the
  per-position kernel (flat vs pairwise accumulation) — bit-parity restored.
- `QTensor::from_model` no longer scans the tensor directory linearly
  (O(N²) pipeline build on MoE/skills files).

## [0.2.2] — 2026-07-15

### Added

- **`cortiq ppl --o1 all|deepN|list|off`** (with `--o1-m` / `--o1-window` /
  `--o1-sink` / `--o1-prefill`, `--windows`, `--window-len`) — scores the
  **converted** model through the real streaming kernel and prints the exact
  baseline over the identical tokens next to it. The O(1) path's quality had
  never been measurable natively: the scoring path ran exact attention by
  design, so the only published numbers came from the reference probe, which
  rectifies every estimated far weight individually — a step a streaming
  operator cannot perform — and derives landmarks from the whole scored
  window. Each window's first `--o1-prefill` tokens run the exact pass that
  freezes the landmarks; every scored position then goes through
  `NystromState::step()`, the same code decode runs. The default is
  unchanged: `ppl` scores the backbone exactly even for a model carrying an
  `--o1` hint.
- **`--o1-rect agg|fm`** (and `CMF_O1_RECT`) — selects how the indefinite
  skeleton is rectified. `agg` (default) clamps only the aggregate far
  denominator; `fm` clamps `FM = F_u·M_u` per query row, which is the
  intuitively "correct" per-key guarantee and, measured, the worse one
  (×1.296 vs ×1.414 at the default landmark budget). `agg` wins at every m.
- Prebuilt **Windows** binaries in GitHub Releases — `x86_64-pc-windows-msvc`
  and `aarch64-pc-windows-msvc`, shipped as `.zip` + `.sha256` (the
  convention there) rather than `.tar.gz`; the ARM64 row cross-compiles from
  the x86_64 runner. The runtime needed no porting: Metal is gated behind
  `cfg(target_os = "macos")`, the NEON/SDOT kernels behind
  `cfg(target_arch = "aarch64")`, and `memmap2` covers Windows.
- The release workflow accepts `workflow_dispatch`, so the binaries for an
  existing tag can be rebuilt on demand.

### Changed

- **The O(1) exact window, sink buffer and landmark keys (K̃) are now shared
  per KV group.** The window ring and sink buffer hold the *group's* keys and
  values, and K̃ is `seg_means` over those same keys, so under grouped-query
  attention every query head in a group was storing byte-identical copies.
  `NystromState` is now one state per KV group — a shared `NystromGroup`
  (ring, sinks, K̃, `m_eff`, geometry) plus a `Vec<NystromHead>` for what
  genuinely depends on the head's queries: the far accumulators and their
  running maxima, Q̃, and the mixing matrix `M = pinv(exp(Q̃K̃ᵀ/√d))`.
  Eviction becomes a group event — `advance()` evicts a position once and
  each head then absorbs that key into its own accumulators before the slot
  is reused (one eviction, one insertion per head, which is the invariant the
  partition rests on). **Arithmetic is untouched and the output is
  bit-identical**, proven three ways: a 4-head group and 4 independent
  single-head states agree on `to_bits()`; on a real 4B hybrid, greedy
  generation from a 370-token prompt matches on token ids and top-1
  confidences to 1e-6 — also with a narrow `W=16 m=8 sink=2` window that
  maximizes evictions; and `ppl --o1 all` reproduces to the digit.
  `fcd_runtime_parity` is unmoved at 9.373e-7 against its pinned 9.4e-7. A
  dedicated test asserts each head's `far_len` equals the eviction count and
  closes the books with `far_len + w + sink == t`; it was verified to have
  teeth by injecting a double insert (the bit-identity test alone does *not*
  catch that mutant, since both paths share `advance`). Measured (qwen3_5 4B
  hybrid, 16 q-heads / 4 kv-heads, head_dim 256, `W=128 m=32 sink=4`, Apple
  M4): nystrom state **47.9 → 18.8 MB** (÷2.55), KV+state **153.2 → 124.1
  MB**, and against plain KV at ctx 4096 **÷2.48 → ÷3.06**; the crossover
  where `--o1` starts *saving* memory moves **731 → 287 tokens**.
- **Dynamic row chunking in the thread pool** — `Pool::run_rows` hands out
  row ranges from an atomic cursor instead of a static 1/n split, so a
  performance core takes several chunks per efficiency-core chunk instead of
  waiting at the latch; on an asymmetric-core machine the cores no longer
  wait on each other. Rows stay disjoint, so output is bit-identical.
  Measured: weight-path bandwidth 54.5 → 58.9 GB/s (+8%), decode +4–5% at
  every thread count on a 4B q8_2f model.
- **Corrected O(1) conversion quality figures.** Measured through the shipped
  streaming kernel on held-out wikitext, landmarks sealed from a 256-token
  prefill, scoring only the drift rows (the harshest region): Qwen3-0.6B with
  28/28 layers converted ×1.296; a Qwen3.5-4B hybrid with 8/32
  converted ×1.132. The ×1.177 previously in the docs was the reference
  operator with whole-window landmarks — an upper bound this runtime cannot
  reach by construction. Corrected in the module docs and the
  `convert --o1` help.

- `cortiq run` defaults to the `warn` log level — the loader's INFO lines are
  noise in front of an answer. `RUST_LOG` overrides; every other command
  keeps `info`.
- `convert` / `import-gguf` paint one in-place progress line on a terminal
  instead of several hundred `@PROGRESS` lines. The markers are byte-for-byte
  unchanged when stdout is not a terminal, which is where supervisors parse
  them.

### Fixed

- **`cortiq run` is a chat again.** It advertised "Interactive chat mode" but
  never rendered the container's chat template — `generate()` encodes the
  prompt verbatim — and `generate_from_ids` clears the KV cache per call
  ("Fresh sequence"), so the interactive loop carried no history either. The
  first command a new user runs answered correctly and then repeated "The
  answer is correct." until `max_tokens`; `finish: stop` was unreachable,
  because raw completion never emits `<|im_end|>`. `run` now renders the
  file's template through `apply_chat_template_opts` — the same call the
  server makes — and carries the conversation across turns. The gate is
  `chat_template.is_some()`, **not** the template call itself: with no
  template that helper falls back to hardcoded ChatML, which is not what a
  base model wants, so those still run completion — as does `--state`, whose
  frozen prefix is a raw token replay. A long chat drops its oldest exchange
  (never a system turn) rather than prefill past the RoPE range.
  - `--raw` — skip the template: the previous behavior, verbatim.
  - `--no-think` — render with `enable_thinking=false`; Qwen3/3.5 answer
    directly instead of emitting a `<think>` block.

- **`cortiq fcd` polished an operator the runtime never serves** — the
  trainer built its far field from whole-window landmarks and the per-(t,j)
  clamp. It now seals landmarks from a prompt prefix (`NysCfg.prefill`,
  default `t/2` — the same discipline `ppl --o1` uses), derives `m_eff` from
  the sealed prompt, runs the aggregate far-denominator guard with raw
  negative mass kept on passing rows, and leaves pre-seal rows exact. A new
  `fcd_runtime_parity` test pins the trainer forward against the live
  `NystromState` at 9.4e-7 (tol 2e-5), while the per-key rectifier differs by
  5.7e-2 on the same fixture — the test cannot pass a trainer that reinstates
  the clamp. The trainer-reported zero-shot ratio moves ×1.168 → ×1.146 on
  its own windows (teacher identical, a clean control).
- **`o1_seal`** now requires `num_heads % num_kv_heads == 0` and degrades to
  exact attention instead of panicking on an index overflow.

## [0.2.1] — 2026-07-14

### Added

- **`enable_thinking`** — `/v1/chat/completions` accepts `enable_thinking`
  (top-level) or the vLLM-style `chat_template_kwargs.enable_thinking`.
  `false` renders the chat template with `enable_thinking=false` — Qwen3/3.5
  prefill an empty `<think>` block and answer directly. Absent = the
  template's default. The tokenizer gains `apply_chat_template_opts`; the
  render context defines the variable only when it is set.

### Changed

- README: an O(1) conversion quick-start — the `convert --o1` commands, the
  `run` / `serve` / `bench` overrides, `CMF_O1`, the tuning knobs, and the
  `cortiq fcd` polish stage.

### Fixed

- **Corrupt published crate tarball** — `cargo package` deterministically
  corrupted the tarball on the previous `README.md` byte layout; a trailing
  newline works around it.

## [0.2.0] — 2026-07-14

### Added

- **O(1) constant-memory streaming attention conversion** — `cortiq convert
  --o1 all|deepN|list` (with `--o1-m` / `--o1-window` / `--o1-sink`) converts
  any softmax checkpoint to per-layer O(1) attention in seconds, with the
  **weights byte-identical**: the conversion records a hint in provenance and
  the binary envelope is unchanged. The kernel (`nystrom.rs`) is an exact
  sliding window plus a PSD far-field skeleton under a single joint
  denominator, with permanent sink tokens (the first `S=4`, which never enter
  the far field), per-landmark flash-style running-max accumulators, and
  delayed insertion — a key enters the far state only when it leaves the
  exact window. Guards: short-prompt exact mode, `m_eff = clamp(T/8, 4, m)`,
  and a ridge pseudo-inverse (f64 Cholesky) with jitter fallback. At runtime
  prefill runs exact attention, then `seal()` builds the landmarks and `M`
  per head, replays the prompt into the state and **drops the layer's full
  KV**; seal refuses on q8 KV and masked-sparse heads, the speculative pair
  path is disabled under o1, and eviction no-ops on sealed layers. Dispatch
  priority: CLI > `CMF_O1` env > the `provenance.o1_attn` header hint. Golden
  parity vs the validated reference math: max 1.1e-6 (sink=4). Measured (M4,
  Qwen3-0.6B q8, `--o1 all`): ctx 4096 decode 19.6 → 68.6 tok/s (×3.5) at
  84.9 MB constant state vs 954 MB KV (÷11.2); ctx 1024 ×1.5 / ÷2.9 — decode
  is near-flat in context length. (The zero-shot quality ratios published
  with this release came from the reference probe rather than the shipped
  kernel; corrected in 0.2.2.)
- **Native FCD restoration trainer** — `cortiq fcd <model.cmf> --corpus …`
  (`--steps`, `--eval-every`, `--kl`, `--gen-check`, `--gen-gate`,
  `--gate-threshold`, `--gate-slack`, `--out`): the bounded KL-anchored
  polish stage for `--o1` conversions, with **no ML framework** — one binary
  end to end. `fcd_ops.rs` is a fixed-graph op library with hand-derived
  backwards over an `Fp` trait (pooled f32 GEMMs, RMSNorm plain and
  zero-centered, RoPE, SwiGLU, segment means, exact causal attention,
  Nyström-joint attention, GatedDeltaNet BPTT, and CE + KL(teacher‖student)
  loss); every op carries a central finite-difference gradcheck (rel err
  1e-9…1e-12; whole-graph block checks ≤ 8.9e-4; GDN forward parity vs the
  runtime kernel 3.4e-8). Teacher and student share one frozen mmap and the
  trainable set is only the normalization gains and FFN tensors of converted
  layers (AdamW, grad clip, deterministic held-out eval, best-checkpoint
  restore, `provenance.fcd` on the written tensors). **Generation-gated
  selection**: each eval probes greedy long-context generation through the
  real streaming kernel and admits a checkpoint only if no prompt loops — if
  none passes, the zero-shot state is restored, so the stage cannot make
  generation worse than conversion alone. The motive is measured: on a
  6/24-softmax hybrid, ppl-only selection reached ×0.86 teacher ppl yet
  regressed all three generation probes into loops.
- **hybrid_k core support** — the vmf_phase linear core now honors an
  optional selective-write gate: `model.layers.{i}.vmf_attn.k_gate.weight`
  `[nh, hidden]` + `.bias [nh]`; κ_h = σ(W_k·x + b)_h multiplies the state
  write (`S = decay·S + κ·φk⊗v`). Presence-driven: files without the
  tensors run the classic phase kernel unchanged. Mechanism-level basis
  («phase + input gate», stage 71): fastest convergence and best/tied
  accuracy across the recall grid, correlated-noise robustness the bare
  phase kernel lacks, and an LM crossover vs softmax at SEQ 512.

- **NEON decode attention** — `attention_head` score/weighted-sum loops and
  the q8-KV `attend` branches now run through NEON kernels (`dot_f32`,
  `axpy_f32`, per-group `dot_i8_f32`, `axpy_i8_f32`). Measured on
  Qwen3-0.6B q8 (28 full-attention layers, teacher-forced 1536 tokens,
  interleaved rounds): **×1.61 wall-time** (29.5 s → 18.3 s); the gain grows
  linearly with context depth. PPL 22.053 → 22.084 (+0.14%, summation
  regrouping only).
- **Long-context bench mode** — `cortiq bench --ctx N` builds a synthetic
  N-token prompt, raises `CMF_MAX_SEQ` so eviction cannot skew the curve,
  and prints `Memory: KV+state X MB at seq_len N` (O(context) KV for
  full-attention vs O(1) state for the linear core, measured).
- Hot-path hygiene: `row_dot` (active-neuron path) NEON for q8_row/q8_2f
  (new `dot_i8_col_f32` folds the θ col-field without a prescaled copy);
  vbit SDOT per-row heap allocation replaced by a per-worker scratch
  (lm_head ≈ 150k rows/token); `prescale` returns borrowed activations
  for non-q8_2f dtypes (was an unconditional copy per matvec). Short-ctx
  q4 decode +4% (64.0 vs 61.6 tok/s, interleaved).

- **q4 SDOT decode path** — `q4_block` matvec now runs through the A8W8
  int8 `sdot` kernel on ARMv8.2+ (nibbles → centered i8 per 32-group, exact
  outlier correction), replacing the scalar inner loop. Measured on
  Qwen3.5-0.8B q4 (M4, interleaved runs): decode 5.3 → 14.4 tok/s (×2.7),
  prefill 7.3 → 24 tok/s (×3.3), PPL 4.008 → 4.022 (+0.35%, bounded A8W8
  noise — the same contract as q8/vbit). `CMF_SDOT=0` keeps the exact
  scalar path.

### Fixed

- **The `bench` memory line under-reported a fully-folded model** — an
  all-linear model reported `KV+state 0.0 MB` because the recurrent state
  (f64, constant in context) was not counted. Both cache kinds are now
  honest: the folded 0.6B reports its analytic 58.7 MB constant state against
  242 → 946 MB of growing KV for the softmax original.
- **The `x86_64-apple-darwin` release binary is published again** — the
  retired `macos-13` runner pool left the Intel job queued with zero steps
  for 24 h before being auto-cancelled, losing that asset on v0.1.8, v0.1.9
  and v0.1.10. It now cross-compiles on `macos-latest`, with a 30-minute
  timeout so a stuck pool fails loudly instead of silently dropping the
  binary.

## [0.1.10] — 2026-07-09

### Added

- **Physical defragmentation** — `cortiq convert --defrag <skill_dir>` drops
  pruned FFN neurons so they are neither stored nor computed (Patent 2 claims
  9/10; spec §11). The mask overlay (§5) is virtual sparsity — the full tensors
  stay on disk; defrag bakes one task's keep-set into the weights and emits a
  standalone, smaller dense `.cmf`. Per-layer variable: each layer shrinks to
  its own live-neuron count (no global-max bottleneck). The keep-set comes from
  an explicit `ffn_keep.npy`, or is autodetected from zeroed `down_proj` columns.
  Native Rust (minimal `.npy` reader); masks are dropped; provenance records the
  pre/post neuron counts. FFN output is bit-identical to the masked model before
  quantization.

### Changed

- The FFN dims are derived from tensor shapes throughout; the loader now
  enforces the FFN triple invariant (`gate.rows == up.rows == down.cols`,
  `down.rows == hidden`) loudly, and runtime telemetry reports per-layer neuron
  counts from the actual shapes rather than the nominal `intermediate_size`.

## [0.1.9] — 2026-07-08

### Added

- **Native v-bit quantization** — `cortiq convert --quant vbit` /
  `cortiq import-gguf … --quant vbit` now encode the grouped variable-bit format
  in Rust (no Python): per-row bit-width (3–8, water-filled by log2 row
  amplitude toward a 4.25-bit budget), per-32-group f16 scale, MSB-first packing
  — byte-compatible with the `cortiq-core` v-bit reader. A round-trip unit test
  and a real-model convert→run confirm it (≈40% smaller than q8, coherent
  output). Only the **GPTQ-calibrated** v-bit variant (which needs an activation
  Hessian) still uses the Python converter; the weight-only path is fully native.

## [0.1.8] — 2026-07-08

### Fixed

- **f16 subnormal decode bug** (`cortiq-core`) — `f16_to_f32` computed the
  subnormal exponent as `127-15-e`, one too small, which **halved every
  subnormal half-float**. This corrupted GGUF K-quant super-block scales (which
  are frequently subnormal), producing garbage output. The biased exponent is
  now `113-e`; covered by new round-trip tests. It also slightly affects any
  runtime f16 weight that happened to be subnormal.

### Added

- **Full GGUF quant coverage** in `cortiq import-gguf` — every common ggml type
  is now dequantized natively (no Python): `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`,
  `Q8_0`, the K-quants `Q2_K`/`Q3_K`/`Q4_K`/`Q5_K`/`Q6_K`, `Q8_K`, `BF16`, and
  the non-linear-codebook `IQ4_NL` / `IQ4_XS` (used inside `q2_k`/`q3_k` mixes).
  Each codec is a faithful port of ggml `dequantize_row_*`; Q4_K/Q5_K/Q6_K have
  unit tests against fp16 ground truth, and all nine Qwen2.5 GGUF quantizations
  convert and generate coherently. Only the `IQ1`/`IQ2`/`IQ3` grid codebooks
  remain unsupported (an honest error, never silent garbage).
- **`cortiq import-gguf <owner/repo>`** now accepts a Hugging Face repo id (the
  best natively-supported `.gguf` is picked and downloaded in parallel), or
  `owner/repo/file.gguf` for a specific file, or a local path. `--hf-token` for
  gated repos. A linear-attention / SSM (GatedDeltaNet) GGUF is refused with a
  clear message pointing at the safetensors path — never silently mangled.
- **Native fused-GatedDeltaNet split** in `cortiq convert` — qwen3_next /
  AgentWorld checkpoints that fuse the GDN projections (`in_proj_qkvz` /
  `in_proj_ba`, group-interleaved) are split into the canonical hub tensors
  natively, so those models no longer need the Python converter. The split is a
  pure row permutation with a unit test; it is not yet generation-verified on
  real fused weights (no small public fused checkpoint exists).
- A GGUF-only repo passed to `cortiq convert` now returns an actionable error
  (use `import-gguf`, or convert the source safetensors repo) instead of a raw
  404 on the missing `config.json`.

## [0.1.7] — 2026-07-07

### Added

- **GatedDeltaNet linear attention** (Qwen3.5 hub layout) in `cortiq convert` —
  the per-layer linear/full schedule, the canonical GatedDeltaNet core, the
  zero-centered `(1+w)` norms, and the multimodal-wrapper tensor names are all
  handled natively. Validated: Qwen3.5-0.8B converts and generates identically
  to the reference Python converter. Fused qwen3_next / AgentWorld checkpoints
  (interleaved `in_proj_qkvz`/`in_proj_ba`) still use the Python path.

## [0.1.6] — 2026-07-07

### Added

- **`cortiq import-gguf <file.gguf> --output model.cmf`** — a native Rust GGUF
  importer (F32 / F16 / Q8_0; llama / qwen2 / qwen3), which also reconstructs a
  Hugging Face tokenizer.json from the embedded ggml metadata. No Python.
  K-quants (Q4_K / Q5_K / Q6_K) still use the Python importer.
- **Mixture-of-experts** in `cortiq convert` — the router and per-expert matrices
  are converted and the runtime dispatches the sparse FFN (qwen2-moe / qwen3-moe).

## [0.1.5] — 2026-07-07

### Added

- `cortiq convert --quant q8_2f` — the two-field (𝒲×θ) int8 quantization that
  recovers most of the int8→fp16 quality gap at the same file size.
- Converter round-trip tests (q8 / q8_2f / q4 encoders + a tiny end-to-end
  convert) run in CI.
- A release workflow that attaches prebuilt `cortiq` binaries (Linux x86_64,
  macOS arm64 / x86_64) to each GitHub Release — usable with no Rust toolchain.

### Changed

- **Byte-faithful, lighter conversion**: round-half-to-even quantization (matches
  numpy — weights are now byte-identical to the reference converter), and the
  input safetensors are memory-mapped and processed one tensor at a time, so peak
  RAM is ≈ the output size rather than the whole model.
- **Resilient downloads**: each byte-range chunk retries with exponential backoff
  and shows a live percentage.

## [0.1.4] — 2026-07-07

### Added

- `cortiq convert --model <owner/name>` now accepts a **Hugging Face repo id**
  directly and downloads it (config, tokenizer, and safetensors weights) before
  converting — the whole HF → `.cmf` pipeline lives in one place, no external
  tooling. `--hf-token` for gated/private repos.
- **Parallel downloads**: weight files are fetched in concurrent 32 MiB
  byte-range chunks over reused connections (saturates bandwidth for both a
  single large file and sharded models). Tunable via `CORTIQ_HF_THREADS`
  (default 8). Downloads are cached under `~/.cache/cortiq/hf`.

## [0.1.3] — 2026-07-07

### Added

- **`cortiq convert`** — a native Rust converter from a Hugging Face checkpoint
  (`config.json` + `*.safetensors` + `tokenizer.json`) to `.cmf`, with **no
  Python / numpy / torch dependency**. Reads safetensors and quantizes in Rust
  (q8 / q4 / f16), embeds the tokenizer and chat template, and writes via
  `cortiq_core::CmfModel::write`. Standard dense transformers (qwen2 / qwen3 /
  llama / mistral-style); output is generation-identical to the reference
  Python converter. MoE / linear-attention models still use the Python path.

## [0.1.2] — 2026-07-07

### Added

- `cortiq serve --host <HOST>` to control the bind address (default `0.0.0.0`;
  set `127.0.0.1` for a local-only server).
- A `/healthz` liveness endpoint on the server — for process managers that embed
  `cortiq serve` as a local model backend (e.g. an LLM gateway).

## [0.1.1] — 2026-07-07

### Added

- `cortiq run --max-tokens <N>` (short `-n`) to cap the number of generated
  tokens (default 256); previously the generation length was fixed at 256.

## [0.1.0] — 2026-07-07

Initial public release.

### Added

- **`cortiq-core`** — the CMF v2 on-disk format: 128-byte envelope, section
  table, memory-mappable tensor directory, tokenizer and chat-template records,
  per-task mask records, and per-skill full-shape replacement-tensor delta
  records with a byte-offset delta index.
- **Quantization codecs** — including the two-field `q8_2f` (scale × phase)
  path and v-bit stacking, with golden round-trip and parity tests.
- **`cortiq-engine`** — a dependency-free runtime that memory-maps a container
  and runs inference on **CPU or GPU**. Overlay execution reads per-skill
  replacement tensors *in place of* the shared backbone at forward time without
  materializing a separate model. Optional `gpu` feature uses a portable wgpu
  backend (Vulkan / Metal / DX12) with CPU/GPU parity.
- **`cortiq-server`** — an optional axum-based HTTP serving layer.
- **`cortiq-cli`** — the `cortiq` command-line binary for inspecting, converting,
  and running containers.
- **Converters** — self-contained Python tooling to produce `.cmf` files from
  source models, plus a pure-Python reader for inspecting containers.
- **Documentation** — the CMF v2 specification and a comparison against GGUF,
  safetensors, ONNX, PyTorch `.pt`, GGML, and TensorRT, in English, Russian,
  and Chinese.
- **Developer tooling** — `Makefile` and `justfile` shortcuts, a pinned
  `rust-toolchain.toml`, GitHub Actions CI (build + test on Linux and macOS,
  clippy, rustfmt), and contributor / community-health docs
  (`CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, issue/PR templates).
- **Licensing** — Apache-2.0 with an explicit patent-grant explanation
  (`LICENSE`, `NOTICE`, `PATENTS.md`).

[Unreleased]: https://github.com/infosave2007/cmf/compare/v0.5.62...HEAD
[0.5.62]: https://github.com/infosave2007/cmf/compare/v0.5.61...v0.5.62
[0.5.61]: https://github.com/infosave2007/cmf/compare/v0.5.60...v0.5.61
[0.5.60]: https://github.com/infosave2007/cmf/compare/v0.5.59...v0.5.60
[0.5.59]: https://github.com/infosave2007/cmf/compare/v0.2.2...v0.5.59
[0.2.2]: https://github.com/infosave2007/cmf/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/infosave2007/cmf/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/infosave2007/cmf/compare/v0.1.10...v0.2.0
[0.1.10]: https://github.com/infosave2007/cmf/compare/v0.1.9...v0.1.10
[0.1.9]: https://github.com/infosave2007/cmf/compare/v0.1.8...v0.1.9
[0.1.8]: https://github.com/infosave2007/cmf/compare/v0.1.7...v0.1.8
[0.1.7]: https://github.com/infosave2007/cmf/compare/v0.1.6...v0.1.7
[0.1.6]: https://github.com/infosave2007/cmf/compare/v0.1.5...v0.1.6
[0.1.5]: https://github.com/infosave2007/cmf/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/infosave2007/cmf/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/infosave2007/cmf/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/infosave2007/cmf/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/infosave2007/cmf/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/infosave2007/cmf/releases/tag/v0.1.0
