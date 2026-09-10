# Adapters at runtime

A LoRA is `y = x·Wᵀ + s·(x·Aᵀ)·Bᵀ`. `cortiq` evaluates that second term
beside the base projection instead of folding it in, for both video models:

```bash
cortiq animate    model.cmf --prompt "…" --lora adapter.safetensors --lora-strength 0.8
cortiq ltx-video  model.cmf --prompt "…" --lora adapter.safetensors --ref subject.ppm
```

## Why the branch is not merged

The container's weights are `q4tp` — four-bit nibbles on a per-row scale
ladder. Folding a rank-32 update into them means dequantizing the whole DiT,
adding, and requantizing: the file's size in RAM, and the codec's error budget
spent on an update the size of a rounding step. Merging happens at *pack*
time instead, where the source checkpoint is still bf16
(`cortiq animate-pack --lora … --lora-scale …`, which is how the Turbo LoRA
gets into the MiniMax-H3 files). At *run* time the branch rides beside the
base and the adapter stays a 130 MB file you can swap per render.

## Which names bind

Three naming conventions are accepted and normalized to the container's own
tensor names:

| adapter writes | example |
|---|---|
| `diffusion_model.…` | ComfyUI single-file adapters |
| `base_model.model.dit.…` | PEFT / `peft`-trained adapters |
| `transformer.…` or the bare module path | diffusers exports |

Both `lora_A`/`lora_B` and `lora_down`/`lora_up` spellings are read, at F32,
F16 or BF16. `alpha`/`lora_alpha` in the file's metadata is honoured as
`scale = strength · alpha / rank`; a file that records neither is taken at
`strength` as-is, which is what the diffusers loaders do.

**MiniMax-H3** binds `blocks.N.attn.qkv_proj`, `blocks.N.attn.out_proj`,
`blocks.N.mlp.fc1`, `blocks.N.mlp.fc2` and the same four on
`token_refiner.blocks.N`. **LTX-2.5** binds the `transformer_blocks.N`
attention and feed-forward projections, plus the `reference_slot_embedding`
that the multi-subject adapters carry.

`adaln_proj.linear` does **not** bind on a MiniMax-H3 container. That weight
is [96768, 2688] per block — 13 B parameters, 40% of the released model — and
the packer collapses it onto a rank-24 curve over the timestep, which is what
makes the file 14 GB instead of 60. An adaLN update can be folded into that
curve exactly (its rank-r update becomes r extra basis columns and r extra
table columns), but only where the time embedding is still available, i.e. at
pack time: `animate-pack --lora … --time-embedder …`. At runtime the container
no longer carries the embedding, so those branches are reported as not applied
rather than silently dropped — an adaLN-carrying adapter reports it like this:

```
lora: rank 16, 208/258 branches bound; not applied: blocks…linear ×50
```

## What an adapter costs, and why

Not its arithmetic. At rank 32 against MiniMax-H3's qkv projection the branch
is 2·(5376·32 + 32·21504) multiply-adds against the base's 5376·21504 — 0.5%.
What it costs is **placement**:

1. **Never write the branch as scalar loops.** Rank 128 over LTX's 480
   branches cost 40 s a step written that way, against 9 s for every base
   GEMM in the block. Both halves are N·Kᵀ products; they go through
   `fcd_ops::gemm_nt` like everything else.

2. **Pin the branch to the host when it stands beside a device-resident
   quantized GEMM.** The generic `GemmNt` probe will send these small f32
   products to the GPU, where they queue behind the q4tp projection they
   accompany and pay a submit and a readback each: 39.7 s a step through the
   probe against 22.2 under `gpu::cpu_scope`, for arithmetic worth 1.1 s.

3. **A branch has to read the panel its projection produced — so it stands
   the fused kernels down.** This is the real cost, and it is not the branch's
   own GEMM: on Metal both models put the branch INSIDE the base GEMM's
   submission (`q4tp_matmat_lora`), reading the activation already uploaded
   and accumulating into the output already written, for no transfer at all.
   What an adapter still costs on MiniMax-H3 is the *attention* fusion:
   `dit_qkv_attn_out` keeps qkv, the attention and the output projection on
   the card with nothing in between, and a branch on `qkv_proj` or
   `out_proj` needs exactly those panels, so the block falls back to the
   chain. Measured on an M4 at 512×288, 22 frames, back to back with memory
   freed: **denoise 133.6 s with the adapter against 126.1 s without — 1.06×**,
   and the video VAE (no adapter anywhere near it) moved 53.4 → 56.9 s between
   the same two runs, so the cost is at or below the machine's own noise.
   Under memory pressure earlier in the session the same pair read 1.13× and
   1.37×, because the base render itself drifted from 22.2 to 32.5 s a step —
   a reminder that on a machine where weights stream, an adapter's extra work
   hides inside the streaming.

## The router

`CMF_LORA_ROUTE=<r>` measures each branch's contribution on the first step it
runs — `‖s·ΔY‖ / ‖Y‖`, the branch against the base panel it is correcting —
and switches off every branch below `r` for the rest of the render. A branch
that is off gives its projection's fused device path back, which is where the
saving comes from: routing does not save the branch's flops, it saves the
fusion the branch was costing.

`CMF_LORA_PROBE=1` prints the same measurement for every branch, loudest
first, without switching any of them off. Both make a branch take the split
path until it has been measured — the fused kernel never separates the branch
from the base, and a probe that cannot see the branch reports `0.0000` for
every one of them, which is what the first run of it did.

Adapters are not uniform across depth, and that is what makes routing worth
doing. `fal/MiniMax-H3-Realism-People-LoRA` on MiniMax-H3, 512×288, 22 frames,
measured on an M4 — 104 branches at rank 32, of which **41 survive `r = 0.02`**:

```
lora branches by contribution ‖sΔY‖/‖Y‖ (41 of 104 live):
    0.1633  on   blocks.13.attn.qkv_proj
    0.0946  on   blocks.9.attn.qkv_proj
    0.0937  on   blocks.14.attn.qkv_proj
    0.0747  on   blocks.16.attn.qkv_proj
    0.0744  on   token_refiner.blocks.1.attn.qkv_proj
    …
    0.0031  off  blocks.2.attn.qkv_proj
    0.0025  off  blocks.0.attn.qkv_proj
    0.0011  off  blocks.1.attn.qkv_proj
```

The loudest branch is **150× the quietest**, the first two blocks contribute
nothing this adapter would miss, and what survives sits in the middle third of
the stack (blocks 9–17 and a tail around 30–35). The picture agrees: against
the base render, the full adapter is 13.76 dB PSNR and the routed one 13.34 —
routing keeps the adapter's look (16.75 dB against the full adapter, i.e.
much closer to it than to the base), on 41 branches instead of 104.

What routing does **not** buy, measured: on Metal it did not shorten the step
at `r = 0.02` (33.8 s routed against 33.7 s with every branch live). The
reason is visible in the design — the branch's own cost is already fused into
the base GEMM, so the only thing left to win back is the *attention* fusion,
and a block regains that only when its qkv AND its out branch are both off.
At this threshold that is true for about half the blocks, and the rest still
pay. Routing per block rather than per branch is the obvious next step; on a
backend with no fused branch kernel (wgpu, CPU) the branch's GEMMs disappear
outright and the saving should be direct, which is untested here and therefore
not claimed.

What routing buys today is the map: which parts of an adapter matter.

Three caveats, stated rather than hidden: the measurement happens on the first
step, where the latent is nearly pure noise, so a branch that only matters
late is judged early; the branch does contribute on that first step before it
is switched off; and the first step is slower under routing, because every
branch takes the split path until it has been measured (59.6 s against 47.1).
The default is off.

## Reference conditioning

LTX's multi-subject adapters ship a `reference_slot_embedding` beside the
branches: a Fourier feature of the slot index through a two-layer MLP, added
to the latent channels of a reference image before it is prepended to the
sequence. `--ref subject.ppm` (up to five, in slot order) needs the adapter
that trained it, so it is refused without `--lora`.

MiniMax-H3 has no equivalent yet: its release's `ref2va` path is not ported,
and `--first-frame` is the only image conditioning it takes.
