# MiniMax-Music-3 → CMF: what the port needs

Reconnaissance for packing [Comfy-Org/MiniMax-Music-3](https://huggingface.co/Comfy-Org/MiniMax-Music-3)
into one `.cmf` the way `animate-pack` does MiniMax-H3. Everything below was
read from the safetensors headers by range request — no file was downloaded
to write it, which is also the cheapest way to answer "how much of this do we
already have?"

## The three stacks

| source file | bytes | what it is |
|---|---|---|
| `diffusion_models/minimax_music3_dit_fp16.safetensors` | 4 914 197 682 | the DiT, 36 layers |
| `text_encoders/minimax_music3_text_encoder_pruned_bf16.safetensors` | 16 706 629 398 | a 36-layer LLM with audio heads |
| `vae/minimax_music3_dav.safetensors` | 216 696 128 | the audio decoder |

`__metadata__` on each carries `comfy_model: minimax_music3_dit | minimax_music3_dav`,
so these are Comfy-Org repackages, same provenance shape as the H3 files.

There are `fp32` and `int8_convrot` variants of the DiT and an unpruned text
encoder; the pruned bf16 pair above is the one to pack, exactly as H3 packs the
pruned DiT.

## DiT — 374 tensors, 36 layers, hidden 2048

Per layer, and this is NOT H3's block:

```
transformer.layers.N.self_attn.to_qkv.weight   [6144, 2048]   fused q|k|v
transformer.layers.N.self_attn.to_out.weight   [2048, 2048]
transformer.layers.N.ff.ff.0.proj.{weight,bias}[16384, 2048]  GEGLU: 2 x 8192
transformer.layers.N.ff.ff.2.{weight,bias}     [2048, 8192]
transformer.layers.N.pre_norm.{gamma,beta}     [2048]
transformer.layers.N.ff_norm.{gamma,beta}      [2048]
```

Outside the stack:

```
preprocess_conv.weight    [2304, 2304, 1]
postprocess_conv.weight   [128, 128, 1]
timestep_features.weight  [128, 1]
to_timestep_embed.0.{weight,bias} [2048, 256]
cond_layer_logits         [8]
cond_layer_scale          [1]
latent_conditioners       (2 tensors)
```

Three differences from H3 that decide how much code is new:

1. **LayerNorm with `gamma`/`beta`, not RMSNorm**, and **no per-block adaLN**.
   H3's whole conversion story is the adaLN collapse — 40% of that file was one
   modulation matrix per block. Music-3 has none of it: conditioning enters
   through `to_timestep_embed` and the `cond_layer_logits`/`latent_conditioners`
   path, which is 8 weights mixing conditioner layers and needs its semantics
   read off the reference, not guessed.
2. **Fused `to_qkv` and GEGLU `ff.0.proj`.** The packer splits or the kernel
   takes them fused; H3's packer assumes neither.
3. **1-D convs at the ends.** `preprocess_conv` is 2304→2304 and
   `postprocess_conv` 128→128, so the latent is 128-wide and 2304 is the
   conditioned width. Both are 1×1, which `push_exact` already flattens to a
   matrix — the same "5-D convs are matrices wearing a hat" case the VAE packer
   handles.

## Text encoder — 328 tensors, 36 layers

```
model.layers.N.self_attn.qkv_proj.weight    fused
model.layers.N.self_attn.k_norm.weight      per-head norm, Qwen3-style
model.layers.N.mlp.gate_up_proj.weight      fused gate|up
model.layers.N.mlp.down_proj.weight
model.layers.N.{input_layernorm,post_attention_layernorm}.weight
model.audio_decoder.audio_heads.{0..3}.weight  [1024, 4096]
```

The block shape is a Qwen3 (RMSNorm, per-head q/k norm, SwiGLU), so
`qwen3te.rs` is the right runtime — but every projection is FUSED where
`pack_te` expects `q_proj`/`k_proj`/`v_proj` and `gate_proj`/`up_proj`
separately. That is a packer arm, not a new forward.

The `audio_decoder.audio_heads` are the reason this file is 16 GB and not 8:
it is a multimodal model, and a text-to-music conditioning path may not need
them at all. Worth checking against the reference before packing 4 GB of
weights the sampler never reads — the same question the ClipProj work asked of
H3's 12.2 GB encoder, with the same kind of answer available.

## Audio VAE — 121 tensors, decoder only

```
dec_in_proj.weight [1024, 64, 1]        latent 64 -> 1024
decoder.model.0.weight_{g,v}            weight-normed conv, [1536, 1024, 7]
decoder.model.N.block.M.alpha           Snake activation, 29 of them
decoder.model.N.block.M.weight_{g,v}    30 weight-normed convs
```

This is a BigVGAN/Vocos-shaped vocoder, which cortiq already decodes for H3's
audio branch — including the lesson recorded there: **do not quantize a
vocoder**, it buys 45 MB and costs audible hiss. Weight normalisation
(`weight_g` × `weight_v`/‖`weight_v`‖) has to be folded at pack time, which is
mechanical and testable in isolation.

## What the packed file would weigh

| | source | packed |
|---|---|---|
| DiT, ~2.46 B params | 4.58 GiB bf16 | ~1.3 GB at `q4tp` |
| text encoder, ~8.4 B params | 15.56 GiB bf16 | ~4.3 GB at `q4tp` |
| audio VAE | 207 MiB | 207 MiB, exact |
| **total** | **20.3 GiB, three files** | **≈5.9 GB, one file** |

Under 6 GB puts music generation on hardware that cannot hold the video model
at all, which is the reason to do it.

## Why this is not a weekend of packing

The H3 card's parity section is the standard this has to meet: *"Established,
not assumed, and separately for each of the four stacks. The reference is
ComfyUI's own module, run on a toy checkpoint carrying the release's real
tensor names and the release's real schedules."* `tools/mmh3_toy_gate.sh` is
that harness.

Music-3 needs the same, because the parts that cannot be read off a header are
exactly the parts that fail silently:

- what `cond_layer_logits` (8 weights) and `cond_layer_scale` (1) actually mix,
  and at which layers
- what `latent_conditioners` conditions on and how it enters the sequence
- the flow schedule and the sampler's step grid — H3 needed two clocks and a
  remap, and getting that wrong is not visible in a tensor name
- whether the audio heads participate in text-to-music at all
- the VAE's frame rate, hop and any tiling contract

Each of those is a convention that passes at one token and fails at a hundred,
which is the sentence the H3 port already paid for once.

## Milestones, in the order they de-risk each other

1. **VAE first.** Smallest, self-contained, and independently gateable: fold
   weight norm at pack time, decode a known latent, compare against the
   reference decoder. It also answers the frame-rate question the sampler needs.
2. **Text encoder.** A `pack_te` arm that splits fused `qkv_proj`/`gate_up_proj`,
   then greedy-parity of the hidden state against the reference — the same
   `CMF_TE_DUMP` cosine gate the ClipProj work used.
3. **DiT forward + sampler**, gated on a toy checkpoint with the real names and
   schedules, per stack, before any full render.
4. Only then pack, render, and publish.

Steps 1 and 2 are contained work against a testable reference. Step 3 is where
the H3 port spent most of its time, and skipping its gate is how a 5.9 GB file
gets published that renders noise.
