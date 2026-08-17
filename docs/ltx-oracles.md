# Gating the LTX-2.5 port against the reference

Every stage of the port is checked against the reference implementation's own
activations rather than against a picture that looks plausible. This file is
the recipe: three hook scripts that dump what the reference computed, and the
`cortiq` commands that walk our numbers against them.

The reference is [Lightricks/LTX-2](https://github.com/Lightricks/LTX-2)
(`packages/ltx-core`, `packages/ltx-pipelines`) with the released checkpoints.
Nothing below is needed to *use* the model — only to re-verify it.

## Why the reference's own dumps

A video that looks right is not evidence: a 4-bit repack of a 21 B model can
look right and still be wrong in a way that costs quality everywhere. What
the gates catch is the opposite — a stage that is numerically wrong long
before it is visibly wrong. Two of the three bugs found this way were in the
*packing policy*, not in the port, and neither would have been visible by eye:

* the adaLN-single projections quantized to 4 bits put 3.6·10⁻² of relative
  error into the very first normalization of block 0;
* the token table quantized to 4 bits put 11 % into every hidden state the
  prompt encoder produced.

The third was a porting error: HF's hidden-state tuple ends with the
*normalized* final state, not the raw one, so one forty-ninth of every
token's features — the entry with the largest magnitude — was wrong.

## 1. The video VAE

Hook the decoder and save every stage:

```python
# dump_stages.py — run inside the reference checkout
import torch
from safetensors.torch import save_file
T = {}
def put(n, x): T[n] = x.detach().float().cpu().contiguous()

dec = ...  # the video VAE decoder, built from the checkpoint
latent = torch.randn(1, 128, 3, 8, 12)          # or a real one
put("latent", latent)
h = dec.conv_in(latent);            put("after_conv_in", h)
for i, blk in enumerate(dec.up_blocks):
    h = blk(h);                     put(f"after_block_{i}", h)
h = dec.conv_out(h);                put("after_conv_out", h)
put("frames", dec(latent))
save_file(T, "vae_gate.safetensors")
```

```bash
cortiq ltx-decode --model ltx25-q4tp.cmf --latent vae_gate.safetensors --gate
```

Every traced stage is compared in pipeline order and the first divergence is
named. This is how the im2col patch buffer was found keeping stale rows
across chunks: invisible below 8192 positions, 8 % error above.

## 2. The transformer

Patch `LTXModel.forward` to save its inputs, the prepared arguments and a few
block outputs on the first call, then run any pipeline:

```python
# gen2.py
from ltx_core.model.transformer.model import LTXModel
orig = LTXModel.forward
def forward(self, video, audio, perturbations):
    for tag, m in (("v", video), ("a", audio)):
        for f in ("latent", "sigma", "timesteps", "positions", "context", "keyframes_mask"):
            put(f"{tag}.{f}", getattr(m, f))
    hooks = [self.transformer_blocks[i].register_forward_hook(mk(i)) for i in (0, 1, 47)]
    vx, ax = orig(self, video, audio, perturbations)
    put("v.out", vx); put("a.out", ax)
    save_file(T, "dit_oracle.safetensors")
    return vx, ax
LTXModel.forward = forward
```

```bash
cortiq ltx-dit --model ltx25-q4tp.cmf --oracle dit_oracle.safetensors --gate
```

Run the reference **without** `--quantization`: an fp8-cast dump measures the
reference's own codec, not the port.

To localize a divergence inside a block, hook its submodules instead
(`attn1`, `attn2`, `ff`, `audio_to_video_attn`, …) with `with_kwargs=True` so
the inputs come along; `ltx-dit` compares `v.b0.sa.in`, `v.b0.sa.out`,
`v.b0.ca.ctx` and the rest by name. That dissection is what showed the
divergence entering at the adaLN modulation and not in attention at all.

## 3. The prompt encoder

Hook `EmbeddingsProcessor.process_hidden_states` and save all forty-nine
Gemma hidden states, the feature-extractor output and the connector output:

```python
# gen3.py
from ltx_core.text_encoders.gemma.embeddings_processor import EmbeddingsProcessor
op = EmbeddingsProcessor.process_hidden_states
def process(self, hidden_states, attention_mask, padding_side="left"):
    for i, h in enumerate(hidden_states):
        put(f"gemma.h{i}", h, half=True)
    put("gemma.attention_mask", attention_mask)
    vf, af = self.feature_extractor(hidden_states, attention_mask, padding_side)
    put("feat.video", vf); put("feat.audio", af)
    out = op(self, hidden_states, attention_mask, padding_side)
    put("enc.video", out.video_encoding); put("enc.audio", out.audio_encoding)
    save_file(T, "te_oracle.safetensors")
    return out
EmbeddingsProcessor.process_hidden_states = process
```

```bash
cortiq ltx-encode --model ltx25-q4tp.cmf --prompt "…" --oracle te_oracle.safetensors
```

Only the valid rows are compared. The prompt is left-padded and a pad
position attends to nothing at all: the reference's masked softmax leaves
those rows a uniform average of the values while ours leaves them zero. Both
are dead weight — the feature extractor masks them and the connector
overwrites them with its learnable registers — but comparing them would
drown the signal.

Our tokenization is checked the same way: the ids printed by `ltx-encode` are
identical to the reference's, which is what makes the hidden-state comparison
meaningful in the first place.

## What the numbers mean

After the packing fixes, on a real prompt and a real latent:

| stage | relative difference |
|---|---|
| video VAE, every stage | 2.7·10⁻⁴ |
| patchified transformer input | 2.3·10⁻³ |
| block 0, ada-zero output | 5.9·10⁻³ |
| prompt context (`enc.video`) | 2.2·10⁻² |

The residue is the 4-bit codec, and it is what a 4-bit repack costs. The
reference at bf16 is the ceiling; these numbers say how far below it this
file sits, measured rather than assumed.
