# q2tp: channel scale and code refinement

A better recipe for the 2-bit `q2tp` codec. File size, format and inference speed stay the same: everything
happens inside `cortiq quantize-gptq`. Measured on MiniCPM5-2B, 13.11.2026.

[Русская версия](Q2TP_SCALE_REFINE.ru.md) · [中文版](Q2TP_SCALE_REFINE.zh.md)

## Result

The same 1 314 231 679-byte file and the same Hessians (32 768 tokens of WikiText-2 train), perplexity:

| Text (windows of 512 tokens) | Before | After | Change |
|---|---|---|---|
| WikiText-2 test (160) | 28.431 | 26.426 | −7.05 % ± 1.20 |
| WikiText-2 validation (160) | 28.337 | 26.384 | −6.89 % ± 1.43 |
| English documentation (64) | 69.185 | 60.445 | −12.6 % ± 2.5 |
| Rust source (64) | 27.515 | 23.706 | −13.8 % ± 3.0 |
| WikiText-2 test, the native file (12) | 26.106 | 23.898 | −8.5 % |

Intervals are 95 % over paired differences on the same windows. The parameters were chosen on the test split;
validation, documentation and code were not used for the choice. The 64- and 160-window rows were scored
through a q8_2f copy of each file (see the last section), the 12-window row on the q2tp file itself.

## Principle

Two steps on top of the usual GPTQ. Both cost no bits.

### 1. A channel scale folded into the norm

A 2-bit code has one step per group of 32 neighbouring input channels. When the channels of a group differ a
lot in magnitude, the small ones are lost. Before quantization every input channel `j` is multiplied by its
own factor, and the inverse factor goes into the gain of the RMSNorm in front of the layer:

```
t_j = σ_j^0.75 / rms_j^0.25          σ_j   — RMS of column j over all rows of the tensor group
                                     rms_j — RMS of input j on the calibration data (√H_jj)
W' = W · diag(1/t)                   the weights the quantizer sees
γ' = γ ⊙ t                           the norm gain written to the output file
H' = diag(t) · H · diag(t)           the Hessian in the new basis
```

`t` is normalized to a unit geometric mean and kept in `[1e-3, 1e3]`; a dead channel gets `t = 1`. Before
quantization the model is mathematically the same: `W'·(t⊙x) = W·x`.

One scale is shared by the tensors that read one input:

| Tensors | Norm that takes `t` |
|---|---|
| `mlp.gate_proj` + `mlp.up_proj` | `post_attention_layernorm` |
| `self_attn.q_proj` + `k_proj` + `v_proj` (`--channel-scale-qkv`) | `input_layernorm` |

The exponent matters: at 0.5 the gain is gone. At 1.0 (with ridge 1.0) the result was no worse than at 0.75:
26.72 against 26.99.

### 2. Code refinement after GPTQ

GPTQ picks each weight's level once, left to right. Afterwards single weights move to the adjacent level
where that lowers the layer's output error. Scale ladders and group rungs stay.

```
r   = Ŵ_i − W_i                      the row's residual
H_f = H + 0.3 · mean(diag H) · I
g   = H_f · r
moving weight j by Δ = ±s_group changes  r·H_f·rᵀ  by   2Δ·g_j + Δ²·H_f[j,j]
```

Each round applies the single best move of a row, then `g += Δ·H_f[:, j]`. At most 256 rounds; a row stops
earlier when no move helps. The usual debias (the row's `lo` shift) runs after it.

The ridge was measured: 0.1 and 0.3 give the same result, 1.0 is worse, 3.0 is worse than no refinement.

## Command

```bash
cortiq quantize-gptq MODEL.f16.cmf --codec q2tp \
  --calib CALIB.txt --calib-model MODEL.q8_2f.cmf \
  --tokens 32768 --window 2048 --lambda 0.01 \
  --tensor-quant lm_head.weight=q8_2f --tensor-quant model.embed_tokens.weight=q8_2f \
  --channel-scale 0.75 --channel-scale-qkv \
  --refine 256 --refine-ridge 0.3 \
  --output MODEL.q2tp.cmf
```

Everything else is the default: the `placed` ladder, debias on. Use at least 32 768 calibration tokens.

## What to check on a new model

1. **The fold is an identity.** The same run with `CMF_CHSCALE_FP=1` writes every linear layer as f16, not
   quantized, with the scale folded in. Its perplexity must equal the same run without `--channel-scale`
   (MiniCPM5: 16.903 and 16.902). A difference means the architecture's norm gain works differently (for
   example `1 + γ`) or something else reads the norm's output.
2. **The number of folds.** The log has one `channel scale <norm name>` line per folded group. A group that
   does not match by names or members is skipped silently: GDN layers (`linear_attn.*`), MoE experts.
3. **A paired comparison with the baseline** on the same Hessians: the same run without the four new flags.
   Judge by the model's perplexity, not by the layer error: at 2 bits the layer error does not predict it.
4. **Text outside the calibration data, and greedy generation.**

## Limits

- Measured on one model (MiniCPM5-2B, a dense Llama-like stack, GQA 16/2) with WikiText calibration.
- The scale is not applied to the inputs of `down_proj` and `o_proj`, to MoE experts or to GDN layers.
- The groups the new flags touch are folded and swept on the CPU. The other groups stay on the card, but
  this mixed path has not been run on a card: on a Mac everything goes through the CPU.
- With an existing `--hessians` cache every group goes through the CPU.
- The exact least-squares gain (`--channel-refit`, the input-channel gain of the NVG encoder) adds nothing:
  −0.23 %. It is not part of the recipe.
- The cortiq rule for layer 7 (`--q2tp-rule-tensor`) does not help in this pipeline: −0.03 % ± 0.43.
- `--refine-q4tp` runs the same refinement on the 16-level grid of q4tp tensors; it is not part of this recipe.

## Code and tools

`crates/cortiq-cli/src/gptq.rs`: `chscale_vector`, `chscale_norm_for`, `q2tp_refine_codes`,
`q4tp_refine_codes`, `channel_refit_gain`.

Scoring a q2tp/q4tp file on the CPU is slow (about 18 tok/s on a 2B model). `cortiq requant FILE --quant
tiled-q8_2f --output COPY` writes a q8_2f copy that scores 7 times faster and reads 0.4–0.7 % higher, the
same for every variant.
