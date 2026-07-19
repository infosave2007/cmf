# q1t — training-free ternary post-training quantization

An experimental, **training-free** compression path that quantizes an
ordinary checkpoint to **~2.25–3.5 bits/weight** — below `q4` (4.5 bpw) —
while staying coherent. Built on the *holographic transfer* idea (the CMF
patents): preserve the layer **output** `W·x`, not the weights.

The ternary base is packed **base-3, 5 values per byte** (`3^5 = 243 ≤ 256`),
so a 32-weight group is `[f16 scale][7 B codes]` = **2.25 bpw** (vs a naïve
2-bit 2.5 bpw) — a lossless size win, same reconstructed values.

It is **not** wired into the default `--quant` flags; it is a separate
calibration-driven command. The engine is untouched apart from the `Q1T`
codec + its fused kernel.

## The method

For each linear layer, given calibration statistics of its input:

1. **Ternary bulk (`Q1T`, BitNet b1.58).** Each 32-weight group →
   `{−s, 0, +s}` with `s = abs-mean` of the group. Capturing the many
   near-zero weights *exactly* (the zero level) is the decisive win over
   1-bit binary — measured ×7 better at matched budget.
2. **Two-field outlier mask.** Keep the top `--keep` fraction of weights by
   `|W|·RMS(x)` (amplitude × activation) at f16 in a sparse overlay. This
   is the SpQR/AWQ salience idea; a *weight* mask beats a *column* mask
   here (3988 vs 615 on 0.5B at 10%).
3. **Докрутка — per-row output stabilization.** After quantizing a row,
   rescale it by the closed-form `α` that minimizes the activation-weighted
   output error `‖α·Q(x) − W(x)‖²_d` (`d` = per-channel activation power).
   One scalar per row, folded into the row's scales — **zero extra size**.
   This is the single biggest lever (0.5B keep-5%: 7344 → 547, ×13).
4. **Keep the bit-sensitive tensors precise.** `embed_tokens`, `lm_head`
   and `down_proj` (the gated-intermediate output) stay at the input dtype
   — cheaper *and* higher quality than flooding them with outliers.

What was tried and **rejected** (measured): the GPTQ/holographic fold
`Σ_PS·Σ_SS⁻¹` *backfires* at extreme low bit (a single-pass, rank-deficient
Hessian injects more error than it removes); a column mask; a finer 4-level
base (ternary is near-optimal for a single scale that includes zero).

## Usage

```sh
# input should be a high-precision CMF (f16 or q8; q4 also works)
CMF_GPTQ_TERNARY=1 \
CMF_GPTQ_SKIP=embed_tokens,lm_head,down_proj \
cortiq quantize-gptq model-q8.cmf \
    --calib corpus.txt \        # .txt, or a JSON array of [prompt, text] pairs
    --output model-q1t.cmf \
    --keep 0.03 \               # outlier budget (2–3% ⇒ below q4 size)
    --tokens 1024               # calibration tokens (diminishing returns past ~2k)
```

Env knobs: `CMF_GPTQ_SKIP` (keep-precise substrings), `CMF_GPTQ_DOWN_KEEP`
(extra down_proj mask if it can't be skipped), `CMF_GPTQ_NOCORRECT=1`
(disable докрутка), `CMF_GPTQ_MAXCOL` (leave wide tensors at input dtype).
The Hessian capture is diagonal-only (fits a 12B); the quantizer streams
one tensor per worker, so RAM stays bounded.

## Measured

Qwen2.5-0.5B (PPL on held-out spec text; q8 = 34):

| build | PPL |
|---|---|
| naive 1-bit | 3.4M |
| ternary + mask, keep 10% | 108 |
| + skip down_proj, keep 10% | 84 |

qwopus-nvg-12b (Qwen3.5 GDN hybrid, 14.8B; q4 baseline = 42.3 @ 7.8 GB):

| build | size | PPL |
|---|---|---|
| ternary + докрутка, keep 10% | 12.7 GB | 83 |
| ternary, keep 2% (below q4 size) | 6.3 GB | 196 |

Larger models degrade **far less** at low bit (12B ~2× vs 0.5B ~4.5×), and
the 12B generates correct, coherent code (docstring, type hints) at ternary
— the recipe is validated end-to-end on a real 15B model. The keep-2% point
is **19 % smaller than q4**; `skip{embed,lm_head,down_proj}` trades a little
of that size back for a large quality gain (measured −39 % PPL on 0.5B at
keep-2%).

**Honest positioning:** q1t does not dominate `q4` at equal size (dense
4-bit is denser); it opens a **smaller operating point** (~2.5–3.5 bpw) that
`q4` cannot reach, with graceful degradation — valuable where size is the
binding constraint (on-device / mobile).

**Decode speed** (three levers, on top of a slow per-row-dequant baseline):

1. **Base matvec** — decode+dot per group straight from mmap (no per-row
   buffer), a 256-entry byte→signs LUT instead of the base-3 divide/modulo
   per weight, and each group's signs unpacked into a tiny stack buffer so
   the 32-wide dot lowers to f32x4 FMAs. **6.9×** over the division decode on
   an 8192×4096 tensor (single thread, 9.25 → 7.64 ms after the SIMD unpack).
2. **Overlay** — dominates at high keep; since the encoder writes ternary
   code 0 at every outlier position, the correction is a plain `value·x`
   with **no scattered per-outlier scale read** (+45 % at keep-10 %).
3. **int8 SDOT** on ARM dotprod (default; `CMF_SDOT=0`/x86 keep the exact f32
   path) for BOTH decode (`matvec`) and prefill (`matmat`): x → i8 once
   (`split_act`, activation outliers stay f32), signs unpacked base-3 → i8,
   two `sdot`s per 32-group; in matmat each row's signs decode once and sdot
   against the whole batch. Decode **+30 % at keep-2 %**, prefill **2.6× at
   keep-2 %** (TTFT 0.87 → 0.32 s). Cost: PPL 89.08 vs 88.66 exact (**+0.5 %**,
   the standard a8w8 activation-quant tradeoff q8/q1/q4 also take by default).
   Levers 1–2 stay bit-identical to `dequant_q1t`; only the i8 path perturbs.

End to end, 0.5B q1t went from ~2.5 tok/s decode (the div-decode regression
the packing introduced) to **~38 tok/s decode / ~58 prefill at keep-10 %, ~60
decode / ~120 prefill at keep-2 %** — a noisy-machine, fair-load read; the
isolated kernel gains above are cleaner.

**Follow-ups:** a cheaper overlay encoding (per-row indices) for higher
keep; an int8-SDOT ternary kernel (signs as `{−1,0,+1}` i8) to close the
remaining gap to dense `q8` decode.
