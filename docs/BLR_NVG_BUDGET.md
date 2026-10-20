# The bit budget is the design space: a non-uniform allocation beats a trained uniform one

*A measured study on Qwen3.5-0.8B: what a sub-1 bpw BLR/NVG conversion really
costs, why the weight-error metric cannot see it, and how moving the same bits
to the right layers recovers a quarter of the loss.*

---

## 0. The claim in one table

Every row is the same model, the same skeleton, the same encoder; only the
**bit budget per layer** changes. Perplexity: wiki 16×512 tokens, c4 16×512.

| model | size | body bpw | wiki ppl | c4 ppl |
|---|---|---|---|---|
| nvg08s — the product's encoder, 1.0 bpw, no training | 336.0 MiB | 0.989 | 672.518 | 932.470 |
| nvg08_trained — the same + the metric training | 336.0 MiB | 0.989 | 135.76 | 182.0 |
| uniform 0.55 bpw, no training | 309.4 MiB | 0.545 | 1266.144 | 1500.364 |
| allocated 0.55 bpw, decided per layer | 309.6 MiB | 0.547 | 965.491 | 1156.508 |
| **allocated 0.55 bpw, decided at the model level** | **310.8 MiB** | **0.566** | **901.491** | **1119.006** |

The allocated model has the **same bytes** as the uniform one (+0.4 %) and is
**−364.7 ppl** better on wiki and **−381.4** on c4. Nothing about the encoder
changed: only *where the bits went* and *how the choice was made*.

And the second table is the one that should stop a reader:

| L0 (the pole layer) encoded at | wiki ppl (4×512) | vs the sign-trained student |
|---|---|---|
| 0.55 bpw | 688.9 | +230.9 |
| 1.0 bpw (the student's own budget) | 502.1 | +44.1 |
| **1.5 bpw** | **409.5** | **−48.5** |
| **2.5 bpw** | **353.3** | **−104.6** |

A **plain, untrained NVG encode** of the teacher's weights for layer 0, given
1.5 bits per weight, makes the model **better than the sign-trained student**
(409.5 vs 457.9). The sign training of the pole layer made it *worse* than a
fresh encode with a larger budget.

---

## 1. What is being quantized

A weight matrix `W ∈ R^{d_out×d_in}` is stored as **P low-rank paths** (two
thin factors each) plus **16-bit scales** (one per row, one per column, one per
latent) plus **one bit per weight** (the sign). The cost:

```
bits(r) = P · [ r(d_out + d_in) + 16(d_out + d_in) + 16 r ]        b = bits / (d_out·d_in)
```

At `r = 320`, `1024×2048` costs 259 KB — **0.99 bit per weight**. The decode
never materializes `W`: the input is scaled per column, the latent rows are
binary dot products (sums of ±1), and the outputs are scaled per row.

The encoder (NVG) is data-aware. It reads a calibration text, captures the
**input Gram** `G = E[x·xᵀ]` of the layer, and minimizes the *output* error

```
min_E  tr(E G Eᵀ),     E = W − Ŵ
```

— "how much does this layer's answer change on real inputs", not "how far is
Ŵ from W in the weight space". The signs are then chosen by a column-sequential
GPTQ-style search, and `P = 1` vs `P = 2` is picked on a **held-out** slice of
the calibration text.

Everything below is a measurement of that pipeline on Qwen3.5-0.8B (24 layers,
150 BLR tensors), with the training-free student `nvg08s` (1.0 bpw, sign-trained)
as the reference.

---

## 2. The degradation law

For each layer and each budget, the layer's BLR tensors were replaced by a
fresh encode and the perplexity measured (24 layers × 2 variants × 3 budgets,
144 measurements, one patch per layer).

| budget | Σ damage (sum over the 24 layers) |
|---|---|
| 1.0 bpw | +95.4 ppl |
| 0.55 bpw | +505.4 ppl |
| 0.1 bpw | +4552.0 ppl |

Fitting `Δnll ∝ B^−α` over the layers gives

> **α = 1.250, R² = 0.834** (22 layers)

A 5.5× budget cut (0.55 → 0.1) costs 9.0× more damage. The curve is convex:
the last bits are worth much more than the first ones.

The damage is also extremely concentrated: **layer 0 alone is 46 % of it at
0.55 bpw and 73 % at 0.1 bpw**; layer 23 is second.

## 3. The damage is not monotone in the budget

For layers 7, 8, 10 and 17 the *coarsest* budget wins:

| layer | 1.0 | 0.55 | 0.1 |
|---|---|---|---|
| L7 | −2.4 | +4.4 | **−2.9** |
| L8 | **−5.3** | −0.3 | **−5.2** |
| L13 | **−5.4** | +6.7 | +21.4 |
| L14 | **−5.4** | +9.6 | +110.5 |
| L19 | **−4.8** | +14.2 | +90.0 |

Two readings:

* On 8 of the 24 layers a fresh encode at the trained budget is **better than
  the trained payload**. The sign training optimizes the end-to-end loss, and
  on those layers it optimized *worse* than a plain encode.
* A 0.1-bpw payload can be *better* than a 0.55-bpw one: a coarse code injects
  a small, nearly harmless error, while a mid-rank code can inject a large and
  *wrong* update into a layer the rest of the model was not adapted to.

This is why a uniform budget is not a neutral choice: it is a bet that every
layer needs the same number of bits, and the data says that bet is wrong.

---

## 4. The weight-error metric cannot see any of this

Twelve local predictors of the per-tensor damage were measured against the
observed ΔNLL (14 unique measurements, Pearson/Spearman):

| predictor | pearson | spearman |
|---|---|---|
| `D_rel` = tr(Δ G Δᵀ)/tr(Ŵ_s G Ŵ_sᵀ) — the swap error in the Gram metric | +0.05 | +0.06 |
| `D_abs` — the absolute swap error | −0.15 | −0.19 |
| `E_rel` — the candidate's distance to the teacher | +0.00 | +0.12 |
| `S_rel` — the trained payload's distance to the teacher | +0.09 | +0.11 |
| `frob_rel` — the weight-space distance | +0.35 | +0.31 |
| `sel` — the encoder's own held-out metric | — | −0.07 |
| `gain = S − E`, `gain·sens`, `cost·sens`, `gain_drift`, … | ≤ +0.20 | ≤ +0.15 |

The pairwise ranking of two variants of one tensor: **1/2** — chance.

The sharpest illustration is the *re-encode paradox*: re-encoding a layer's own
decoded payload (the trained code, decoded, encoded again at the same rank)
**improves the weight error** and **destroys the model**:

| L0 tensor | weight error (sel) | wiki ppl |
|---|---|---|
| stored (sign-trained) | 0.337 | 457.9 |
| re-encoded from its own decode | **0.249** | **553.7** (+95.8) |
| re-encoded from the teacher's weights | 0.337 | 467.3 |

> A lower weight error with a worse model is not a paradox once you know what
> the training optimized: the sign training balanced the code against the
> *network*, and re-deriving the code from the decoded weights throws that
> balance away while improving the local metric.

The conclusion that shapes the whole method: **the only reliable way to choose
a payload is an end-to-end measurement** — and the way to make that affordable
is to make one measurement cheap (≈20 s per tensor on the GPU pipeline) and to
use it to *allocate*, not to search.

---

## 5. The allocation formula

Everything the formula needs is measured once per layer: the damage at two or
three budgets.

```
m_L(B_low → B_high) = ( Δ(B_low) − Δ(B_high) ) / (B_high − B_low)      [nats per bpw]
v_L                 = m_L / P_L                                        [nats per byte]
```

`Δ` is the measured damage, `P_L` the layer's weight count. The budget is then
moved from the layer with the smallest `v` to the layer with the largest,
in steps that keep `Σ bpw·P_L` constant:

```
δ_recipient · P_recipient = δ_donor · P_donor
```

The donor's floor is its lowest measured budget; the recipient's ceiling is
the highest measured one (extrapolation past a measured point is not a
measurement).

Applied to the Qwen3.5-0.8B curve, the knapsack wants:

* **L0 → 1.5 bpw** (it is the only layer whose curve is still steep at 1.0),
* **L23 → 1.0 bpw** (the second pole),
* **L7, L8, L10 → 0.1 bpw** (their damage *falls* at 0.1 — they are the donors,
  and they pay for the recipients),
* everything else stays at 0.55 bpw.

The byte budget is preserved to 0.06 %, and the result is the −252.6 / −294.8
ppl of the first table.

### The model-level descent (what the formula buys in practice)

The allocation above is computed from per-layer measurements. Acting on it
*at the model level* — patch one layer's choice into the assembled model,
measure, keep or roll back, next layer — gives the champion of the study:

| model | body bpw | wiki (4×512) | wiki (16×512) | c4 (16×512) |
|---|---|---|---|---|
| uniform 0.55 (the encoder alone) | 0.545 | 993.2 | 1266.1 | 1500.4 |
| the allocation, assembled from the per-layer winners | 0.547 | 751.1 | 965.5 | 1156.5 |
| **the same allocation, decided at the model level** | **0.566** | **656.1** | **901.5** | **1119.0** |
| the same descent without a budget cap | 1.163 | 324.0 | 480.0 | 725.4 |

Two things to read here. The model-level arbitration is worth **−64 ppl** over
the per-layer winners *at the same budget* (the row-gauge reversal of §5 is
part of it). And left unconstrained the descent buys its quality with bytes:
at 1.16 bpw it reaches 480.0, better than the *trained* 1.0-bpw model — the
budget cap is what makes the comparison fair.

### The per-layer measurements do not compose

The allocation above was measured **per layer** (one layer patched into the
trained model, the rest intact). Assembling the per-layer winners is *not* the
same as the best model: with the identical allocation and the identical byte
count, a model built from the per-layer winners scores **1013.5** (wiki, 16×512)
and a model built from the plain encodes of the same plan scores **965.5**.

A byte-level diff shows the two models differ **only** in the head-structured
tensors — the ones the per-head row gauge touches. The gauge won in isolation
(L0 at 1.5 bpw: −11 ppl; L0 at 0.55 bpw: −51 ppl) and **loses ~38–48 ppl in
the assembled model**.

> A layer patched in isolation changes the stream for every other layer, so
> the per-layer deltas are not additive — and a variant that wins in isolation
> can lose in the model. The arbiter has to be the model.

The practical consequence is a procedure, not a formula: the correct selection
is a **coordinate descent over the model** (patch one layer into the assembled
model, measure, keep or roll back), and the per-layer curve is only good enough
to decide *where* the bits go — which layer deserves an order of magnitude more
of them, and which one does not care.

### Why the pole layer deserves more than the trained model

Layer 0's input is the embedding stream: normalized, and with **no drift**
between the teacher and the student (the student's layer-0 activations are the
teacher's, up to the quantization of layer 0 itself). Everywhere else the
student's stream has drifted, and a payload that reproduces the teacher's
weights fights that drift.

That is why the pole layer is the one place where "more bits, no training"
beats "fewer bits, with training": with enough rank, the fresh encode *is* the
teacher's function, and nothing else in the model has to be re-fitted.

---

## 6. Two free improvements (0 extra bits)

**The per-head row gauge.** The NVG fit minimizes the error in the *whole*
input metric. A GDN `z` projection is consumed by a **per-head** normalizer, so
what matters is the *relative* error inside each head, not in the aggregate.
Rescaling the target's rows by `sqrt(median_head_energy / head_energy)` before
the fit (and undoing it on the stored scales afterwards — 0 extra bits) flips
the sign of the effect:

| L0 `z` @ 1.0 bpw | plain | with the row gauge |
|---|---|---|
| wiki ppl | 467.3 (+9.3) | **456.0 (−2.0)** |

and at 0.55 bpw it is worth **−51 ppl** on the whole of layer 0. The gauge
applies to the head-structured projections only (GDN `qkv`/`z`, attention
`q`/`k`/`v`); on the plain matrices it is a no-op.

**But it does not survive the assembly** (§5): in the model built from the
per-layer winners the gauged payloads lose ~38–48 ppl against the plain
encodes of the same plan. The gauge is a per-layer win and a model-level loss —
another instance of the composition problem, and a good illustration of why
the arbiter has to be the model and not the layer.

**The encode target.** Encoding the *teacher's* weights into the student's
payload is better than keeping the student's own decoded weights wherever the
input has not drifted, and worse where it has — the measurement decides, per
tensor.

---

## 7. The frontier: which codec wins where

The budget is not the only axis — the *code* is. The same model, converted
without any training, with every codec the toolbox has:

| code | what it is | body bpw | wiki ppl | c4 ppl |
|---|---|---|---|---|
| q8_2f | the teacher | 8.02 | 24.27 | 34.47 |
| q4tp | 4-bit tiled planar, static group scales | 4.18 | 28.45 | 39.94 |
| GPTQ q2tp (mixed) | 2-bit gate/up + 4-bit rest, Hessian-placed scales + GPTQ | 3.47 | 34.96 | 49.58 |
| **GPTQ q2tp (pure)** | **every linear at 2 bits, Hessian-placed scales + GPTQ** | **2.79** | **48.74** | **75.98** |
| BLR 1.5 | NVG, no training | 1.49 | 512.71 | 746.05 |
| BLR 1.0 | NVG, no training | 0.99 | 853.06 | 1082.65 |
| BLR 1.0 trained | NVG + sign training | 0.99 | 672.52 | 932.47 |
| BLR 0.55 allocated | NVG, no training, the budget moved | 0.547 | 965.49 | 1156.51 |
| BLR 0.55 | NVG, no training | 0.545 | 1266.14 | 1500.36 |
| q2tp (converter rule) | 2-bit tiled planar, static scales, no calibration | 2.18 | 4269.78 | 4296.50 |
| GPTQ q1s | 1-bit ternary + an f16 outlier overlay | 1.98 | 20130.01 | — |

There are **two different cliffs**, and the distance between them is *the data*:

* the **converter's static 2-bit rule** (no calibration) collapses — 4269.78 ppl
  at 2.18 bpw;
* the **same 2 bits with the Hessian** — the scales placed at the H-weighted
  MSE-optimal rungs and the rounding error folded back through H⁻¹ (GPTQ) —
  gives **48.74 ppl at 2.79 bpw**, practically free of loss;
* below ~2 bits even the data-aware path collapses (the 1-bit ternary codec
  with an outlier overlay: 20130 ppl): there is no longer enough capacity to
  carry the function at all.

So the frontier has two regimes:

* **above ~2.5 bits the integer codecs win outright** — a data-aware 2-bit
  model sits at 48.7 ppl, an order of magnitude better than BLR at the same
  budget, and at 4 bits q4tp is at 28.45, practically lossless and much cheaper
  to run than a low-rank binary code;
* **below ~2 bits BLR/NVG is the only workable structure** — the integer codes
  have already fallen off their cliff, and BLR degrades gracefully
  (1266 → 965 → 853 → 513 ppl at 0.55 → 0.99 → 1.49 bpw).

This is the answer to "would a dynamically quantized q4tp be better": the
*data-aware* version is dramatically better than the static one — but it needs
the calibration, and it lands at ~2.8 bits. At the sub-1-bpw target there is
nothing to switch to.

### How the training compares with the bits (measured on the same model)

Three models, the same skeleton, the same teacher, the same encoder — only
the payload and the training differ. The training numbers are the published
A/B of the NVG campaign (1 M tokens, the same schedule):

| model | how it was made | body bpw | wiki (16×512) | c4 |
|---|---|---|---|---|
| NVG 1.0, no training | the encoder alone | 0.99 | 672.52 | 932.47 |
| NVG 1.0, + metric training | the training moves the f16 scales | 0.99 | **135.76** | **182.0** |
| NVG 1.0, + sign training | the training also moves the bits | 0.99 | **80.35** | 116.6 |
| NVG 1.5, no training | the encoder alone, a bigger budget | 1.49 | 512.71 | 746.05 |

So the honest price list on this model is:

* the **metric training** buys 536.8 ppl — about **1.7 bits per weight** of
  equivalent quality (half a bit buys ~160 ppl);
* the **sign training** adds another 55.4 ppl on top (135.8 → 80.4);
* a fresh encode at 1.5 bpw buys 159.8 ppl.

> The training is worth about 1.7 bits per weight here — far more than the
> half bit a fresh encode at 1.5 bpw buys. Bits and training are *not*
> substitutes: the training optimises what the code already is, the bits
> change what it can be.

That is why the campaign's subject is the *no-training* conversion: with the
training available, the budget question is secondary (train the model you
have); without it, the budget is the only lever — and the levers that matter
there are the allocation and the calibration protocol, not the encoder.

The two are nevertheless complementary, and the combination is now a single
command: `cortiq nvg-train --budget-plan FILE` re-encodes the planned layers
and then trains on top of them.

## 8. Where this sits against the earlier campaigns

The NVG/BLR work did not start here. The experiment store (`cmf-experiments`)
holds three campaigns that a new study has to answer to.

**The MiniCPM5-2B q4tp study (the "genesis").** The method there is not a
different code but a *refit of the scales the file already stores*: the 4-bit
codes do not change, the f16 row scales and the norms are fine-tuned against
the original model. Same file size, and it removes ~60 % of the remaining gap
to the original (19.13 vs 18.89 = +1.3 %, where the converter's own q4tp is
+7.1 %). The lesson for a budget study: **the last zero-bit lever is the
parameter-level refit, not the encoder.**

**The MiniCPM5-2B q2tp study.** The ordinary 2-bit path breaks that model
(ppl 206.6 against the original's 18.9). The same study at the same file size
reaches 21.9 — 9.4× better — and 21.6 with +0.1 bits per weight. What moved
the number, in order: enough calibration data (32–131 k tokens; on 2 k tokens
GPTQ is *worse* than no calibration at all), the 7th layer (weights 94× the
average), the group-scale placement, the output "shrinkage" debias, and finally
the scale refit (−15 % at the same size).

That study also states the methodological point this article re-derives
quantitatively: *the per-tensor proxy is known to misrank at 2 bits, while the
model-level held-out NLL tracked ppl.* Here that becomes 12 measured
predictors, all |r| ≤ 0.4, and a pairwise ranking at chance (§4).

**The NVG training campaigns.** On the 0.8B at 1.0 bpw: the encoder alone
672.52, the metric training 135.76, the sign training 80.35. On the 27B at
0.55 bpw: 148.6 → 48.31 → 31.86 (and 32.41 with 40 % chat-format text in the
training mix, which moves the chat test from 167.0 to 92.7). The training is
not optional at these budgets — see the price list in §7.

**What this study adds.**

| | earlier campaigns | here |
|---|---|---|
| the proxy question | stated as an observation ("the per-tensor proxy misranks") | **quantified**: 12 predictors, \|r\| ≤ 0.4, pairwise 1/2, and the reason (the missing `∂L/∂y`) |
| the budget | a uniform number | a **law** (Δnll ∝ B^−1.25), a non-monotone damage curve, and a **byte-preserving allocation** by the marginal value per byte |
| the selection | the per-tensor proxy | the **model level** as the arbiter: a coordinate descent over the payload choices |
| the frontier | the q2tp/q4tp arms at 2 and 4 bits | the **whole codec frontier** with two cliffs: the static 2-bit rule collapses (4270) while the data-aware 2-bit model is at 48.7; below 2 bits BLR is the only structure left |
| the product | a script that reproduces the recipe | `--blr-plan` and `nvg-train --budget-plan`: the measured allocation is a file the tools read |

**One correction the store forces.** The campaign's numbers are measured with
the *non-sequential* calibration (every layer's Grams captured from the final
model) and the teacher's weights as the encode target. The product's
converter calibrates *sequentially* (each layer on the activations the layers
below it actually produce) and encodes the model's own weights: the same
1.0 bpw budget comes out at **672.52** instead of the 853.06 measured here.
The protocol is therefore worth ~180 ppl — comparable to everything else in
this article — and the absolute numbers below 1 bpw in this study are a
*lower bound* on what the product's protocol would deliver at the same
allocation. The allocation map itself is a statement about the *relative*
value of a bit per layer and should carry over, but the honest thing is to
re-measure the curve under the product protocol before the numbers go into a
release note.

## 9. What became a product feature

The allocation is a first-class part of the tooling, because the measured
allocation must be replayable verbatim:

* `--bpw` accepts **any** value in `(0, 16)` — `0.11`, `0.56`, `1.37` are as
  valid as `0.55`.
* `--blr-plan FILE` gives the budget **per layer and per tensor**:
  ```
  layer 0 1.5
  tensor model.layers.0.mlp.down_proj.weight 0.11
  ```
  A malformed rule is an error, never a silent fallback: a plan that is
  ignored would spend the wrong bytes.
* `cortiq nvg-train --budget-plan FILE` applies the plan **before** the
  training: the planned layers are re-encoded (the teacher's weights as the
  target, the student's own Grams as the calibration) and the sign training
  then optimizes the metric of the re-planned payloads. The budget of the pole
  layer no longer has to be "extracted" after the fact.
* The budget **contract** is reported, not assumed: the rank moves in steps of
  8, so the delivered budget sits within one step *below* the request, and a
  request below the container's own floor (rank 8 plus the 16-bit row and
  column scales) **cannot be met at all** — for a 32×32 tensor the floor is
  1.625 bpw, so "0.11 bpw" is a physical impossibility, and the conversion says
  so instead of silently spending more.

---

## 10. Reproduction

```bash
# 1. the waves of the student (one capture per layer, cached)
NVG_BENCH_CAPTURE_ONLY=1 cortiq nvg-bench --model student.cmf --layer L \
    --gram-cache grams/student_L --calib calib.txt

# 2. the curve: encode every layer at a budget, patch it, measure
for B in 0.1 0.55 1.0; do
  NVG_BENCH_W_FROM=teacher.cmf cortiq nvg-bench --model student.cmf --layer L \
      --bpw $B --device gpu --gram-cache grams/student_L --calib calib.txt
  cortiq patch-tensor student.cmf --set <tensor>=<payload>.blr --raw --output probe.cmf
  cortiq ppl -f wiki.test.raw --windows 4 --window-len 512 probe.cmf
done

# 3. the allocation
python3 tools/alloc3.py --points layer01.tsv:0.1 --points layer055.tsv:0.55 \
    --points layer10.tsv:1.0 --out-map alloc_map.tsv

# 4. the training with the allocation
cortiq nvg-train student.cmf --teacher teacher.cmf --calib calib.txt \
    --budget-plan alloc_map.tsv --train-signs --output trained.cmf
```

The measurements behind every number above are in the campaign logs
(`work/sweep/*.tsv`, one row per layer and variant, with the delivered bits per
weight) and the analysis scripts (`tools/curve.py`, `tools/alloc3.py`,
`tools/predict.py`).

---

## 11. What is still open

* The damage law is measured on one model and one architecture. The exponent
  α = 1.25 and the shape of the layer curve (one pole, a flat middle, a tail
  that does not care) need a second model to be a law and not a fact.
* The allocation is currently *measured* per layer; making it *predicted*
  needs a signal that predicts the damage — and §4 shows that no local weight
  or Gram metric does. The candidate is the first-order loss sensitivity
  `g = ∂L/∂y` at each layer output, which the engine does not compute yet.
* The training itself still runs on a uniform budget: the plan moves the bits
  of the *encoding*, and the sign training optimizes the metric inside the
  ranks it is given. Training *with* the allocation (a curriculum over the
  budget) is the natural next step.
