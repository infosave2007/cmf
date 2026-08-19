---
license: apache-2.0
tags:
- cortiq
- cmf
- quantization
---

# Which file to take — the CMF quantization ladder

Every `.cmf` on this account is the same model twice over: the same weights,
written at a different number of bits per weight. This page says what those
names mean, what each one costs you, and how to pick without downloading two.

Short version:

| suffix | bits/weight | take it when |
|---|---:|---|
| `f16` | 16 | you are checking a port against the reference |
| `q8` | 8.1 | quality is the whole point and the file still fits |
| `q4tp` | **4.17** | **the default.** Everything published here uses it |
| `q4t` | 4.50 | same values as `q4tp`, older scale layout |
| `q2tp` | 2.17 | the model is a MoE and you are trading experts for size |
| `q1t` | ~2.25 + overlay | a normal checkpoint pushed below q4, with calibration |
| `q1` | 1.5 | the model was *trained* at 1 bit (BitNet/Bonsai class) |
| `vbit` | 3–8, per row | one file has to fit an awkward memory budget exactly |

Normalization weights, biases and anything one-dimensional are **always f16**,
in every file. Compression buys nothing there and precision costs nothing.

## q4tp, and why the scale is the interesting part

A four-bit quantizer stores a nibble per weight and a scale per group of 32.
At f16 that scale is 16 bits per 32 weights — **11% of the file**, spent on a
number that varies smoothly along a row.

`q4tp` keeps the nibbles byte-identical to `q4t` and replaces the scale with a
5-bit rung on a per-row geometric ladder: `scale = 2^(lo + code·step)`, with
`lo` and `step` taken from the row's own min and max log-scale. A reader
expands one row's 32 rungs once and then reads scales by table lookup. The
file drops from 4.50 to **4.17 bits/weight** — 7.3% — for 1.14% RMS
perturbation of the weights, against the ~10% that four-bit quantization
itself already costs. It is, in short, free.

`q2tp` is the same ladder with a two-bit weight plane (8 bytes a group instead
of 16). One detail matters more than it sounds: a symmetric 4-level grid
cannot spell **zero**, so rung 0 is reserved to mean "this group is exactly
zero" and the geometric rungs start at 1. Without that a pruned or masked
group comes back as noise.

## q1t — below four bits without training

`q1t` is ternary `{−s, 0, +s}`, packed base-3 at five values per byte
(3⁵ = 243 ≤ 256) for 2.25 bits/weight, plus a per-row overlay that keeps the
salient weights at f16. Two things make it work where plain 1-bit does not:
capturing the many near-zero weights *exactly* (the zero level is the win over
binary — measured ×7 better at matched budget), and choosing what to keep in
the overlay by `|W|·RMS(x)` — amplitude times activation, not amplitude alone.

It is calibration-driven, not a flag on `convert`: see
[`docs/Q1T_PTQ.md`](https://github.com/infosave2007/cmf/blob/master/docs/Q1T_PTQ.md).

## q1 — only for models trained at one bit

For a BitNet-class checkpoint (Bonsai, and the 1-bit trained families) the
weights already sit on two levels per group, so 1.5 bits/weight is nearly
lossless. As post-training quantization of a normal checkpoint the same
encoding destroys the model. The converter therefore exposes it as an explicit
opt-in and never as a default.

## What the packer decides for you in a video model

Text models are uniform: one dtype for every matrix. Generative containers are
not, and the choices are worth knowing because they are where the size goes.

**The modulation weight is a curve, not a matrix.** MiniMax-H3's
`adaln_proj.linear` is [96768, 2688] per block — 13 B parameters, 40% of the
released model — for a map whose input is one number, the timestep. `cortiq
animate-pack` collapses it onto a rank-24 basis of that curve plus a
[1025, 24] table to interpolate: rms 8.7e-5 on a signal of rms 0.46. This is
most of the difference between a 60 GB checkout and a 14 GB file, and it is
lossless in the way that matters — the map is genuinely one-dimensional.

**Some tensors are kept exact because nothing downstream blurs them.** In the
LTX-2.5 pack, three policies were found by the parity gate rather than by
taste: the adaLN-single projections stay high precision (q4tp put 3.6e-2 into
the first normalization of block 0), and the token table goes to q8 (q4tp put
11% into every hidden state). A quantizer that is right for a 4096×4096
projection is wrong for a lookup table.

**A published dead end.** `mmh3-turbo-fl2va-q2tp.cmf` exists so that the
result is reproducible, not because you should render with it: two bits across
a dense video DiT changes the *subject* of the clip. The size class it was
aiming at is better served by the ClipProj build, which keeps four bits
everywhere and shrinks the prompt encoder instead.

## Checking what you got

```bash
cortiq verify model.cmf   # per-tensor hashes — the file, not the download
cortiq info   model.cmf   # arch, parameter count, dtype per tensor family
```

`verify` is the reason a truncated or mis-mirrored download cannot quietly
become a "bad model": every tensor carries its own hash, and the file says so
before the first kernel runs.

## Where the ladder is honest

- `q4tp` numbers above are measured on the published containers, not
  estimated: 4.17 vs 4.50 bits/weight is arithmetic, and the 1.14% RMS figure
  is against the same weights encoded as q4t.
- Quality claims per model live in that model's card, with the corpus named.
  Where a number was not measured, the card says so.
- The full dtype × execution-path matrix — which format is fast on which
  backend, and where the honest gaps are — is
  [`docs/QUANT_COVERAGE.ru.md`](https://github.com/infosave2007/cmf/blob/master/docs/QUANT_COVERAGE.ru.md).
- The normative layout of every dtype is §3.2 of
  [`CMF_V2_SPEC.md`](https://github.com/infosave2007/cmf/blob/master/docs/CMF_V2_SPEC.md).
