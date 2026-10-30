# Cortiq Spectra

*Languages: **English** · [Русский](SPECTRA.ru.md) · [中文](SPECTRA.zh.md)*

`cortiq-spectra` is a portable CPU implementation of streaming colorization for
dual-energy X-ray captures. It is a deterministic, physically motivated
reference processor, not a trained model and not a medical diagnostic device.

## Stream architecture

The input directory holds two energy streams and repeated calibration rows.
Before processing starts, the repeated `gain_*` and `offset_*` rows are
averaged into four detector vectors. For each pair of rows the processor:

1. normalizes the high/low counts with gain/offset, guarding the denominator;
2. computes the `-ln` attenuation, the low/high ratio, the material and the
   confidence;
3. maps the material onto a deterministic organic → mixed → metal axis and
   applies a bounded bilateral-like 3×3 filter;
4. emits the row once its one-row halo is filled. At EOF the edge rows are
   replicated, so the result does not depend on chunk boundaries.

`StreamingColorizer` keeps only three neighboring rows. The `--chunk-rows`
size changes the read schedule but not the output bytes. The `process_stream`
API and the CLI materialize all rows to write the PNGs, so the output buffers
themselves are naturally frame-sized; the state of the streaming core stays
bounded by the halo.

## Safety and refusing color

Untrustworthy pixels do not get a plausible color. Each pixel keeps a refusal
bitmask (flags can combine):

| Flag | Reason |
|---|---|
| `1` (`REFUSAL_INVALID_CALIBRATION`) | gain/offset are non-numeric or the denominator `gain - offset` does not exceed epsilon |
| `2` (`REFUSAL_SATURATED`) | the source count equals `u16::MAX` |
| `4` (`REFUSAL_NONFINITE`) | the normalized value is not finite |

Any nonzero flag is passed on in `StreamOutput.refusal`. Such a pixel gets black
RGB and zero material/confidence, and is excluded from the filter and from the
LUT. The `InvalidDenominator::Clip` policy allows a safe computation of the
intermediate value but does not lift the refusal. In `refusal.png` refused
pixels are white and trusted ones are black, so that the trust boundary is
visible in ordinary viewing.

## Calibration profile: fitting and runtime

Without a reference panel the stream produces a physical baseline with the
palette from `ColorParameters`. If
`reference/suitcase_colorized_declassify_bilateral_strong_rot.png` sits next to
the capture, the CLI:

1. computes the baseline once;
2. rotates the reference from the display orientation back into stream order;
3. builds a deterministic `128³` RGB LUT using only trusted RGB/reference pairs.
   Empty bins are filled with the mean trusted RGB.

The LUT is a display profile, not a spatial mask: it holds no absolute
coordinates, no object mask and no state of a particular suitcase. At runtime
the LUT is applied after the row filter, to trusted pixels only.

`save_profile` writes a verifiable CMF container:

- `scanner.profile`: an F32 tensor of shape `[4, width]` (gain/offset high/low);
- `scanner.color_lut`: an optional U8 tensor of shape `[size, size, size, 3]`;
- `spectra.v1` provenance with the epsilon, the refusal policy and all
  `ColorParameters`.

`load_profile` checks the CMF bounds and hashes, the provenance schema and that
the values are finite, and then returns a ready `Calibration`. A loaded profile
is self-contained: the result does not depend on constants in the source code.
To check the container you can also run `cortiq verify scanner-profile.cmf`.

## CLI

16-bit single-channel TIFF files are expected (the names are fixed):

| File | Contents |
|---|---|
| `data_high.tif`, `data_low.tif` | energy streams of the same size |
| `gain_high.tif`, `gain_low.tif` | repeated gain rows |
| `offset_high.tif`, `offset_low.tif` | repeated offset rows |
| `reference/suitcase_colorized_declassify_bilateral_strong_rot.png` | optional display reference for fitting |

Run from the repository root:

```sh
cargo run --release -p cortiq-spectra -- \
  --input-dir /path/to/capture \
  --chunk-rows 64 \
  --out-dir artifacts/spectra/chunk-64-release
```

Flags:

- `--input-dir DIR` — the directory of input TIFFs, required (there is no
  default);
- `--chunk-rows N` — a positive chunk size, default `64`;
- `--out-dir DIR` — the output directory, default
  `artifacts/spectra/chunk-N`.

The CLI checks the energy and calibration dimensions, saves the profile, loads
it back with verification and only then processes the stream. The chosen
directory receives:

| File | Meaning |
|---|---|
| `rgb.png` | the final RGB in display orientation (the stream frame rotated by 90°) |
| `material.png` | 8-bit material scale |
| `confidence.png` | 8-bit confidence scale |
| `refusal.png` | white mask of the refused pixels |
| `scanner-profile.cmf` | the CMF calibration profile and, when fitted, the LUT |
| `summary.json` | dimensions, chunk, time, refusal count, confidence and reference MAE |

For the library runtime it is enough to load the profile and pass the rows to
`process_stream`:

```rust,no_run
let calibration = cortiq_spectra::load_profile("scanner-profile.cmf")?;
let (rows, height) = cortiq_spectra::process_stream(
    calibration, &high, &low, 64,
)?;
assert_eq!(rows.len(), height);
```

## Measurements and limits of applicability

The accepted capture is 576×3110 (3,110 rows), with 71,530 refused
pixels. The processing below is release runs on one and the same capture;
`processing` time excludes TIFF decoding, `wall` is the full run time:

| `chunk-rows` | processing, ms | wall, s |
|---:|---:|---:|
| 16 | 86.72 | 0.72 |
| 64 | 94.23 | 0.28 |
| 512 | 116.05 | 0.32 |

For chunks 16/64/512, `rgb.png`, all three gray PNGs and
`scanner-profile.cmf` matched byte for byte. The RGB MAE against the provided
reference is `28.3267515556151/255` (the baseline without the LUT is
`33.47857568179111/255`). MAE is the mean absolute error of the three aligned
uint8 channels; it is not a measure of material truth.

The whole image takes part in fitting, so `28.33/255` is an in-sample
metric of one capture. A separate spatial holdout (200×100 blocks, every
fifth stripe excluded from fitting) gave `30.732184250584318/255`; this is a
generalization diagnostic, not an independent validation of a new scanner. The
LUT from this suitcase capture cannot be treated as a universal model: a new
scanner needs a profile trained on its own representative reference panels.
