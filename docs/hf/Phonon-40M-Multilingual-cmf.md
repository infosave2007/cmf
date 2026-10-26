---
license: cc-by-nc-4.0
library_name: cortiq
base_model: gradium/phonon-40m-multilingual
base_model_relation: quantized
pipeline_tag: text-to-speech
language:
  - en
  - fr
  - de
  - es
  - pt
tags:
  - cmf
  - cortiq
  - phonon
  - text-to-speech
  - multilingual
  - gguf
---

# Phonon 40M Multilingual — CMF

This is a CMF distribution of
[Gradium's Phonon 40M Multilingual](https://huggingface.co/gradium/phonon-40m-multilingual).
It accepts English, French, German, Spanish, and Portuguese and writes mono
24 kHz WAV audio.

The package does **not** claim a new quantization or a native reimplementation
of Phonon. It keeps the upstream `model.q8.gguf` byte-for-byte together with
its `config.json`, `tokenizer.json`, and `default-voice.safetensors` in
integrity-checked CMF `U8` entries. Cortiq verifies those entries,
materializes them in a private temporary directory, and invokes Gradium's
maintained [`ptts`](https://github.com/gradium-ai/xn-ptts) runtime for the
model-specific flow and Mimi-decoder inference path.

## Published files

| file | bytes | SHA-256 |
|---|---:|---|
| `phonon-40m-multilingual-q8.cmf` | 75,380,093 | `c622ed46bf67a2f50965d4f5b7ee502b8d8580105f60530148b83b2cbf9a743b` |

The immutable artifact release is commit
`fcbc781f9108da6ea6c3a90b00e5c25875faac52`.

Five short deterministic smoke renders (`--temperature 0 --seed 42`) are in
[`samples/`](samples/): English, French, German, Spanish, and Portuguese. They
confirm that the packaged runtime wrote mono 24 kHz WAV output for every
advertised normalization language; they are not a quality benchmark.

## CLI release status

This card and the commands below are prepared for **cortiq-cli 0.8.36**. The
published 0.8.35 crate and earlier releases do **not** contain `phonon-pack` or
`cortiq tts`. Until the `v0.8.36` tag and the corresponding crates.io
publication exist, a bare `cargo install cortiq-cli` is not a way to obtain
these commands; use the public
[test Space](https://huggingface.co/spaces/infosave/phonon-40m-multilingual-cmf-demo)
for the deployed demo instead.

After 0.8.36 is published, install that exact release:

```bash
# Default build: GPU support plus the Phonon bridge.
cargo install cortiq-cli --version 0.8.36 --locked

# Portable CPU-only build, retaining Phonon support.
cargo install cortiq-cli --version 0.8.36 --locked \
  --no-default-features --features phonon
```

To download the immutable CMF artifact for archival or integrity verification
through the Hugging Face API CLI:

```bash
hf download infosave/phonon-40m-multilingual-cmf \
  phonon-40m-multilingual-q8.cmf --revision fcbc781f9108da6ea6c3a90b00e5c25875faac52 \
  --local-dir .
cortiq verify phonon-40m-multilingual-q8.cmf
```

To create a CMF yourself, first accept the upstream model terms and download
its complete checkout, then package it:

```bash
hf download gradium/phonon-40m-multilingual --local-dir phonon-source
cortiq phonon-pack phonon-source --output phonon-40m-multilingual-q8.cmf
```

`cortiq tts` requires an explicit `--lang` of `en`, `fr`, `de`, `es`, or `pt`.
It does not perform automatic language detection; Russian is not supported by
this upstream model.

## Useful controls

```bash
cortiq tts phonon-40m-multilingual-q8.cmf "A calm, clear sentence." \
  --lang en --temperature 0.3 --seed 4242424242424242 \
  --duration-delta -0.25 --out fast.wav
```

- `--temperature` controls flow-noise variation; `0.3` is the upstream
  default and `0` is deterministic for a fixed seed.
- `--seed` makes the stochastic path reproducible.
- `--duration-delta` is the model's native rate conditioner: negative values
  ask for faster speech and positive values ask for slower speech. It is not a
  pitch-changing post-processing resampler.
- `--guidance`, `--eos-threshold`, and `--max-tokens-per-chunk` are available
  for advanced use.
- `--voice-file` accepts only an already computed Phonon-compatible
  `.safetensors` voice latent. Recording-based voice cloning is not exposed:
  the needed speaker codec is not distributed by the upstream model.

Only the upstream default voice is in this artifact. Do not assume that a
voice from Gradium's separate voice catalog may be redistributed here; check
that voice's own terms first.

## Contents and integrity

| CMF entry | upstream file | purpose |
|---|---|---|
| `phonon.model_q8_gguf` | `model.q8.gguf` | upstream Q8 model payload |
| `phonon.config_json` | `config.json` | model configuration |
| `phonon.tokenizer_json` | `tokenizer.json` | normalization/tokenization data |
| `phonon.default_voice_safetensors` | `default-voice.safetensors` | default precomputed voice latent |

`cortiq verify` validates the CMF envelope, directory, and every stored-entry
hash before the TTS backend is invoked. The companion Space validates this
restricted CMF contract before extracting its assets.

## Provenance and license

The source checkpoint is `gradium/phonon-40m-multilingual` at revision
`1537ce46c64f42903e0d608b132ccb073dac46d2`. It is a gated upstream model:
you must accept its terms to obtain the source checkpoint.

The source payload SHA-256 values preserved in the CMF provenance are:

```text
e312ab694d5ec1f79ccea520a2eeec989969ce07e9b4189ae7e9ad8e035ef82e  config.json
96fa067813244e0c6c5997b98ec2596b65ec8574b14916b48e97db056ff97e27  model.q8.gguf
a5e8c6060bf927ed367314f21355148fdf4acff6fadb94e49c86bc7182449433  tokenizer.json
dc537478eaadc1a5257fcda2e3501ab81e14c111a5d95a5838f350db2afd0f1a  default-voice.safetensors
```

The weights and this derived redistribution are under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/): attribution
is required and commercial use is not permitted. The Cortiq bridge uses the
upstream `ptts` code under its MIT-or-Apache-2.0 terms; that does not change
the model-weight license.

No quality, latency, or voice-similarity claim is made here without a
reproducible measurement accompanying the published artifact.
