# MiMo q4tp distributions and media integration

Two distributions share the **unchanged q4tp text backbone**:

* **Text-only:** `<stem>.cmf` plus `<stem>.mtp.cmf` (automatic greedy MTP).
* **Full:** the same two files plus `<stem>.mm.cmf` (vision, video, audio).

This avoids a second 164-GB download. A standalone full container is also
supported by `convert --mimo-towers multimodal`; the converter's default
`--mimo-towers text` excludes all media towers. A q4tp profile is not a claim
that every tensor is 4-bit: the backbone's established q8 projections and
float norms stay unchanged, and tower codec fallbacks must pass quality gates.

## Runtime

`mimo_ingress` is shared by CLI and OpenAI chat ingress. It preserves ordered
content blocks through strict template rendering, expands the nine pinned media
tokens, runs only requested towers, and replaces pad rows with embeddings.
Pure-text requests do not discover/load a companion. Media prefill uses
`PrefillIn::Hidden`, including the GPU prefix, never a device re-embedding.
Token-only KV reuse is disabled both into and out of media requests.

Examples (after qualification):

```sh
cortiq run MiMo-V2.6-Flash-RL-q4tp.cmf --prompt 'Read this image.' --image page.png
cortiq run MiMo-V2.6-Flash-RL-q4tp.cmf --prompt 'What is shown at 00:04?' --video frames --video-fps 1
cortiq run MiMo-V2.6-Flash-RL-q4tp.cmf --prompt 'Transcribe the speech.' --audio speech.wav
```

`--mm PATH` overrides sibling discovery. `--image-max-pixels` bounds image/video
preprocessing resolution. CLI flags place text first, followed by images, videos
and audio in their respective flag order. OpenAI multipart messages support
interleaving, `image_url`, `input_audio {data, format: "wav"}`, `audio_url`, and
`video {path, fps}`. Video currently means a local Y4M file/frame directory;
MP4/audio interleave and compressed audio are not implemented. The server
prepares media before opening SSE so preparation failures return HTTP 400.
Network splits, class-token classification, raw/resumed CLI media and GPU layer
splits are rejected rather than silently using text placeholders as embeddings.

## Assembly and gates

The `mimo_mm_assemble` diagnostic example copies selected **already-quantized**
vision/audio tensors into the base companion inventory without requantization.
It checks names/shapes, refreshes codec/count/byte provenance, clears inherited
end-to-end gate claims and validates the assembled companion. It refuses to
overwrite an existing output. Final quality gates must run on the actual output.

Current integration is a candidate, **not a qualified/published release**:

* Mac `cargo check` with GPU, engine tests, CLI and server: passes.
* Synthetic ingress tests cover interleaving, strict unsupported blocks, pad
  count/embedding alignment and token-only KV-reuse isolation.
* Pod engine checks: 44 MiMo unit tests and 7 GPU bank tests pass. The newer
  multipart regression cases still need runtime execution.
* The actual assembled companion passes OCR (`CORTIQ 7392`), ordered colored
  shapes, the chart question and five ASR clips (WER 0%, including 16 kHz and
  stereo 44.1 kHz inputs). Real-model ID/embedded-row/MTP 32-token parity passes.
* Video timestamp acceptance remains open: the eight-frame `00:04` fixture
  expects 5 but both the candidate and exact BF16 tower answer 6. Identical
  failure in the baseline does not turn the gate into a pass.
* Strict vision row-cosine acceptance also remains open: q8 mean is 0.99774,
  but 6/1488 rows fall below 0.98 (minimum 0.85990). The HF bf16/fp32 baseline
  has comparable outliers; the specified minimum is not silently relaxed.
* Previous tower work found plain q4tp/GPTQ vision below its quality threshold;
  q8_2f vision and audio-tokenizer fallback candidates are used alongside the
  GPTQ-q4tp audio encoder. The base text q4tp is not requantized.

Final ingress repeat on `26c02083`: 46 MiMo unit tests, 7 actual-device bank
checks, the CLI media-flag parser and server content-preservation test pass.
The repeated real-media suite retains identical ID/Hidden/MTP output, exact
OCR and zero WER on all five speech clips; the video answer remains 6.

Both staged variants were exercised through the HTTP server. Multipart text
IDs equal concatenated single-text IDs; neither server startup nor pure text
maps the MM companion. Full-package image OCR returns `CORTIQ 7392`; malformed
streamed media fails with HTTP 400 before SSE. Text-only correctly rejects
media with HTTP 400. CLI and server expanded image prompt IDs are identical.
The packages contain regular hardlinks to the shared backbone/MTP (not
absolute symlinks); text-only contains no `.mm.cmf`.
