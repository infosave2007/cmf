---
library_name: cortiq
license: mit
base_model:
- deepseek-ai/DeepSeek-V4-Flash-0731
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- cmf
- cortiq
- moe
- 4-bit
- 2-bit
---

# DeepSeek-V4-Flash-0731 — CMF

[DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
(304B, 43 layers, 256 routed experts top-6 + shared) in the
[CMF container](https://github.com/infosave2007/cmf), running on `cortiq`.

**Requires cortiq 0.5.47 or newer.** No configuration: the runtime detects
the GPU and VRAM itself and picks the fast path, including speculative
decode. The automatic q4tp A40 path described below is in current CMF master
after 0.5.100 and will be included in the next packaged release.

| variant | size | expert planes | folder |
|---|---|---|---|
| `q2tp` | 112 GB | 2-bit gate/up, 4-bit down | `parts-q2tp-v3/` |
| `q4tp` | 158 GB | 4-bit throughout | `parts-q4tp/` |

(`parts-q2tp-v2` is superseded by `-v3`: same trunk byte for byte, the
draft stack requantized for speculative decode.)

## Install the runtime

```sh
cargo install cortiq-cli   # or a prebuilt binary: github.com/infosave2007/cmf/releases
```

## Download

Files ship in slices that concatenate back byte for byte:

```bash
huggingface-cli download infosave/DeepSeek-V4-Flash-0731-cmf \
  --include 'parts-q2tp-v3/part_*' --local-dir .
cat parts-q2tp-v3/part_* > dsv4-flash-q2tp.cmf

# speculative-decode routing tally — keep it next to the model file
huggingface-cli download infosave/DeepSeek-V4-Flash-0731-cmf \
  dsv4-flash-q2tp.cmf.dspark.tsv --local-dir .
```

For q4tp, replace `parts-q2tp-v3` with `parts-q4tp`. The routing tally is
format-independent; copy or rename it to `dsv4-flash-q4tp.cmf.dspark.tsv`.
Without the sidecar everything still runs; speculative acceptance is lower.

## Run

Get `cortiq` from the [releases](https://github.com/infosave2007/cmf/releases)
or `cargo build --release --features gpu`. Then:

```bash
# one-shot
cortiq run dsv4-flash-q2tp.cmf \
  --prompt $'<｜begin▁of▁sentence｜><｜User｜>What is 2 + 2? Answer with just the number.<｜Assistant｜></think>' \
  --max-tokens 8

# server (OpenAI-compatible)
cortiq serve dsv4-flash-q2tp.cmf --port 8080

# benchmark (greedy, the numbers below)
cortiq bench dsv4-flash-q2tp.cmf --core --tokens 128
```

Prompt format: BOS is required, and ordinary chat closes the reasoning
block **in the prompt** — end with `</think>`. For reasoning mode end with
`<think>` instead. Without BOS the output is noise; that is
out-of-distribution input, not a broken model.

```
<｜begin▁of▁sentence｜><｜User｜>your question<｜Assistant｜></think>
```

**Generating code: use `--temperature 0.6 --rep-penalty 1.0`.** The default
repetition penalty is harmful for code because it penalizes repeated
identifiers and punctuation.

## Measured speed: q2tp

`cortiq bench --core --tokens 128`, one RTX PRO 6000 Blackwell, no
environment variables. The smaller points were measured by capping VRAM to
what a card of that size would auto-detect.

| VRAM | tok/s | mode |
|---|---:|---|
| 96 GB | **40.2** | speculative decode, 63% acceptance |
| 64 GB | 6.9 | GPU walk, cold experts from mmap |
| 32 GB | 3.9 | GPU walk |
| 16 GB | 3.2 | GPU walk |
| CPU only (48 cores) | 2.7 | host |

Speculation engages automatically when the budget packs the trunk far
enough for the draft's capture layers; below that it declines and the whole
budget goes to expert packs. Plan on system RAM at least the file size.

## Development result: q4tp on A40

Current master auto-selects chain-of-one, the VRAM margin, batched prefill,
native-format MTP and a calibrated `<model>.moe-mass.json` sidecar when one
is installed. On a 46 GiB NVIDIA A40, driver 580.173.02:

| mode | steady tok/s | notes |
|---|---:|---|
| old default | 0.665 | exact routing |
| exact dynamic LRU | **1.5–1.8** | full routing; exact CPU cold correction |
| weighted mask 97.5%, no MTP | **11.0** | all 43 layers in the GPU chain |
| weighted mask 95% + automatic MTP | **19.5** | 34/60 accepted |
| weighted mask 92.5% + automatic MTP | **20.5** | 34/60 accepted, 48-token run |
| bare command + sidecar | **21.5** | 48/80 accepted, exactly 64/64 tokens |

The old mask counter gave every top-k winner one vote. That was wrong for
DeepSeek V4: a weak eighth route and the dominant route consumed the same
coverage. `CMF_MOE_STATS` now records normalized routing-weight mass. At
92.5% it keeps an average of 40 experts in each learned-router layer; the
three hash-router layers stay complete.

The aquarium sidecar was calibrated only on the first 96 prompt tokens.
Paired quality on three later, evenly spaced 32-token windows (93 scored)
was PPL **94.217 masked vs 242.799 unmasked**. The high absolute values come
from resetting context at arbitrary HTML/JS windows; the paired comparison
uses identical tokens. This is a **task-specialist** sidecar, not a universal
DeepSeek mask: do not attach it to a general-purpose q4tp download. Without
a sidecar, exact dynamic LRU remains the safe default and a cache miss never
changes the output.

The zero-knob speculative gate counts the sidecar's open experts rather than
all 256 rows when reserving VRAM. The 97.5% mask plus verify workspace OOMed
the A40; 95% and 92.5% fit. `CMF_DSV4_SPEC=1` remains a diagnostic override,
not a launch requirement. The benchmark now performs a real untimed decode
warmup, so first-use DSpark upload is not charged to the steady window.

No aquarium output or third-party example file is published from this test.

## Quality gate

```bash
cortiq ppl dsv4-flash-q2tp.cmf --file docs/ppl_nat.txt --tokens 128
# PPL = 4.578
```

4.578 on the repository's reference text, identical to the last digit on
CPU and GPU and at every VRAM budget from 16 to 96 GB. Any change that moves
it is a defect. (`cortiq ppl` pins the strict kernels itself; generation uses
the fast ones.)

Known trade-off of `q2tp`: it reasons and writes like `q4tp` but is weaker at
arithmetic (2+2 can come out wrong; `q4tp` answers 4). Do not restrict routing
with `CMF_MOE_MASK` in production — generation degrades within a few dozen
tokens.

## FCD and the draft

The 21.5 tok/s result does not use FCD. The current `cortiq fcd` polishes
dense FFNs and deliberately rejects MoE layers; DeepSeek V4 and its MTP stack
are MoE. A correct future `draft-fcd` should distill q2 MTP experts from the
native q4 MTP on real activations and optimize held-out speculative
acceptance. Since the trunk still verifies every candidate, this can improve
speed without changing a single confirmed output token.

## Provenance

Weights derive from DeepSeek's release and remain under its licence. The CMF
container and the cortiq runtime are Apache-2.0 (see the repository's LICENSE
and PATENTS.md).
