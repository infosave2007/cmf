# Embryo chat model: how it is assembled

A sealed Embryo genome, knowledge records next to it, and a resonance router that picks the record. All steps are scripted in `scripts/embryo/`.

## Training (GPU)

```bash
# brief answers: every assistant turn cut to its first sentences, T256
uv run --with pyarrow python scripts/embryo/build_brief_mix.py --smol ... --oasst ... --q27 ... --ru ... --out mix.messages.jsonl
cortiq-embryo sft-prepare --input mix.messages.jsonl --tokenizer tok.json --format messages --group-field group --open-truncated --seq 256 \
  --train-out train.sft --dev-out dev.sft --final-out final.sft --manifest-out manifest.json
# one trainer at a time on a 24 GB Mac; 3/5 chat records + 2/5 raw windows, the base keeps learning
cortiq-embryo sft --ckpt in.ckpt --shard train.sft --val dev.sft --replay raw.u16 --raw-val rawval.u16 --out out.ckpt \
  --batch 40 --seq 256 --steps N --lr 3e-5 --dropless --allow-extended --sampler-skip S --replay-fifths 2 --skip-eval-before
```

Chat training wears out long-range recall. Healing brings it back without touching the rest of the network: only the anchors train, on raw windows with retrieval tasks, and every other tensor and all routing descriptors stay byte-identical.

```bash
cortiq-embryo birth --resume out.ckpt --train-only layers.3.attn,layers.7.attn --carry --carry-reset-every 16 --carry-eot 32760 --dropless \
  --retrieval-rate 3 --retrieval-dists 128,2048 --retrieval-pairs 4 --shard en.u16:0.6 ru.u16:0.2 code.u16:0.12 math.u16:0.08 \
  --val rawval.u16 --batch 8 --seq 1024 --steps M --lr 5e-5 --lr-restart --out healed.ckpt
```

## Assembly

`scripts/embryo/assemble_chat.sh <ckpt> <dir>` exports the sealed genome (`DTYPE`, q8_2f in the release) with the chat template and file sampling defaults (`--chat`, `--sampling`), builds the lookup records, computes φ on the GPU and fits the router. `scripts/embryo/eval_chat.sh <dir>` runs the gates.

## The router

`router.phi.pool = key_span`. The engine finds the table key in the message and pools the hidden states after the φ layer: φ = unit(unit(mean over the key tokens) + α·unit(mean over the rest of the message)). Every record is an isolated subspace (mean + top-k principal directions); the request goes to the record with the smallest reconstruction error if it beats the backbone subspace by the margin. Only records whose table holds the key are candidates, and a message without a key runs the backbone with no φ forward.

A message that names no key and is no follow-up is decided on the whole-message φ between two topologies (`router.keyless`): questions for a specific fact and ordinary chat. A question for a fact goes to the `dontknow` record with an honest answer, since the network has no facts of its own; talk, tasks and advice go to the network. A `--keyless` record answers with the card of its first entry.

A message without a key after a turn that named one (a follow-up) is decided on φ = unit(unit(key tokens of the earlier turn) + `followup_alpha`·unit(the message)) at `followup_margin`, on isolated topologies of its own (`router.followup`): follow-up questions about the record against unrelated messages with the same key in front of them. The first-turn topologies never saw this form: plain follow-ups passed 1–2 % of the time. For cards with fields only.

```bash
cortiq-embryo phi-pool --ckpt c.ckpt --cmf m.cmf --prompts skill.jsonl --layer 5 --alpha 0.5 --out s     # GPU batches
cortiq-embryo phi-pool --ckpt c.ckpt --cmf m.cmf --prompts general.jsonl --layer 5 --alpha 0.5 --out g
cortiq route-fit m.cmf --id herbs --skill-prompts s.prompts.jsonl --general-prompts g.prompts.jsonl --skill-phi s.phi.f32 --general-phi g.phi.f32 \
  --phi-layer 5 --rank 16 --max 0 --key-alpha 0.5 --margin 0.01 --target-fpr 0.001
# follow-ups: f32 rows of unit(unit(key) + 4·unit(message)), scripts/embryo/mk_followup.py
cortiq route-fit-followup m.cmf --skill-phi herbs=fu-herbs.f32 --general-phi fu-general.f32 --rank 32 --alpha 4 --margin 0
# keyless: whole-message φ (phi-pool --whole-message) of the two classes, scripts/embryo/mk_keyless.py
cortiq route-fit-keyless m.cmf --record dontknow --fact-phi kl-fact.phi.f32 --chat-phi kl-chat.phi.f32 --rank 32 --margin -0.06
```

The general set must hold prompts that name table keys in ordinary speech, capped per key: otherwise one frequent word fills the backbone subspace.

## File defaults for a small model

`sampling.loop_stop = 1`: the answer stops when its tail repeats itself (finish reason `loop`), and the returned text says the block once. `sampling.history_turns = 1`: the network sees only the last user turn, because a small model copies its earlier answers. The lookup records keep their conversation memory either way.

## Runtime

`cortiq run` and `cortiq serve` put an Embryo file on the resident GPU graph (`CMF_GPU=wgpu`, `CMF_EMBRYO_RESIDENT=1`, wgpu over Metal on a Mac) unless you chose a backend. `CMF_GPU=0` runs the CPU path; the greedy text is the same. An Embryo file keeps float activations on the CPU, because its router is fitted on the exact φ and the 8-bit activations of the other q8 models move φ by 4e-3 (`CMF_SDOT=1` turns them on: a 5× faster prefill, a different text). Both paths stay within 3e-5 of exact arithmetic.

The matrices stay on the device in the form they have in the file, f16 or q8_2f (`gpu_wgpu/embryo_wide.wgsl` with `embryo_wide_f16.wgsl` or `embryo_wide_q8.wgsl`). A token is about 80 dispatches and a boundary between two costs more than most kernels (8 µs on an M4, 5 µs on an RTX 2000 Ada), and a GPU only fills up with many short threads. So a thread walks at most 128 values of a row and the next norm stage adds up the partial sums; the column field of a q8_2f matrix goes into its input vector once, in the stage that makes the vector, and the codes unpack as integers; an anchor rotates its query and keys by the absolute position, so the ring holds rotated keys and a score is a plain dot product; the value sum goes by blocks over the keys above e^-16 of the largest weight; the delta rule and its norm share one workgroup per head. The head reads only the clusters above a log probability of −9.21 (`CMF_EMBRYO_HEAD_GATE`; `-1e30` reads every row, and the rows of a skipped cluster come back as −1e30). `CMF_EMBRYO_WIDE=0` runs the f32 path.
