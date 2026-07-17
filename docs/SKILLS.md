# Skills: a swarm of specialists in one file

*Languages: [Русский](SKILLS.ru.md) · [中文](SKILLS.zh.md)*

One `.cmf` file can carry a shared backbone plus any number of **skills** —
sets of replacement tensors grafted from real fine-tuned checkpoints. The
runtime reads a skill's tensors *in place of* the backbone's (never adds,
never assembles a second model), so switching specialists is free: same
mmap, same RAM, different weights at the same addresses. A prompt can be
routed to the right skill automatically — the file itself knows which of
its specialists fits, with no trained gate and no external router.

## Terms — the three words you need

- **Backbone (base model)** — an ordinary LLM converted to CMF
  (`cortiq convert`). It is the model's shared weights *without* skills;
  skills live in the same file next to it and replace some of its
  tensors when active.
- **Skill** — a set of replacement tensors ("in these layers take the
  specialist's FFN weights, everything else comes from the backbone")
  plus a ~4 KB routing descriptor that lets the file recognize "this
  prompt is mine".
- **Donor** — a *fine-tuned checkpoint of the very same backbone* (SFT
  or merged LoRA) a skill can be taken from ready-made. A donor is NOT
  a different model (a "coder", a "philosopher"): grafting from a
  continued-pretrain relative does not work, and that is measured below.

## Where a skill comes from: two paths

**Path 1 — a dataset, no donor: `cortiq skill bake`.** If all you have
is the backbone and domain text (a corpus, benchmark tasks), the engine
finds the relevant neurons itself. This is exactly the imatrix-style
"highlighting" intuition taken to its conclusion: run the corpus through
the model and watch which FFN neurons carry the task — except instead of
OR-ing the highlights the mask is *trained* (L1 regularization per
DTG-MA — USPTO application 19/452,464, [PATENTS.md](../PATENTS.md)), because the honest measurement shows untrained top-K
highlighting collapses quality (PPL 59 vs 11.9 — see below). The trained
mask drops noise neurons — the model first gets *better* on the domain —
then the last layers are polished against the exact teacher (FCD). No
second model is needed.

**Path 2 — a ready donor: `cortiq skill add`.** If someone already
fine-tuned your backbone (a text-to-SQL tune, a Russian SFT), no
training is needed at all: the "highlighting" was already done by the
donor's author — their fine-tune IS the training result. What remains is
extracting the changed part: each donor tensor is compared against the
backbone, tensors the tune never touched are dropped (`--min-delta`),
the changed ones are grafted into the file as a skill. Minutes on a
laptop, no GPU.

Both paths end the same way: the file gains a skill with an id, a name
and a routing descriptor; `skill list` shows it, `--quality` writes the
measured verdict into the registry.

### FAQ

**Is the backbone the weights including masks/skills?** No. The backbone
is the shared model weights. Skills and masks are separate directory
entries in the same file; when a skill is active the runtime reads its
tensors instead of the backbone's.

**How do you extract just the relevant weights from a donor without
training?** The training already happened — inside the donor. We do not
search for "skill neurons" in it; we take the *difference*: the tensors
the fine-tune changed (a reference decoder measures relative delta
against the backbone), and store only those. The threshold is
`--min-delta`.

**And if there is no donor?** Then path 1: dataset → `skill bake` →
trained mask + polish. That is precisely the "ran 50 benchmark tasks —
got an swe-skill" scenario: `--files` takes your texts, the engine does
the rest (a real step-by-step run log is below).

**Can skills be shared between people?** Yes. A skill is a set of
directory entries inside a `.cmf`; you can hand over the whole file with
its swarm, and moving a single skill between files of the same backbone
is a catalog operation (no weight recomputation). A skill-hub is exactly
where this design points.

Everything below is reproducible with three commands' worth of real
checkpoints from Hugging Face; the prompt and eval files live in
[`docs/skills-demo/`](skills-demo/). Measured on `cortiq` 0.3.4+ with
Qwen2.5-0.5B-Instruct (q8) as the backbone.

## The demo: one file, four personalities

```sh
# 1. A backbone
cortiq convert --model Qwen/Qwen2.5-0.5B-Instruct --quant q8 --output swarm.cmf

# 2. Three skills, grafted from real fine-tunes of that same base
cortiq skill add swarm.cmf --from vindows/qwen2.5-0.5b-text-to-sql-merged \
  --id sql --name "SQL assistant" --prompts docs/skills-demo/prompts-sql.txt

cortiq skill add swarm.cmf --from Vikhrmodels/Vikhr-Qwen-2.5-0.5b-Instruct \
  --id ru --name "Русский ассистент" --prompts docs/skills-demo/prompts-ru.txt \
  --quality docs/skills-demo/eval-ru.txt

cortiq skill add swarm.cmf --from ewre324/ewre324-Thinker-Qwen2.5-0.5B-Instruct-Reasoning \
  --id thinker --name "Step-by-step verifier" --prompts docs/skills-demo/prompts-think.txt
```

`skill list` shows what the file now carries:

```
3 skill(s):
  sql        SQL assistant            72 tensor(s), 314.3 MB, layers [0..23], routable
  ru         Русский ассистент        72 tensor(s), 314.3 MB, layers [0..23], routable
      quality: {"backbone":11.875,"overlaid":11.027,"metric":"ppl","file":"eval-ru.txt"}
  thinker    Step-by-step verifier    72 tensor(s), 314.3 MB, layers [0..23], routable
```

One 1.45 GB file instead of four separate 479 MB models (1.9 GB), one
mmap, and the switch between personalities costs nothing at load time —
skill tensors are ordinary directory entries the runtime points at
instead of the backbone's.

## What measurably improved

**Routing: 6/6 on prompts the router never saw.** The file routes fresh
prompts to the right specialist with a 3–5× error margin
(`E = ‖r−BBᵀr‖²/‖φ‖²`, lower is closer):

```
$ cortiq route swarm.cmf -p "Show me the SQL to find orders with no matching invoice."
  sql      E = 0.0016   ← winner
  thinker  E = 0.0073
  ru       E = 0.0112

$ cortiq route swarm.cmf -p "Расскажи, как приготовить сырники из творога."
  ru       E = 0.0050   ← winner
  thinker  E = 0.0168
  sql      E = 0.0204

$ cortiq route swarm.cmf -p "Verify step by step whether 91 is divisible by 7 and by 13."
  thinker  E = 0.0047   ← winner
  sql      E = 0.0165
  ru       E = 0.0190
```

**`ru`: −7.1% perplexity on Russian prose**, recorded in the file's own
registry at bake time (backbone 11.88 → overlaid 11.03 on a held-out
Russian text). Generation quality is visibly different — the backbone
produces circular list-points («Стоимость аренды обычно выше, чем
лизинг… Стоимость лизинга обычно ниже»), the skill writes coherent
structured prose.

**`thinker`: checks instead of asserting.** Ask *"Is 17077 a prime
number?"* — the backbone confidently invents a wrong factorization
(`17077 = 3 × 5699`, which is false: 17077 is prime); the skill starts a
systematic divisibility check. Same file, `--skill thinker`.

**`sql`: text-to-SQL structure.** The skill answers schema questions with
window functions and CTEs where the backbone writes flat joins.

A note on metrics, honestly: perplexity is the right gate for a skill
that shifts the *distribution* of a domain (like `ru` on Russian text).
For instruction-style SFT donors (`sql`, `thinker`) the win is in
*generation behavior*, not text likelihood — measure those with A/B
generations or task accuracy, and expect PPL on hand-written corpora to
be flat or even slightly worse. `--quality` records whatever you measure
into the registry; it never silently blesses a skill.

## Using the skills

```sh
cortiq run swarm.cmf -p "..."                      # routes automatically: the file picks its specialist
cortiq run swarm.cmf -p "..." --skill thinker      # pin one explicitly
cortiq run swarm.cmf -p "..." --skill none         # force the backbone
cortiq route swarm.cmf -p "..."                    # score all skills
cortiq explain swarm.cmf --prompt "..."            # routing + first-token distribution
cortiq run swarm.cmf -p "..." --blend auto         # soft superposition of the top-2
cortiq run swarm.cmf -p "..." --route-dynamic --trace   # per-token switching
```

`--blend auto` mixes the top-2 skills with softmax(−E/T) weights — a
Croatian math-check prompt lands between `ru` and `thinker` at ~50/50.
`--route-dynamic` switches the active skill *mid-stream* as the context
evolves; with `--trace` you watch it happen (a mixed Russian/SQL prompt
switched `thinker → sql` exactly as the generation entered the
`… WHERE o.id IS NULL` clause, then back to the backbone for the Russian
explanation). Switching thresholds are hysteresis-guarded; tune with
`CMF_ROUTE_EON` / `CMF_ROUTE_EOFF` (activation / abandonment E) if your
skills sit closer together than these defaults.

A real per-token trace with numbers (`--trace`, mixed prompt "Найди в
базе всех клиентов без заказов: напиши SQL-запрос и объясни его
по-русски", `CMF_ROUTE_EON=0.9 CMF_ROUTE_EOFF=1.0`):

```
  #  token        skill       E
  0  Вот          —           —       first tokens: router builds φ-EMA
  7  ,            —           0.857   E < EON(0.9) → switch on
  8  который      thinker     0.857   thinker active
 23  sql          thinker     0.662   E floor — the SQL block
 31  c            thinker     0.834   Russian explanation: E climbs…
 47  ет           thinker     0.865   …but E < EOFF(1.0) — hysteresis holds
```

Read the numbers as margins: between switch-on (0.857 vs the 0.9
threshold) and drop-out (0.865 vs 1.0) there is ~0.13 of slack — no
flapping. The defaults (EON 0.62 / EOFF 0.74) are conservative: this
swarm's E floor is 0.662, so an untuned file honestly stays on the
backbone. Raise the thresholds when your skills' φ statistics sit
further from live prompts than in this demo.

## Creating a skill: what actually matters

`cortiq skill add` grafts tensors from a **donor** checkpoint into the
backbone's container:

```sh
cortiq skill add <backbone.cmf> \
  --from <hf-repo-or-local-dir>   # donor, same architecture
  --id <id> [--name "..."]        # registry identity
  --layers all|A-B|i,j,k          # which layers (default all)
  --tensors ffn|attn|all          # which families (default ffn)
  --prompts file.txt              # 8+ example prompts → routing subspace
  --quality held-out.txt          # PPL gate, recorded in the registry
  --min-delta 0.02                # drop tensors the tune never touched
  --skill-quant vbit --mean-bits 6  # cheaper encoding for the overlay
  --output out.cmf                # default: rewrite in place
```

The donor's tensors are re-quantized with the backbone's own per-tensor
encoding, so a q8 backbone gets q8 skills; the graft is bit-faithful
(re-baking a skill from the backbone's *own* source checkpoint changes
PPL by exactly +0.0%).

**Pick donors that are fine-tunes of your backbone.** This is the one
rule that decides success. FFN tensors co-adapt with the attention
around them:

- **SFT / merged-LoRA on the same base — works.** All three demo donors
  are fine-tunes of Qwen2.5-0.5B-Instruct; their FFN grafts are coherent
  and behaviorally distinct.
- **Continued-pretrain relatives — do not work as partial grafts.** We
  measured it so you don't have to: grafting FFN (or even FFN+attention)
  from Qwen2.5-**Coder**-0.5B-Instruct — same shapes, same family, but a
  heavily continued-pretrained sibling — produced PPL 238 vs the
  backbone's 2.6. Its weights co-adapted with norms and embeddings the
  skill doesn't carry. If the donor diverged that far, you want a
  separate model file, not a skill.

**Routing prompts** (`--prompts`): 8 short prompts that *sound like the
skill's users*. The φ(x) statistics (mean + rank-2 PCA basis over the
mean-pooled hidden state at 2/3 depth) are fitted from them and stored in
the header — ~4 KB per skill, and the file routes itself from then on.
Skills that replace non-FFN tensors are excluded from *dynamic* routing
(static `--skill` still works) — the runtime warns when that happens.

## Resources: what you need, on what hardware

**Path 2 (`skill add`) is light.** No GPU at all:

| What | Cost |
|---|---|
| Donor download | one HF checkpoint (~1 GB bf16 for a 0.5B), cached in `~/.cache/cortiq/hf` |
| Time | minutes on a laptop: quantize the grafted tensors + rewrite the file |
| RAM | ~backbone size + one donor tensor (donor is mmap-streamed) |
| File growth | + the grafted tensors only: 314 MB per FFN-all-layers skill on a q8 0.5B (backbone stays byte-identical) |
| Runtime cost of a skill | zero when inactive; active skill = same speed as the backbone (identical shapes, identical kernels) |
| Routing cost | one prefill pass over the prompt up to the φ layer |

**Path 1 (`skill bake`) is CPU training.** The bake holds an f32 replica
of the model (4 bytes per parameter) plus optimizer state only for the
polished layers, so plain RAM sets the ceiling:

| Backbone | Bake RAM (estimate: 4 B/param + optimizer) | Time (M4, 240+120 steps) |
|---|---:|---:|
| 0.5B | ~3 GB | 8.8 min (measured) |
| 1–2B | ~6–10 GB | ~20–40 min |
| 4B | ~18 GB | ~1.5 h |
| 7–8B | ~35 GB | ~3 h |
| 27B | ~115 GB — a 128 GB+ Mac Studio, or the torch recipe on CUDA | overnight |

Corpus: 100–200 KB of domain text is enough (the real run above used 82
chunks of 256 tokens). For a "benchmark skill", collect the task
statements and reference solutions into text files and pass them to
`--files`.

## One command, no Python: `cortiq skill bake`

The whole DTG-MA recipe runs natively — mask training, FCD polish and
the defrag bake in a single command on the CPU (the training GEMMs ride
the same Accelerate path as prefill; attention is frozen, so the
backward walks only the FFN chain):

```sh
cortiq skill bake backbone.cmf \
  --files docs/CMF_V2_SPEC.ru.md README.ru.md docs/COMPARISON.ru.md \
  --output rutech-specialist.cmf
```

A real run, step by step — Qwen2.5-0.5B-Instruct (q8) on this
repository's own Russian docs, Apple M4, **8.8 minutes end to end**:

```
bake: 70 calib + 12 held chunks of 256 tokens | FCD last 4 layer(s)
baseline (full): 24.157
  [A] step  30: L1=0.015 pruned= 0% hard-PPL=23.648 (bottom 23.648@0%)
  [A] step  60: L1=0.020 pruned= 2% hard-PPL=21.110 (bottom 21.110@2%)   <- the denoising bottom
  [A] step  90: L1=0.025 pruned= 6% hard-PPL=22.778 (bottom 21.110@2%)
  [A] step 120: L1=0.030 pruned=10% hard-PPL=25.610 (bottom 21.110@2%)
  [A] step 180: L1=0.040 pruned=16% hard-PPL=49.659 (bottom 21.110@2%)   <- past the bottom quality collapses
[A] 314s: masked-PPL 21.110                                              <- the bottom checkpoint is restored
  [B] step  30: held-PPL 18.304
  [B] step  60: held-PPL 17.840
  [B] step  90: held-PPL 17.474
  [B] step 120: held-PPL 17.423                                          <- FCD keeps digging
=== bake: baseline 24.157 | mask 21.110 | mask+FCD 17.423 | pruned 2% -> SPECIALIST <= baseline
runtime gate (held-out, real engine): backbone 24.173 -> specialist 19.039 (-21.2%)
```

Three verdicts worth reading twice:

- The training replica and the real engine agree (baseline 24.157 vs
  24.173) — what the bake optimizes is what the runtime serves. The gap
  between 17.4 (the f32 replica) and 19.0 (the written file) is the q8
  re-quantization of the trained FFN — measured, not hidden.
- **Generalization**: on a Russian tech document that was never in the
  corpus (`PERFORMANCE_ROADMAP.ru.md`) the specialist scores 22.56 vs
  the backbone's 25.62 — **−12.0% on unseen text**.
- The denoising bottom landed at 2% here (backbone PPL 24 — not that
  weak on this domain), so the size win is small (479 → 472 MB); the
  quality win is the story. On a weak domain (backbone PPL 70) the same
  recipe prunes 11% and cuts PPL by a quarter — see the next section.

## Smaller than the original — and better: the DTG-MA bake

The strongest form of a skill doesn't ride next to the backbone — it
*replaces* it for one domain. The DTG-MA recipe (USPTO application
19/452,464 — legal context in [PATENTS.md](../PATENTS.md)) trains an
L1-regularized mask over the FFN neurons on your task corpus, rides the
*denoising bottom* (pruning noise neurons first IMPROVES the model
before it starts to hurt), polishes the last layers against the exact
teacher (FCD), and then `--defrag` bakes a standalone file where pruned
neurons are physically absent — neither stored nor computed
(claims 9/10 of application 19/452,464):

```sh
# native, one command (see the previous section):
cortiq skill bake backbone.cmf --files corpus1.txt corpus2.txt --output specialist.cmf

# or the original torch recipe (identical phases; useful on CUDA boxes):
python3 converter/make_skill_l1fcd.py --model <hf_snapshot_dir>   --id ru --files corpus1.txt corpus2.txt --out skill-ru
cortiq convert --model <hf_snapshot_dir> --defrag skill-ru   --quant q8_2f --output ru-specialist.cmf
```

Measured end-to-end through this repository's runtime on Qwen3.5-0.8B,
scored on a held-out Russian technical document the recipe never
trained on:

| | size | PPL (ru tech, held-out) | decode |
|---|---:|---:|---:|
| original checkpoint (bf16) | 1.6 GB | — | — |
| CMF q8_2f baseline | 733 MB | 13.97 | 86.0 tok/s |
| **ru-specialist (mask 11% + FCD, defragged)** | **705 MB** | **11.92 (−14.7%)** | **89.7 tok/s** |

Smaller than the original LLM by 2.3×, better on the domain, and faster
— from one bake. The recipe's own report on its in-domain held-out went
further still: masked bottom 70.5 → 54.2 PPL at 11% pruning, −38.7%
after FCD, −19.4% on an independent unseen tech corpus.

Two honest rules the measurements taught:

- **The bake shines where the backbone is weak.** Russian tech text
  (backbone PPL 70) responded with the full effect; the same recipe on
  a domain the backbone already handles well (code, PPL 8.7) made
  things *worse*. Check your backbone's PPL on the domain first.
- **The denoising bottom is real and narrow.** The mask's held-out PPL
  during training: 67.9 at 0% pruned → **54.2 at 11% (the bottom)** →
  77.1 at 18%. The recipe stops at the bottom automatically; don't push
  past it for size.

`cortiq skill add --sparse <keep>` offers a quick *training-free*
approximation of the mask (per-layer top-K by task activation mass) —
useful for probing, but measured honestly: with an 8-prompt calibration
it collapsed at keep 0.5 (PPL 59 vs 11.9). The trained mask is the real
method; the flag is a scout.

## Shrinking a skill on disk

A skill should cost what the fine-tune actually changed — untouched
neurons are not worth bytes. Two flags control that, both measured, both
gated by `--quality`:

**`--min-delta <x>` — don't store what the tune never touched.** Every
candidate tensor is compared against the backbone through the reference
decoder; tensors whose relative change is below the threshold are
dropped — the runtime reads the backbone entry there, which is exactly
what the donor holds anyway. The gate prints the full delta
distribution, so you see before you cut:

```
# the ru donor, measured:
delta gate ≥ 0.0000001: kept 72 / dropped 0 unchanged tensor(s) (−0.0 MB);
rel-delta min 0.0980 / median 0.2196 / max 0.3724
```

Calibrate the threshold against your backbone's *quantization noise
floor*: bake a self-graft (`--from` = the backbone's own source
checkpoint) with `--min-delta 0.0000001` and read the printed median —
that is pure quant error (measured median 0.92% for our q8 backbone; a
self-graft with `--min-delta 0.02` correctly drops all 72 tensors and
refuses to bake). Deltas at that level are noise, not signal. Our three
demo donors all sit above it everywhere (sql 0.9–2.6%, thinker
1.8–3.5%, ru 9.8–37%), so the gate rightly kept them whole — it pays
when the tune is localized (a LoRA merged into a few projections, a
tune that only touched deep layers), and it never lies: it measures
first.

**`--skill-quant <enc> [--mean-bits N]` — a cheaper encoding for the
overlay.** Per-tensor dtypes are native to CMF (spec §3), so a skill may
live in fewer bits than the backbone. Measured on the `ru` skill over
the q8 backbone (held-out Russian prose, backbone PPL 11.88):

| encoding | size | skill PPL | verdict |
|---|---:|---:|---|
| q8 (same as backbone) | 314 MB | 11.03 (−7.1%) | full quality |
| `--skill-quant vbit --mean-bits 6` | 257 MB (−18%) | 11.19 (−5.8%) | keeps most of the win |
| `--skill-quant vbit` (4.25 bits) | 184 MB (−41%) | 12.17 (+2.5%) | **worse than backbone** |
| `--skill-quant q4` | 176 MB (−44%) | 12.20 (+2.7%) | **worse than backbone** |

The lesson is the same one quantization always teaches, one level down:
the *fine-tune's signal* has to survive the extra noise. A 0.5B's SFT
delta (here 10–37% relative) survives ~6 bits and drowns at 4. Always
re-measure with `--quality` after shrinking — the number lands in the
registry either way, so the file carries its own verdict.

You can also restrict layers by hand (`--layers 12-23` halves the size);
same rule — measure whether the behavior you care about survives.

## A skill's passport — what to ship alongside

The registry already carries almost everything needed for honest
exchange; if you publish a skill (or prepare for a future skill-hub),
the minimal passport is:

| Field | Source | Why |
|---|---|---|
| `id`, `name` | `skill add --id/--name` | registry identity |
| provenance | donor repo+revision, or the bake corpus file list | reproducibility |
| donor license | the HF checkpoint's card | the right to redistribute delta weights |
| `quality` | `--quality held-out.txt` — written into the registry | the measured verdict: backbone → overlaid, on what |
| `--min-delta` threshold | the bake command | how many tensors were dropped as quant noise |
| encoding | `--skill-quant`/`--mean-bits` | readers see whether the signal survived compression |
| routing prompts | `--prompts file.txt` | the φ descriptor: how the file recognizes its prompts |

Everything except the license and provenance travels inside the file
(`skill list` shows it); state those two in the description next to it.
Moving a skill between files of the same backbone is a catalog
operation — no weight recomputation.

## How it works underneath

- Skill tensors are ordinary directory entries named
  `skill.{id}.{tensor}` — mmap'd like everything else, paged in on first
  touch. Storage scales as |backbone| + Σ|deltas| ([spec §9](CMF_V2_SPEC.md)).
- The runtime resolves every tensor through source indirection: backbone
  entry or `skill.{active}.{name}` — replacement, never addition; a full
  per-skill model is never assembled.
- Routing is recon-argmin over affine subspaces: no trained gate, no
  classifier — selection is a *property of the file*. `route`, `explain`,
  `--blend`, `--route-dynamic` all read the same 4 KB descriptors.
- The registry's `quality` field is the claim-16 honest contract: what
  was measured, on what, with what result. `skill list` shows it.
