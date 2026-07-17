# Skills: a swarm of specialists in one file

*Languages: [Русский](SKILLS.ru.md) · [中文](SKILLS.zh.md)*

One `.cmf` file can carry a shared backbone plus any number of **skills** —
sets of replacement tensors grafted from real fine-tuned checkpoints. The
runtime reads a skill's tensors *in place of* the backbone's (never adds,
never assembles a second model), so switching specialists is free: same
mmap, same RAM, different weights at the same addresses. A prompt can be
routed to the right skill automatically — the file itself knows which of
its specialists fits, with no trained gate and no external router.

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
cortiq run swarm.cmf -p "..."                      # backbone
cortiq run swarm.cmf -p "..." --skill thinker      # explicit overlay
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

## Resources

| What | Cost |
|---|---|
| Donor download | one HF checkpoint (~1 GB bf16 for a 0.5B), cached in `~/.cache/cortiq/hf` |
| Bake time | minutes on a laptop: quantize the grafted tensors + rewrite the file |
| File growth | + the grafted tensors only: 314 MB per FFN-all-layers skill on a q8 0.5B (backbone stays byte-identical) |
| RAM while baking | ~backbone size + one donor tensor (donor is mmap-streamed) |
| Runtime cost of a skill | zero when inactive; active skill = same speed as the backbone (identical shapes, identical kernels) |
| Routing cost | one prefill pass over the prompt up to the φ layer |

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
