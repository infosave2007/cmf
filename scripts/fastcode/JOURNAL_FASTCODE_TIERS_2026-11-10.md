# Fast Code on 12 / 16 / 24 / 32 GB cards — journal, 2026-11-09/10 (night)

Pod: RunPod RTX 5090 32 GB (PCIe 5.0 x16), 2× EPYC 9554 with a 31-core quota, 62 GB cgroup (the page cache counts),
300 GB local disk. Run dir `/root/fc`; scripts in `scripts/fastcode/pod2/`. A smaller card is the same 5090 capped:
cortiq gets `CMF_GPU_VRAM_MB` = the budget it would pick on that card (card − clamp(card/8, 2.5, 4) GiB); Strata, whose
expert cache sizes itself from free VRAM, gets a VRAM hog that leaves the card's size free (`hog.py`, cudaMalloc).

## Files

- `fcaq256` — the published cut (q4tp, 256 experts a layer by code + task routing), 66.8 GB, 40.9 GB on the card.
- `fcq2` — the same expert set cut from `qwen38-flash-next-q2tp-v2.cmf` (2-bit gate/up, 4-bit down): 56.2 GB, 30.7 GB on
  the card, so it runs whole on a 32 GB card.
- Strata Coder IQ1_M on Strata engine 0.1.59 (`strata update`), its own config (MTP, int8 K/V, PLE in RAM).

## Baseline before tonight's changes (cortiq.v1 = PR #103), fcq2, prompt tok/s

| card | 1K | 4K | 16K | 32K |
|---|---:|---:|---:|---:|
| 32 GB (whole) | 497 | 658 | 684 | 756 |
| 24 GB | 71 | 102 | | |
| 16 GB | 35 | 37 | | |
| 12 GB | 17 | 17 | | |

Below a whole card the prompt went in eight-token frames with a fence and a cold-list readback per layer.

## 1. A loaded routing profile kept every frame reading back (fixed)

`CMF_QWEN_PROFILE` (the warm start every card and test uses) left `state.profile` live: it only counted this run's
routing for nobody, but a live profile turns the lean frames off (each layer copied its cold lists and MoE inputs to the
host) and stopped prompt chunks from overlapping the host. The profile now stays live only with `CMF_QWEN_PROFILE_SAVE`.
fcq2 on the whole 32 GB card: prompt 497 / 658 / 684 / 756 → **1208 / 1173 / 1269 / 1138** tok/s at 1K / 4K / 16K /
32K; generation 146 / 143 / 150 at 4K / 16K / 32K.

## 2. Streamed prompt on a card that cannot hold every expert (`forward_stream_device`)

A prompt chunk (16 frames of 256 tokens, 8 under a 14 GiB budget) runs layer by layer: each layer's whole expert bank
is admitted once (`QwenGpuPool::ensure_layer`) and runs every frame of the chunk on the resident model's kernels; the
frames' cross-layer state (hyper rows, MoE output, inject weights) is parked in `Dev::stream_bank` between them. No cold
pass, no readback, one queued layer ahead of the host. `CMF_QWEN_STREAM=0` restores the eight-token frames.
Greedy text of a 3K-token prompt: streamed = the eight-token frames word for word (48 tokens); the resident chained
path differs past token ~40 (its matrix-unit frames round to f16).
fcq2, 24 GB: prompt 4K 102 → 405, 16K → 376 tok/s; 12 GB, 3K prompt: first token ~3 min → 40 s.

## 3. Strata Coder on Strata 0.1.59, the same pod and client (speedtest2: a code base + "write a module and its tests")

| card | prompt 1K / 4K / 16K / 32K, tok/s | generation 1K / 4K / 16K / 32K, tok/s | VRAM |
|---|---|---|---:|
| 32 GB | 1254 / 3637 / 5430 / 6424 | 259 / 269 / 270 / 233 | 31.0 GB |
| 24 GB | — / 4610 / 6144 / 6738 | — / 223 / 234 / 187 | 23.6 GB |
| 16 GB | 1291 / 4461 / 6003 / 6425 | 151 / 182 / 189 / 182 | 15.6 GB |
| 12 GB | 1187 / 4516 / 5986 / 6300 | 125 / 166 / 174 / 168 | 11.6 GB |

On this host (31 EPYC cores, DDR5) Strata's expert cache plus CPU pool hardly depends on the card: its prompt streams the
experts at large batches, its CPU pool computes the misses of a token. Its published RTX 5070 + 64 GB DDR5 figures are
about a third of these.

fcq2 on cortiq.s1 (fixes 1–2), same client:

| card | prompt 1K / 4K / 16K / 32K | generation 4K / 16K / 32K |
|---|---|---|
| 32 GB | 1208 / 1173 / 1269 / 1138 | 146 / 143 / 150 |
| 24 GB | 100 / 405 / 376 | — / 73 |
| 16 GB | 68 / 110 / 109 | — / 35 |
| 12 GB | 44 / 95 / 104 | 13 / 14 |

Below a whole card the streamed prompt is bound by expert admission (`write_buffer` per expert, under 1 GB/s); generation
by the cold experts of every token. Whole card: equal at 1K, Strata's prompt grows with its batch (ours stays at the
256-token frame).

## 4. A dropped expert segment in resident prompt frames (fixed)

The matrix-unit expert GEMMs of a prompt frame ran `nseg = min(winners, n_experts)` segment workgroups, but a frame's
winners carry one more distinct slot, the shared expert's. On a 256-expert cut the frame's last segment (the layer's
least-profiled expert, the highest slot) was never computed and its winners summed the part rows of an earlier frame.
Now `n_experts + 1`. It is why the resident path differed from the eight-token frames past token ~40.

## 5. fcq2 answers the 1K request with "Writing repo_summary.py" and stops

Also with MTP off and on the streamed path; fcaq256 (q4tp) writes the module. A q2tp quality loss, not the engine.

Correction to §4: s4 (the fix) gives the same greedy text and top logits as s1 — a frame rarely holds all 256 experts
of a layer, so the dropped segment almost never existed. The resident path's difference from the eight-token frames is
the f16 rounding of its matrix-unit frames. The fix stays (a full frame would have lost a segment).

## 6. Expert packs by task (the patent-1 router)

Texts: python (stdlib), rust (cargo sources), c (redis, sqlite, headers), web (three.js examples and sources, Bootstrap
examples and CSS), php (WordPress, Laravel), go (stdlib), java (commons-lang, gson), ts (VS Code base) — files of
well-known projects, each with short task prompts in that language between them; ~200 KB to fit, 35–80 KB held out
(other files). fcq2 on the whole card dumps every 128-token window's routing (`CMF_QWEN_PHI_DUMP`).
`cortiq qwen-packs packs --layers 8 --rank 16 --blocks 1,2,4`: one isolated topology per task (mean + 16 principal rows
of its own windows' φ, `router::fit_descriptor`); decision = argmin reconstruction error.

- Router on the held-out texts: 87.5 % of 128-token windows (87.4 % at 256, 87.9 % at 512); go 170/171, ts 73/73,
  java 151/154, php 172/180, rust 141/150, c 150/158, web 233/247, python 54/175 (to ts 109). Z-scoring the errors by
  the topologies' fit statistics: 84.3 % (python unchanged) — the pure argmin stays.
- Coverage of a task's held-out routing by the top-K experts per layer — own pack / the generic code+task profile the
  cut was made by / the best other pack: K=16 25.8/14.3/23.1 %, K=27 35.4/20.9/32.2, K=48 49.2/31.7/45.6,
  K=64 57.3/39.3/53.6, K=128 80.1/63.4/76.7. A pack holds ~15–20 points more of the routing than the generic profile;
  the languages share most of their experts (the best other pack is 3–4 points behind).
- A 12 GB card's arena holds ~27 experts a layer: a decode kept inside the pack would drop ~65 % of the routing. The
  masked decode is for cards from ~24 GB (~170 a layer, ~90 %); below that decode stays exact.
- Trap: `CMF_QWEN_PROFILE_SAVE` starts from the loaded `CMF_QWEN_PROFILE`, so the first task profiles were the generic
  counts plus the task's; rebuilt from the windows (`fix_profiles.py`).
- Prompts past ~65K tokens (web, php fit texts) ran out of VRAM on the 32 GB card: fcq2 resident leaves ~1 GB.

## 7. Where a resident prompt frame goes (fcq2, whole card, 16K prompt, one stage left out at a time)

experts 3.8 s (36 %), GDN 3.0 s (29 %), QSA attention 2.3 s (22 %), hyper-connections 1.0 s (10 %), PLE 0.4 s (4 %)
of 10.4 s. The q8_2f projections of GDN and QSA ran the scalar int8 tile GEMM (`q8_2f_mm_r2`, ~18 TFLOP/s).

## 8. 512-token prompt frames (TROWS 512, counting sort past 4096 winners)

Greedy text and top logits = 256-token frames. Whole card: prompt 1114/1295/1262 → 1267/1465/1438 tok/s at 1K/4K/16K
(+13 %), but a 32K prompt runs out of VRAM (31 997 MiB): the frame buffers double and fcq2 fills the card. Not the
default. 16 GB card with the host tier: prompt 221/356/366 (from 68/110/109), generation at 16K 63 tok/s (from 35).

## 9. The q8_2f projections of prompt frames on the matrix units (cortiq.s6)

`q82_t` for frames past eight tokens now runs the dense models' eight-bit prompt GEMM (`q82_mm_coop_s`, f16 operands,
f32 sums, the activation scale reduced on the card first). Top logits move in the f16 rounding (760: 21.4286 →
21.4593), greedy text the same. Whole card: 1114/1295/1262/1234 → 1208/1308/1272/1257 tok/s — nothing. Every GEMM of
our prompt frame (the projections, the experts at ~10 rows an expert of a 256-token frame) runs at ~18 TFLOP/s;
Strata does about four times our work per token. The levers are the matrix-unit GEMM kernels themselves and running
a layer's experts over the whole streamed chunk instead of each frame — work beyond this night.

## 10. Quality of fcq2 (q2tp) on the whole card

Aquarium spec, greedy: a 17 KB page that imports three.js and OrbitControls by URL with no importmap (OrbitControls
imports the bare `three`: it does not load); fcaq256 made a working page 4/4. With the 1K early stop (§5) the q2 cut
that fits a 32 GB card whole is the weaker file. HumanEval needs pyarrow on this pod (installed, rerun queued).

## 11. Masked decode (the narrow bundle) on a 24 GB card, fcq2, cortiq.s6, host tier on

| arm | prompt 1K / 4K / 16K | generation at 16K |
|---|---|---|
| exact, eight-token frames (v1) | 71 / 102 / — | — |
| exact, streamed (s1) | 100 / 405 / 376 | 73 |
| masked, the prompt's own routing | 131 / 422 / 596 | 157 |
| masked, patent-1 packs (python chosen for the code base, web for "Say hi") | 130 / 438 / 601 | 149 |

The repack after each prompt (2 000–2 700 experts out and in) took ~3 s inside the first token: `prefill` wrote every
expert through the queue. s7 refills through `fill_slot` in parallel (DMA from the host tier).

## 12. Exact decode on 16 and 12 GB cards (cortiq.s6, fcq2)

| arm | prompt 1K / 4K / 16K | generation at 16K |
|---|---|---|
| 16 GB, streamed (s1, no host tier) | 68 / 110 / 109 | 35 |
| 16 GB, host tier | 225 / 322 / 345 | 65 |
| 16 GB, host tier, cold experts on the CPU (`CMF_QWEN_COLD_DEVICE=0`) | 159 / 267 / 322 | 32 |
| 16 GB, masked decode with packs | 191 / 326 / 347 | stops after 14 tokens |
| 12 GB, streamed (s1) | 44 / 95 / 104 | 14 |
| 12 GB, host tier | 325 / 587 / 656 | 46 |
| 12 GB, host tier, cold experts on the CPU | 247 / 498 / 644 | answers of one token |

- The CPU's cold experts are slower and on 12 GB broken (one-token answers): not pursued.
- Masked decode at 16 GB (~60 % of the routing in the arena) breaks the answer: the narrow bundle needs ~24 GB.
- Stream profile, 16 GB: a 2 040-token chunk 2.3 s = admission 0.9 s + the card 0.9 s + encode 0.3 s; the chunk past
  2 048 tokens 5.3 s with 10 581 experts written through the queue — the host tier was still filling in the
  background (measured 20 s after the server started). A long-lived server's numbers come from a warmed tier.

## 13. Quality of the masked decode, and the pinned staging ring

- 24 GB, masked decode with the patent-1 packs (fcq2): HumanEval **119/164 = 72.6 %** (fcaq256 93.3 %, Strata Coder
  94.5 %), the aquarium request answered in 9 tokens. Keeping the routing inside the arena's ~170 experts a layer
  costs code quality far more than its coverage suggests; with the repack inside every short request HumanEval ran at
  23 tok/s. The mask stays an opt-in experiment; the packs stay useful as the warm set an exact decode starts from.
- 12 GB, pinned staging ring (`CMF_QWEN_STAGE_MB=768`, no host tier): prompt 420 / 748 / 840 tok/s at 1K / 4K / 16K
  against the host tier's 325 / 587 / 656; generation at 16K 40 against 46.

## 14. cortiq.s7: the host tier on by default below a whole card, the repack through it, 4096-token chunks on 16 GB

The request's repack now takes 10–40 ms (was ~3 s): the freed slots are refilled through `fill_slot` in parallel,
a DMA from the host tier. "Warm": a first request, then 90 s for the tier's background fill (a long-lived server).

| card, arm | prompt 1K / 4K / 16K | generation at 16K |
|---|---|---|
| 24 GB, exact decode, the arena refilled by the request's pack (warm) | 896 / 1090 / 752 | 98 |
| 24 GB, masked decode with packs (warm; quality broken, §13) | 895 / 1034 / 775 | 147 |
| 16 GB, defaults | 561 / 821 / 546 | 68 |
| 16 GB, defaults, warm | 587 / 942 / 573 | 68 |
| 16 GB, staging ring instead of the tier, warm | 248 / 699 / 823 | 63 |
| 12 GB, defaults | 426 / 642 / 693 | — |
| 12 GB, defaults, warm | 452 / 683 / 689 | 46 |
| 12 GB, staging ring instead of the tier, warm | 607 / 864 / 860 | 37 |
| 12 GB, staging ring + packs, warm | 587 / 689 / 872 | 37 |

First token of the 1K request on 12–24 GB cards: 1.0–2.1 s (Strata 0.74–0.80 s).

## 15. fcq2 quality on the whole card, MTP draft length (cortiq.s7)

- HumanEval **134/164 = 81.7 %** (fcaq256 93.3 %, Strata Coder 94.5 %); the aquarium page the same broken one (§10).
  The 2-bit cut that fits a 32 GB card whole is not a Strata replacement on quality.
- Generation on the whole card by MTP draft length: k=3 (default) 146–169 tok/s, k=4 138–142, MTP off 101.5;
  k=2 ended the answers early. k=3 stays.

## 16. The published q4tp cut fcaq256 below a whole card (cortiq.s7, host tier on, not warmed)

| card | prompt 1K / 4K / 16K | generation 4K / 16K | HumanEval | aquarium |
|---|---|---|---|---|
| 32 GB (needs 41 GB whole) | 130 / 736 / 509 | 87 / 100 | **153/164 = 93.3 %** | working-page shape (importmap) |
| 24 GB | 602 / 719 / 274 | 67 / 68 | | |

Same HumanEval as on the 96 GB card: the streamed prompt and the exact decode keep the quality. The 1K first request
paid the tier's cold fill (admission 4.6 s of 7.0 s).

## 17. fcaq160: 160 q4tp experts a layer from fcaq256 (54.9 GB), to fit a 32 GB card whole

`moe-defrag out/fcaq256.cmf --stats out/fcaq256.profile --keep 160` (a cut of a cut: the source numbers compose).
Whole card, resident: prompt 746 / 1148 / 1203 / 1185 tok/s at 1K / 4K / 16K / 32K, generation 149–163, 30.4 GB —
but HumanEval **104/164 = 63.4 %** and the aquarium request ended after 2 tokens. 160 of 256 is below the quality
floor: rejected (deleted). The q4tp file with Strata's quality keeps all 256 experts and streams below 41 GB.

## 18. fcaq256 (the q4tp cut with Strata Coder's quality) by card size, cortiq.s7, warmed servers

| card | prompt 1K / 4K / 16K | generation 4K / 16K | Strata prompt 1K / 4K / 16K | Strata generation |
|---|---|---|---|---|
| 32 GB | 153 / 772 / 464 | 96 / 100 | 1254 / 3637 / 5430 | 269 / 270 |
| 24 GB | 650 / 768 / 335 | 65 / 72 | — / 4610 / 6144 | 223 / 234 |
| 16 GB | 187 / 389 / 407 | 49 / 48 | 1291 / 4461 / 6003 | 182 / 189 |
| 12 GB | 121 / 240 / 251 | 32 / 30 | 1187 / 4516 / 5986 | 166 / 174 |

Before tonight (cortiq.v1, eight-token frames below a whole card) fcq2 alone took 17–102 tok/s of prompt below 32 GB
and 13–14 tok/s of generation at 12 GB; fcaq256 could not stream at all.

## Summary of the night

- Engine (all exact, uncommitted): a loaded profile no longer kills lean frames (resident prompt ×1.7–2.4); the
  streamed prompt below a whole card; the host tier on by default there; the request's arena repack (10–40 ms);
  patent-1 expert packs (router 87.5 % on held-out texts) as the warm set; q8_2f prompt projections on the matrix units;
  counting sort past 4096 winners; the segment-count fix. Opt-in: the masked decode (breaks quality), 512-row frames
  (+13 %, OOM at 32K on a full card) — not defaults.
- Strata Coder on Strata 0.1.59 wins every speed column on this host at every card size; quality: fcaq256 matches it
  (HumanEval 93.3 vs 94.5, a working aquarium page on the first try); fcq2 (q2, whole on 32 GB) and fcaq160 do not.
- What separates us: our matrix-unit GEMMs run ~18 TFLOP/s in a prompt frame (Strata ~4× our work per token); experts
  are computed per 256-token frame (~10 rows an expert) instead of over the whole chunk; below a whole card Strata's
  CPU pool computes a token's misses, we copy them over PCIe.
- Results: `/Users/oleg/dev/fastcode-runs/pod2_final/` (no φ files), private HF
  `infosave/cmf-experiments/fastcode-tiers-2026-11-10/fastcode_tiers_2026-11-10.tgz` (58 975 141 B, API-checked).
