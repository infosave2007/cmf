# Fast Code (Qwen3.8-Flash-Next for code) — journal, 2026-11-09

Pod: RunPod RTX PRO 6000 Blackwell Server 96 GB, EPYC 9355, 13.6-core CPU quota, 188 GB cgroup, 300 GB local disk.
Run dir `/root/fc`; scripts in `scripts/fastcode/pod/`.

## Strata (what we compare against)

- Strata (github Niko1221/Strata, MIT) is an inference ENGINE (llama.cpp/ggml-derived, CUDA/HIP/SYCL/Metal) for
  Qwen3.8-Flash-Next on consumer cards; experts in RAM + hot experts in VRAM, MTP / prompt-lookup / DFlash2 drafting.
- "Strata Coder" = ISTA-DASLab `Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF`, IQ1_M file type: 256 of 512 routed experts
  per layer (RCO selection that minimises KL on code + agent + vision calibration), per-tensor mixed GSQ quants
  (expert banks IQ2_S / IQ3_XXS / IQ3_S / IQ4_XS gate-up, IQ4_NL / Q2_0 down); shard 1 29.6 GB (resident), shard 2
  28.8 GB = the PLE n-gram table at IQ4_NL. No MTP in the GGUF; Strata setup downloads the original bf16 MTP layer
  (5.2 GB) and packs it.
- Published accuracy (ISTA card, xhigh reasoning): SWE-bench Verified 79.20 (BF16 82.80), LiveCodeBench v6 87.43
  (= BF16). Published speed (Strata DETAILS, engine 0.1.26, RTX 5070 12 GB + 64 GB DDR5, MTP on, 256 new tokens):
  decode 58.9/55.1/54.9/53.2/43.0 tok/s at 1K/4K/32K/64K/128K, prefill 656/1583/2177/2236/2208 tok/s.
- Honest comparison here: the same pod, the same client, the same prompts — Strata engine 0.1.58 (release v0.2.12)
  with its own setup (`--native`, expert cache auto, `--prefill auto`, `--spec 4`, int8 KV, PLE in RAM), and the
  Strata Coder weights imported into CMF (`import-gguf --template`) on our engine.

## Engine facts found on the way

- qwen4 device path: a prompt is fed in frames of TMAX = 8 tokens; by default one fenced submission per layer
  (49 syncs a token) because a cold expert must be completed on the host between layers. ppl/profile of the full
  q4tp runs ~62 tok/s on this card.
- `CMF_QWEN_PROFILE_SAVE` records per-(layer, expert) routing counts on the device path (the code profile).

## Engine optimization, round 2 (f13–f17), RTX PRO 6000, fc256 (K = 256 code experts, q4tp), 4000-token prompt

| build | change | prompt tok/s 1K / 4K / 16K | decode tok/s plain / MTP k=3 (4K) |
|---|---|---|---|
| f12 | 128-token prompt frames | — / 389 / — | 68.7 / — |
| f13 | 256-token frames (TROWS), sorted experts, tiles opt-in | 595 / 551 / 529 | 68.3 / 122.5 |
| f14 | 16-row gate/up kernel (opt-in, `CMF_QWEN_GU16=1`): x read 4× less, no gain | 593 / 546 / 525 | 68.4 / 122.0 |
| f15 | top-k: keep-all path when k ≥ n, 8-bit digit radix otherwise (exact) | — / 551 / 532 | 67.5 / 123.8 |

- Greedy text identical across f11–f15 (and 256- vs 128-token frames).
- MTP depth at 4K: k=2 107.8 (81 % accepted), k=3 122.5 (80 %), k=4 109.6 (70 %), k=5 101.4 (59 %) → k=3 stays.
- Radix top-k vs the ranking kernel: 16K decode 48.8 → 67.9 tok/s (f13), 4K equal.
- Stage cost at 4K (f14, `CMF_Q4_SKIP`), prompt ms/token: experts 0.74, QSA 0.39, GDN 0.23, HC 0.19, PLE 0.10,
  rest 0.18 (of 1.82). Decode ms/token: GDN 3.8, QSA 3.4, HC 2.7, experts 2.6, PLE 0.55 (of 14.8).
- GPU-timestamp decode profile (per-layer frames, 9K prompt), ms/token: qsa_attn 2.0, gdn_proj 1.6, gu 1.3,
  moe_hc 1.2, attn_hc 1.1, dn 1.1, gdn_out 0.75, gdn_rec 0.73, route 0.73, router 0.62, head 0.48, qsa_proj 0.40.
- Bytes a decode token reads (fc256): HC f16 mixes 1.26 GB, GDN projections 2.1 GB, routed experts 1.23 GB,
  QSA 0.6 GB, head 0.64 GB, shared experts 0.12 GB → ~6 GB, 3.4 ms at 1.79 TB/s; we run 14.8 ms.
- Per-step cost probe (`CMF_Q4_SKIP_STEP`, f16): every decode dispatch, even a one-workgroup one, costs about
  10–25 µs of token time; the HC v3 pairs (steps 103/104, 203/204) ≈ 3.3 ms a token together, GDN qkv 0.85,
  z 0.73, out 0.91, conv 0.45, step 0.34 ms.

## Round 3 (f17–f20): matrix units for prompt experts, attention, decode kernels

| build | change | prompt tok/s 1K / 4K / 16K | decode plain / MTP (4K) | MTP 16K |
|---|---|---|---|---|
| f17 | experts of prompt frames on the matrix units (`q4g_mm_gu/dn`, coop 16×16×16 f16→f32, q4tp dequant into the tile); QSA identity list ≤ 2K; coalesced split attention (`q4t_attend_sp2`) | 754 / 766 / 770 | 68.6 / — | — |
| f18 | decode experts read nibbles as vec4 (`q4_gu_q4tp4v`, `q4_dn_q4tp4sv`, exact) | 756 / 768 / 773 | 69.3 / 125.6 | 130.5 |
| f19 | coop experts with register prefetch (exact, no gain), HC one-pass down on 2–4 token frames | 746 / 738–759 / 753–764 | 63.7–? / 124.9 | 130.5 |

- Greedy text identical f17 ≡ f18 ≡ f19 (matrix-unit experts vs scalar: identical 64-token text; top-5 logits after a
  9K prompt the same tokens, values within 0.7 % — the f16 rounding of activations).
- Prompt cost at 4K after f17: experts 0.46 ms/token (was 0.74); without experts the prompt runs 1185 tok/s.
- Frame size (f15): 64 → 429, 128 → 504, 256 → 553 tok/s at 4K.
- Bench noise between runs is 2–3 %: decisions on small changes come from the in-process kernel bench below.

Kernel bench on the card (`decode_kernel_bench`, `--ignored`, best of 4, weight sets past the 128 MB L2):

| kernel (one token) | µs | GB/s |
|---|---|---|
| one-workgroup dispatch in a dependent chain | 2.0 | — |
| HC v3 pair (down 11.5 + up-fold 4.8 + barrier) | 20.7 | 636 |
| HC stream-split pair (`hc3_down4` + `hc3_upfold4`) | 18.8 | 704 |
| HC pair at 4 tokens (verify window), one-pass down | 40.2 → 36.0 | — |
| router f16 257 × 2560 | 6.5 | 200 |
| split attention, 2051 keys: old / coalesced | 64.7 / 19.9 | — |
| q8_2f [10240, 2560] / [12288, 2560] | 22.2 / 26.4 | 1255 / 1263 |
| q8_2f [2560, 6144]: 4 rows a WG / one row a WG | 10.7 / 8.9 | (partly L2) |
| decode experts gate/up, 11 slots (vec4 loads) | 23.8 | 789 |
| decode experts down, 11 slots | 15.3 | 620 |

- Decode token at 4K: 13.2 ms = host encode 2.1 ms (wgpu finish 1.6 ms) + card 10.7 ms, one after the other
  (`CMF_QWEN_PROF=1`). The card time matches the kernel sum (~9.9 ms).
- Trap: the kernel bench's bind cache keys on (layer, step, token slot); two shapes on the same key measured the first
  shape twice (the first "41 % out-projection" reading was that).

## Round 4 (f20–f22): host encode overlap, matrix-unit pair GEMMs, kernel balance

| build | change | prompt 1K / 4K / 16K | decode plain 4K | MTP 4K / 16K |
|---|---|---|---|---|
| f20 | a decode/verify token in 3 chains, each encoded while the card runs the previous (`CMF_QWEN_CHAINS`); HC stream-split down default; long-row q8_2f matvec | 768 / 770 / 778 | 78.3 | 149.1 / 140.8 |
| f21 | prompt f16 pair GEMMs (HC mixes, router, GDN gates) on the matrix units (`q4t_f16_pair_coop`); 128-lane decode down; 4 chains | 830 / 839 / 844 | 81.1 | 146.3 / 148.7 |
| f22 | 64-column K slices in the expert GEMM (bit-exact), pair GEMM vec4 loads + prefetch (bit-exact); dead variants removed | 888 / 893 / 896 | 81.5 | 145.4 / 149.5 |

- Chains at 4K, plain decode: 1 → 72.6, 2 → 78.3, 3 → 78.3, 4 → 79.7 tok/s; MTP 1 chain 140.6 vs 3 chains 149.1.
  The token: 13.2 ms (encode 2.1 serial) → 11.6 ms.
- Kernel bench: 128-lane decode down 15.3 → 12.0 µs; 96-lane gate/up 23.86 → 23.84 (no gain, removed);
  long-row q8_2f [2560, 6144] 10.7 → 8.9 µs (outputs identical).
- f21 toplog after the 9K prompt, pair GEMM coop vs scalar: same top-5 tokens, logits within 0.15 (f16 inputs);
  greedy text identical. Card unit test: coop vs scalar pair rel 3.2e-3, gate rows 2e-4.
- Stage costs at 4K (f20 sweep), prompt ms/token: experts 0.45, HC 0.30, GDN 0.28, QSA 0.14, PLE 0.10;
  decode ms/token: GDN 3.2, QSA 2.4, experts 2.3, HC 2.3.
- Greedy text identical f17 → f22 (`ef0901ea`), 0 panics; all 22 qwen4 card tests pass on f22.

## Final numbers (f22, server, the client that measured Strata)

- `speedtest.py` puts a random session tag in front of every prompt, so no prefix cache (ours or llama.cpp's) reuses
  anything but the chat template's first tokens; one short untimed warm-up request. The neighbour branch found the
  pod's GPU starting at 330 MHz and reaching 1890 MHz only after ~1 s: the 1K rows (TTFT ~0.5–1.3 s) of BOTH engines
  are likely understated; 4K and longer are not much affected.
- Fast Code (f22): prompt 688 / 855 / 873 / 860 tok/s at 1K / 4K / 16K / 32K (921 / 3613 / 13238 / 25965 tokens);
  generation 188 / 167 / 157 / 150 tok/s; HumanEval 156/164 = 95.12 %; peak VRAM 40.9 GB.
- Strata (13:05, same client): prompt 1916 / 5492 / 7229 / 7659; generation 224 / 220 / 223 / 222; HumanEval 94.51 %;
  31.1 GB.
- Fresh-code ppl (8192 tokens of code written after 20.10.2026): fc256 3.416, Strata Coder weights on cortiq 3.436.
  Rust ppl (1536): 4.519 (unchanged by the engine work).
- HumanEval fell from 158 to 156 of 164 against the f7-era binary; an A/B with the prompt frames on the scalar f32 kernels
  (`he_ab.sh`) checks whether the f16 matrix-unit prompt costs it.
- Aquarium, first turn: greedy and three T = 0.7 pages all fail on the same import (OrbitControls from unpkg without an
  importmap: `Failed to resolve module specifier "three"`); thinking ran out of 30 000 tokens. A second turn with the
  console error (`aqfix.sh`, both models) is the developer's loop.
- Full 512-expert q4tp file (97.1 GB + 1.5 GB MTP) on f22, same server client: prompt 171 / 208 / 206 / 210 tok/s,
  generation 127 / 118 / 138 / 116 tok/s, HumanEval 155/164 = 94.51 %, fresh-code ppl 3.366, Rust ppl 4.111,
  peak VRAM 59.6 GB. bench: prompt ~79 tok/s (the experts stream; the server path reads 2.6× faster), decode plain
  77.5, MTP 129 at 16K; MTP acceptance 84 % at 1K where fc256 has 54 % (its head was trained on the full trunk).
- Fast Code against the full file on the same card: prompt 4.0–4.2×, generation +14…48 %, VRAM 41 vs 60 GB,
  HumanEval 95.1 vs 94.5 %, fresh-code ppl 3.416 vs 3.366.
- HF: private repo infosave/Qwen3.8-Flash-Next-Fast-Code-cmf (`up_final.sh`, API-checked): model as
  `qwen38-flash-next-fast-code.cmf`, `fast-code.profile`, sha256, README (docs/hf/Qwen3.8-Flash-Next-Fast-Code-cmf.md),
  assets/fc_{speed,quality,size}_{en,ru,zh}.png. Public only after the owner's yes (the card names cortiq 0.8.65).
- Aquarium (browser check): the full 512-expert file writes a WORKING page on the first greedy try (importmap for
  `three` and `three/addons/…`, glass tank, three clownfish, plants, stones, no console errors). Strata's first page
  fails (`c3 is not defined`, `uTime` undeclared in a shader); after one repair turn with its console errors it renders.
  fc256: 4 first tries (greedy, T 0.7 × 3) and 4 repair turns, never an importmap — OrbitControls from unpkg imports the
  bare `three` and the page stays blank. The code-corpus cut dropped what the full model uses for web code.
- HumanEval A/B on f22: prompt frames on the scalar f32 kernels 158/164 = 96.34 % against 156/164 with the f16
  matrix-unit prompt paths (the same binary).
- Next (owner, 09.11 17:4x): the expert choice from the full model's routing on the TASK (the aquarium prompt and
  answers) plus code, and more experts at heavier quantization like the hot file (q2tp-v2 + 8-bit tubes) at the same
  size — `v2.sh`: merged profile `hot/code_aq.profile`, hot file cut to 320 experts (`out/fch320.cmf`).
- Task routing (full q4tp file, 55 405 tokens: the aquarium prompt with six answers of the full model, Fast Code and
  Strata; ppl 6.652, 824 s): 35.2 % of the task's routing mass (max 45.8 % in a layer) lands on experts fc256 dropped.
  Merged profile (code 108 480 + task 58 336 tokens, each normalized, equal weight) top-320 covers 88.9 % of the task's
  mass (min 81.7 %) and 90.6 % of code's (fc256's top-256: 87.1 %).
- Hot cut `out/fch320.cmf` (`moe-defrag hotsrc/qwen38-flash-next-q2tp-hot.cmf --stats hot/code_aq.profile --keep 320
  --merge …hot.mtp.cmf`): 66 846 962 792 bytes (fc256: 66 755 957 864), 320 q2tp-v2 experts a layer with the hot
  file's 8-bit tubes of the kept experts. Rust ppl 4.460 (fc256 4.519), fresh-code ppl 3.226 (fc256 3.416, full q4tp
  3.366). Caveat: both texts are this repository's code and the hot tubes were baked on a probe of 16 KB of code
  whose source is not recorded here, so the decision rests on HumanEval and the aquarium.
- fch320 on f23 (q2tp gate/up on the matrix units, `q4g_mm_gu64q2`; logits after a 9K prompt within 0.09 of the
  per-token q2tp kernels, top-2 a near tie): bench prompt 772 / 789 / 797 at 1K / 4K / 16K, decode plain ~59, MTP
  107–116; server prompt 496 / 628 / 672 / 732, generation 112–116; **HumanEval 152/164 = 92.68 %** (fc256 95.12,
  full 94.51); greedy aquarium again without an importmap. Verdict: at this size the 2-bit experts lose exact-code
  accuracy that perplexity does not show; hot-style is not the Fast Code base.
- HumanEval of the f16 matrix-unit prompt paths: all coop 156, experts coop + scalar pair 157, all scalar 158 (of 164).
- Next: `fcaq.sh` — q4tp, 256 experts by the code + task profile (fc256's size and precision, task-aware choice).
- hi + lo pair GEMM (f32-close HC/router inputs): HumanEval 156/164, the same 36 025 tokens as the plain coop pair →
  removed. The 156/157/158 spread of the prompt paths is greedy-decoding noise.
- **fcaq256** (q4tp, 256 experts by code + task at weight 1; 66 755 974 248 bytes): Rust ppl 4.900, fresh-code 3.337;
  server prompt 716 / 760 / 796 / 838, generation 157 / 147 / 127 at 4K / 16K / 32K; HumanEval 153/164 = 93.29 %;
  **greedy aquarium page WORKS on the first try** (importmap, tank, three clownfish, plants, stones; browser check).
  The task-aware choice restores what the code-only cut lost, at a cost in code (HumanEval −1.8, Rust ppl +0.38).
- Queued: `fcrco.sh` (our q4tp weights, Strata Coder's RCO expert set) and `fcw.sh` (task weight 0.35).
- fcrco256 (our q4tp weights, Strata Coder's RCO set): fresh-code ppl 3.406, HumanEval 154/164 = 93.90 %, greedy
  aquarium answer without an html block; server generation 173–190 tok/s.
- fcw035 (task weight 0.35): Rust ppl 4.605, fresh-code 3.304; HumanEval 153/164 = 93.29 %; greedy aquarium without an
  importmap. fcaq256's sampled pages (T 0.7, seeds 1–3): importmap in 3 of 3 (seed 3's page 517 bytes, cut short).
- Summary of the cuts at 66.8 GB (HumanEval of 164; aquarium = the importmap/working page): fc256 156 / no;
  fcw035 153 / no; fcaq256 153 / **yes, 4 of 4**; fcrco256 154 / no; fch320 (q2 hot) 152 / no; full q4tp 155 / yes;
  Strata 155 / after a repair turn. HumanEval spreads within noise (σ ≈ 3 tasks); the task decides: fcaq256.
- PR #102 (hybrid-fixes 5fd6a6dc) merged in: worktree `/Users/oleg/dev/cmf-fastcode2`, branch `fast-code-v2` from
  origin/hybrid-fixes, the fast-code diff applied 3-way without conflicts, 22 qwen4 tests pass. qwen4 prompt frames
  take the staged GDN scan (`gdn_prep_k` + `gdn_scan3_k`, `CMF_QWEN_SCAN3=0` off). PR #102 alone still indexes
  `weights.layers` in `graph_gdn_layer_count`: `cortiq run/serve` on qwen4 files panic without this branch's guard.
- f24 (PR #102 base): with `gdn_scan_k` the greedy text equals f22's (`ef0901ea`); prompt 897 / 910 / 886 tok/s at
  1K / 4K / 16K, MTP 153.4 at 4K (89 %). The staged scan read 4K at 893 and moved the top logits by ~0.2 (text
  changes): off by default, `CMF_QWEN_SCAN3=1` turns it on.
- Hot method on fcaq256 (owner: «метод которым мы выбирали экспертов в …-hot-cmf и 5 % q8_2»): `hot-probe --experts`
  on fcaq256 over 9 KB of code + 9 KB of the aquarium task (4096 tokens, CPU), then `hot-bake --experts
  --whole-experts 28 --shared-q8 --bf16-down` from the bf16 checkpoint. **Bug caught before the bake ran:** hot-bake
  read checkpoint expert `e` into slot `e`, but a moe-defrag cut numbers its experts afresh, so every lifted expert of
  a cut would have been another expert's weights. Fix: moe-defrag records `provenance.moe_defrag.kept_ids` (the
  checkpoint expert behind every slot; a cut of a cut composes), hot-bake reads the cut's slots by those numbers, and
  `--defrag-stats PROFILE` recomputes them for cuts written before (checked against `stats_hash64`). fch320 was not
  affected: it was cut from the already baked hot file (tubes renumbered with their experts).
- **fcaqhot** (fcaq256 + 28 busiest experts a layer and the shared expert whole in q8_2f from bf16, source numbers
  fixed; 73 658 709 096 bytes, +10.3 %, the 28 hold 48.7 % of the layers' Σ(w·act)²): Rust ppl 4.826 (fcaq256 4.900),
  fresh-code 3.297 (3.337); server prompt 491 / 628 / 650 / 668, generation 120 / 113 / 111 / 112 (the tube path);
  **HumanEval 148/164 = 90.24 %** (fcaq256 153); greedy aquarium without an importmap; peak VRAM 47.8 GB. Rejected:
  as with fch320, the 8-bit lift lowers perplexity and loses exact-code accuracy and the task.
- **Release cut: fcaq256.** cortiq f26 (this branch on PR #102), server, two runs with fresh session tags: prompt
  709 / 811 / 831 / 840 and 888 / 842 / 856 / 845 at 1K / 4K / 16K / 32K; generation 151 / 161 / 154 and 172 / 172 / 118
  at 4K / 16K / 32K (the 1K answer is 1–17 tokens, no rate); peak VRAM 40.9 GB. Routing cover of its 256 experts:
  code 83.2 % (min 69.8 %), task 81.5 % (min 71.2 %); fc256: 87.1 % / 64.8 %.
- Private HF `infosave/Qwen3.8-Flash-Next-Fast-Code-cmf` now holds fcaq256 as `qwen38-flash-next-fast-code.cmf`
  (66 755 974 248 B, sha256 adf47c92b9c9d10e…, API-checked), `fast-code.profile` (fcaq256.profile), the checksum, the
  rewritten card (aquarium screenshot first, speed and accuracy table vs Strata Coder and the full file) and the
  figures; still private. Pod data copied to `fastcode-runs/pod_final/`; the pod stopped 20:06 UTC (owner's order).
