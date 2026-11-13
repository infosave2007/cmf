# Edge0 → CMF, 13.11.2026

Pod: RTX PRO 6000 Blackwell Server 96 GB, EPYC 9355 (128 threads), 188 GB cgroup, 400 GB disk; driver 595.91, CUDA 12.8
libraries. Work dir `/root/e0`; scripts in `pod/` here.

## Sources

- `Edge0/Edge0-145B-NVFP4` rev 6aab5f43, `Edge0/Edge0-120B-NVFP4` rev 291b1ee0 (the native 4-bit builds; the Q3_K builds are
  a requant of them). Skeleton base: `infosave/Qwen3.8-Flash-Next-Fast-Code-cmf` (its dense part is Qwen's bf16 in q8_2f).
- Edge0's dense part against `Qwen/Qwen3.8-Flash-Next` rev 812d96e1 (`pod/e0_cmp.py`): the router (bf16) is identical bit for
  bit, the hyper-connections too; attention and the shared expert are Q8_0 of Qwen's weights (max error 0.537 of a block
  scale); GDN tensors differ only by llama.cpp's head order, norms are stored as 1+w. Only the routed experts and D/U are
  Edge0's.

## Conversion (`cortiq edge0-import BASE DIR OUT --ple edge0`)

| | 145B | 120B |
|---|---:|---:|
| latent width | 1792 | 1280 |
| NVFP4 experts (bit for bit) | 47.56 GB | 33.97 GB |
| file | 110.23 GB, sha256 48365a63… | 95.64 GB, sha256 e411fbac… |
| n-gram table vs the base's (min cosine of sampled rows) | 0.9920 | 0.9920 |

Both on private HF (`infosave/Edge0-145B-cmf`, `infosave/Edge0-120B-cmf`), verified by size + LFS sha256.

## First runs (145B, `cortiq run`, greedy, thinking off)

- CPU: correct palindrome function (reference for the device paths).
- CUDA, MTP off / on (3 drafts): 70.2 / 153.4 tok/s on a 400-token Rust answer, 76 % of drafts accepted.
- Vulkan, MTP off / on: 64.6 / 109.5 tok/s, 77 % accepted (after the fix of the latent row stride in `lat_gu`).

## Server benchmarks (b5 binary, `e0_bench.sh`: speedtest 1K/4K/16K/32K two runs, HumanEval 164)

CUDA 145B: prompt 1437 / 2778 / 2641 / 2543 and 1808 / 2842 / 2730 / 2629 tok/s; generation 146 / 143 / 134 / 135 and
147 / 143 / 135 / 141 tok/s. HumanEval pass@1 156/164 = 95.12 % (159 tok/s end to end).

Vulkan 145B (b5, per-pick kernels in prompt frames): prompt 443 / 447 / 461 / 456 and 465 / 464 / 462 / 456 tok/s;
generation 105 / 107 / 109 / 118 and 112 / 116 / 109 / 118 tok/s.
CUDA 120B: prompt 1689 / 3316 / 3154 / 3027 and 2333 / 3516 / 3322 / 3114 tok/s; generation 172 / 162 / 159 / 138 and
177 / 166 / 155 / 138 tok/s. HumanEval 153/164 = 93.29 % (198 tok/s end to end).
Vulkan 120B: prompt 557 / 567 / 577 / 569 and 585 / 584 / 578 / 570 tok/s; generation 113 / 111 / 119 / 124 and
124 / 115 / 121 / 115 tok/s; card 45 795 MiB after the 32K test, server RSS 54.9 GB.

## b6 (in chain B)

- CUDA small rounds: `q4_lat_mv` reads D and U once for every row of the round and skips the groups no row picked
  (`mvb` re-read U's 14 336-wide rows once per row). Chunks: each sub-batch writes its picks' down rows
  (`q4_lat_put`), one `q4_lat_sum` in pick order after the last sub-batch (was a full scatter per sub-batch).
- Vulkan prompt frames: picks sorted by expert on the card (`lat_sort`), `lat_gu_g` / `lat_dn_g` decode an expert's
  32-row tile once per tile of 32 picks (`CMF_QWEN_LAT_GROUPED=0` keeps the per-pick kernels).

b6 results, both rejected:
- CUDA `q4_lat_mv` for U as one block row: 160 blocks on a 188-SM card; decode with drafts fell 153 → 107 tok/s.
  b7: grid y = the group of U's columns (1 280 blocks), partial sums `[G][n][h]` added in group order (`q4_lat_red`).
- Vulkan grouped prompt kernels with tiles of 32 picks: prompt reading 455 → 265 tok/s. A 256-token frame gives an
  expert about five picks, so most of a 32-pick tile was padding. b7: tiles of 8 picks, one row and one pick a thread.

b8 (CUDA transposed staging in `q4_lat_mv`; Vulkan stage probes `CMF_Q4_SKIP=lat_*`), 145B, one 4K request each:
CUDA prompt 2 893, generation 147.9 tok/s (b5 142.7, b7 107). Vulkan prompt / generation: all 445 / 99; no experts
(`lat_x`) 1 378 / 103; no D and U 489 / 71 (wrong output); no router 464 / 106; no shared expert 456 / 101; no MoE at
all 2 604 / 192. The prompt frames are bound by the routed experts' kernels.

Vulkan prompt kernels, 145B, one 4K request each: per-pick 445; shared-memory tiles of 32 picks 265, of 8 picks 380;
a row a thread with array sums 553; a row a thread with vec4 decode and sums (b10, default) 601 tok/s. Generation is
the per-pick kernels' in every variant (99–103 tok/s at 4K).

Final chain (b10): `pod/e0_chainD.sh`, numbers in `docs/hf/figures/edge0/figures_data.json` (`collect.py`).

## Second pass (same day): CUDA speed on the full card, 12–32 GB cards

Owner's request: compare with Edge0's own engine once, then make cortiq fast and make it run on 12 / 16 / 24 / 32 GB
cards (CUDA and Vulkan), and publish. Reference numbers of Edge0's engine on this stand (its `model.json` config, our
`speedtest.py`, two runs, 1K / 4K / 16K / 32K; it hangs on the first answer on a 96 GB card unless
`STRATA_VERIFY_ALL_RESIDENT=0`): 145B prompt 1651–2154 / 3280–3698 / 4177–4562 / 4667–4713 tok/s, generation
167 / 166 / 157 / 157–159; 120B prompt 1683–2339 / 3583–4185 / 4746–5314 / 5367–5474, generation 172–180 / 181–185 /
176 / 174–179. Not rerun afterwards (owner: no more comparison runs).

Where the time went (b11, nsys over one 3.6K prompt and 128 tokens; `CMF_QWEN_NCU=<round>` prints one decode round's
kernels): prompt = NVFP4 → f16 expansion 222 ms + cuBLAS batched GEMMs ~310 ms + GDN scan ~200 ms of 1.1 s;
decode round (4 rows, 23 ms) = D/U matvecs 64 µs a call × 96, per-pick gate/up 53 µs and down 42 µs × 48.

What changed, with the measurement that kept or rejected it (145B, RTX PRO 6000, CUDA):
- Fused prompt GEMMs from the NVFP4 slots (`q4_lat_ggu` / `q4_lat_gdn`): 4K prompt 2873 → 4610 tok/s, same text.
- GDN scan with a state column split over lanes (`q4_gdn_scan_p`): 16K prompt 4045 (column a thread) → 4261 (4 parts)
  → 4426 tok/s (8 parts, default).
- D/U matvecs side by side with 128-bit loads (`q4_lat_mv2`): 64 → 33 µs a call, decode 133 → 166 tok/s on a short
  prompt. Rows per block (`LAT_MV2_RPW` 1 / 2 / 4) made no difference; the old kernel with 32 or 64 rows a block
  was slower (76 µs).
- Rejected: `q4_lat_gu2` / `q4_lat_dn2` (several rows a warp, 16-byte loads): 63 µs against 53; picks grouped by
  expert on the card (`CMF_QWEN_LAT_UNITS=1`): gate/up 68.6 µs against 54.3, decode 157 against 166 tok/s;
  programmatic dependent launch: no change.
- Full card, final build (b18, `pod/e0_chainH`): 145B prompt 3590 / 5252 / 4758 / 4419, generation 175 / 167 / 163 /
  162, card 61.6 GB, HumanEval 155/164 = 94.51 %; 120B prompt 4208 / 5924 / 5264 / 4797, generation 188 / 178 / 174 /
  149, card 47.4 GB.

Smaller cards (emulated: `vramfill` holds the rest of the card, `CMF_GPU_VRAM_MB` = the size; the pod's CPU quota is
13.6 cores, so 12 threads compute the host's experts):
- Edge0's `profile.bin` ranks the experts poorly for our routing: its top 25 % carry 38 % of the picks of our
  calibration run. Our own profile (`CMF_QWEN_LAT_PROFILE_SAVE`, 40 code prompts of the calibration corpus and 24
  chat questions, `tools/profile_feed.py`): top 25 % carry 65 %, top 35 % 76 %, top 50 % 88 %. Shipped as
  `<model>.profile`.
- Host kernel: the first AVX-512 VNNI version allocated per row and ran ten times slower (13 tok/s at 32 GB);
  without the allocations a round's host share is 25.9 ms against 30.3 ms for the AVX2 kernel.
- Prompt chunk on a lean card, 145B at 24 GB: 1024 tokens 852 / 903 tok/s (4K / 16K), 2048 1270 / 1337, 4096
  2072 / 1845, for 80–90 against 82–88 tok/s of generation. Default for a latent model: 4096 from 20 GB, 2048 from
  14 GB, 1024 below.
- Final (b21, `pod/e0_chainJ`), prompt 4K / 16K and generation, tok/s: 145B 32 GB 2380 / 1873, 102–110; 24 GB
  1855 / 1838, 78–81; 16 GB 1099 / 1116, 50; 12 GB 643 / (16K: out of card memory, the sequence falls to the CPU),
  34–36 without the draft head. 120B 32 GB 3352 / 3135, 130–145; 24 GB 2823 / 2394, 100–118; 16 GB 1571 / 1571,
  71–73; 12 GB 904 / 928, 40–42 without the draft head. HumanEval of the 145B at 24 GB: 155/164 = 94.51 %.
- Vulkan (banks of the layers that do not fit in host-visible system memory, read over the bus): 145B at 24 GB,
  9.7 GB of banks on the card and 35.9 GB in system memory: generation 35–36 tok/s, prompt 25 tok/s. The rest of
  the Vulkan matrix was not run (pod cost).

Traps: `pgrep -f` / `pkill -f` with a pattern that occurs in the ssh command line kills that ssh (again); a background
`cargo build` on the pod skews anything the host computes; the scratch folder of the session was emptied between
turns (the upload script and license copies were rebuilt from the pod).
