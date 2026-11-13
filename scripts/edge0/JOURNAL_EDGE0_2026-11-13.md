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
