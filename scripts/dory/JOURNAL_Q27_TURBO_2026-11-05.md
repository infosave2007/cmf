# Qwen3.8-27B Turbo, 05.11.2026

Goal: what Dory's turbo gave, on the widely used 27B, and above all a fixed device memory for small cards.
Stand: RTX 5090, 32 GB, Vulkan, EPYC host, a disk with 4.5 GB/s write and 5.2 GB/s read.

## Starting point

- Exact, drafts on: prefill 470/463/382/301 tok/s and decode 92/98/84/68 at 1000/4000/16000/32000. Memory 18.3 GB at
  4000, 23.8 GB at 32000. A far field looked needless on speed; the point is the memory of a small card.
- A far field alone on 100 000 characters: facts 0 of 6, exact 4 of 4.

## Far field on the batch graph

- `CMF_O1_GPU=1` missing: per-op, 2.4 tok/s.
- `bench` did an extra host pass over the prompt; the bounded route fixes it.
- The exact prefix past about 31 000 rows declined the batch graph (score planes of k×context, 131 GB at 32 000 on
  Dory's plane). The first seal is capped at 8192 rows.
- A panic in `o1-qcap-stage`: 9.4 GB over the 4 GB limit. Queries are drained per chunk, one row in 8 kept.
- The trap of the day: garbage text, facts 0 of 8. A typo in `o1s_sum`, `sb + i*hpg + h` for `(sb + i)*hpg + h`.

## A ceiling, RAM and SSD

Rows past the first seal go through host-visible staging to an archive, in RAM or in files. The far field is rebuilt one
layer at a time, so memory never holds two layers' rows.

## Exact rows

The rank features of Dory work on the 27B: fact rows in the top 0.22 % on average, 0.96 % at worst, leave-one-out, over 128
ranks (16 layers × 4 KV heads × 2). Facts: 8 of 8 with the choice before the prompt's last 512 rows.

## Short threads, one row and a chunk

A workgroup stays on one SM. One row: 31.5 → 56 tok/s. A chunk of rows, `o1c_*`, replaces `o1b_attend`:

| | before | after |
|---|---:|---:|
| verify of five rows | 57 ms | 41 ms |
| bench decode, 5000 rows | 54.5 | 80.2 |
| sealed prompt at 64 000 | about 230 tok/s | 360 tok/s |

`o1_batch_matches_serial_steps`: max|Δ| 4.8e-6 against 3.7e-6 for the old kernel. Greedy text between the two kernels can
part ways at a near tie: the first difference came at a token with two close choices, and both answers are correct.

## Drafts over a far field

- A verify was refused outright, "o1 batch does not support speculative tail".
- Then the trunk rewind failed: a sealed layer has no mirror.
- Then it worked: 52 % accepted in bench, text identical over 60 772 characters, 56.9 → 72 tok/s before the chunk
  stages, 92 after.
- Real text accepts about half of the drafts, and `bench --core` far more at times: 98 % in one run. Real answers are the
  number to quote: 50.9 → 60.4 tok/s at 100 000 characters.

## Where 4 GB of the card went

`nvidia-smi` over a run: 16.9 GB right after the weights, 12.6 GB of them. A build that logged every buffer showed no 4 GB
buffer, and `wgpu_core=trace` gave labels without sizes. `MemoryHints::MemoryUsage` grows blocks to 64 MiB and a tensor of
33 to 64 MiB takes a whole one. `Manual` with blocks of 16 MiB: 17.6 → 14.1 GB at ctx 1000, 60 tok/s both ways. Allocations
stay below VRAM / 16 MiB, far under the 4096 some drivers allow.

## Seal against the ceiling

| seal | memory at 32 000 | prefill | decode |
|---:|---:|---:|---:|
| 8192 | 15.7 GB | 348 | 91 |
| 4096 | 15.0 GB | 368 | 92 |

With chunk stages a sealed prompt reads about as fast as an exact one past 16 000. 8192 stays the default where it fits,
because a prompt under it stays exact and draws drafts at 104 tok/s. `o1_seal_for_card` picks 4096 for a 16 GB budget.

## Measurement traps

- A compile on the Mac during a measure gave false numbers (26.7 against 25.8; clean 32.2 against 25.5).
- A new binary with `CMF_NLL_SKIP` against an old one without it compares different rows.
- `pgrep -f` inside `bash -c` matches itself.
- A single argument on Linux stops at 128 KB: `run --prompt-file`.
- `run` has no `--ignore-eos`: the first real-answer pass printed nothing.
- `CMF_O1=...` does nothing for `ppl`, which takes `--o1` and prints "exact attention" otherwise.

## Metal

`o1_attend` on Metal stopped at 196 keys. `gpu_metal/o1c.rs` ports the chunk kernels with a near window up to 2052, and the prompt
path takes the rows graph, with queries and archive rows copied out of its scratch. It ran at the first try: 5000 rows, 31.7 tok/s,
decode 4.8. At 16 000: 29.3 tok/s, decode 4.0, footprint 6.3 GB. A planted fact in 40 000 characters came back whole.

A plain `bench --ctx` on Metal does an extra host pass at 5.6 tok/s: 16 000 rows would take 48 minutes, and its RSS of 13 GB comes from
that pass. Compare with `run`.

## Not done

- The CPU runs the file as the standard one.
- No exact run at 16 000 rows on Metal to set against the far field.
- The turbo file itself was not read on a Mac for lack of disk: the far field ran over the plain file by environment.
- Dory's rows still sit on the card while a long prompt is read.
- No real 16 GB card: the budget was set on the 5090.
