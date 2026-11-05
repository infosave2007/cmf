# Qwen3.8-27B Turbo, a prompt on the Neural Engine, 05.11.2026

Goal: a faster first token on a Mac, where the GPU alone reads a prompt at about 48 tok/s.
Stand: MacBook Air M4, 24 GB, fanless. Prompt of 3 600 tokens (`m4k.txt`), greedy, the same text in every row below.

## Where it stands

| path | first token |
|---|---:|
| GPU alone, before | 87.0 s |
| GPU alone, now | 73.9 s |
| ANE takes the FFNs, a chunk at a time | 45.3 s |
| ANE takes the FFNs, two halves of a chunk | 32.2, 33.0 s |
| ANE takes the FFNs, a GDN mixer's `z` and an attention layer's `q` (default) | 27.5 to 28.3 s |

10 000 tokens with the rows on the SSD: 217.6 → 113.5 s, peak footprint 6.7 → 3.4 GB, the planted pass key comes back.
Against the GPU's numbers: `z` and `q` 4e-4 relative rms (fp16 holds q4tp whole), an FFN 1.4 % (int8 under a row's scale).

## What gave it

- Attention over prompt rows as three GEMMs per KV group (scores, causal softmax, P·V) on the chunk graph's kernels: the
  loop 41-48 → 9-12 ms a half. It serves the GPU path too.
- The a and b projections of a GDN mixer in one dispatch over rows: 12 → 5.8 ms.
- The ANE's pack, join and unpack in tiles of 32 × 8: unpack 5-6 → 1.5 ms.
- A GDN layer in two commands: the GPU computes qkv, a, b, the conv and the gates while the ANE computes `z`. Scratch
  per half, since one half's commands have the other's between them.
- The half that ran an FFN keeps the ANE for the next layer's short call, and lets it go if it has to wait for weights.
- The weight arena in 477 windows cut at tensors: the driver prepares a whole window at its first command, and one
  window over 13.6 GB held the first command 7.2 s.
- A read ahead of the file in 8 MB blocks from a thread: a cold file comes in at 3.3 GB/s, against 0.7 GB/s through
  page faults on one thread and 2 GB/s on eight. Cold start 35.3, 35.2 → 33.6, 32.9 s (measured with the FFNs alone).
- An attention layer's `q` on the ANE, `k` and `v` on the GPU ahead of it: the GPU's share of such a layer 39 → 23 ms a
  half, first token 29.1-29.4 → 27.5-28.0 s.

## Rates

GPU GEMM 1.2-1.3 G MAC/ms. ANE with weights as inputs: int8 3.2-3.5, fp16 2.0. An FFN of 256 rows 19-21 ms, of 512 rows
43 ms: the ANE's time is linear in rows, so larger calls gain nothing. A layer of 512 rows takes 56-57 ms; the GPU's
share is 44 ms and the ANE's 47 ms, so a perfect schedule would give 48.

## What lost

- A third set of FFN weights, so that the next layer is staged before the second half lets one go: 31.3 s, and 28.6 s
  with a third set of mixer weights too, against 27.7 with two. It also deadlocked until a waiting half let the ANE go.
- Having the driver prepare the windows behind the read-ahead: 28.0 against 27.5 s. A window of mapped pages is prepared
  in milliseconds.
- Every mixer projection on the ANE (`CMF_ANE_MIX=all`): the ANE becomes the busier device.
- `z` as int8: 28.2 against 28.7 s, at 1.4 % instead of 4e-4. Left as a switch.
- f32 tiles for the attention GEMM: no faster than fp16.

## Memory

During a prompt the file cache holds 12.6 GB of the model, the process 2.7 GB of its own, and other programs are
squeezed into the compressor (2.1 → 4.7 GB). At the first token the driver wires the FFN windows as well, 16 GB wired in
all, as on the GPU path from the start.

## Measurement traps

- A fanless Air throttles: the same build gave 28.9 s cool and 34.7 s hot. A four-minute compile heats it for minutes.
  `q27_cool.sh` pauses before every run; a pause of 70 s after a compile is not enough.
- The file cache decides the first chunk: 8.9 s cold against 4.0 s warm. `filecache.c` drops a file from the cache
  (`msync(MS_INVALIDATE)`) or reads it in, so every run starts in a known state.
- A run during a compile: 60.8 s.
- Swap moved by other programs: one run of a pair at 37.4 s against 28.2 s.

## Tools

`CMF_ANE_TRACE=1` prints every wait, ANE call and GPU command on one clock; `q27_ane_gaps.py` reads it. `CMF_VERIFY_CUT=1`
runs each stage as its own command. `CMF_ANE_CHECK=1` computes what the ANE did on the GPU as well and prints the distance.

## Not done

- The next layer's `z` inside the FFN's call, with the norm in the program: one call a layer instead of two would bring
  a layer near 48 ms. The norm in fp16 needs a scale against overflow.
- The CPU's matrix unit as a third engine.
- Decode is the GPU's, 6 tok/s.
- No Mac with a fan and no other chip: every number is one M4.
