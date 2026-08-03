# Batched prefill for DeepSeek-V4

## Why

A 7.4 KB prompt is ~2500 tokens. Today prefill runs the decode path once per
token, so it costs `2500 × per-token`. At the current 36.8 ms that is 92
seconds before the first word — and the user sees that number, not the
27.2 tok/s that follows it.

Batching does not help because it amortises DISPATCHES. It was measured:
a dispatch costs 2.67 µs on this card and there are ~1400 a token, so
launches are 3.7 ms of 36.8 — a tenth. Batching helps because it amortises
**weight reads**:

| what a token reads | bytes |
|---|---|
| experts (8 routed + shared, 43 layers) | ~2.9 GB |
| projections (wq_a/b, wkv, wo_a/b, compressors, indexer) | ~2.4 GB |

At 1.3 TB/s that floor is ~4 ms, and the chain measures 28 — nine times off,
which is what a matvec looks like when it streams a weight to use it once.
A chunk of C tokens reads each weight ONCE and uses it C times. For C = 64
the prompt's weight traffic drops from ~13 TB to ~200 GB.

## What is sequential and what is not

Within one layer, given that layer's KV cache, the tokens are independent.
The cache itself depends only on the layer's inputs, which are all known
before the layer runs. So per layer:

1. **Batched.** The projections that read the layer input: `wq_a`, `wkv`,
   both compressors, the indexer's two. One GEMM each instead of C matvecs.
2. **Sequential.** The window append, the compressor's state machine (it
   folds every `ratio` tokens), the indexer's top-k, and attention itself —
   token t attends to the cache as of token t.
3. **Batched.** The output projection, the router, the experts, and the
   whole hyper-connection half: all per-token functions of the attention
   output, with no cross-token dependency.

Only (2) stays per token, and (2) is the cheap part: attention measures
1.8 ms of the 28 ms chain.

## Stages

Each stage is separately measurable and separately revertible. The gate is
the same one decode uses: five stands, the host comparison, and
`CMF_DSV4_SLOT_CHECK`.

- **S1 — the chunk walk.** `forward_chunk` over C tokens with everything
  still per token. No win; it is the scaffolding the rest hangs on, and it
  proves the chunked state bookkeeping (positions, caches, hc state) before
  any kernel changes.
- **S2 — batched projections.** The seven q4tp projections of (1) and (3)
  through `q4tp_matmat`, which exists and has a parity test. Expected: the
  projection half of the weight traffic, ~45%.
- **S3 — batched experts.** The largest single reader. Tokens are grouped by
  chosen expert and each expert's weights are read once per chunk. Needs a
  new kernel: the current one takes one token's activations.
- **S4 — batched attention.** A grid over (token, head) with each token's own
  index list. Smallest of the four by measurement, last by priority.

## What this does not change

The decode path. `forward_token` stays exactly as it is; the chunk path is
entered only for prompt tokens and only when every layer of the run is
eligible, with the same all-or-nothing preflight the chain uses — a path
that refuses half way after advancing a cache is not slower, it is wrong.
