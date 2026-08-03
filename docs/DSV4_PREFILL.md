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

## The second caller: speculative verify

The draft (`docs`-less for now; see `dsv4::dspark_draft`) proposes five
tokens per trunk step, and the trunk has to check them. Checking them one at
a time costs five trunk passes — 5 x 36.8 ms against 5.36 tokens gained,
which is SLOWER than not speculating at all. So the batched pass is not an
optimisation for speculation; it is the whole of it.

That makes the two callers one piece of work:

| | prefill | verify |
|---|---|---|
| batch | 64+ prompt tokens | 5 drafted tokens |
| causality inside the batch | yes | yes |
| state committed | all of it | only the accepted prefix |
| wrong answer costs | a bad prompt | a wrong token stream |

Only the last row differs, and it differs only in the rollback: verify has
to write B positions speculatively and keep k of them. The transaction is at
most five positions deep — save the logical lengths and the ring slots it
will overwrite, journal any compressed entry the block creates, and on
rejection restore the tail. A full clone of the caches is not needed and
would not be affordable.

## What the batch has to beat

Per token, of the 28.1 ms chain: projections 9.25 (q-proj 1.37, compressors
2.90, indexer 1.36, wo_b 1.24, o_lora 1.40, next-q 0.98), MoE 4.96, the
hyper-connection glue 4.14, attention 1.80, and ~8 unattributed.

Batching only the projections leaves the other 19 ms paying five times over,
which lands around 38 tok/s — under the target, after all this. The glue and
the MoE have to come too. That is the order the work is in.

## What this does not change

The decode path. `forward_token` stays exactly as it is; the chunk path is
entered only for prompt tokens and only when every layer of the run is
eligible, with the same all-or-nothing preflight the chain uses — a path
that refuses half way after advancing a cache is not slower, it is wrong.
