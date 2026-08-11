# More than one GPU

A `.cmf` file does not need to be told about your cards. One flag picks
between the two things people actually mean by "use both GPUs", and they
solve different problems:

```sh
cortiq serve model.cmf --gpus 2   # throughput — one full replica per card
cortiq run   model.cmf --gpus 2   # capacity  — the layer stack split across cards
cortiq bench model.cmf --gpus 2   # the same split, timed honestly
```

`cortiq gpu` lists the cards and their indices. `CMF_GPU_ADAPTER=<index
or name substring>` pins one card for a process that should only see
one.

## Which mode you want

**Replicas — `serve --gpus N`.** A whole copy of the model on each card,
N requests decoding at once. This is the mode that scales: nothing
crosses between cards, so N cards do N times the work as long as
requests arrive to fill them. Use it whenever the model fits on one
card, which is almost always the case for a quantized `.cmf`.

**Layer split — `run/bench --gpus N`.** Segment *i* pinned to card *i*
inside ONE process; only a hidden vector crosses each boundary and it
never leaves the address space. This exists for models bigger than one
card. On a model that already fits, it costs a few percent rather than
saving any — the handoff is real work and there was nothing to relieve.
`--peer-split N` moves the boundary off the halfway point.

The server picks for you and says so: it compares the file against the
per-card VRAM budget and prints which mode it took and why. `run` and
`bench` always split, because a single stream cannot use a replica.

**Across machines.** `--peer <addr>` speaks to a `cortiq worker` holding
the tail layers of the same file — the same split, one socket further
away. The two sides verify they hold the same weights by `dir_hash`
before a token moves. `--net-dtype f16` halves the bytes on the wire;
`f32` (the default) is bit-exact.

## Measured, 2×RTX 5090

Single-stream decode, one card against the split, 96 tokens:

| model | file | 1 GPU | split, 2 GPU |
|---|---|---|---|
| Bonsai 1.7B (q1) | 0.3 GB | 191.6 tok/s | 174.8 tok/s |
| Nanbeige 4.2 3B (q4t) | 2.2 GB | 77.4 tok/s | 71.8 tok/s |
| Qwen3.6 35B-A3B W2 (q2tp) | 12.0 GB | 129.6 tok/s | 118.9 tok/s |

Every one of them runs correctly on the split — same output, no
refusals — and every one of them is slower for a *single* stream, by 7
to 9%. That is the honest shape of a layer split: it buys room, not
speed. Reach for it when a model does not fit, and for nothing else.

Throughput with replicas, the mode that does scale — 8 concurrent
requests, aggregate:

| model | 1 GPU | replicas, 2 GPU |
|---|---|---|
| Qwen3.6 35B-A3B W2 | 115.3 tok/s | **218.5 tok/s** (1.9×) |

## Video

`cortiq animate` renders on one card and does not split: the DiT's
attention and both VAE decoders are already resident, and a second card
would only add a bus crossing. Two cards help video by rendering two
clips at once — run two processes, each pinned with
`CMF_GPU_ADAPTER=0` and `=1`.

## When a card is not used

- `cortiq gpu` shows nothing → no Vulkan/Metal/DX12 adapter. On RunPod
  images, four GLVND packages and `XDG_RUNTIME_DIR=/tmp` are usually
  what is missing.
- `--gpus 2` on a machine with one card is an error, not a silent
  downgrade.
- The engine keeps weights in VRAM up to `CMF_GPU_VRAM_MB` (default
  8192 on discrete cards) and makes layers resident in first-touch
  order — so a small budget behaves like `-ngl`: the first N layers on
  the card, the rest on the CPU. Raise it to hold a whole model.
