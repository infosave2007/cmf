# The CUDA backend

*Languages: **English** · [Русский](CUDA.ru.md) · [中文](CUDA.zh.md)*

cortiq runs on GPUs through wgpu (Vulkan, DX12, Metal) and through its own Metal path. On an NVIDIA card
there is a third way, a native CUDA backend. It is experimental and off by default: a build feature turns
it on and an environment variable selects it.

```bash
cargo install cortiq-cli --features cuda      # or, in a checkout: cargo build --release -p cortiq-cli --features cuda
CMF_GPU=cuda cortiq run spark-x2.6-4b-q4tp-turbo.cmf --prompt "..." --greedy --no-think
CMF_GPU=cuda cortiq bench spark-x2.6-4b-q4tp-turbo.cmf --ctx 4000 --tokens 60 --core
cortiq gpu        # says whether the card and the libraries are there
```

The build needs no CUDA toolkit: nothing is linked. At run time the backend opens the driver's `libcuda`
and two toolkit libraries, `libnvrtc` (it compiles the kernels) and `libcublas`, version 12.x. If one of
them is missing, `CMF_GPU=cuda` runs on the CPU and says why (`cortiq gpu`, or `RUST_LOG=warn`).

## The server

`cortiq serve` runs on this backend as on any other: an OpenAI-compatible HTTP API with a web dashboard on
the same port.

```bash
CMF_GPU=cuda CMF_MAX_SEQ=131072 cortiq serve spark-x2.6-4b-q4tp-turbo.cmf --port 8080 --host 127.0.0.1
```

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "spark",
  "messages": [{"role": "user", "content": "Explain how a hash map works."}],
  "max_tokens": 256,
  "temperature": 0,
  "enable_thinking": false,
  "stream": true
}'
```

| endpoint | what it does |
|---|---|
| `POST /v1/chat/completions` | a chat turn; `stream: true` sends the answer as it is decoded and ends with `data: [DONE]` |
| `POST /v1/completions` | the continuation of a `prompt` |
| `GET /v1/models` | the served model |
| `GET /healthz` | 200 while the server is up |
| `GET /` | the web dashboard |
| `GET /v1/cortiq/status` | the engine's state |

- `model` is required in a request and may be any name: one file is served.
- The default `--host` is `0.0.0.0`, every interface, and the server checks no keys; `--host 127.0.0.1`
  keeps it local.
- `CMF_MAX_SEQ` is the context cap of every request, 32 768 by default.
- A request that continues the previous one reads only what is new: the K/V rows of the shared beginning
  stay on the card (20 new tokens in 42 ms with 87 cached). Past the seal of a turbo file a new turn reads
  the conversation again.
- `CMF_SERVE_SLOTS=N` keeps N conversations at once, each with its own rows on the card: one for every four
  CPU cores by default, four at most. Their requests take turns on the card. Two at once gave the same
  texts as one after the other, at 38 and 64 tok/s where one alone decodes at 117.
- After each request the server prints what it took, as llama-server does. A document of 20 283 tokens on
  an A40, read, sealed and answered:

```
prompt eval time =    8334.96 ms / 20283 tokens (    0.41 ms per token,  2433.49 tokens per second)
       eval time =     710.31 ms /    63 tokens (   11.27 ms per token,    88.69 tokens per second)
      total time =    9045.26 ms / 20347 tokens
```

`scripts/cuda/pod/serve_turbo_check.sh` runs all of this against a card.

## What runs on it

Dense stacks with full or sliding-window attention, the Spark-X2.x class: weights in q4tp, q8_2f or f32,
RMS norms, rotary positions per layer, Spark's head-wise output gate, SiLU or GELU FFNs. For those the
backend serves the two graph contracts the wgpu backend serves:

- **decode**: every layer and the head on the card, the hidden row resident, one read-back a token;
- **the prompt**: chunks of 1024 rows through every layer, K/V rows left in the mirrors decode reads;
- **the far field** of a turbo file: a prompt past the file's seal is read whole with exact attention and
  sealed once after it; from there a token reads the far-field state the card keeps, and its cost and the
  card's memory stop growing with the text. `CMF_O1=off` keeps exact attention at any length.

Hybrids of the Qwen3.5 family (`qwen3_5`: GDN mixers and gated full attention with Q/K norms, dense FFNs,
an MTP block; Qwen3.8-27B) run on a second arm of the backend with drafts and their verify on the card,
see [Hybrids](#hybrids-the-qwen35-family). The Fast Code stack (`qwen4_exp`) has its own path
(`gpu_cuda/qwen4.rs`). Anything else (other MoE stacks, sandwich norms, output gates other than these) is
declined before any state changes, and the model runs on the CPU path; `CMF_CUDA_DEBUG=1` prints the
reason.

## How it works

The Rust side is [cudarc](https://crates.io/crates/cudarc): the driver API for the context, memory, module
loading and kernel launches, NVRTC and cuBLAS. The kernels are CUDA C in
`crates/cortiq-engine/src/gpu_cuda/`, compiled on first use for the card's architecture; the decode
attention is compiled for the model's head geometry.

**Decode** keeps the weights in their codec and reads each matrix once a token. One kernel, `mv_multi`,
takes several matrices against one input row in a launch (q, k, v and the head gate; gate and up). A warp
owns a row of the matrix; the input row is staged in shared memory once per block with the RMS norm and a
q8_2f column field folded in. The staged row is stored element-major with an odd stride, so the 32 lanes
of a warp read 32 different banks. A q4tp row's 32 scales sit in the lanes' registers and a group's scale
is a warp shuffle. Attention is three small kernels over the K/V mirror: scores with a warp per key row,
read once for the four query heads of a KV group; the exponentials and value sums per block; the sum over
blocks with the head gate.

**The prompt** expands each weight to a dense f16 plane once per chunk and multiplies with cuBLAS on the
tensor cores: f16 operands, f32 accumulation and f32 output. Attention is two batched GEMMs per layer
(scores and values, the four KV groups in one call) around a causal softmax, in runs of rows that keep
the score matrix near 384 MB.

**The far field** (`nystrom.rs`) keeps, for a sealed layer, the rows it reads exactly — the first rows,
the rows chosen to stay exact, the ring of the last 1800 — in a buffer laid out as a K/V mirror, and each
head's far sums over 32 landmarks: 18 MB a layer. A token adds two launches to the three of exact
attention: the ring row it replaces goes into the far sums, and the far part of the row's attention comes
out as one more block of the partial sums the near rows' kernels leave, under one shift with them. The
seal is the host's: the prompt's K/V rows and one query row in eight come off the card for it, and it
chooses the 248 exact rows among all the rows of the prompt. This backend reads the whole prompt before
the seal, where the wgpu path seals at 8 192 rows and reads the rest through the far field;
`CMF_O1_WHOLE=1` takes the wgpu path the same way.

## Precision

Decode is f32 in every mode. For the prompt, `CMF_CUDA_PREC` chooses:

| value | weight GEMMs | attention GEMMs |
|---|---|---|
| `fast` (default) | f16 operands, f32 accumulation | TF32 |
| `tf32` | TF32 | TF32 |
| `exact` | f32 | f32 |

`fast` is the class of the wgpu path's matrix-unit kernels. The FFN's middle, `act(gate) · up`, goes to
f16 divided by 16 and the GEMM multiplies it back, so a half holds magnitudes up to a million.

## Numbers

NVIDIA A40 (46 GB), driver 595.91, CUDA 12.8 libraries; a container with 7.65 CPUs of a Xeon Gold 6342.
`cortiq bench --core`, 60 tokens after a prompt of that length, medians of three runs, the two paths
taking turns. Vulkan is cortiq 0.8.66 with `CMF_O1=off` (exact attention, what the CUDA path runs).

| file | prompt, tokens | prompt, tok/s (Vulkan) | prompt, tok/s (**CUDA**) | first token, s (Vulkan) | first token, s (**CUDA**) | decode, tok/s (Vulkan) | decode, tok/s (**CUDA**) | peak VRAM, GB (Vulkan) | peak VRAM, GB (**CUDA**) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| q4tp | 1 000 | 3177 | **6132** | 0.31 | **0.16** | 82.9 | **116.9** | 3.6 | **3.5** |
| q4tp | 4 000 | 3789 | **6096** | 1.06 | **0.66** | 80.2 | **111.5** | 4.5 | **3.9** |
| q4tp | 16 000 | 2949 | **4808** | 5.43 | **3.33** | 54.3 | **92.7** | 5.5 | **4.9** |
| q8_2f | 1 000 | 513 | **5926** | 1.95 | **0.17** | 63.5 | **82.7** | 5.2 | **5.2** |
| q8_2f | 4 000 | 528 | **5927** | 7.58 | **0.67** | 60.1 | **80.0** | 6.1 | **5.6** |
| q8_2f | 16 000 | 507 | **4732** | 31.53 | **3.38** | 45.3 | **69.4** | 7.1 | **6.6** |

With `CMF_CUDA_PREC=exact` the q4tp prompt runs at 1807 / 1788 / 1587 tok/s at the three lengths; decode
is the same code in every mode. A 1 000-token prompt is one chunk, read in 0.16–0.3 s on either path, so
its tok/s moves more between runs than the others (CUDA 4524–6165, Vulkan 3132–3784).

### Past the seal

The 4-bit turbo file at 16 000 to 64 000 tokens: the far field over Vulkan (sealed at 8 192 rows), the far
field on CUDA (the prompt read whole, one seal) and CUDA with exact attention (`CMF_O1=off`), taking turns.
Medians of three runs of 60 tokens, then one run of 500 tokens a path for the long decode and the memory
the card holds while it decodes.

| prompt, tokens | path | prompt, tok/s | first token, s | decode, tok/s | decode over 500 tokens, tok/s | peak VRAM, GB | VRAM while decoding, GB |
|---:|---|---:|---:|---:|---:|---:|---:|
| 16 000 | Vulkan, far field | 1001 | 15.99 | 83.6 | 85.2 | 4.6 | 4.2 |
| 16 000 | **CUDA, far field** | **2622** | **6.10** | 95.1 | **103.0** | 4.9 | **3.9** |
| 16 000 | CUDA, exact attention | 4828 | 3.31 | 92.6 | 92.1 | 4.9 | 4.9 |
| 32 000 | Vulkan, far field | 1030 | 31.07 | 83.8 | 87.0 | 4.6 | 4.2 |
| 32 000 | **CUDA, far field** | **2161** | **14.81** | 91.1 | **102.5** | 6.0 | **3.9** |
| 32 000 | CUDA, exact attention | 3533 | 9.06 | 74.7 | 74.8 | 6.0 | 6.0 |
| 64 000 | Vulkan, far field | 1061 | 60.29 | 86.0 | 85.9 | 4.5 | 4.2 |
| 64 000 | **CUDA, far field** | **1511** | **42.35** | 84.9 | **100.7** | 8.2 | **3.9** |
| 64 000 | CUDA, exact attention | 2149 | 29.79 | 53.5 | 53.4 | 8.2 | 8.2 |

The first token after the seal moves the far-field state onto the card and frees the prompt's rows, so a
60-token decode reads lower than a long one. The 8-bit file's far field on CUDA: 2535 tok/s, first token in
6.31 s and 68.5 tok/s at 16 000 tokens; 2113 tok/s, 15.15 s and 68.4 tok/s at 32 000.

Where the far field's first token goes at 32 000 tokens (14.8 s): the prompt's chunks 9.0 s, the queries
off the card 0.7, the K/V rows off the card 1.1, their way into the host's cache 1.0, the choice of the
exact rows 0.3, the seal 2.7 s. The seal takes layers two at a time on the pod's 7.65 CPUs; one at a
time it is 4.1 s (`CMF_O1_SEAL_PAR`).

## Agreement

The reference is the host with float activations (`CMF_GPU=0 CMF_SDOT=0`). Four prompts, 96 decoded tokens
each, every path forced along the same ids, the largest logit difference to the reference:

| file | path | top-1 as the reference | first token | later tokens |
|---|---|---:|---:|---:|
| q4tp | CUDA, `exact` | 384/384 | 0.0001 | 0.0002 |
| q4tp | **CUDA** | 384/384 | 0.020 | 0.016 |
| q4tp | Vulkan | 384/384 | 0.020 | 0.017 |
| q8_2f | CUDA, `exact` | 384/384 | 0.0001 | 0.0002 |
| q8_2f | **CUDA** | 384/384 | 0.019 | 0.023 |
| q8_2f | Vulkan | 384/384 | 0.014 | 0.015 |

Free greedy runs with exact attention on six prompts of 20 to 20 255 tokens (`CMF_O1=off` for the longest),
96 tokens each: CUDA and Vulkan give the same 576 ids for each file. A chat of three turns and two server requests that share a prefix (its K/V rows reused)
give the same text on both. The Vulkan rows are 0.8.67's: its first token after a prompt takes the
token graph's head, where cortiq 0.8.66 used the hybrid head for a q8_2f file (0.29 at the first token).

The far field has three checks. Its token kernels against `NystromState::step_group` on random states,
144 rows through a whole turn of the ring: 3.5e-5 at values up to 2.3 (a unit test on the card). From one
and the same seal of a 20 255-token prompt, the decode on the card against the decode on the host with
float activations, 96 tokens: the same top-1 at every step, logits within 0.0002 (q4tp) and 0.0004 (q8_2f).
Against the wgpu path taken the same way, both prompts read in f32: 96/96 and 0.025 (q4tp), 95/96 and 0.50
(q8_2f). At the default precisions the two paths' far fields part ways sooner, and so do two precisions of
one path: on that prompt the wgpu path's default and its f32 prompt differ by 0.04 in the mean logit
(q4tp) and by 0.65 (q8_2f), the CUDA path's the same. The seal amplifies the prompt's rounding; exact
attention does not. Facts planted in 60 000, 120 000 and 240 000 characters of Wikipedia text are answered
8 times of 8 with the far field on CUDA (`scripts/cuda/pod/chain_cuda_passkey.sh`).

## Hybrids: the Qwen3.5 family

A stack of GDN mixers and gated full attention with Q/K norms (`qwen3_5`, Qwen3.8-27B among them) runs on
its own arm, `gpu_cuda/hyb.rs` and `hyb.cu`, behind the same two graph contracts. The pipeline's draft
rounds, prompt lookup and the server run on it unchanged.

```bash
CMF_GPU=cuda cortiq run qwen38-27b-q4tp.cmf --prompt "..." --greedy --no-think
CMF_GPU=cuda cortiq serve qwen38-27b-q4tp.cmf --port 8080 --host 127.0.0.1
```

**Decode and verify** use one tensor-core kernel: `mma.sync` m16n8k16 with bf16 operands and f32 sums.
- At load the card repacks each q4tp matrix, so a lane loads its eight bytes of a 16-row tile at once.
- An input row reaches the kernel as two bf16 planes, about 17 significant bits, written by the kernel
  that produced the row.
- Up to sixteen rows cost about one: eight rows 1.05× a single row on a 3090, sixteen about 1.6×.
- A row's arithmetic does not depend on how many rows share the launch. A verify window therefore gives
  the plain token's bits, and the text with drafts is the text without them.
- Gate and up are one matrix whose tiles hold eight rows of each. The kernel's epilogue writes
  silu(gate) · up for the down projection.

**GDN and attention.**
- A GDN layer is two launches: the conv, its ring, the L2 norms and the gates in one, the delta rule in
  the other.
- Attention reads f16 K/V mirrors.
- A token or a verify window is one CUDA graph, replayed with a record of its positions.

**Drafts.**
- The model's MTP block drafts on the card, its input projection included.
- Prompt lookup continues a draft that repeats an earlier stretch of the text. Its window is 8 rows, and
  16 after a lookup round whose drafts all held.
- After a partial accept, every GDN state goes back to the window's start and replays the accepted rows:
  two launches for all layers.

**The prompt** expands the repacked weights to f16 for cuBLAS, in chunks of 1024 rows. Attention runs on
the f16 K/V; the GDN glue runs in f32 kernels.

**Load.** Each kernel's PTX is cached on disk (`~/.cache/cortiq/cuda`, `CMF_CUDA_CACHE`), and the repack
runs on the card. The 27B file's plan builds in under 4 s.

**A chat's next turn.** A Qwen3.5 template renders a past answer without its think block, so the next
turn never extends the cached ids. The recurrent states cannot be rewound either. Instead, while it reads
a prompt, the card copies the GDN states at the prompt's last `<|im_start|>` (the assistant header), about
150 MB for the 27B file. A turn that starts with the ids before it goes back there and reads only what
follows. The attention rows before that point are prompt rows no decode wrote, so the copy stays exact
after any draft round. The draft head's rows before that point stay too. `CMF_KV_REUSE=0` reads every
turn from the start.

Three turns over a 10 KB file (`scripts/q27cuda/serve_chat.sh`), each turn carrying the model's own
answers:

| turn | prompt, tokens | read again | **from the snapshot** |
|---:|---:|---:|---:|
| 2 | 4 048 | 4 048 in 4.13 s | **338 in 0.50 s** |
| 3 | 4 119 | 4 119 in 4.29 s | **78 in 0.24 s** |

The texts of all three turns are the same both ways, and so is the decode rate (99 against 97 tok/s on
turn 2).

### Numbers on an RTX 3090

Qwen3.8-27B q4tp on an RTX 3090 (24 GB), driver 595.91, CUDA 12.8 libraries, a container with 6.8 CPUs.
`cortiq run --greedy --no-think`, 512 tokens; the Vulkan column is the wgpu path on the same card and
build. Eight chat prompts, then a copy task (60 lines of a 14 KB document reproduced) and an edit task (a
rename in a 10 KB file).

| prompt | Vulkan, drafts | **CUDA, drafts** |
|---|---:|---:|
| chat0: an LRU cache module | 53.6 | **101.2** |
| chat1: an SSD's FTL | 53.4 | **90.7** |
| chat2: an article in Danish | 40.8 | **67.1** |
| chat3: a refactor | 62.9 | **118.4** |
| chat4: trains | 65.5 | **118.5** |
| chat5: a JSON schema | 73.9 | **143.4** |
| chat6: a blog post | 48.0 | **84.4** |
| chat7: a summary | 63.3 | **105.2** |
| **chat, mean** | 57.7 | **103.6** |
| copy0 | 74.8 | **168.5** |
| edit0 | 71.6 | **185.7** |

- Without drafts (`CMF_GRAPH_SPEC=0`): 44.5 tok/s on CUDA, 40.8 on Vulkan.
- The text with drafts equals the text without them on every prompt checked (chat0, chat3, copy0, edit0).
  On Vulkan it differs on chat0, whose verify is int8.
- Prompt: 983 tok/s at 4 000 tokens (Vulkan 560), 929 at 16 000, 819 at 32 000.
- First token of chat0, the load included: 2.5 s.
- Through `cortiq serve`, four greedy requests with and without thinking: 82–151 tok/s against 44.6–44.8
  without drafts, the same texts (`scripts/q27cuda/serve_ab.sh`).
- The turbo file: facts planted in 60 000 characters of Wikipedia text are answered 8 times of 8. The first
  token comes in 21–23 s, against 52–58 s over Vulkan; past the seal it decodes without drafts.

## Options

| variable | meaning |
|---|---|
| `CMF_GPU=cuda` | select the backend (a build with `--features cuda`) |
| `CMF_CUDA_DEVICE=N` | the card's ordinal, 0 by default |
| `CMF_CUDA_PREC=fast\|tf32\|exact` | the prompt's precision, see above |
| `CMF_CUDA_BATCH=N` | prompt rows per chunk, 1024 by default |
| `CMF_GPU_VRAM_MB=N` | the weight budget instead of the card's memory less 2 GB |
| `CMF_O1=off` | a turbo file with exact attention at any length |
| `CMF_CUDA_PROF=1` | stage timers of the prompt and of a token, with a wait after each stage |
| `CMF_CUDA_CACHE=DIR` | where compiled PTX is kept, `~/.cache/cortiq/cuda` by default |
| `CMF_GRAPH_SPEC=0` | a hybrid without drafts |
| `CMF_KV_REUSE=0` | every chat turn read from the start |
| `CMF_CUDA_DEBUG=1` | why a graph declined, and a decode / prompt summary at exit |

## Limits

- Three architecture classes are covered: dense Spark-class stacks (measured on an A40), Qwen3.5-family
  hybrids (measured on an RTX 3090) and the Fast Code stack. Other MoE stacks and Gemma-style stacks run on
  the CPU under `CMF_GPU=cuda`, not on wgpu.
- The far field's prompt is read whole: its K/V rows are on the card until the seal (8.2 GB at
  64 000 tokens, 4.5 GB on the wgpu path), and its time grows faster than the text. A new turn of a
  sealed conversation reads the conversation again, as on the wgpu path.
- Drafts run on the hybrid arm and on the Fast Code stack, not on the dense arm. Past a turbo file's seal
  the hybrid decodes without them.
- The toolkit libraries must be on the loader's path; the bindings are for CUDA 12.x.
- Decode is bound by its own arithmetic, about 350 GB/s of weights on an A40 against 500 for a kernel
  that only loads them (`scripts/cuda/bench/mv_bench.cu`).

The run that produced these numbers: `scripts/cuda/` (pod scripts, prompts, kernel benchmarks).
