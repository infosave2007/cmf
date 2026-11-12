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
  card's memory stop growing with the text. On the hybrid arm (below) the prompt itself stays within the
  seal's memory. `CMF_O1=off` keeps exact attention at any length.

Hybrids of the Qwen3.5 family (`qwen3_5`: GDN mixers and gated full attention with Q/K norms, dense FFNs,
an MTP block; Qwen3.8-27B) run on a second arm of the backend with drafts and their verify on the card,
see [Hybrids](#hybrids-the-qwen35-family). The Fast Code stack (`qwen4_exp`) has its own path
(`gpu_cuda/qwen4.rs`). Nemotron Labs Dory (`dory`, a looped nGPT) runs on the hybrid arm's kernels, its
turbo files included, see [Dory](#dory-the-looped-ngpt). Anything else (other MoE stacks, sandwich norms, output gates other than these) is
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
out as one more block of the partial sums the near rows' kernels leave, under one shift with them. For
the seal the prompt's K/V rows and one query row in eight come off the card, and the host chooses the 248
exact rows among all the rows of the prompt. The dense arm reads the whole prompt before the seal.

The hybrid arm reads a long prompt the way the wgpu path does, so the card's memory stays at the seal's:

1. The first rows go exact, as many as the card holds beside the weights (8 192; 4 096 on a 16 GB card
   under the 27B).
2. The seal: the host builds each group's landmarks, sinks and window from the rows by index, and the card
   sums the far rows into each head's accumulators (`hy_far_insert_n`, 8 192 rows a tile). Before, the host
   replayed every row; at 32 000 tokens that took 7.3 s on 7 cores.
3. The rest of the prompt goes in chunks of 1 024 rows on the GEMMs. A chunk's rows sit after the far
   field's near rows. The near part of their attention is an exact layer's two batched GEMMs, the far part
   comes from the landmarks a row at a time (`hy_far_rows`), and a softmax takes both (`hy_softmax_hf`).
   The chunk's rows then move into the window, and the rows they push out go into the far sums. Each
   chunk's rows go to the host.
4. Before the prompt's last 512 rows the far field is built again over all the rows, the exact rows
   chosen among all of them. The last rows read it.

`CMF_O1_WHOLE=1` reads the whole prompt first on either arm and on the wgpu path, so the two can be
compared row for row.

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
- In the delta rule a thread owns a column of the 3 MB state, so a warp reads 32 consecutive floats; the
  verify's copy of the state for a step back is written by the same launch.
- Attention reads f16 K/V mirrors.
- A token or a verify window is one CUDA graph, replayed with a record of its positions.
- A one-row call over a long matrix set takes one tile a block (gate/up 801 against 745 GB/s); the
  arithmetic does not depend on it.

**Drafts.**
- A round's MTP drafts are one CUDA graph: each step gathers its token's embedding row on the card (the
  first 65 536 rows of the embedding, repacked), runs the block and the shortlist head and takes the
  argmax there. One upload, one launch, the ids back.
- The verify picks each row's argmax on the card with the host's tie rule, so the logits stay there.
- A round that lands two tokens or more speculates without timing plain tokens first: a round costs
  about 1.25 plain tokens on this arm.
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
| chat0: an LRU cache module | 53.6 | **118.5** |
| chat1: an SSD's FTL | 53.4 | **104.0** |
| chat2: an article in Danish | 40.8 | **72.7** |
| chat3: a refactor | 62.9 | **141.5** |
| chat4: trains | 65.5 | **138.6** |
| chat5: a JSON schema | 73.9 | **167.5** |
| chat6: a blog post | 48.0 | **89.0** |
| chat7: a summary | 63.3 | **125.9** |
| **chat, mean** | 57.7 | **119.7** |
| copy0 | 74.8 | **201.0** |
| edit0 | 71.6 | **220.1** |

- Without drafts (`CMF_GRAPH_SPEC=0`): 49.1 tok/s on CUDA, 40.8 on Vulkan.
- The text with drafts equals the text without them on every prompt checked (chat0, chat3, copy0, edit0).
  On Vulkan it differs on chat0, whose verify is int8.
- Prompt: 983 tok/s at 4 000 tokens (Vulkan 560), 929 at 16 000, 819 at 32 000.
- First token of chat0, the load included: 2.5 s.
- Through `cortiq serve`, four greedy requests with and without thinking: 86–173 tok/s against 48.9–49.1
  without drafts, the same texts (`scripts/q27cuda/serve_ab.sh`).
- The turbo file: facts planted in 60 000 characters of Wikipedia text are answered 8 times of 8. The first
  token comes in 21–23 s, against 52–58 s over Vulkan.

**Drafts over the far field.** Past a turbo file's seal a verify window's rows sit right after the near
rows (the sinks, the rows kept exact, the window), so a row reads its window's earlier rows as near ones and
the far field as it stood before the window (`hy_far_prep_b`, one body with the one-row kernel). The state
moves over the accepted rows only (`o1_spec_commit`): each takes the ring's next slot, and once the ring is
full the row it replaces goes into the far sums first. The ring's evicted row is folded in by a warp per
(head, landmark), the far part by 256 threads a head: past the seal a token costs 21.9 ms of kernels against
23.6 before, and 21.5 for the standard file at 1 000 tokens. `CMF_O1_SPEC=0` decodes past the seal without
drafts.

Past the seal on the RTX 3090 the turbo file decodes at 118–128 tok/s in `cortiq bench --core` and at 80.3 tok/s
on a real summary of 100 000 characters (42 % of the drafts kept, 45.2 without them; facts in 100 000
characters 8 of 8). The draft head reads the prompt's last 512 rows, warmed after the seal as over the wgpu
path's far field.

### Turbo on a 16 GB card

Qwen3.8-27B Turbo on an RTX 2000 Ada (16 GB, 15.6 GB usable), release build, `cortiq bench --core`: 60 tokens
after a prompt of that length (the bench's text repeats, so most drafts hold). The turbo file over Vulkan on
the same card and build, and the standard file on CUDA. Memory is in GB of 1024 MiB, the peak of the whole
run.

| prompt, tokens | first token, s (**CUDA**) | first token, s (Vulkan) | decode, tok/s (**CUDA**) | CUDA without drafts | decode, tok/s (Vulkan) | peak VRAM, GB (**CUDA** / Vulkan) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 000 | **2.55** | 3.58 | **87.3** | 13.9 | 27.9 | 14.6 / 14.1 |
| 4 000 | **9.71** | 14.88 | **86.9** | 13.6 | 26.7 | 14.9 / 14.3 |
| 16 000 | **43.2** | 70.4 | **94.4** | 13.8 | 25.4 | 15.0 / 14.8 |
| 32 000 | **86.2** | 135.6 | **93.5** | 13.8 | 26.7 | 15.0 / 14.8 |
| 64 000 | **174.6** | 277.9 | **93.8** | 13.8 | 26.5 | 15.0 / 14.8 |

- The standard file on CUDA: 2.56 s and 9.59 s to the first token at 1 000 and 4 000 tokens, then it runs
  out of the card's memory at 16 000.
- Real answers, up to 400 tokens: a summary of 30 000 characters of Wikipedia, first token 21.8 s on CUDA
  and 39.4 s over Vulkan, then 23.7 and 17.5 tok/s. A summary of 100 000 characters: 69.8 s and 114.3 s, then
  31.5 and 21.3 tok/s. Without drafts CUDA decodes at 13.7–13.8 tok/s and writes the same text. Vulkan
  writes the same text at 30 000 characters and differs by one word at 100 000.
- Facts planted in 60 000 and 100 000 characters: 8 of 8 each.
- A chat whose first turn is 40 000 characters, then two questions in one process: each turn reads the
  conversation again (26–28 s on CUDA, 40–49 s over Vulkan).
- Where a 1 000-row prompt chunk's 2.2 s go (`CMF_CUDA_PROF=1`): GEMMs 1.35 s (about 40 TFLOPS, the card's
  limit), the weights expanded to f16 0.33 s, the GDN scan 0.25 s, attention 0.15 s. A 32 000-token prompt
  spends 3.5 s building the far field again before its last rows: 0.5 s choosing the exact rows on the host,
  0.8 s putting the rows into the caches, 1.3 s summing the far rows on the card.

What keeps the 27B under 16 GB on CUDA:

- One allocation per repacked matrix, and one per plan for all of them: the card rounds each allocation up to
  2 MiB, which cost 0.95 GB on 64 layers.
- One allocation per GDN layer's state, and small constants in shared 8 MiB blocks.
- A prompt chunk's weight expands to f16 in 64 MiB bands, not whole: the gate/up pair of a layer is 356 MB.
- The decode attention's scratch is sized by the keys a call reads, not by the context cap: 181 MB → 30 MB.

### DFlash 2 beside the file

The published DFlash 2 drafter (`z-lab/Qwen3.8-27B-DFlash2`) drafts a round of seven tokens in one pass. It
has five layers, and its context is the trunk's rows after layers 5, 19, 33, 47 and 61. On this arm the
block runs on the card:
- the four-bit matrices go through the same tensor-core matvec;
- fc, the kernel projections and the selector run in f16 on cuBLAS;
- the row convolutions, the head norms and the attention over the context have kernels of their own.

The trunk's graphs copy the tapped rows out on their way, and the drafter's context is made from them on
the card.

```bash
hf download z-lab/Qwen3.8-27B-DFlash2 --local-dir dflash2
cortiq convert --model dflash2 --output qwen38-27b-q4tp.cmf --dflash-sidecar --quant q4tp   # writes qwen38-27b-q4tp.dflash.cmf
CMF_GPU=cuda CMF_DFLASH=1 cortiq run qwen38-27b-q4tp.cmf --prompt "..." --greedy --no-think
```

Each round goes to the drafter that lands more tokens at this card's prices. A block costs about 0.19 of a
plain token, and a verify of eight rows 1.24.
- A router in the sidecar's header reads the trunk's row at the tip. It is fitted on logged rounds at those
  prices (`cortiq dflash-route-fit … --prices`).
- Without a router, the drafters' measured holds decide.
- A copy stays with the MTP chain: a tail that repeats eight tokens or more, or lookups that keep landing.
  So does text from the high part of the vocabulary.

The ten prompts on the RTX 3090, release build, tok/s:

| prompt | MTP chain | **DFlash + router** |
|---|---:|---:|
| chat0 | 121.2 | **125.8** |
| chat1 | 102.1 | **103.4** |
| chat2 | 72.8 | **69.9** |
| chat3 | 141.5 | **152.9** |
| chat4 | 138.8 | **156.5** |
| chat5 | 167.0 | **188.2** |
| chat6 | 93.7 | **97.1** |
| chat7 | 124.7 | **133.0** |
| **chat, mean** | 120.2 | **128.4** |
| copy0 | 198.9 | **183.5** |
| edit0 | 220.3 | **197.6** |

- The texts are the same as without drafts, and so are four server requests and a three-turn chat.
- The block is built once the first prompt has been read (0.9 s), so the first token comes 0.5 s later.
- A copy or an edit loses 8–10 %. The block's first round makes the prompt's context once: 100 ms on the
  card, and more in host copies of the tapped rows.
- A trunk without an MTP head ([Arete-27B Turbo](https://huggingface.co/infosave/Arete-27B-Turbo-cmf)) drafts
  every round with the block: 49.2 → 102.1 tok/s on chat0, the same text. The sidecars of Qwen3.8-27B, Arete-27B
  Turbo and OrcaCyber-Nano-1.0 Turbo, with routers at these prices, are on Hugging Face beside the files.

## Dory: the looped nGPT

Nemotron Labs Dory has an input block, a recurrent block that runs several times over the same weights and
an output block; a sub-layer is attention or a SwiGLU MLP, and every row lives on the unit sphere. It runs
on the hybrid arm's kernels (`gpu_cuda/hyb/dory.rs`, `dory.cu`), q4tp files only.

```bash
CMF_GPU=cuda cortiq run dory-turbo.cmf --prompt "..." --no-think
CMF_GPU=cuda CMF_MAX_SEQ=131072 cortiq serve dory-turbo.cmf --port 8080 --host 127.0.0.1
# in another shell, once it listens
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "dory", "messages": [{"role": "user", "content": "Hello"}], "enable_thinking": false, "stream": true}'
```

The server's endpoints and settings are in [The server](#the-server) and in
[docs/DORY.md](DORY.md#server); `"enable_thinking": false` asks Dory for a direct answer.

- **A token**: every matrix through the arm's tensor-core matvec, eight warps a tile and a tile a block
  (Dory's matrices are 2560 to 7168 columns wide). Gate and up are one pair with their scales in the q4tp
  ladders, so the matvec's epilogue writes silu(g) · u. Between the matvecs: the rotation with the unit
  heads and the query scale, one pass over the keys (flash decoding, each block under its own maximum), the
  gate, the nGPT residual with the row in registers. K and V are f16: a sliding layer's in a ring of its
  window and a turn's rewind, a full layer's at their position. Each stage of a token is one CUDA graph.
- **The router** of a turbo file decides after the first pass: the card runs the input block and that
  pass, the host scores the state (10 KB), and the card runs the output block, or the second pass first.
- **A prompt** goes by GEMMs, 512 rows at a time: each weight expanded to f16 once a chunk, a full layer's
  attention as two batched GEMMs around a masked softmax (128 query rows a pass, `CMF_DORY_ATT_ROWS`), a
  sliding layer's the same over its window unwrapped from the ring. On a turbo file these rows take one
  pass. The last ((n − 1) mod 64) + 1 rows take every pass on the tensor-core path, as the wgpu path's last
  chunk does.
- **The seal** of a turbo file: the full layers' rows come to the host and leave the card, the rows the far
  field would lose are chosen and kept exact (`pick_exact_rows`), and the far fields go up as the arm's. The
  host builds each KV head's landmarks, sinks and window from the rows by index (`seal_gathered_rows`), the
  card sums the far rows (`hy_far_insert_n`). Past it a token reads its exact rows (sinks, the chosen ones,
  the window) and 32 landmarks a head.
- **A long turbo prompt** stays within a fixed ceiling, as on the wgpu path and the hybrid arm:
  1. The first 8 192 rows go exact (fewer on a card without room for them; `CMF_O1_PREFIX_MAX` by hand), then
     seal. Their rows open an archive on the host (RAM, or files under `CMF_O1_SPILL`).
  2. The rest goes in chunks of 512 rows on the GEMMs. A full layer's rows sit after the far field's near
     rows: the near part of their attention is an exact layer's batched GEMMs, the far part comes from the
     landmarks a row at a time (`hy_far_rows`), and a softmax takes both (`hy_softmax_hf`, `hy_attn_out_f`).
     The chunk then moves into the window, and the rows it pushes out go into the far sums. Each chunk's rows
     go to the archive.
  3. Before the prompt's last 512 rows the far field is built again over all the rows, the exact rows chosen
     among all of them. The last rows read it: the last ((n − 1) mod 64) + 1 take every pass, up to 16 at a
     time after the near rows, as a verify window does on the hybrid arm.

  A prompt that ends within 512 rows of the prefix is read exact and sealed at its end.

### Dory on an RTX 3090

Release build, the same card and build for both columns; Vulkan is the wgpu path.

| | CUDA | Vulkan |
|---|---:|---:|
| `dory-turbo`, 12 real prompts of 200 tokens, tok/s | **201.4** | 120.9 |
| `dory-q4tp-v2`, the same, tok/s | **159.6** | 86.9 |

`cortiq bench --core`, 60 tokens after a prompt of that length:

| prompt, tokens | first token, s (CUDA) | first token, s (Vulkan) | decode, tok/s (CUDA) | decode, tok/s (Vulkan) |
|---:|---:|---:|---:|---:|
| `dory-turbo` 1 000 | **0.26** | 1.09 | **231.9** | 110.1 |
| 4 000 | **0.78** | 4.90 | **215.2** | 88.0 |
| 16 000 | **6.11** | 39.05 | **212.7** | 84.1 |
| 32 000 | **13.64** | 78.42 | **216.0** | 85.2 |
| 64 000 | **32.95** | 162.58 | **214.1** | 91.6 |
| `dory-q4tp-v2` 1 000 | **0.36** | 1.66 | **156.9** | 72.1 |
| 4 000 | **1.17** | 7.64 | **146.0** | 58.3 |
| 16 000 | **5.38** | 60.07 | **127.9** | 33.3 |
| 32 000 | **12.82** | 197.27 | **111.3** | 21.4 |

- The greedy texts of the 12 prompts are the CUDA path's own in a second run, and the wgpu path's on two
  files of 128 tokens; the prompt's last logits agree with Vulkan's within a KL of 4e-6 to 7e-4.
- Facts planted in 60 000 characters of Wikipedia (a pass key, an access code, a password, a room number):
  8 of 8, first token 5.7–6.4 s, decode 163–172 tok/s past the seal.
- The far field holds 257 MB of K/V past the seal (Vulkan 502 MB). In 0.8.70 and 0.8.71 the card's peak
  grew while a long prompt was read, `dory-turbo` 3.7 GB at 1 000 tokens, 5.2 at 16 000, 7.7 at 64 000 (Vulkan
  2.9, 3.6, 3.6): the prompt was read whole before the seal. It stays flat now, see
  [Dory on a 16 GB card](#dory-on-a-16-gb-card).
- Facts in 100 000 characters: 8 of 8 as well, first token 9.9–10.5 s. Wikitext-2 perplexity, 12 windows of
  512 tokens read by `cortiq ppl`: 13.689 against Vulkan's 13.683 (`dory-turbo`), 13.696 against 13.696
  (`dory-q4tp-v2`).

### Dory on a 16 GB card

`dory-turbo` on an RTX 2000 Ada (16 GB), release builds: this one, 0.8.71 (the prompt read whole, then
sealed) and the wgpu path (Vulkan) on the same card, taking turns. `cortiq bench --core`, 60 tokens after a
prompt of that length; the peak is the card's over the whole run, in GB of 1024 MiB
(`scripts/dory_cuda/ceiling.sh`).

| prompt, tokens | first token, s (**CUDA**) | 0.8.71 | Vulkan | decode, tok/s (**CUDA**) | 0.8.71 | Vulkan | peak, GB (**CUDA**) | 0.8.71 | Vulkan |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 000 | **0.51** | 0.51 | 1.84 | **84.0** | 84.0 | 63.9 | **3.44** | 3.49 | 2.92 |
| 4 000 | **1.72** | 1.73 | 8.52 | **79.0** | 79.0 | 56.0 | **3.49** | 3.66 | 2.93 |
| 16 000 | **8.84** | 10.80 | 70.5 | **80.7** | 80.3 | 55.9 | **3.80** | 4.99 | 3.52 |
| 32 000 | **17.07** | 25.21 | 149.0 | **80.2** | 80.1 | 56.1 | **3.80** | 5.72 | 3.52 |
| 64 000 | **33.41** | 69.14 | 306.0 | **80.2** | 80.3 | 53.9 | **3.80** | 7.47 | 3.52 |

- From 16 000 tokens on the card holds 3.80 GB whatever the prompt's length: the peak is the exact prefix's
  last chunk, with the nine full layers' 8 192 rows on the card (288 MiB of K/V and 144 MiB of queries) and the
  attention's score buffers. The rows leave the card before the far fields go up. Past the prefix a 512-row
  chunk costs the same at any position, so the first token at 64 000 tokens comes in half 0.8.71's time.
- 12 real prompts of 200 tokens: 75.5 tok/s on CUDA, 58.9 over Vulkan, 99.1 % of the text the same as
  Vulkan's, as in 0.8.71 (a short prompt takes the same path as before). Wikitext-2 perplexity, 12 windows of
  512 tokens: 13.689 (Vulkan 13.683), as before.
- Facts planted in 60 000 and 100 000 characters: 8 of 8 each, first token 8.5–9.5 s and 13.1–13.7 s (0.8.71
  9.9–11.4 and 17.2–18.1 s). A summary of 100 000 characters (about 25 000 tokens): first token 14.2 s, 19.5 in
  0.8.71, 115.5 over Vulkan, then 80 tok/s.
- Past the prefix the prompt is read through the far field, as the wgpu path reads it, where 0.8.71 read
  every row exact. A long answer is therefore closer to Vulkan's: on that summary the two share their first 221
  characters and go on nearly word for word, while 0.8.71 shares 76 and words the summary its own way.
- The host holds the archive while the prompt is read, 92 KB a row: at 64 000 tokens the process peaks at
  8.0 GB of RAM, against 21.1 GB in 0.8.71. With `CMF_O1_SPILL=DIR` the rows wait in files (5.45 GB, removed
  once the far field is built again) and the process keeps 1.25 GB of its own memory beside the mapped file;
  the first token and the text are the same.
- `cortiq serve` with a document of 60 000 characters (15 277 tokens): read in 8.3–9.4 s and answered at
  79 tok/s, the card at 3.77 GB at most over the server's whole run. Over Vulkan: 66.4 s, 55 tok/s, 4.34 GB,
  and an answer that begins the same (`scripts/dory_cuda/serve_check.sh`; a follow-up turn, a tool call, a
  streamed answer and two slots at once too).

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
| `CMF_O1_SPEC=0` | a turbo file without drafts past the seal |
| `CMF_O1_CHUNKS=0` | the hybrid arm and Dory read a turbo prompt whole, then seal (0.8.70's way) |
| `CMF_O1_FAR_GPU=0` | a seal's far sums on the host |
| `CMF_O1_PREFIX_MAX=N` | the first `N` rows of a long prompt exact, instead of the choice for the card |
| `CMF_CUDA_MEM=1` | what the card holds after each plan, prompt chunk and 256th token |
| `CMF_HYB_WD_MB=N` | the band (MiB) a prompt chunk's weight expands into, 64 by default |
| `CMF_HYB_ATT_ROWS=N` | a prompt chunk's rows per attention GEMM pass, 128 by default |
| `CMF_HYB_KT=N` | kernel times of the hybrid's N-th token |
| `CMF_KV_REUSE=0` | every chat turn read from the start |
| `CMF_DFLASH=1` | the DFlash sidecar beside the file drafts the rounds it wins (`CMF_DFLASH_PICK=dflash\|mtp` pins one) |
| `CMF_CUDA_DEBUG=1` | why a graph declined, and a decode / prompt summary at exit |
| `CMF_DORY_CUDA=0` | Dory on the CPU under `CMF_GPU=cuda` |
| `CMF_DORY_BIG=0` | a Dory prompt by the tensor-core matvec, 16 rows at a time |
| `CMF_DORY_KT=N`, `CMF_DORY_KT_AT=P` | kernel times of Dory's N-th token, of its prompt chunk at position P |
| `CMF_DORY_ATT_ROWS=N` | query rows of a Dory prompt chunk per attention GEMM pass, 128 by default (512: the 0.8.71 way) |

## Limits

- Four architecture classes are covered: dense Spark-class stacks (measured on an A40), Qwen3.5-family
  hybrids and Dory (measured on an RTX 3090), and the Fast Code stack. Dory's q8_2f file runs on the CPU. Other MoE stacks and Gemma-style stacks run on
  the CPU under `CMF_GPU=cuda`, not on wgpu.
- The dense arm reads a turbo prompt whole: its K/V rows stay on the card until the seal (8.2 GB at 64 000
  tokens, 4.5 GB on the wgpu path). The hybrid arm's and Dory's prompts stay within the seal's memory. A new
  turn of a sealed conversation reads the conversation again, as on the wgpu path.
- Drafts run on the hybrid arm and on the Fast Code stack, not on the dense arm.
- The toolkit libraries must be on the loader's path; the bindings are for CUDA 12.x.
- Decode is bound by its own arithmetic, about 350 GB/s of weights on an A40 against 500 for a kernel
  that only loads them (`scripts/cuda/bench/mv_bench.cu`).

The run that produced these numbers: `scripts/cuda/` (pod scripts, prompts, kernel benchmarks).
