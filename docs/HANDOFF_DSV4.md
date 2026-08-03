# DeepSeek-V4-Flash on cortiq — stand, state, open problems

Written to be handed to someone who has none of the context.

## 1. The stand

Rented Colab box, RTX PRO 6000 Blackwell Server Edition, 97887 MiB VRAM,
176 GB RAM, Ubuntu. Reached over an SSH-in-a-tunnel; the port changes every
time the tunnel restarts.

```
ssh -p <PORT> root@bore.pub
```

The port was 39791 at the time of writing and 46973 the session before. If it
refuses, the tunnel is down and has to be re-made from inside the Colab
notebook:

```bash
# in the notebook, once:
apt-get install -y openssh-server
ssh-keygen -A                      # REQUIRED — sshd will not start without host keys
mkdir -p /root/.ssh && echo "<your public key>" >> /root/.ssh/authorized_keys
sed -i 's/^#\?PermitRootLogin.*/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config
service ssh start
# bore tunnels the SSH port, which is 2222 here, NOT 22:
curl -fsSL https://github.com/ekzhang/bore/releases/latest/download/bore-linux -o /usr/local/bin/bore
chmod +x /usr/local/bin/bore
bore local 2222 --to bore.pub      # prints the public port
```

Two traps that cost hours:

- **bore must forward the port sshd actually listens on** (2222 in this
  image, not 22).
- **cloudflared cannot do this.** It tunnels HTTP, not arbitrary TCP, on the
  free tier. Do not retry it.

## 2. Vulkan on Colab

The NVIDIA driver is present but its Vulkan and EGL manifests are not, so
wgpu finds no adapter and silently falls back to CPU — which looks like "the
model is slow", not like "there is no GPU". Fix, in this order:

```bash
apt-get install -y libvulkan1
echo /usr/lib64-nvidia > /etc/ld.so.conf.d/nvidia-colab.conf && ldconfig
mkdir -p /usr/share/vulkan/icd.d /usr/share/glvnd/egl_vendor.d
printf '{"file_format_version":"1.0.0","ICD":{"library_path":"/usr/lib64-nvidia/libGLX_nvidia.so.0","api_version":"1.3.0"}}' \
  > /usr/share/vulkan/icd.d/nvidia_icd.json
printf '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' \
  > /usr/share/glvnd/egl_vendor.d/10_nvidia.json
```

Then **always** run with `WGPU_BACKEND=vulkan`. Without it wgpu picks the GL
backend and every timing measured is of the wrong device. A whole
dispatch-cost study was taken on the GL backend before this was noticed.

Verify the device is really up by the ABSENCE of `wgpu init failed` on
stderr, not by the presence of any "GPU path: on" line — the latter prints
either way.

## 3. Model and build

```bash
# 112 GB, 15 parts, concatenated:
BASE=https://huggingface.co/infosave/DeepSeek-V4-Flash-0731-cmf/resolve/main/parts-q2tp-v2
: > /content/dsv4-q2tp.cmf
for i in $(seq 0 14); do
  aria2c -x8 -s8 -q --allow-overwrite=true -d /content -o tmp "$BASE/$(printf part_%03d $i)"
  cat /content/tmp >> /content/dsv4-q2tp.cmf && rm /content/tmp
done
dd if=/content/dsv4-q2tp.cmf of=/dev/null bs=64M   # warm the page cache; mmap faults read at ~50 MB/s cold

curl --proto '=https' -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.85.0 --profile minimal
git clone https://github.com/infosave2007/cmf.git /root/cmf
cd /root/cmf && cargo build --release --features gpu
```

Canonical run:

```bash
export CMF_GPU=wgpu WGPU_BACKEND=vulkan CMF_DSV4_GPU_ATTN=1 CMF_DSV4_GPU_MOE2=1
export CMF_GPU_VRAM_MB=96500 CMF_GPU_UPLOAD=staged CMF_UPLOAD_EVICT=0
export CMF_DSV4_GPU_LAYER=1 CMF_SDOT=0 CMF_DSV4_CHAIN=1
./target/release/cortiq bench /content/dsv4-q2tp.cmf --tokens 128
```

Expected on the 97887 MiB stand: **30.2 tok/s steady** (128-token canonical run). All 43 layers use the
card: 42 have complete expert packs and the last pack contains 199/256 routed
experts; cold winners are completed from the mmap-backed CMF on the CPU.
Perplexity gold is **3.282** (`cortiq ppl … --file /root/ppl.txt --tokens
128`), which equals the CPU exactly — any change that moves it is a defect.

`CMF_UPLOAD_EVICT=0` above is for a hot benchmark on a machine with enough
RAM. Omit it (the default is eviction) for the out-of-core production mode:
pages uploaded to a discrete GPU are dropped from the host page cache, and a
later cold expert faults only its own ranges back from the CMF file. The 112
GB model is always mmap-backed; it is not copied into a 112 GB heap buffer.

`CMF_GPU_VRAM_MB` also emulates a smaller card, which is how the degradation
ladder below was measured without owning one.

## 4. Where the time goes

Per token, about 33.1 ms total. The 42-layer full chain is 1.32 ms host encode
and 28.03 ms wait. The final layer is a budget-sized partial GPU layer; only
its non-resident selected experts run on the CPU. This replaced the previous
whole host layer; the refined workspace reserve then moved 28.6 to 30.2 tok/s
while keeping the same exact cold-expert correction.

Inside the chain: hyper-connection glue 4.14, compressors 2.90, gate/up 2.49,
down 2.47, attention 1.80, o_lora 1.40, q-proj 1.37, indexer 1.36, wo_b 1.24,
next-q 0.98, and about 8 ms unattributed. The unattributed part is the
largest single item and nobody has measured it; the timestamp-query machinery
exists (`CMF_GPU_TS=1`, `ts_pair`) but only three passes are instrumented.

Degradation by VRAM (emulated):

| budget | layers on card | tok/s |
|---|---|---|
| 96500 | 42 full + 1 partial (140/256) | 28.6 |
| 48000 | 21/43 | 4.7 |
| 24000 | 10/43 | 3.8 |
| CPU only | 0/43 | 2.1 |

The sub-96-GB rows predate adaptive partial packs and are historical; re-run
them before using them as current capacity-planning numbers.

## 5. The speed target and why it is where it is

The goal is 40+ tok/s. Kernel work alone cannot reach it: the whole token is
36.8 ms and even zeroing the unattributed 8 ms lands near 31-32.

The only multiplier is speculative decoding, and the checkpoint already
carries the draft model: `model.mtp.{0,1,2}`, a three-stage DSpark block that
proposes five tokens at once. It was invisible to the loader until this
session because the loader looked for DeepSeek-V3 names. It is loaded now
(`dsv4::load_mtp`) and a CPU oracle measures it (`CMF_DSV4_DRAFT_PROBE=1`).

Measured: **acceptance 1.58 of 5, i.e. 2.58 tokens per trunk pass**, prefix
survival [0.67 0.50 0.29 0.08 0.04]. (An earlier figure of 4.36 came from the
benchmark's repetitive output and is wrong — always report the count of
distinct tokens beside any acceptance number.)

Cost model, measured rather than assumed:

```
verify(B) = 35.0 + marginal_batch_cost * (B - 1)
```

The earlier 4.1 ms marginal estimate came from isolated q4tp batch kernels.
It does **not** describe the current production batch: the B=5 frame records
most token bodies separately and relies on L2. On a persistent server a
64-token prompt measured 2421 ms at B=1 and 2615 ms at B=5. Keep B=1 as the
default until the large projections and MoE bodies use real B-axis kernels.

The honest five-position CPU/disk draft is now batched by weight and costs
about **156 ms/block** (down from 185–190 ms), not the earlier 36 ms estimate;
with it there is no gain. With the draft on the
card (~10 ms) it is about 38 tok/s. With the draft AND layer 42 on the card
it is about 49. Both need VRAM this card does not have: the draft is 10.34 GB
in q4tp (about 5.4 after requantising to q2tp) and layer 42 needs 2.18 GB,
against roughly 3.3 GB physically free. **Short by 4-5 GB.** On a ~110 GB
card the whole thing fits.

## 5a. State after the speculative-decode session (2026-08-03, commits d7431ea + 7374e34)

What now exists and is measured on the stand:

- **The batch is a token-axis frame** (`dsv4_chain_batch_bt`, `CMF_DSV4_BT=0`
  reverts): glue, MoE and projections carry the token on a grid axis; prep
  and attention stay per token in causal order, because a full window SHIFTS
  on append and an attend must read the cache as of its own token. Text
  parity against the walk holds on ten toy stands (widths 3 and 5,
  multi-chunk) and on the release checkpoint. Speed did NOT move
  (verify(5) ≈ 181 ms ≈ the old 204): the pass is dispatch-latency-bound at
  ~60 µs a dispatch, and the per-token attention interleave keeps the count
  high. The fix is a STAGED-append batch — batch appends into staging rows
  at the cache tail, per-token index lists built on the host, every attend
  batched, one shift-by-k at commit. NB: naive ring-slot writes are wrong at
  a full window (token t' destroys a row inside earlier token t's view).
- **The draft runs on the card**: `CMF_DSPARK_GPU=1` +
  `CMF_DSPARK_PACK=<freq.tsv>` + `CMF_DSPARK_RESIDENT=64` packs the top-64
  experts per stage by measured frequency (masked routing — acceptance
  pays, correctness never does) and `dspark_graph` runs all three stages ×
  five positions in one submission. **11.1 ms a block** (was 156–176 on the
  CPU tier), acceptance intact: 0.99/5 natural, 4.77/5 on the bench prompt.
  The markov bigram is NOT optional (without it 1.02 → 0.42) and rides on
  the host over the one B-wide head submission. VRAM: the pack needs
  `CMF_DSV4_PACK_MAX_LI=40` (two host tail layers) at budget 96500; budget
  97200 was a real device OOM again.
- **The speculative loop is wired**: `CMF_DSV4_SPEC=1` (greedy only,
  default off). Verify = one batched pass over `[t_next, drafts...]`;
  rollback = device photograph restore + replay of accepted tokens' state
  appends from retained hiddens + host-tail re-walk (`gpu_spec_txn` pins
  six all-rejected rounds at drift ≤ 2e-3).
- **The honesty line**: the speculative output is greedy-equivalent to
  ROUND-OFF, not bit-identical. One ulp of reassociation survives in the
  attention half (x2 after one device layer differs by 8.9e-8 with MoE
  skipped; near-tied expert selection amplifies it). Measured on the
  natural prompt: ~110 tokens of exact agreement, then a near-tie flip.
  PPL is untouched (ppl never speculates). Chasing the ulp further needs a
  two-process buffer diff — the in-process one is poisoned by the shared
  frame-buffer pool.
- **Where the clock stands**: walk bench 24.6 tok/s at PACK_MAX_LI=40
  (30.2 at the canonical 96500/41 config without the draft pack), spec
  bench 24.5 — break-even, entirely because verify(5) ≈ 181 ms against the
  11 ms draft. With the staged-append batch bringing verify(5) to 60–90 ms
  the bench lands at 40–55 tok/s (acceptance 4.77). Natural text stays near
  baseline (tokens/pass ≈ 2.0) — speculation pays on predictable text.

## 6. Open problems, in the order they should be taken

### 6.1 Make the production batch a real weight-sharing batch

The production entry point is connected and B=1/B=5 greedy parity is exact.
Per-token state, positions, forced hash rows and cache growth are separated;
q and next-q use the real q4tp batch kernel. That is correctness scaffolding,
not yet acceleration: B=5 is 8% slower on a hot 64-token prompt. Convert the
large projections and selected-expert bodies to true B-axis kernels before
enabling it by default.

Beware: **two checks in this area have already been vacuous.** The gate
compared a batched prompt against a walked one and reported agreement while
the batch never ran, and a generation comparison did the same. Any comparison
of a fast path against a reference must first prove the fast path executed —
`tools/dsv4_toy_gate.sh` now demands that and prints NOT TESTED otherwise.
Keep it that way.

### 6.2 Improve partial-pack selection and cold completion

The packer now adapts to the live VRAM budget for every layer. It reserves
workspace, flushes upload staging per projection, routes over all experts,
returns cold scored or hash-forced winners, and applies their exact
`post[j] * cold` correction before the next layer. The simple policy packs
the first N expert IDs. A frequency-aware stable pack (or an evictable,
reusable disk-to-GPU cold slot) is the next improvement; do not grow an
unbounded second GPU cache.

### 6.3 Speculative verify and its state transaction

Once the batch runs, verification needs to write B positions into the caches
speculatively and keep only the accepted prefix. Needed: a snapshot of the
logical lengths, a shadow tail for the window ring, a journal of new
compressed and index entries, commit on acceptance, restore on rejection.
Test rejections of length 0 through 5, at a compressor ratio boundary, and at
a window wrap, and check that the next ordinary token matches the sequential
reference afterwards.

### 6.4 Out-of-core ownership (implemented; keep it invariant)

There is no layer-number cutoff and no requirement that the model fit in RAM
or VRAM. Full GPU runs are chained; partial layers use resident experts plus
mmap-backed CPU completion; layers with no room remain wholly mmap-backed.
The pack cache key is `(model_uid, ordinal, first_expert_idx)`, so long-lived
multi-model servers and MTP stages cannot inherit another layer's pack.

### 6.5 The unattributed 8 ms

Instrument the chain's passes with `ts_pair`. The device has
`TIMESTAMP_QUERY_INSIDE_PASSES`, so per-stage timing inside the fused passes
is possible. This is the largest unexplained item in the token.

### 6.6 Workspace reserve is part of admission

The expert packer keeps a geometry-independent device workspace reserve. Its
default is budget/384 clamped to 256–512 MiB; `CMF_GPU_WORKSPACE_MB` overrides
it. On the 97,887 MiB stand (96,500 MiB weight budget), 768 MiB packed 140
experts in the tail layer at 29.6 tok/s, while 256 MiB packed 199 at 30.2
(30.6 in the shorter 32-token A/B).
Zero packed 229 at 31.1 but is not a safe universal default. Raising the total
budget to 96,800 MiB and packing the last layer completely caused a real
device OOM, so do not infer physical headroom from the weight budget alone.

## 7. Traps that have already cost time

- **A uniform field repurposed without moving every writer.** Word 2 of
  `Q1Params` was `cols` and became a batch count; two encoders kept writing
  `cols`, the kernel read a batch of 4096, and decode fell from 27.2 to 0.35
  tok/s — while still producing the RIGHT answer, because the storage bounds
  clamp swallowed the strays. Nothing failed; only the clock said so. All
  such params now go through `q4tp_mv_params`.
- **Device caches keyed by a host address.** The GPU weight cache keyed on
  the mmap's address, so a reloaded model inherited the previous one's
  buffers. Now keyed on `CmfModel::uid()`. The same class: `pack_for` keyed
  on the layer ordinal, which would hand the draft's three stages the trunk's
  first three packs.
- **Parity harnesses bound to buffers they had dropped.** Three tests created
  a fresh device buffer per call while the step under test caches its bind
  group by key, so every call after the first wrote into freed memory. They
  read exactly like broken kernels. Pool the buffers.
- **A test that adds a bias the device also adds.** The compressor parity
  test counted the positional bias twice and blamed the kernel; the growth of
  the error across folds was the clue.
- **`timeout` does not exist on macOS**, and `pgrep -f <pattern>` matches the
  shell running it — a watchdog loop waiting on `pgrep -f 'cortiq run'`
  waits on itself forever. Kill by PID.
- **Backticks in a heredoc commit message** get executed by the shell.
