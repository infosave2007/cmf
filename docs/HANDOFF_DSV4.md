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

Expected: **27.2 tok/s steady, 42 of 43 layers on the card.** Perplexity gold
is **3.282** (`cortiq ppl … --file /root/ppl.txt --tokens 128`), which equals
the CPU exactly — any change that moves it is a defect.

`CMF_GPU_VRAM_MB` also emulates a smaller card, which is how the degradation
ladder below was measured without owning one.

## 4. Where the time goes

Per token, 36.8 ms total: chain 30.5 ms of GPU + 1.3 ms host encode, plus
5.0 ms for layer 42, which does not fit on the card and runs on the host.

Inside the chain: hyper-connection glue 4.14, compressors 2.90, gate/up 2.49,
down 2.47, attention 1.80, o_lora 1.40, q-proj 1.37, indexer 1.36, wo_b 1.24,
next-q 0.98, and about 8 ms unattributed. The unattributed part is the
largest single item and nobody has measured it; the timestamp-query machinery
exists (`CMF_GPU_TS=1`, `ts_pair`) but only three passes are instrumented.

Degradation by VRAM (emulated):

| budget | layers on card | tok/s |
|---|---|---|
| 96500 | 42/43 | 27.2 |
| 48000 | 21/43 | 4.7 |
| 24000 | 10/43 | 3.8 |
| CPU only | 0/43 | 2.1 |

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
verify(B) = 36.8 + (4.1 + h43) * (B - 1)
```

4.1 ms is the marginal weight traffic of an extra token (2.4 GB of
projections plus 1.66 GB of distinct experts at the ~1 TB/s the kernels
achieve). `h43` is layer 42's per-token host cost, about 5 ms, and nobody has
measured whether grouping it across a batch helps.

With the draft on the host (~36 ms) there is no gain. With the draft on the
card (~10 ms) it is about 38 tok/s. With the draft AND layer 42 on the card
it is about 49. Both need VRAM this card does not have: the draft is 10.34 GB
in q4tp (about 5.4 after requantising to q2tp) and layer 42 needs 2.18 GB,
against roughly 3.3 GB physically free. **Short by 4-5 GB.** On a ~110 GB
card the whole thing fits.

## 6. Open problems, in the order they should be taken

### 6.1 The batched prefill does not run — find its entry point

This is the immediate blocker and it is small.

`dsv4::forward_chunk` has a batched path (`CMF_DSV4_BATCH=N`,
`forward_chunk_batched`) that sends N prompt tokens through the card in one
submission. Everything under it is written and green: per-token buffers,
per-token seeding, per-token forced expert rows for hash layers, the batched
chain itself.

**It has never executed.** Neither its "batching by N" line nor its refusal
line appears — under `cortiq ppl` or under `cortiq run`, at any batch width,
and not because of the log level (both were switched to `warn!` to rule that
out). So `forward_chunk` is not on the prompt path at all.

`pipeline.rs` has a chunked dsv4 prompt loop around line 1660 guarded by
`self.dsv4.is_some() && mtp.is_none() && pos < input_ids.len() && !cancel`,
and a separate generic batched path above it gated on `CMF_BATCH_K`. One of
them is being taken, or the prompt is consumed before either. Find which,
and put the batch there.

Beware: **two checks in this area have already been vacuous.** The gate
compared a batched prompt against a walked one and reported agreement while
the batch never ran, and a generation comparison did the same. Any comparison
of a fast path against a reference must first prove the fast path executed —
`tools/dsv4_toy_gate.sh` now demands that and prints NOT TESTED otherwise.
Keep it that way.

### 6.2 Measure h43

Layer 42's MoE on the host costs about 4 ms a token. Grouping the batch's
tokens by chosen expert should cut it — 16 distinct experts per layer across
5 tokens against 28 picks was measured — but it has not been implemented or
timed. `h43` decides whether speculation lands near 38 or near 45.

### 6.3 Speculative verify and its state transaction

Once the batch runs, verification needs to write B positions into the caches
speculatively and keep only the accepted prefix. Needed: a snapshot of the
logical lengths, a shadow tail for the window ring, a journal of new
compressed and index entries, commit on acceptance, restore on rejection.
Test rejections of length 0 through 5, at a compressor ratio boundary, and at
a window wrap, and check that the next ordinary token matches the sequential
reference afterwards.

### 6.4 The 43rd layer

It needs 2184 MB and there is not that much free. `pack_for` plans all 256 of
its experts, so the partial-pack path (`CMF_DSV4_COLD_CPU=1`) never engages
for it. That path is exact — PPL 3.282 with it and without — but measured
SLOWER where it does engage (3.5 against 3.9 tok/s on an emulated 24 GB
card), because a cold pick forces a fence per layer per token. Do not expect
it to fix anything without solving the fence.

### 6.5 The unattributed 8 ms

Instrument the chain's passes with `ts_pair`. The device has
`TIMESTAMP_QUERY_INSIDE_PASSES`, so per-stage timing inside the fused passes
is possible. This is the largest unexplained item in the token.

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
