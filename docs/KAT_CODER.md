# Running a 35B MoE coder from GGUF to GPU — step by step

*[Русская версия](KAT_CODER.ru.md)*

This walkthrough takes **KAT-Coder-V2.5-Dev** (Kwaipilot, = Qwen3.6-35B-A3B:
40 layers of which 30 are GatedDeltaNet linear attention, 256 routed experts
top-8 plus an always-on shared expert) from the public GGUF on Hugging Face
to GPU decode, with every command as actually run. The same `.cmf` file and
the same commands work on both backends; only step 4 differs.

Verified end-to-end on:

| | hardware | decode (steady) |
|---|---|---|
| Vulkan | RTX 5090 32 GB, Ubuntu 24.04 | **32.8 tok/s** (CPU on the same box: 14.4) |
| CPU | 32-core EPYC-class | 14.4–16.6 tok/s (llama.cpp on the same file: 4.7) |
| Metal | Apple M4, 24 GB | ~7 tok/s (probe-arbitrated, see §4b) |

## 0. What you need

- **Disk**: ~41 GB free during conversion — 21.4 GB GGUF + 19.6 GB `.cmf`
  (delete the GGUF afterwards).
- **RAM**: 24 GB is enough for both the import (the GGUF is memory-mapped,
  never read whole) and CPU inference (decode streams ~600 MB of expert
  weights per token from the page cache).
- **Vulkan path**: a discrete GPU. The *whole-model* graph wants a 32 GB
  card (17.4 GB of expert weights + KV mirror + attention weights ≈ 31 GB
  resident); on smaller cards the VRAM budget refuses what does not fit
  and those layers honestly run on the CPU — it degrades, it does not break.
- **Metal path**: Apple silicon with 24 GB+ unified memory.

## 1. Install cortiq

The simplest route is crates.io (Rust installs with one command from
[rustup.rs](https://rustup.rs)):

```sh
# macOS — Metal is always built in, nothing more to enable
cargo install cortiq-cli

# Linux / Windows — add the wgpu backend (Vulkan / DX12) for step 4a
cargo install cortiq-cli --features gpu
```

Already installed once? Add `--force` to upgrade in place.

Alternatively, release binaries ship with both GPU backends built in:

```sh
# Linux x86_64
curl -LO https://github.com/infosave2007/cmf/releases/latest/download/cortiq-x86_64-unknown-linux-gnu.tar.gz
tar xzf cortiq-x86_64-unknown-linux-gnu.tar.gz && sudo mv cortiq /usr/local/bin/

# macOS Apple silicon
curl -LO https://github.com/infosave2007/cmf/releases/latest/download/cortiq-aarch64-apple-darwin.tar.gz
tar xzf cortiq-aarch64-apple-darwin.tar.gz && sudo mv cortiq /usr/local/bin/
```

Or build from a source checkout (for hacking on cortiq itself):

```sh
git clone https://github.com/infosave2007/cmf && cd cmf
cargo build --release -p cortiq-cli --features gpu   # gpu = wgpu (Vulkan/DX12); Metal is always in on macOS
```

Check: `cortiq --version` → 0.5.25 or newer.

## 2. Convert: GGUF → .cmf (identical on both platforms)

One command — cortiq downloads the GGUF from Hugging Face itself and imports
it in pure Rust (no Python, no llama.cpp needed):

```sh
cortiq import-gguf bartowski/Kwaipilot_KAT-Coder-V2.5-Dev-GGUF/Kwaipilot_KAT-Coder-V2.5-Dev-Q4_K_M.gguf \
    --output kat-q4t.cmf --quant q4t
```

If you prefer to download yourself (resumable, any mirror), fetch
`Kwaipilot_KAT-Coder-V2.5-Dev-Q4_K_M.gguf` (21.4 GB) from
[bartowski/Kwaipilot_KAT-Coder-V2.5-Dev-GGUF](https://huggingface.co/bartowski/Kwaipilot_KAT-Coder-V2.5-Dev-GGUF)
and pass the local path as the first argument.

Import takes ~7 minutes on a 32-core box (mostly requantization). What it
does under the hood: derives the layer schedule (GDN vs full attention) from
tensor presence, splits the 3-D routed-expert tensors into per-expert
matrices, and undoes every llama.cpp storage convention — the `+1` baked
into RMS-norm weights (except the GDN gated norm, which is stored raw),
`ssm_a` stored as −exp(A_log), the tiled V-head reordering on every
V-indexed tensor including out_proj columns, and the fused QKV layout. The
result is a standard 19.6 GB `.cmf` with the tokenizer and chat template
embedded — a single self-contained file.

## 3. Smoke test on CPU (identical on both platforms)

```sh
cortiq run kat-q4t.cmf --prompt "Write a Python function that checks if a number is prime." --max-tokens 120
cortiq bench kat-q4t.cmf
```

You should see coherent code with reasoning. Expected decode: ~16 tok/s on
a 32-core server CPU, ~7 tok/s on an M4. If the output is garbage, the
import went wrong — re-run step 2 and check its log.

## 4a. Vulkan (Linux, discrete GPU)

Headless server images (RunPod/vast-style PyTorch containers) are
compute-only and miss the GL vendor libraries the NVIDIA Vulkan driver
links against. Fix once:

```sh
apt-get update && apt-get install -y vulkan-tools libglvnd0 libegl1 libgl1 libglx0
vulkaninfo --summary | grep deviceName   # must print your GPU, not an error
```

Run — size the weight budget to your card (a 32 GB card shown; the default
is a conservative 8 GB):

```sh
CMF_GPU=1 CMF_GPU_VRAM_MB=26000 cortiq run kat-q4t.cmf \
    --prompt "Write a Python function that checks if a number is prime." --max-tokens 200

CMF_GPU=1 CMF_GPU_VRAM_MB=26000 cortiq bench kat-q4t.cmf
```

What happens: the whole 40-layer stack decodes as **one GPU submit per
token** — GDN recurrence, attention, the MoE router, the top-k expert
selection and every selected expert included. The first token pays a
one-time expert upload (~20 s with a warm page cache; a few minutes if the
model file is on a cold network disk) — keep the process alive (`serve`,
interactive `run`) rather than restarting per prompt.

Expected on an RTX 5090: **32.8 tok/s steady** vs 14.4 CPU-only on the same
machine. `RUST_LOG=cortiq_engine=info` prints
`wgpu GPU path: on (NVIDIA ... / Vulkan, discrete, weight budget ...)` and
the probe verdicts. `CMF_GPU_WGPU_GRAPH=0` reverts to per-op offload,
`CMF_GPU=0` to pure CPU.

## 4b. Metal (macOS Apple silicon)

Same file, same command — no flags beyond `CMF_GPU=1`:

```sh
CMF_GPU=1 cortiq run kat-q4t.cmf \
    --prompt "Write a Python function that checks if a number is prime." --max-tokens 200
```

Honest expectations: the MoE whole-token graph is a Vulkan/DX12 feature
today — on Metal the runtime probe arbitrates per-op GPU against CPU and
keeps whichever side wins. On an M4 with 24 GB that lands at ≈ CPU speed
(~7 tok/s): a 19.6 GB model near-saturates a 24 GB machine's memory
bandwidth, and the probe correctly refuses losing offloads. Dense and q1
models get the full Metal graph (a 27B q1 decodes at 11–12 tok/s on the
same M4); porting the MoE graph to Metal is on the roadmap.

## 5. Optional levers (both platforms)

```sh
CMF_MOE_TAU=0.9  cortiq run kat-q4t.cmf ...   # confidence-adaptive routing: ~+12% decode,
                                              # equal-or-better perplexity than fixed top-8 (CPU path)
CMF_MOE_TOPK=4   cortiq run kat-q4t.cmf ...   # fixed smaller top-k (faster, mild quality cost)
cortiq run kat-q4t.cmf --o1 all ...           # O(1)-context attention: KV+state at 4K ctx
                                              # drops 238 → 83 MB; essential at 32K+
```

Diagnostics: `CMF_DEBUG_LAYERS=1` traces per-layer hidden-state rms/max;
`CMF_GRAPH_PROF=1` prints per-token graph timings (build / encode /
submit+readback).

## Troubleshooting

| symptom | cause / fix |
|---|---|
| `vulkaninfo`: `ERROR_INCOMPATIBLE_DRIVER` | missing GL vendor libs — the `apt-get install` line in §4a |
| GPU run ≈ CPU speed on Vulkan | budget too small for the experts — raise `CMF_GPU_VRAM_MB`; check `RUST_LOG=cortiq_engine=info` |
| very slow first token | one-time expert upload from a cold disk; page cache makes the next start ~20 s |
| `unknown quant 'q4t'` | cortiq older than 0.5.24 — update |
| incoherent output | broken import — redo step 2 with cortiq ≥ 0.5.24 (earlier versions predate the qwen35moe importer) |
