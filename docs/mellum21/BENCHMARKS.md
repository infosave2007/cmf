# Mellum2.1 CMF: release benchmark protocol

This is a release gate, not a marketing template. A number belongs on the model
card only when its raw command output, artifact hash and hardware record are
kept with the release. Do not compare a cold run with a warm run, CPU with GPU,
or different context lengths in one column.

## 1. Freeze inputs before measuring

Record these once for each candidate artifact:

```bash
MODEL='mellum2.1-12b-a2.5b-thinking-q4tp.cmf'
sha256sum "$MODEL"
cortiq --version
cortiq info "$MODEL"
cortiq verify "$MODEL"
```

Record the upstream revision and hashes used by conversion:

```bash
sha256sum mellum-upstream/config.json \
          mellum-upstream/tokenizer.json \
          mellum-upstream/chat_template.jinja
```

The upstream revision for this release candidate is
`92ddae9fc7665e9f801d141d2e5a6b2caf2460c4`. If the conversion used another
revision, publish that fact rather than relabeling the artifact.

## 2. Correctness gate

1. `cortiq verify "$MODEL"` must pass after conversion **and again after the
   artifact is downloaded from Hugging Face**.
2. Save deterministic CPU greedy outputs for a fixed prompt set. Re-run the
   same commands with the GPU selection being tested. Report whether each
   output is identical; if not, retain both outputs and explain the observed
   difference rather than calling it parity.
3. Record the converter's architecture/template/tokenizer test result and the
   exact test command. Do not claim upstream-model parity unless a published
   harness and its raw results measure it.
4. For a quantization-quality comparison, score the same held-out text and
   token budget for every profile. `cortiq ppl` provides a reproducible
   CMF-side gate; it is not a substitute for an upstream task benchmark.

Example fixed CPU/GPU smoke contract:

```bash
mkdir -p raw/outputs
PROMPT='Implement a Python function that returns the first non-repeating character.'

CMF_GPU=0 cortiq run "$MODEL" --prompt "$PROMPT" --greedy --max-tokens 256 \
  > raw/outputs/cpu.txt
CMF_GPU=wgpu XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp}" \
  cortiq run "$MODEL" --prompt "$PROMPT" --greedy --max-tokens 256 \
  > raw/outputs/vulkan.txt

diff -u raw/outputs/cpu.txt raw/outputs/vulkan.txt \
  | tee raw/outputs/cpu-vulkan.diff
```

For a Metal run, replace `CMF_GPU=wgpu` with `CMF_GPU=1`. A non-empty diff is
evidence to investigate, not a reason to hide the output. On a quantized file,
upstream BF16 and CMF need not produce byte-identical completions; label any
source-vs-CMF comparison with its actual metric and prompt suite.

Example CMF-side perplexity record (supply a redistributable held-out text
file and its SHA-256):

```bash
sha256sum evaluation/heldout.txt
CMF_GPU=0 cortiq ppl "$MODEL" --file evaluation/heldout.txt --tokens 4096 \
  | tee raw/ppl-cpu.txt
```

## 3. Throughput protocol

Use the same artifact, context length, generated-token count and benchmark mode
for every backend. The command below uses Cortiq's synthetic 512-token context,
256 generated tokens, greedy core timing, and EOS suppression. It reports
machine-readable JSON. Run five independent processes so that run-to-run
variation is visible; `cortiq bench` itself performs an untimed warm-up.

```bash
mkdir -p raw/bench

# CPU reference
for i in 1 2 3 4 5; do
  CMF_GPU=0 cortiq bench "$MODEL" --ctx 512 --tokens 256 --core --ignore-eos --json \
    | tee "raw/bench/cpu-${i}.json"
done

# Linux Vulkan / wgpu candidate
for i in 1 2 3 4 5; do
  CMF_GPU=wgpu XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp}" \
    cortiq bench "$MODEL" --ctx 512 --tokens 256 --core --ignore-eos --json \
    | tee "raw/bench/vulkan-${i}.json"
done

# macOS Metal candidate
for i in 1 2 3 4 5; do
  CMF_GPU=1 cortiq bench "$MODEL" --ctx 512 --tokens 256 --core --ignore-eos --json \
    | tee "raw/bench/metal-${i}.json"
done
```

Do not publish a GPU value when the JSON or stderr says the graph fell back,
weights were repeatedly uploaded, or the selected adapter is unknown. Keep
`CMF_GPU_VRAM_MB` unset for an automatic-budget result; if it is set, record
the exact value and label the result as a constrained-budget run.

For production-loop measurements, repeat the same protocol **without**
`--core`, place it in a separate table, and do not present it as the same
metric as core decode.

## 4. Hardware and software record

Save the output that applies to the test machine; unavailable commands are
allowed to fail, but should be noted rather than silently omitted:

```bash
{
  date -u +'%Y-%m-%dT%H:%M:%SZ'
  uname -a
  cortiq --version
  cortiq gpu
  command -v lscpu >/dev/null && lscpu
  command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  command -v vulkaninfo >/dev/null && vulkaninfo --summary
  command -v system_profiler >/dev/null && system_profiler SPDisplaysDataType
} | tee raw/hardware.txt
```

Also record the OS image/container, driver, adapter name, RAM, declared VRAM,
Cortiq version, artifact SHA-256, environment variables and whether the model
was reloaded between samples.

## 5. Card table to fill from raw results

Publish medians over the five JSON files, plus the range. Values below are
intentionally blank until measured.

| Artifact SHA-256 | Backend | Adapter / CPU | Context | Mode | Prefill tok/s | Steady decode tok/s (median; range) | TTFT | VRAM / RAM observation | Raw files |
|---|---|---|---:|---|---:|---:|---:|---|---|
| **FILL** | CPU | **FILL** | 512 | core | **FILL** | **FILL** | **FILL** | **FILL** | `raw/bench/cpu-*.json` |
| **FILL** | Vulkan | **FILL** | 512 | core | **FILL** | **FILL** | **FILL** | **FILL** | `raw/bench/vulkan-*.json` |
| **FILL** | Metal | **FILL** | 512 | core | **FILL** | **FILL** | **FILL** | **FILL** | `raw/bench/metal-*.json` |

Attach `measurements.template.json` as `measurements.json` only after replacing
all `FILL_ME` values and adding the raw-file paths. Never convert an empty
field into a performance claim.
