# DeepSeek-V4.1-Flash on a pod: the Vulkan path and the numbers

The model is ours: `infosave/DeepSeek-V4.1-Flash-1017-cmf` (FP4 463 GiB,
q2tp 378 GiB, q4tp 464 GiB; 755B params, 40 layers, MoE with mmap-backed
experts). This page is what a pod needs before the model runs at speed.

## 1. The container's Vulkan must work first — and it usually does not

`cortiq gpu` printing "адаптеров не найдено" is not "no GPU here". On a
compute-only container the NVIDIA Vulkan ICD fails to initialize:

* `libvulkan.so.1` is present and `/etc/vulkan/icd.d/nvidia_icd.json` names
  `libGLX_nvidia.so.0`, which even exports `vk_icdGetInstanceProcAddr`;
* the ICD's `vk_icdNegotiateLoaderICDInterfaceVersion` returns
  `VK_ERROR_INITIALIZATION_FAILED`;
* `strace` shows the ICD loading **libEGL.so.1**, which the image does not
  ship (no GLVND dispatch library).

So the ICD cannot start, wgpu reports **zero adapters**, and the engine falls
back to the CPU — silently, except for one warning line.

```bash
apt-get update -q
apt-get install -y libegl1 libgl1 libglx0 libgles2 libglvnd0
vulkaninfo --summary | grep deviceName      # must show the GPU
cortiq gpu                                    # must list a Vulkan adapter
```

Measured on one RTX PRO 6000 (96 GB) with the same FP4 model:

| | decode | note |
|---|---|---|
| without the GLVND dispatch libs | ~0.6 tok/s | CPU fallback, no error anywhere else |
| with them | **8.7 tok/s** | Vulkan path |

A newer Vulkan loader does **not** help — the ICD, not the loader, was failing.

## 2. The first run is not the speed

Run the benchmark twice; take the second run. The first run compiles the
shaders and the compile lands inside the timed decode:

```
run 1: 1.0 tok/s
run 2: 7.8 tok/s
run 3: 8.5 tok/s
```

The pipeline cache is a 3.9 MB blob next to the model
(`CMF_PIPELINE_CACHE=<path>` pins it). Keep it — it is the difference between
a one-off and a service.

## 3. The knobs that matter

`CMF_DSV4_FETCH_MAX` — how many missing experts a layer may pull in before the
frame runs (the FreeToken-style fetch). `CMF_DSV41_POOL_PCT` — the share of
VRAM the expert bank takes.

| configuration | FP4 decode | q2tp decode | FP4 prefill |
|---|---|---|---|
| default | 8.7 | 10.1 | 3.2 |
| `PCT=85` | 8.7 | 10.3 | — |
| `FETCH=8` | 8.9 | — | 11.1 |
| `PCT=85 FETCH=8` | 8.7 | 10.3 | 11.8 |
| `PCT=90 FETCH=16` | **9.0** | **10.5** | 11.4 |

The decode gains 3–4 %; the **prefill gains ×3.7**. Since 0.8.28 the quota
follows the link (8 at ≥10 GB/s, 1 otherwise).

`cortiq bench --bw` measures the machine's two offload arms and is the ceiling
of the whole approach:

```
copy to RAM:    107.5 GB/s   (the CPU arm's ceiling)
upload to VRAM:  40.2 GB/s   (the fetch arm's ceiling)
storage read:     2.9 GB/s
```

A 463 GiB model on a 96 GB card streams most of its experts every token, so the
fetch arm (40 GB/s) is the wall — not the arithmetic.

## 4. What DeepSeek's own stack says about this workload

DeepInfer (`deepseek-ai/DeepInfer`, Apache-2.0) is the inference stack behind
DeepSeek-V4.1-Flash. Its relevant pieces:

* **MoE EP + EPLB** — experts placed across GPUs with a frequency-balanced
  layout. This is the main lever, and it needs more than one GPU: the host of
  this pod has four (`/dev/dri/renderD132…135`), the container was given one.
  Four GPUs mean 384 GB of VRAM against a 463 GiB model: the offload nearly
  disappears. **Recreate the pod with all four GPUs.**
* tilelang kernels (FP4/FP8 grouped GEMM, flash-MLA) are CUDA/SM100-only; our
  path is Vulkan/wgpu, so they are a reference rather than a module.
* SWA bounded replay, batch-invariant inference and deterministic sampling are
  algorithms and transfer — we have our own equivalents.

## 5. A working pod recipe

```bash
# one-time: the Vulkan dispatch libraries
apt-get update -q && apt-get install -y libegl1 libgl1 libglx0 libgles2 libglvnd0
vulkaninfo --summary | grep deviceName

# the model (xet: ~1.5 GB/s, 463 GiB in ~5 minutes)
HF_HUB_ENABLE_HF_TRANSFER=1 python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('infosave/DeepSeek-V4.1-Flash-1017-cmf','dsv41-1017-fp4.cmf',local_dir='/root/models')"

# speed: two runs, take the second
for i in 1 2; do
  CMF_DSV41_POOL_PCT=85 CMF_DSV4_FETCH_MAX=8 \
    cortiq bench --tokens 40 --ignore-eos /root/models/dsv41-1017-fp4.cmf | grep -E 'tok/s steady'
done

# sanity: the answer is still an answer
CMF_DSV41_POOL_PCT=85 CMF_DSV4_FETCH_MAX=8 \
  cortiq run --prompt 'The capital of France is' -n 24 /root/models/dsv41-1017-fp4.cmf
```

Expected: ~9 tok/s decode on one RTX PRO 6000, ~11 tok/s for the q2tp variant,
and the answer "The capital of France is **Paris**."
