# Qwen3.8-27B Turbo on a 16 GB card through CUDA

The measurements of `docs/CUDA.md` (section "Turbo on a 16 GB card"), on an RTX 2000 Ada pod laid out as
`scripts/cuda/README.md` describes (`/root/cuda/{src,bin,out,data}`, models in `/root/models`).

| file | what it does |
|---|---|
| `get_wiki.py` | wikitext-2's test split as `data/wiki.test.raw` (the real prompts and the planted facts read it) |
| `final.sh BIN OUT` | `bench --core` from 1 000 to 64 000 tokens (CUDA turbo with and without drafts, Vulkan turbo, the standard file on CUDA), summaries of 30 000 and 100 000 characters, a chat that starts past the seal |
| `pk.sh` | facts planted in 60 000 and 100 000 characters, eight trials each |
| `alloc_test.cu` | how much card memory small `cuMemAlloc` calls take (the 2 MiB rounding the arena avoids) |

`CMF_CUDA_MEM=1` prints what the card holds after each plan, prompt chunk and 256th token;
`CMF_CUDA_PROF=1` the prompt's stage timers; `CMF_PREFILL_PROF=1` the seal's parts. Figures:
`scripts/cuda/lines_fig.py scripts/q27cuda/fig_turbo16_cuda.json OUT_DIR`.
