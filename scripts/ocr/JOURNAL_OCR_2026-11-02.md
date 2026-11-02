# Unlimited-OCR-2 and MiniCPM-OCR-1.0 → CMF, 02.11.2026

Stand: RTX PRO 4500 Blackwell (32 GB), 10.2-core quota, 62 GB cgroup, 200 GB disk, wgpu/Vulkan. No-LTO fast build.

## MiniCPM-OCR-1.0

- `minicpm_ocr`: a SigLIP tower with `downsample_mode = "4x"`, so `vit_merger` is not used at all; the text model is the
  dense Qwen3.5 hybrid the engine already ran. New code is the tower and the slicing processor (`minicpm_ocr.rs`).
- The resident ViT kernels take it with identity rotations; separate q/k/v are fused into one `[3d, d]` matrix at load.
- Parity with Transformers 5.21 float32: pixels 0, rows cos ≥ 0.9999998. 300 greedy tokens identical for f16 and q8_2f.

## Unlimited-OCR-2

- Vendor infer path = `configure_dash_inference`: DASH in prefill and decode. `indexer.*` is the training stage, only
  `dash_router.*` runs. Router params come from `sparse_attention_config` (scale 128, floor 1e-4, K 12, top-M 16).
- `(tok, page)` rope: frequency pairs 0..63 on the running count inside the page, pair 63 on the page's parity. The page
  of a response token comes from the `<PAGE>` markers; prompt pages count from 1, page 0 holds BOS and the instruction.
- Pixels: Pillow's fixed-point BICUBIC (`pil_bicubic`), not torchvision's, which was 2 levels off.
- Position tables: torch antialiased bicubic is a = -0.5 (PIL's), not -0.75. Min row cosine 0.71 → 0.97 when fixed.

Traps measured on the way:
- Sequential storage loops on a GPU stall per load. The value pass as one warp over 512 rows took 2.6 ms of a 4.4 ms
  token; split over 8 warps: 1.05 ms, decode 175 → 300 tok/s. Pipeline barriers were NOT the cost (7 dispatches → 1
  changed nothing), a closing block as single-thread storage loops cost 1 ms per layer.
- A chunk of 64 rows spans fewer blocks than the local window, so all its remote blocks closed before it starts: router
  and attend run in parallel over the rows.
- `q8_2f` on the graph needed the detached shared expert (1792 wide over 896) in both graphs.
- Text against the float32 original diverges at near-ties: the vendor's own bf16 build leaves float32 at char 35 (a
  coordinate) and takes the same `image_caption` branch our files take at char 1790 of 6 pages. Not a bug.
