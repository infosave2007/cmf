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

## Metal, M4 (10-core GPU, 24 GB MacBook Air, no-LTO fast build)

Baseline before the Metal work: MiniCPM-OCR already rode the GDN token graph (95 / 69 tok/s) with its tower at 1.2 s a
view and the prompt on the host walk (6.8 s); Unlimited-OCR ran on the CPU: decode 71 tok/s, a tiled page's encoder 20 s,
its prompt 12.6 s.

- `gpu_metal/vit.rs`: towers resident on `eg_mm_nt/nn`. SigLIP 0.49 s a view, bound by the GEMM (34 GF a layer at the M4's
  2.7 TF/s). SAM tile 0.32 s, view 1.06 s; CLIP 0.04 / 0.09 s. Rows against the host path: cos ≥ 0.9999998.
- The row cosine against Transformers has the same 0.97 minimum at rows 802 and 854 as on Vulkan and on the host: not the
  kernels.
- `vision_gemm_nt` took SigLIP's k = 588. The tile GEMM needs k on a grid of 32: it read 20 floats past every row, and past
  the end of the scratch on the last one, so one row of the embedded patches held 6.7e19. The per-op tower had hidden it
  behind a mostly right text (common 1175 of 1517 chars). Refused now; text identical.
- The DASH kernels are the WGSL ones with the sequential part of the router on one thread: fine here, the cost is in the
  matvec. Decode on the token graph: q4tp 112–138, q8_2f 95–105 tok/s, against 71 on the CPU, 137 tok/s at 24 pages.
- Chunk prefill: 64 rows, 65 ms of GPU, 79 % of it the routed experts. 909 rows 1.4 s, 6581 rows 8.9 s.
- Rows graph on a q8_2f prompt: a matvec per position took 2511 rows to 19.4 s, worse than the host walk's 7 s. The f32-tile
  GEMM over the weight's own field: 2.5 s, text identical.
- Text: Unlimited-OCR q8_2f identical to float32 on 300 tokens of a page (1121 chars), q4tp ratio 0.986; on six pages both
  stay on the bf16 branch (0.993, 0.997). MiniCPM-OCR q8_2f identical, q4tp same text and 48 chars short as on Vulkan.
- `CMF_GPU=wgpu` on the Mac keeps the host router: the device's and the host's router state are not interchangeable.
- Thermal: the Air loses up to 20 % over a long run (q8_2f MiniCPM 59–75 tok/s). Numbers in the cards are after a pause.
- Peak RSS: Unlimited-OCR 6.17 / 4.73 GB, MiniCPM-OCR 3.96 / 3.71 GB (q8_2f / q4tp).

