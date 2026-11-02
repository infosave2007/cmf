# Four models → CMF (31.10.2026)

Worktree `~/dev/cmf-four` (branch `four-models`, from origin/master dc522a1f). Scripts in the repo: `scripts/four/`.
Pod: RunPod A40 (46 GB, cgroup 50 GB, 96 cores, 200 GB disk). Work dir `/root/four`
(`dl/` sources, `out/` CMF files, `ref/` Transformers references, `logs/`), checkout `/root/cmf` synced by
`~/dev/four-models-runs/sync.sh` (changed files only, ~3 s; a full rsync took 2 min).

## Sources (checked on HF, all Apache-2.0)
- sensenova/SenseNova-Loop-9B-Preview: 38 GB (F32 safetensors), Qwen3.5-9B hybrid (GDN + gated full attn), loop over
  layers 4..28 ×2 (56 virtual layers), mHC 2 streams on 4..28 (Sinkhorn 20), Qwen3-VL-style tower (27×1152, no
  deepstack), interleaved MRoPE 11/11/10. Transformers ≥ 5.12.
- microsoft/Mage-VL-Cowork: 9.5 GB bf16, llava_onevision2 = Qwen3-4B text + OneVision ViT (24×1024, 3-axis RoPE 4:6:6
  with INTERLEAVED rotate_half, LayerNorm, exact GELU) + merger. 1-D text positions.
- speridlabs/idol: 3.1 GB, image→3D Gaussian splats: DiT flow denoiser + octree decoder + Gaussian decoder + BiRefNet,
  plus two GATED encoders (DINOv3 ViT-H+/16, FLUX.2 VAE). Not started (see below).
- ostris/primal_vae: 2 × 321 MB fp32 (2x1, 2x2), conv U-Net, 80 M params each, channel RMSNorm, dilations 1/2/4.

## Traps found
- Transformers 5.22 loads Mage-VL with the WHOLE text model missing (random init, only a MISSING table in the log):
  its `_checkpoint_conversion_mapping` is not applied. Reference runs in `/root/venv457` (transformers 4.57.3, the
  version in the model config). SenseNova needs ≥ 5.12 → system python (5.22).
- Mage-VL's transformers path crashes without explicit `patch_positions`; raster `(t, row, col)` positions give the
  right answers (descriptions match the images).
- `hf download --exclude a b` treats `b` as a filename: repeat `--exclude`.
- `cortiq convert` takes `--model/--output`, `cortiq run` takes the model positionally.
- f16 CMF on CPU decodes single-threaded (known): CPU parity of text goes through q8_2f, f16 only for the tower rows.
- primal_vae checkpoint carries an I32 `arch_spec` tensor: the generic safetensors reader refuses I32 → own reader.

## Results so far
- primal_vae: `cortiq convert --model …/primal_vae_2x2.safetensors --output x.cmf` (f16 kernels, 160.9 MB each),
  `cortiq vae --mode encode|decode|roundtrip`. PSNR 512² (ref f32 → cortiq f16): 2x1 32.12/22.06/21.34 →
  32.13/…; 2x2 37.50/27.09/27.41 → same; latent rel err 1.4e-3. A40 wgpu: encode 0.18 s, decode 0.08 s (CPU 4.2/3.6 s).
- Mage-VL tower (f16 CMF, CPU) vs Transformers f32: pixels 2e-7, rows cosine 1.000000 (rel 1.6e-5).
  Converted: f16 9.50 GB, q8_2f 5.10 GB, q4tp 2.98 GB.
- SenseNova converted: f16 18.86 GB, q8_2f 9.92 GB, q4tp 5.62 GB.

## Engine work
- `onevision2.rs` (Mage-VL tower/prompt), `gpu::vision_gemm_nt` (Metal: egemma2 `eg_mm_nt` one-GEMM door; wgpu:
  `gemm_nt_f32`), CLI `--image` + server image parts.
- `primal_vae.rs` + resident device plans `gpu_metal/primal_vae.rs`, `gpu_wgpu/primal_vae.rs`; `cortiq vae`.
- `sensenova.rs`: plan, mHC routing (host), MRoPE positions (shifted so decode is plain), image prompt;
  pipeline `layer_map`, mHC hooks in the token and batched CPU paths, `attention::MropeView`;
  mHC kernels in the wgpu token graph (`gpu_wgpu/mhc.rs`) and the Metal token graph (`gpu_metal/mhc.rs`).
- idol: not started — six networks, two of them gated, a sparse-octree pipeline and PLY output: a new runtime class.

## Parity (31.10, afternoon)
- SenseNova CPU q8_2f: text "The capital of France is Paris." = ref; haiku = ref (20 tokens); image (img01, 256 tokens)
  48 greedy tokens IDENTICAL to Transformers f32; tower rows cosine 1.000000 (rel 1.6e-5), pixels 1e-7.
- SenseNova wgpu (A40): token graph with mHC kernels gives the same text; batched mHC prefill → haiku = ref.
  Image with the batched MRoPE prefill: one word differs ("stand"/"stands") with the coop (tf32-class) tower GEMMs,
  identical with CMF_COOP=0 (exact f32 tower). First token 25.9 s (host MRoPE prefill) → 8.3 s.
  Decode q8_2f 20–23 tok/s, q4tp 31.8 tok/s (CPU 4.2).
- Mage-VL: CPU q8_2f image = ref (48 tokens), q4tp same content other words; wgpu q8_2f image = ref, decode 48 tok/s.
  Tower GPU rows cosine 0.99998 (min 0.999); tower 3.1 s for 1024 patches (gemm 1.82 s) — GEMM door slow, check.
- primal_vae Metal (M4): PSNR 37.53 / 22.07 dB (= ref); 512² encode ~0.45 s, decode ~0.35–0.44 s; CPU 1.85 / 2.83 s.

## Engine additions since
- Batched mHC in the wgpu batch graph (route one workgroup per row), MRoPE angle tables per row in the batch graph's
  per-row attention loop (`set_batch_mrope`), embedded prompts (vision rows) ride the batch graph when
  `generation_batch_k > 0`. Graphs refuse to start or stop inside an mHC range.
- Vision towers cached per model between requests.
- Trap: `pkill -f test_mage.sh` killed my own ssh (the pattern matched the remote bash argv) — kill by PID.
- Trap: f16 CMF does not run on the GPU graphs → publish q8_2f + q4tp only.

## Vulkan final (A40, release build, before the pod went off)
- Mage q4tp decode 90.2 / q8_2f 66.8; 2k prompt 863 / 855 tok/s (decode at 2k 64.9 / 53.2).
- SenseNova q4tp decode 29.1 / q8_2f 21.3; 2k prompt 338 / 293 tok/s (decode at 2k 27.8 / 20.8).
- Screenshot 2206×1048 (9108 patches, 2277 tokens): Mage tower 4.8 s, SenseNova 5.9 s.
- primal 1024² encode 0.40 / decode 0.29 s (CPU ~13 s); 512² 0.20 / 0.08 s.
- PPL wikitext-2 2048: Mage ref 121.83, q8_2f 125.56, q4tp 128.68; SenseNova q8_2f 13.435, q4tp 13.203.
- Defaults: Mage q8_2f, SenseNova q4tp. Backup: infosave/cmf-experiments/four-models-2026-10-31/.

## Metal (M4, 24 GB)
- SenseNova q4tp: Metal token graph covers all 56 layers with mHC; decode 11.1 tok/s; 512-token prompt 80 tok/s;
  tower 512² 1.2 s, screenshot 17.9 s (attention 11.9 s: hd 72 wastes most of the second 64-row PV tile).
- Mage q4tp: decode 35.6 tok/s, 2k prompt ~295 tok/s; tower 512² 0.69 s (rows cosine vs CPU min 0.999998),
  screenshot 10.8 s (attention 6.6 s, GEMM 3.8 s).
- `CMF_DIT_FLASH=1` (Metal flash attention) is 2.6× slower on both towers (17.3 s vs 6.55 s) — stays off.
- primal: 512² 0.34 / 0.32 s, 1024² 1.27 / 1.23 s (scalar tile; simdgroup conv 2.7 s → opt-in).
- False alarm: "CPU batched ppl ≠ seq at 128 tokens" — re-measured on the merged build, CPU batched = CPU seq = Metal
  (128 tokens 5.254, 256 tokens 5.273). Greedy text of long prompts may still diverge after a few dozen tokens
  between batched and per-token prefill (summation order, q4tp near-ties).
