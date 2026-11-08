# OrcaCyber-Nano-1.0 → CMF Turbo — journal (2026-11-08)

Owner: «сделай https://huggingface.co/orcarouter/OrcaCyber-Nano-1.0 тоже в формате cmf сделай режим турбо выжми
максимальную скорость и опубликуй на HF … сначало сделай для vulcan а потом спроси разрешение на оптимизации на mac
book пока metal загружен».

## Source

orcarouter/OrcaCyber-Nano-1.0, gated «auto» (403 for infosave until the owner clicked access, 08.11). Licence «other /
qwen». GGUF only: Mix-15.7G (15 695 148 832 B, MTP inside), Mix-13G, DFlash2-Q8_0 drafter (2 056 414 112 B — ggml-org's
dflash-Qwen3.8-27B-Q8_0.gguf is 2 056 414 752 B, 640 B of metadata apart), mmproj F16.

Header of Mix-15.7G (`scripts/gguf_peek.py`, range reads): arch qwen35, block_count 65 = 64 trunk + 1 nextn
(`nextn_predict_layers` 1), name "OrcaCyber Nano 1.0 AI Defense", tags abliterated / uncensored / qwen3_5 / multimodal.
imatrix mix: IQ4_XS 244 tensors 9.10 GB, Q5_K 106 / 3.73 GB, Q6_K 129 / 1.20 GB, Q4_K 9 / 0.95 GB (token_embd and the
whole MTP block), IQ3_S 18 / 0.69 GB (ffn_up of some layers), F32 360. output.weight Q6_K.
Chat template = Qwen3.8-27B's minus the `{% generation %}` markers (llama.cpp's jinja lacks them): same rendering.
Tokenizer: same counts as Qwen3.8-27B (248 320 tokens, 247 587 merges, pre qwen35).

**Norms:** every F32 norm read (attn/post norms of layers 0, 1, 3, 32, 63, output_norm, q_norm, MTP enorm) equals
1 + Qwen3.8-27B's bf16 w bit for bit (llama.cpp's +1 in f32). Checked the method on ggml-org/Qwen3.8-27B-GGUF Q8_0:
identical too. So no full fine-tune; any change is in the matrices (LoRA merge / abliteration) — see the delta probe.

## Engine

`cortiq import-gguf` (gguf.rs) on master 9b9e11f2:
- qwen35 dense was named `qwen3_5_moe`; the MTP block (blk.64 + nextn.*) became a 65th layer → fixed: trunk = block_count
  − nextn, `blk.64.*` → `model.mtp.layers.0.*`, `nextn.{eh_proj,enorm,hnorm,shared_head_norm}` →
  `model.mtp.{eh_proj,enorm,hnorm,norm}`, arch `qwen3_5`, header `mtp` {1 block, shared embed/head}; the MTP norms get
  the same −1. Test `qwen35_gguf_mtp_block_maps_to_model_mtp`.
- IQ3_S had no dequantizer → port of ggml `dequantize_row_iq3_s` + `iq3s_grid`; test against gguf-py on a real block
  (`iq3s_matches_ggml_reference`).
- The reconstructed GGUF tokenizer uses the GPT-2 regex; Qwen splits digits one by one and NFC-normalizes → import with
  `--tokenizer-dir` holding Qwen3.8-27B's tokenizer.json.

## Pod

root@213.173.110.201 -p 17618 (RunPod): RTX 2000 Ada 16 GB (16 380 MiB), driver 595.91, 48 cores, 251 GB RAM but
cgroup limit 31 GB, / overlay 100 GB, /dev/shm 14 GB (counts in the cgroup), Ubuntu 24.04 (glibc 2.39 → build from
source). Vulkan: GLVND line + XDG_RUNTIME_DIR=/tmp at once. Build master 400 s, branch 223 s. HF: 15.7 GB in 14 s.

## Delta probe: what OrcaCyber changed (`scripts/delta_probe.py`, eigenvalues of D·Dᵀ, D = GGUF − Qwen bf16)

| matrix | type | rel ‖D‖/‖B‖ | sv0/edge | sv1 | sv4 | spikes > 1.5 edge |
|---|---|---:|---:|---:|---:|---:|
| blk.0.ffn_up | IQ3_S | 0.140 | 1.09 | 1.08 | 1.04 | 0 |
| blk.0.ffn_down | IQ4_XS | 0.082 | 8.08 | 7.62 | 1.03 | 2 |
| blk.3.attn_output | Q5_K | 0.041 | 13.1 | 9.7 | 1.05 | 3 |

Quantization noise sits at the Marchenko–Pastur edge (≤ 1.1, flat imatrix plateaus up to 1.7 on input-side
matrices); the edit is a few spikes far above it on the matrices that write into the residual stream:
abliteration-like, consistent with the GGUF tags (abliterated, uncensored) and the identity in the answers.

## Rebuild (`scripts/rebuild.py`, tau 2.0, k 32, 6 workers, ~7 min for 18 shards)

W = B + the components of D above 2 × the noise edge (randomized SVD, spike singular values debiased for the
noise); F32 tensors compared exactly (norms in the 1+w convention, A_log, dt_bias, conv taps, GDN norm): 360 of
360 unchanged. 296 matrices edited, 210 noise only, 333 vision tensors copied from the base (convert drops them).

| kind | n | ranks | energy of D kept | max sv0/edge | GGUF rel |
|---|---:|---|---:|---:|---:|
| linear_attn.out_proj | 48 | 2:13, 3:14, 4:20, 5:1 | 0.232 | 23.1 | 0.043 |
| self_attn.o_proj | 16 | 2:1, 3:5, 4:10 | 0.241 | 21.5 | 0.043 |
| mlp.down_proj | 64 | 1:12, 2:35, 3:16, 4:1 | 0.063 | 22.8 | 0.072 |
| embed_tokens | 1 | 1 | 0.036 | 12.2 | 0.074 |
| mtp.fc / mtp o_proj / mtp down | 1 each | 2 | 0.05 | 8.1 / 7.1 / 8.8 | 0.07 |
| mlp.gate / up | 64 each | mostly 0–1 | 0.005 | 6.4 / 6.0 | 0.070 / 0.087 |
| self_attn.k / v (Q6_K) | 16 each | 1–4 (one v 10) | 0.03 | 4.9 / 4.3 | 0.020 |
| in_proj_qkv / z / a / b, q_proj | | 0–1 | ≤ 0.002 | ≤ 4.3 | 0.02–0.08 |
| lm_head (Q6_K) | 1 | 13 (plateau at 2.0: noise) | 0.013 | 2.0 | 0.018 |

GGUF-direct q4tp (import-gguf, 284 s, peak RSS 29 GB of the 31 GB cgroup, 14 315 197 544 B): "I'm OrcaCyber Nano 1.0
AI Defense Model, an AI model built by the OrcaCyber team for cybersecurity defense work…"; MTP drafts 27–61 %
accepted. (Numbers taken while the rebuild ran on the CPU: not speed numbers.)

## Quality: GGUF-direct against rebuilt (wikitext-2, 12 × 512, `cortiq ppl`, exact attention, same binary)

| file | how | ppl |
|---|---|---:|
| qwen38-27b-turbo.cmf (published, Qwen3.8-27B bf16 → q4tp RTN) | control | 8.794 |
| orca-gguf-q4tp.cmf | import-gguf Mix-15.7G → q4tp (double quantization) | 8.890 |
| orca-q4tp.cmf | rebuild (Qwen bf16 + recovered edits) → convert q4tp | 8.985 |
| orca-turbo.cmf | far-turbo of orca-q4tp | 8.985 |

The control matches August's 8.793, so the pipeline is sound and Orca's edits raise ppl over the base. That the
double-quantized file beats bf16→RTN by 1 % says the imatrix rounding of the mix survives the re-encoding better than
RTN does from continuous weights; the rebuild was a measured hypothesis that did not pay on ppl.
Both answer "I am OrcaCyber Nano 1.0 AI Defense Model…" (rebuilt: "…focused on cybersecurity defense").
`cortiq convert` of the rebuilt dir: 261 s, 14 271 151 208 B. far-turbo: 14 264 213 096 B, "far field 32,1800,4 from 8192
rows, 248 exact rows" (byte size identical to the published Qwen3.8-27B Turbo).

Trap: a chain started without the HF token cannot re-fetch the gated GGUF (import-gguf then reads the local path as an
HF spec "root/o/…" → 401). Fetch with the token through stdin, never in the chain file.

Code corpus (gguf.rs + convert.rs of the repo, first 400 000 bytes, 12 × 512): base 3.711, rebuilt 3.776, GGUF-direct
3.786. So the two Orca files trade places: GGUF-direct −1.1 % on wikitext, rebuilt −0.3 % on code (the mix's imatrix was
calibrated on `cal_imatrix.txt`, presumably prose). A wash on quality; the tie is broken by real decode with MTP drafts.
The redundant turbo-file ppl was dropped (the turbo file is exact under the seal: 8.985 = its q4tp).

## Speed, RTX 2000 Ada 16 GB (70 W, 22 SMs, 224 GB/s), Vulkan

### Real prompts, one answer a process, 220 tokens, `q27_dflash_ab.sh` arms (cortiq 0.8.63 + import branch)

| prompt | plain | MTP default | k2 | k3 | k4 | (rebuilt / GGUF-direct turbo) |
|---|---:|---:|---:|---:|---:|---|
| code | 13.6 / 13.5 | 27.5 / 28.5 | 27.5 / 26.3 | 29.6 / 31.2 | 29.1 / 31.5 | |
| review | 13.5 / 13.6 | 22.4 / 27.0 | 22.9 / 26.0 | 23.2 / 28.5 | 21.9 / 28.0 | |
| logs | 13.4 / 13.5 | 25.4 / 24.1 | 26.5 / 24.3 | 28.6 / 27.8 | 28.6 / 26.9 | |
| prose | 13.5 / 13.5 | 19.6 / 20.0 | 21.0 / 21.0 | 20.3 / 20.1 | 18.0 / 18.0 | |
| ru | 13.5 / 13.5 | 20.2 / 19.5 | 21.2 / 19.9 | 20.2 / 20.3 | 19.4 / 18.3 | |

Means: MTP default 23.0 / 23.8, k3 24.4 / 25.6. The GGUF-direct file's drafts hold more (code/review 73–83 % against
51–72 %). → **The published file is the GGUF-direct turbo** (`orca-gguf-turbo.cmf`): better wikitext ppl, faster
drafts; the rebuild stays here as a measured alternative. A round of 3 drafts: draft 8 ms + verify 95 ms + commit 4 ms
against a plain token's 74 ms. "first token ~8 s" on short prompts = 14 GB read + upload of one process (disk-bound
~1.8 GB/s on this pod); `serve` pays it once.

### Prompt reading (prefill)

Default: 74–75 tok/s at 1000–6000 tokens, 69 at 16 000 (bounded route past the 4096-row seal), GPU at 100 % and
68–69 W (the card's 70 W cap), host idle. `CMF_BATCH_TS` per 128-row chunk (coop on): gate-up|gdn-proj 374 ms,
down 148, attn-core|recur 124–163, gdn-out 36, qkv 33, o-proj 12 → GEMMs at ~10 TFLOP/s.
Traps: `bench` runs a CPU `measure_pair_fusion` (≈20 s, GPU idle) before its prompt — not a user path; `ppl` scores
token by token (~11 tok/s), not the batched path.

1. `CMF_BATCH_COOP=1` (tensor-core GEMM for the batched prompt, was opt-in "until measured on more than one card"):
   74.3 → 85.8 tok/s at 2000 tokens, same greedy text (md5 c81e385a). → default ON on NVIDIA.
2. `CMF_BATCH_K=256`: +2 % on top — not taken (bigger scratch on a full 16 GB card).
3. New `q4tp_mm_coop128_s`: the coop GEMM with a 128-row M tile (8 subgroups, 44 KB workgroup memory), so a 128-row
   chunk unpacks each weight tile once instead of once per 64-row half: 85.1–85.4 → **98.1–98.2 tok/s** (two passes),
   text identical to the old default's (md5 5e223704). Total 74.5 → 98.2 = **+32 %**.
4. Eight-k weight staging in the 128-row tile (one aligned word of nibbles and one group scale a lane): 98.3–99.3 →
   107.0–107.6 tok/s, text identical. 5. vec4 activation loads: 107.5–108.0 → 111.8–112.5, text identical.
   Parity: `wgpu_q4tp_mm_coop128_matches_coop64` bit for bit + the two existing coop tests pass on the card
   (`--features gpu`). Per 128-row chunk after: gate-up 283 ms (374), down 108 (148), attn-core|recur 127, the rest ~62.
   Not done (measured, not built): K step 64 needs > 48 KB of workgroup memory; the GDN recurrence (127 ms/chunk,
   sequential over positions) would want a chunked WY form; an f16 activation pre-pass.

### Final numbers (binary cortiq-final = master + PR #99, file orca-gguf-turbo.cmf, final.log)

| ctx | standard: first token s / decode / VRAM MiB | turbo |
|---:|---|---|
| 1000 | 9.14 / 27.4 / 14467 | 8.89 / 29.9 / 14467 (prefill 112.5 tok/s) |
| 4000 | 36.99 / 25.8 / 14883 | 36.69 / 26.7 / 14883 (109.0) |
| 16000 | wgpu panic, out of device memory (15745 MiB) | 159.13 / 26.7 / 15107, RAM 4154 MiB (100.5) |
| 32000 | wgpu panic (15779) | 323.78 / 26.3 / 15107, RAM 7314 (98.8) |
| 64000 | wgpu panic (15779) | 652.47 / 26.0 / 15123, RAM 13629 (98.1) |

Before the kernels (same file, 0.8.63 path): turbo 16000 first token 231 s at 69.2 tok/s.
Real prompts (turbo, final binary): code 13.5 → 28.2 (76 %), review 13.4 → 26.5 (70 %), logs 13.3 → 23.6 (64 %),
prose 13.5 → 19.6 (48 %), ru 13.5 → 19.5 (48 %); greedy text plain = MTP on four of five (ru differs: int8 verify).
Real answer 100 000 chars (24 600 tokens): "The last article is about the Second Battle of Naktong Bulge … September 1 to 15,
1950 … US 2nd Infantry Division and the 1st Provisional Marine Brigade … Pusan Perimeter." first token 252.3 s,
decode 20.8 tok/s (with CMF_GRAPH_SPEC_TIME logging), drafts 57 %.

Open (not done, for the owner): the standard file panics instead of failing cleanly when it outgrows the card; the
one-shot MTP depth rule loses 4–11 % against the best pinned depth (k3 ~ +8 % on a first answer), a per-device price
cache across processes would close most of it; `bench` spends ~20 s in a CPU `measure_pair_fusion`.

HF: infosave/OrcaCyber-Nano-1.0-Turbo-cmf, orcacyber-nano-1.0-turbo.cmf 14 315 202 152 B, sha256
6ff230576c09672102f1f0dbc6020721bc1bad548aa4df9bb31e3c7d1251d5f7 (MATCH by get_paths_info from the pod).
