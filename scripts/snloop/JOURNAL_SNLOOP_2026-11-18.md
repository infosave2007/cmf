# SenseNova-Loop-9B → CMF (release 2026-11-17), Turbo, CUDA/Vulkan/Metal

Started 2026-11-18. Worktree /Users/oleg/dev/cmf-snloop, branch sensenova-loop9b from master 64b8d3a2 (0.9.4).
Pod: RunPod RTX A5000 24 GB, 96 CPUs, 503 GB RAM, 100 GB overlay disk; work dir /root/sn (dl, src, out, ref, logs, bin).
Address only in ~/dev/snloop-runs/p (never committed).

## Release vs Preview (HF config/index diff)
- modeling_sensenova_loop.py, chat template, generation_config: byte-identical.
- rope_parameters: `rope_type: yarn`, factor 2.0, original_max_position_embeddings 262144; max_position_embeddings 524288
  (Preview: default rope, 262144). Transformers applies YaRN at every length: inv_freq indices >= 22 of 32 halved
  (ramp 14..22), cos/sin x 1.0693 (attention_scaling printed by the reference).
- MTP head moved out of the index into `mtp.safetensors` (15 tensors, 486 MB bf16): fc 4096x8192, one gated full
  attention layer, MLP 12288, norms. Convert used to drop it.
- New tensor `model.language_model.loop_gate.logits` [4096] f32, constant -3.8918 (= logit 0.02); the reference never
  reads it.

## Convert changes
- `mtp_sidecar_shard`: a sharded checkpoint's `mtp.safetensors` outside the index joins counts, names and the open list.
- sensenova_loop: MTP kept (canon `model.mtp.*`), `loop_gate.logits` dropped.

## Loop exit router (Dory-style), verdict: rejected
loop_probe.py, bf16 on the A5000, wikitext-2 test first 1024 tokens, teacher-forced:
- 2 passes ppl 10.138; 1 pass everywhere 19.699; top-1 agree 68.7 %; KL(2||1) mean 1.06, median 0.43.
- Mixed (a token leaves after pass 1; its pass-2 cache rows = its pass-1 projections; hidden after the block = pass 1's):
  share 0.50 oracle 11.474 (+13 %) / random 14.002; 0.70 oracle 13.044 / random 17.015; 0.85 oracle 15.754 / random 18.212.
- Dory lost 1.9 % at 85 % one-pass tokens. Here even the oracle loses 13 % at 50 %: the second pass and the per-visit
  caches matter. No router; speed comes from the MTP head (lossless) and the far field.

## Parity and the YaRN decline (fast build)
- CPU f16 (CMF_GPU=0 CMF_SDOT=0), unfolded build: text / haiku / code greedy = Transformers f32, word for word.
- First GPU run: "wgpu token graph declined: scaled RoPE without per-layer geometry" -> per-op path on CUDA:
  q8_2f 0.7 tok/s, first token 51.8 s (text right). The uniform graph arms decline a rope scale != 1.
- Fix `loader::fold_yarn_scale_into_qk_norms`: q/k norms run before RoPE and RoPE is linear, so the YaRN cos/sin scale
  (1.0693) moves into the norms' 64 rotary channels (Gemma style: s*(1+w)-1), rope_scale = 1, MTP block too.
  `CMF_YARN_FOLD=0` keeps the old way. YaRN inv_freq: Python port of `yarn_inv_freq` = HF inv_freq exactly (low 14, high 22).
- After the fold, CUDA q4tp RTN haiku: decode 45.7 tok/s, first token 2.6 s, MTP drafts on (4 of 17 accepted).

## Parity after the fold (fast build, A5000), greedy 64 tokens, no thinking
- CUDA q8_2f: text / haiku / code = Transformers f32. q4tp RTN: text, code = ref; haiku other words.
- Vulkan: the same texts as CUDA for both files. Speeds over 64 tokens (MTP on q4tp): CUDA q4tp code 69.2, Vulkan 37.0;
  q8_2f CUDA 28.9, Vulkan 24.9.
- MTP survey CUDA q4tp RTN, 256 tokens (mtp_survey.sh): code 89.5/94.5/91.9 vs plain 51.7/51.5/51.2 (accept 47-48 %);
  general 77.2/78.9/62.0 vs 52.0 (20-42 %); text identical with and without drafts. q8_2f: drafts off by the engine's
  policy (q4tp trunk only).

## Far field on the loop (Turbo groundwork)
- With CMF_O1 env on CUDA the prompt ran on the host (180 % CPU, a 1500-token prompt not done in 100 s): the device
  prefix route required `phys_layer(li) == li` (added 30.10 with the route, before any looped model used it).
- Relaxed to `kv_cache.layers.len() == num_layers` (one cache per executed layer: SenseNova keeps one per visit).
  CUDA q4tp RTN: prefix 1024 rows on the batch graph 3.0 s, 14 virtual full layers sealed on the card in 0.09 s,
  the rest in o1 batches, decode 44.6 tok/s through the far field. 4905-token prompt: prefix 4096 rows 5.98 s,
  first token 9.2 s, decode 43.4. Without exact rows the password at row ~700 of 1352 is lost (expected).
- Feature dump: 112 features a row (14 visits x 4 KV heads x 2). Fit: q27_feat.py 20 prompts x 21000 chars, seal 4096.
- Exact-row fit (keep_sn9.bin, 112 weights): leave one prompt out best 0.00 % mean 0.11 % worst 0.28 % of rows
  (27B 0.22 / 0.96). Vulkan far field on the loop works too (prefix 1024 rows 15.3 s under GPTQ load; to re-measure).
- TRAP: the pod's cgroup memory limit is 50 GB (memory.limit_in_bytes), not the 503 GB `free` shows: the f32
  reference (38 GB) died silently next to GPTQ. hero.png is 4465x2280 (~10k vision tokens): use hero1024.png.

## Quant variants (CUDA ppl; wikitext-2 test first 2048 tokens; chat_eval = the q8_2f's own answers to 12 held-out
prompts in its template, 2484 tokens)
- q8_2f 11.784 / 2.094; q4tp RTN 11.347 / 2.228; q4tp GPTQ v1 12.462 / 2.200 (v1 = the loop calibration bug:
  layers 28-31 rounded, loop layers saw only first-visit inputs).
- GPTQ fix `calib_upto_visit`: chunk forward to the last visit of its top layer. v2 passes 75-160 s a window
  (forward through all 56 visits), ~65 min in all on the A5000 with --calib-chunk-layers 8, 16384 tokens.

## Trial turbo (far-turbo of q4tp RTN, keep_sn9.bin, 32,1800,4 from 8192, 248 exact rows), CUDA under GPTQ load
- facts 60 000 chars (13.2-15.1k tokens): 4/4 (reference bf16 4/4), first token 18-20 s, decode 27-38 tok/s.
- Vulkan, one prompt: OK, first token 49.8 s, decode 11.6 (under load; re-measure).
- TRAP: facts_engine.py first judged the whole stdout, which echoes the prompt (fact included): every answer
  "OK". Now only the text after the question counts. 240 000 chars, first prompt: the answer was "0120" (miss).
- GPTQ v2 (all visits): wiki 2048 12.950. Transformers f32 on the same 2048 tokens: 11.900 (sn_ref.py, CPU).
  So q8_2f 11.784 (-1 %), q4tp RTN 11.347 (-4.6 %), GPTQ v1 12.462, v2 12.950 (+8.8 %): one wiki window is noise-
  level for picking; chat texts decide (see below). `cortiq ppl --windows 8` on CUDA runs one host thread
  (>20 min a file) — dropped.
- Image prompts with the MTP head: "embedded prompts require ... without ... a generic MTP head" -> every image
  request failed once the file carried MTP. Fix in `generate_from_embeds`: the head is set aside for that request
  (state is rebuilt from the next prompt), and a turbo file applies its length gate first.
  Image (hero1024.png, 533 prompt tokens) q8_2f: CUDA and Vulkan start as the f32 reference and part at near ties
  after 7 / 18 words; CPU f16 check running.
- Facts at 240 000 chars (~53k tokens), CUDA: standard q4tp RTN 4/4; trial turbo (short-prompt fit) 2/4
  ("0120" for 012060; one "I can't provide the password."). Refit on 24 prompts x 60 000 chars (wiki train,
  CMF_O1_PREFILL=8192, rows scored up to len-512): LOO best 0.00 mean 0.04 worst 0.16 % -> keep_sn9_long.bin.
- Chat texts (CUDA ppl): chat_eval (2484 tokens) q8_2f 2.094 / RTN 2.228 / GPTQ v1 2.200 / GPTQ v2 2.160;
  chat_eval2 (30 held-out answers of the q8_2f, 4096 tokens) RTN 2.000 / GPTQ v2 1.942.
  Choice: q4tp = GPTQ v2 (chat calibration 50 % + wiki 50 %, every visit of the loop), the half of the chat gap.
- Image, CPU f16 (host float activations): 48 greedy tokens = Transformers f32 word for word. GPU near-ties only.
- Turbo with keep_sn9_long.bin, 240 000 chars: 3/4 (the miss: "I can't provide the password."), standard 4/4.

## Final chain (release build, GPTQ v2 q4tp, keep_sn9_long.bin), CUDA, RTX A5000
- Private backup verified: q4tp 1ad6e559…, turbo 392a246e…, keep_sn9_long.bin da4ccd4c… (q8_2f 8cebe012… earlier).
- bench --core CMF_MTP=0, 128 tokens. Standard q4tp: 1k 1.47 s/51.9/7.77 GB; 4k 2.33/50.4/7.99; 16k 10.14/47.1/8.85;
  32k 22.62/43.5/10.03; 64k 53.02/38.1/12.19; 128k 141.86/30.5/16.48. q8_2f: 1k 0.70/34.0/9.06; 4k 2.38/33.7/9.28;
  16k 10.25/32.1/10.14. Turbo: 16k 16.47/50.8/8.32; 32k 34.41/50.4/8.32; 64k 70.3/50.5/8.32; 128k 139.91/49.9/8.32
  (host RSS 23.7 GB); 250k with CMF_O1_SPILL 264.59/50.4/8.32 (spill 27 GB, removed after the rebuild).
- Real answers (mtp_survey, 256 tokens): code 88.1 (74.4-98.9) vs 51.0; general 57.9 (52.7-62.4) vs 49.8;
  instructions (short) 38.9 vs 41.6; text identical everywhere.
- Facts CUDA: turbo 60k/120k/240k chars 4/4 each; standard 4/4 each.

## Vulkan (release build), RTX A5000
- Standard q4tp bench (CMF_MTP=0): 1k 1.08 s/33.2 tok/s/4.78 GB; 4k 2.99/32.3/5.34; 16k 11.93/31.3/6.75;
  32k 25.51/28.8/8.47; 64k 60.23/24.3/12.10; 128k 163.33/19.3/19.37. q8_2f: 1k 2.08/25.2/8.21; 4k 8.35/24.4/8.67;
  16k 33.71/23.1/9.96 (q8_2f prompt ~480 tok/s on wgpu vs 1336 for q4tp).
- Turbo: 16k 38.68/33.1/5.74; 32k 67.89/33.1/5.74. Prompt ~470 rows/s vs 1250-1340 standard.
  Profile at 16k (CMF_PREFILL_PROF): exact prefix 8192 rows 19.8 s = chunks 13.6 s (600 rows/s) + queries off the
  card 3.6 s (CUDA gathers them on the card: 0.44 s) + K/V off 1.4 + into the cache 1.1; archive seal 3.7 s; sealed
  chunks ~460 rows/s; rebuild 3.9 s. Chunk size is not it: CMF_BATCH_K default/512/256 -> 43.8/40.3/38.9 s.
  Follow-up: the wgpu turbo prompt path (query capture on the card, the prefix's chunk graph).
- Turbo Vulkan (cont.): 64k 155.68 s/33.1/5.74 GB; 128k 318.66/33.3/5.82; 250k (spill) 602.36/33.5/5.82.
