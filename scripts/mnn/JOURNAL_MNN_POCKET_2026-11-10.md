# MNN-Pocket-0.3B family → CMF (10.11.2026)

Sources: taobao-mnn/MNN-Pocket-0.3B (+ -Base, -SFT, -DFlash2, -Dec, -GGUF Q4_1 = the QAT INT4 weights).
Qwen3ForCausalLM, 28 L, 896/2560, GQA 14/2, hd 64, vocab 65 536, tied (lm_head shipped as a byte copy).
Worktree ~/dev/cmf-mnn, branch mnn-pocket from master 847fc6f7 (0.8.65). Run dir ~/models/mnn-pocket.
HF: infosave/MNN-Pocket-0.3B-cmf (PRIVATE), 9 files + README, each verified by size + sha256 (tree API).

## Engine / converter changes (commit 817db84d)
- tied lm_head byte copy not written: q8_2f 375 → 316 MB, f16 ppl identical (4741.094).
- streaming UTF-8: `decode_token_stream` keeps an unfinished character's bytes; 44 of 192 greedy answers
  had U+FFFD (65k vocab splits CJK and ₆ over tokens). run / serve SSE / net split path.
- DFlash2 sidecar: SpecForge config has no `is_causal` and a top-level `block_size`; tied base accepted.
- `cortiq decide-pointer` + `decision.pointer.*` kept f32 by the converter (Dec model, see below).

## Quality (ppl; source = transformers f32 on the same ids, scripts/mnn/ref_ppl.py)
chat_eval.txt = 9 925 tokens, the bf16 instruct's own sampled answers to 48 EN/ZH requests (chat_eval.py),
16×512; WikiText-2 test 12×512.
| file | chat src → file | wiki src → file |
| instruct f16 | 16.111 = 16.111 | 4741.091 = 4741.094 |
| q8_2f | 16.11 → 16.53 (pod 16.53) | 4741 → 4736 |
| q4tp from QAT, attn+emb q8_2f | QAT f16 13.57 → 13.465 (pod 13.463) | 2733 → 2185 |
| Base q8_2f | 5.369 → 5.404 | 18.245 → 18.31 |
| SFT q8_2f | 10.651 → 10.21 | 783.2 → 767.1 |
The instruct is overconfident (30+ nat misses on raw text). 8-bit variants (q8, emb f16, emb q8) all within
noise by greedy prefix agreement (61–66 %); q8_2f kept as instructed.
q4tp from bf16: wiki calib 23.74, wiki+chat 22.02, +attn q8 18.60 — all bad. From the authors' QAT weights
(GGUF Q4_1 → import-gguf f16 → dequant → raw_to_st.py → convert --sampling): 15.24 plain, 13.48 attn q8,
13.465 attn+emb q8_2f (adopted: Metal 2k TTFT 0.67 vs 1.19 s with plain q8 attention, decode −8 %).
QAT model ≠ bf16 model (greedy 2/48 identical), ppl lower on both corpora.
Sampling in the files: temperature 0.8, top-p 0.8, top-k 20, rep 1.0, loop_stop=1 (bf16 loops on ``` after
code blocks in transformers too, 6/6 seeds; loop_stop ends it with finish `loop`).

## Mac (MacBook Air M4 24 GB), bench --core
q8_2f Metal 161 tok/s, 2k 0.69 s / 144, 8k 5.9 s / 101, RSS 0.68 / 1.31 GB; CPU 178, 2k 2.12 s / 104.
q4tp Metal 216, 2k 0.67 s / 183, 8k 5.7 s / 122, RSS 0.52 / 1.18 GB; CPU 191, 2k 2.04 s / 110.
Trap: on Metal `prefill tok/s` of bench is a separate forward_ids; the generation prompt is TTFT.
Metal q8_2f decode loses to the CPU: 29 submits per token (one per layer + head) at 6.2 ms/token.

## Pod RTX 2000 Ada 16 GB (root@157.157.221.29:57821, /root/mnn), Vulkan
q8_2f 318–338 tok/s, 2k 1.65 s / 292, 8k 10.5 s / 224, VRAM +435 / +705 MiB.
q4tp 389–403, 2k 0.76 s / 342, 8k 7.0 s / 256, +355 / +593 MiB. Base, SFT = q8_2f speed.
8k prefill (779 tok/s) slower than the M4's (1390): long-prompt attention on wgpu is the limit.
Pod-converted Base/SFT/sidecars: every tensor hash equal to the Mac files (header differs: source path).
The first pod (213.192.6.73:40134) was shared with chainQ — its numbers are not used.

## DFlash2 sidecar (q4tp, 90 MB, bound to its base by trunk hash)
Native Metal: no solo DFlash path (needs MTP head). Vulkan greedy, 4 prompts: acceptance 94/60/32/73 %
(q8_2f), 81/68/37/73 % (q4tp), but 320 → 120–209 tok/s, 383 → 139–243: the block stays on the host
(dflash_dev: hs % 256 and q4 cols > 2048 required) and a round costs more than ~3 ms plain tokens.

## Dec (decision model)
Branch mask (state + own branch, positions continue the state) == a causal pass over state+branch, so
`decide-pointer` runs encode_hidden_states per question. dec_merge.py: LoRA merged in f32 (scale 2.0,
196 pairs), head + temperature as decision.pointer.* f32. dec_ref.py = authors' inference.py (peft wheel
in a scratch dir). Parity on 49 questions: f16 49/49, |dp| ≤ 0.001; q8_2f Metal 47/49 (near ties) ≤ 0.04,
Vulkan 36/36 ≤ 0.05. Time per request of 3 questions: q8_2f 0.21 s M4, 0.31 s pod; f16 1.2 / 1.9 s.

## Metal small-model decode (commit 6d2eeed7)
Host waits 5.8 of 6.3 ms (CMF_GRAPH_HOSTPROF); GPU 0.2 ms a layer for 9 MB (CMF_METAL_GPUPROF + CMD_TIMES).
In-process stage elimination (CMF_METAL_AB mcfr/mcfrx + CMF_METAL_SKIP): gate/up 33 %, attend 18 % (~30 us a
layer at ~250 positions), head 13 % (at the bus), q 8.5 %, o 6.6 %. examples/metal_barrier_bench: a dependent
dispatch ~3 us on the M4. examples/metal_attend_bench (warm GPU): split-K in blocks of 16/32/64 is 2-4x the
per-head / per-key kernels up to ~1.5k positions on 14/2/64, 16/2/128, 32/8/128, 8/1/128.
Levers: DENSE_Q8R4 for every dense plan (A/B 0.85x MNN q8_2f, 0.81x MiniCPM5-2B, 0.78x Minico-M1, 0.88x at 2k)
and short_split_bs (A/B 0.91x / 0.89x, 0.98x MiniCPM5). Trap: separate runs on the fanless Air drift by
thermal state (an "old" arm read 199.6 one run, 162 the next) — only CMF_METAL_AB pair medians count.
Alternating runs on an idle Mac: q8_2f 162 -> 202 tok/s (CPU 154-161), q4tp 214 -> 252 (CPU 179-182).
Text identical: 16/16 MNN q8_2f and q4tp, 6/6 MiniCPM5-2B, 6/6 Minico-M1.
Tried and dropped: CMF_GQA_KB_MIN / KB=0 / BLKSZ (worse), ATT_SMALL (per-key, neutral).

## DFlash2 on Vulkan (commit 48b9cb9f), RTX 2000 Ada pod 157.157.221.29:57821
Round before: draft 7.7 ms on the host (block 3.7, head 2.6), verify 5.7 ms (2 rows) .. 10 ms (8 rows).
1. dflash_dev: hs % 64, narrow q4 through encode_q4tp_mm_r -> device block; CMF_DFLASH_DEV_CHECK drafts equal host.
2. batch-ts (CMF_GPU_TS=1 CMF_BATCH_TS=1) of a 6-row verify: gate-up 3.05, attn-core 3.08, down 1.31, qkv 0.95,
   o 0.81, norm 0.42 ms. queue.submit 0.3 ms, readback wait 7.4 (new SPEC_SUBMIT/WAIT print) — GPU-bound.
3. examples/q82_batch_bench: q8_2f_matvec_b is ~4x the decode kernel q8_2f_matvec4k even at ONE row (gate 55
   vs 14 us); rows barely matter. A subgroup-shuffle tree variant was SLOWER (indexed acc array spilled) —
   reverted; 32 KB-shared r4/b4k variants slower. Unrolled q8_2f_matvec_bu (decode kernel x nb, scalar vec4
   accumulators, lane-major partials) bit-equal to q8_2f_matvec4k: 6 rows gate 33 / down 22 / q 14.5 us.
4. Solo controller: summed window wall / tokens from round 3 (SOLO_JUDGE).
Result q8_2f, plain -> DFlash: Python 318-320 -> 413-434, Chinese code 322-327 -> 318-370, arithmetic 322-326
-> 299-315, chat 302-312 -> 235-256; text identical every run. q4tp still slower (int8 q4tp verify).
Next levers: the q4tp batch verify kernels the same way; attn-core of the verify (2.6 ms for 6 rows).
