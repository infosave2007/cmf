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
