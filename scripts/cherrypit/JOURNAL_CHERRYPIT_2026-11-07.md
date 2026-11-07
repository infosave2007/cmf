# CherryPit-1.2-2B → CMF (07.11.2026, MacBook Air M4 24 GB)

Source: DRUPEs/CherryPit-1.2-2B (MiniCPM5-2B finetune, LlamaForCausalLM, 42 L, 2048/6144, GQA 16/2, hd 128,
vocab 130560, rope 5e6, untied, bf16 5.03 GB in 4 shards). Template: chat_template.jinja (default identity system prompt,
enable_thinking, XML tool calls). Downloaded to ~/dev/cherrypit/hf/CherryPit-1.2-2B.

Binary: worktree ~/dev/cmf-cherrypit (branch cherrypit from master b8d89d0f 0.8.61), built into ~/dev/cherrypit/bin/cortiq.

## Convert
`cortiq convert --model hf/CherryPit-1.2-2B --quant f16 --output out/…-f16.cmf --also-output q8_2f=out/…-q8_2f.cmf`
14 s, peak RSS 6.5 GB. f16 5 050 440 063 B, q8_2f 2 537 829 759 B.

## Prompt parity (scripts/ref_prompts.py + scripts/parity.py)
7 conversations rendered by transformers 5.22.1 (plain, enable_thinking false/true, system, two turns, tools, unicode):
`serve` prompt_tokens equal the reference token count in 7/7, and greedy answers from the served prompt equal greedy
answers from the reference ids (CMF_PROMPT_IDS) in 7/7. Tool call parsed into tool_calls.
Bug found and fixed: serve + enable_thinking=false inserted the direct-answer directive as the system message and
thus dropped the template's default identity prompt (prompt 51 vs 211 tokens; "Who are you?" → "a MiniCPM series
model"). Fix commit adc440f1 (Tokenizer::template_has_default_system).

## Perplexity (cortiq ppl --windows 12 --window-len 512)
| file | wikitext-2 test | code_eval (Rust, 2nd half of repo code) | python_eval (transformers sources) |
|---|---:|---:|---:|
| f16 | 18.450 | — | — |
| q8_2f | 18.462 | 6.742 | 5.858 |
| q4tp GPTQ, calib wiki.train 2048 tok | 19.357 | 7.386 | 6.442 |
| q4tp GPTQ, calib wiki 0.5 + code_calib 0.5 | 19.780 | 7.091 | 6.184 |
Chosen q4tp = the mix (code is the model's job). Both GPTQ runs: --tokens 2048 --window 2048 --lambda 0.01
--calib-chunk-layers 8 (all at once wanted 21 GB of Hessians), head + embeddings q8_2f; 624-634 s, peak RSS 13.9 GB.
f16 ppl took 433 s at 785 % CPU after the f32 GEMM row-split fix (before: one core, unfinished after 12 min).

## q4tp tool-call defect and the fix
The mix q4tp answered "Build a React component …" (the vendor's own example, --no-think) with a made-up tool call
(`<function name="code_agent_show">…`) greedy and in 3/3 sampled seeds; q8_2f answered with code in all.
scripts/toolstart.sh (16 coding prompts, greedy + 2 seeds, no tools offered): mix q4tp 4/48 tool-call starts
(prompts 0 and 3), q8_2f 0/48.
Variant attn8 = mix calibration + every `model.layers.*.self_attn.*` at q8_2f (scripts/q4variant.sh):
1 769 428 351 B; wiki 19.310, code 7.017, python 6.125; tool-call starts 0/48. Adopted as the q4tp file.
(edge8 = first/last 4 layers q8 did not run: zsh passed the pattern list as one argument.)
attn8 still started a tool call in 4/20 sampled seeds of the React prompt (q8_2f 0/20) and 2/20 of the dark-mode
prompt (q8_2f 0/20): raw-text calibration never shows the model its chat format, where that first token is chosen.
Variant chat8 = attn8 + chat-format calibration: `--calib wiki.train=0.25 code_calib=0.25 chat_calib=0.5 --tokens
4096`, chat_calib.txt = q8_2f greedy --no-think answers to 20 coding requests NOT in toolstart.sh, rendered with the
model's template by transformers (scripts/chat_calib.sh, 36 KB). 1 769 428 351 B; wiki 19.401, code 7.030,
python 6.148; toolstart 0/48; React 0/20; dark mode 0/20. ADOPTED as CherryPit-1.2-2B-q4tp.cmf.

## Mac (MacBook Air M4 24 GB, fanless), clean window 14:39-15:00
bench --core --ignore-eos, medians of 3: Metal q8_2f 26.3 tok/s, q4tp 42.8; CPU (CMF_GPU=0) q8_2f 25.2, q4tp 30.5.
GPU busy 22 of 23.8 ms (q4tp) / 32 of 34 ms (q8_2f) per token: kernel-bound, ~71 GB/s for both codecs.
Prompt: the dense llama prompt went the host walk on Metal (prefill_batch_metal admitted GDN only). Commit 9337a56f
opens the rows graph to pure-attention models: 2k TTFT q4tp 6.00 -> 4.45 s, q8_2f 9.17 -> 5.47 s; 8k q4tp 58.2 ->
31.4 s, q8_2f 55.1 -> 33.7 s; decode at 8k q4tp 15.6 -> 20.9. ppl through the rows graph = host to 3 decimals.
Greedy text: q4tp 6/6 identical; q8_2f SQL answer flips at a 42 % near-tie (" Cherry" vs " not"), any one shape
class on the exact kernel flips it back: float order, not a kernel bug.
RSS: Metal 2k q8_2f 4.08 GB, q4tp 2.53; 8k 5.27 / 3.87; CPU 2k 3.09 / 2.31.
A later run (16:40) was taken while another session ran a ppl on the Mac (GPU 94-98 % busy): Metal q4tp 34-35,
CPU q8 15-17 tok/s — not used.

## Vulkan, RTX 5090 32 GB (pod), branch build without LTO
q8_2f before: decode 160.9, 2k TTFT 10.93 s / decode 93.8, 8k 49.6 s / 67.9, ppl 18.461.
Cause: dense stacks took the chunked host walk on wgpu (graph_prefill_preferred: GDN/MoE/OneVision only); prefill-split
showed qkv enqueue+wait, attend with readback (3.7 s of readback), o-proj, ffn per op. q8_2f coop threshold
(CMF_Q82_COOP_ROWS 2048/256) and the register-blocked int8 GEMM did not help on the walk or on the graph (worse).
Commit fead1ce7: dense resident stacks take the batch graph (batch_k 128). After:
q8_2f decode 150 (147-150), 2k TTFT 1.21 s / 102.4, 8k 7.19 s / 90.5, VRAM 3035/3069/4397 MiB, ppl 18.462.
q4tp decode 161 (159-162), 2k 0.99 s / 108.1, 8k 6.24 s / 95.0, VRAM 1915/2045/3277 MiB, ppl 19.401.
Greedy text 12/12 identical walk vs graph. q8_2f pod-converted file: all 381 tensor hashes equal the Mac file.
Decode: device span 4.33 ms/token of 6.0 wall (allocs 4796/token); per layer FFN ~46 us, attention stage 70-88 us;
knobs CMF_ATTEND_RKV/GQA/XG=0, CMF_GRAPH_FIRST=2, CMF_Q8_R4=1, CMF_COOP=0 all within +-5 %. At 4k the attention stage
grows 3.3 -> 6.7 ms (81 us/layer for 8.4 MB of KV): gqa_attend_gpart = 256-position chunks per workgroup, 2 KV heads
-> 32 workgroups at 4k on 170 SMs, lanes read their own K rows. Follow-up: a coalesced, finer-split GQA attend.

## Turbo
Exact-row fit (pod/fit.sh, scripts/dory/router/q27_feat.py, 20 prompts x 21000 chars, seal at 4096, 168 features):
leave one out best 0.00 % mean 0.01 % worst 0.02 % (27B: 0.22 / 0.96). keep_cherrypit.bin.
far-turbo q4tp -> CherryPit-1.2-2B-turbo.cmf (32,1800,4 from 8192 rows, 248 exact), 1 769 429 023 B,
sha256 61789f95…; HF private infosave/CherryPit-1.2-2B-Turbo-cmf, verified by API.
Facts (passkey2.py, 100 000 chars ~25k tokens, 8 trials): turbo 8/8 (first token ~65 s), standard q4tp 8/8 (~37 s).

## Turbo, Vulkan (RTX 5090), bench --core --tokens 60 --ctx N (pod/turbobench.sh)
| prompt | standard first token / decode / VRAM MiB | turbo |
|---:|---|---|
| 1 000 | 0.43 s / 108.0 / 2171 | 0.45 s / 109.4 / 1933 |
| 4 000 | 2.31 / 111.9 / 2461 | 2.29 / 108.2 / 2461 |
| 16 000 | 18.11 / 90.5 / 3547 | 48.80 / 66.9 / 2989 |
| 32 000 | 61.67 / 73.7 / 4893 | 86.41 / 67.4 / 2797 |
| 64 000 | 228.53 / 42.9 / 7582 | 171.90 / 67.0 / 3259 |
Host RSS: standard 1.8 GB; turbo 4.3 / 6.5 / 11.0 GB at 16k/32k/64k; CMF_O1_SPILL: 8.3 GB RAM + 8.2 GB disk, 159.5 s,
67.9 tok/s. CMF_GPU_VRAM_MB=3800: 2693 MiB, 161.3 s, 66.0 tok/s.
Facts: 300 000 chars (67 100 tokens) 3/4; the room-number miss gave 990190 with 96 tokens of answer. Release
binary 0.8.60 reads the turbo far field (fact at 100 000 chars OK) and all three files.
Real answer (summary of the last article, 400 tokens): 30 000 chars identical text; 100 000 chars the turbo mixed in
an earlier article. Row scores of a 60 000-char fact prompt: the key's rows at places 0 and 1 of 13 812.
ppl past the seal: `ppl --o1 all` ran per-op (graph refused) and was stopped; the env recipe of q27_quality.sh does
not reach the dense batch NLL path (both printed 12.032 exact).
HF: infosave/CherryPit-1.2-2B-cmf (q8_2f, q4tp, README) and infosave/CherryPit-1.2-2B-Turbo-cmf (turbo, README,
assets/*.png, keep_cherrypit.bin), all verified by the tree API; still PRIVATE (making them public was refused to the
agent — the owner's action).
