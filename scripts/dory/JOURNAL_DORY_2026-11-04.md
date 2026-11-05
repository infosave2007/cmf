# Dory1.0-4B on cortiq: run log

Pod: NVIDIA L4 (23 GB, driver 595.91), EPYC 9254 with a 5.1-CPU quota, 62 GB cgroup. Work dir /root/dory.

## Reference
- No Transformers modeling code exists: the vendor's code is vLLM PR #60149 and llama.cpp PR #30306.
- `scripts/dory/ref_dory.py` transcribes vLLM's forward to plain PyTorch.
- vLLM itself at float32 (`scripts/dory/vllm_ref.py`, FLEX_ATTENTION: Triton's kernel asks an L4 for more
  shared memory than it has): 64/64 tokens equal, first-step log-probabilities within 3e-4 on four prompts,
  including 1500 tokens past the sliding window.

## Conversion and parity
- f16 7.98 GB, q8_2f 4.01 GB, q4tp 2.10 GB. Scale vectors stay f32, BF16 widens exactly.
- f16 on the CPU against the reference: logits within 1e-4, same text.
- q4tp on Vulkan against q4tp on the CPU (float activations): 8.9e-4 max, 1.5e-4 rms.
- `CMF_DORY_DUMP=dir` writes the logits of each forward.

## Quality: wiki 12x512 (6144 tokens), perplexity
| file | GB | ppl |
| reference f32 | | 13.416 |
| q8_2f | 4.01 | 13.440 |
| q4tp + attention, head, embedding q8_2f | 3.06 | 13.864 |
| q4tp + head, embedding q8_2f | 2.42 | 14.410 |
| q4tp + head q8_2f | 2.26 | 14.477 |
| q4tp | 2.10 | 14.600 |

## Vulkan speed, L4, tok/s
q4tp: 51.7 short, 42.0 at 1k, 35.6 at 4k, 22.3 at 16k; prefill 426 at 1k, 373 at 4k, 208 at 16k.
q8_2f: 33.6 short, 29.9 at 1k, 26.2 at 4k; prefill 336 at 1k, 302 at 4k.

Weights cost 13.4 ms of a q4tp token at about 210 GB/s, near this card's bus (q8 the same 220 GB/s).

What moved the rest:
- A workgroup stays on one SM on NVIDIA. A layer has 4 KV heads, so workgroups that held a whole head left most
  of the card idle: 230 µs a layer at a full window. Short threads, one per key or per (piece, vec4), and no shared
  memory: 4k decode 21.9 -> 35.6.
- One 32-lane warp with subgroupAdd stalls on every load: a row-wide gate 61 µs, a residual 16 µs. The norm before
  an output projection only scales what the residual normalizes anyway, so it went: gate and activation became
  short threads of about 3.5 µs. Logit change at the level of rounding (1.5e-4 rms), ppl the same to 3 digits.
- A head's rotation as one workgroup, a lane per vec4 pair, subgroupAdd: 24 -> 4 µs.
- Keep-one table: `CMF_DORY_KEEP=qkv,g,prep,score,soft,sum,gate,o,gu,silu,d,slerp,mix` (`scripts/dory/stages.sh`).

## q4tp-v2
`scripts/dory/gptq.sh`: GPTQ of all 288 projections on 32768 tokens (wiki 0.6, the model's own answers 0.25, code
0.15), the Hessians on the card, the calibration forward on the host stack: 2.09 GB, ppl 13.957 against 14.600 for
plain q4tp at the same size, 13.416 for the reference.

## Metal, M4 Air, q4tp
27.4 tok/s short, 23.4 at 1k, 19.9 at 4k, prefill 138 at 1k. Keep-one (ms): gate+up 12.3, down 6.5, q,k,v 6.3, g 4.0,
o 3.5, residuals 2.0, score 1.3, the rest under 1. The weights run at 87 GB/s of a 120 GB/s bus.
Sliding rings have their exact size now: 998 MB of KV at 2k instead of 1.9 GB.

## CPU
`matmat` with one row was the whole loss: M4 q4tp 1.0 tok/s. One row through the fused matvecs, SIMD for the activation,
softmax and gate: 3.2. Int8 activations (the default for q4tp): 16.7, ppl 14.615 against 14.600. q8_2f stays on float
activations, 14.2 tok/s, because int8 is slower there. A prompt goes by chunks of 64.

## Loops, token by token (`scripts/dory/router`)
80 own answers (40 GSM8K, 15 code, 15 general, 10 instructions), 32754 tokens, teacher-forced, f32:
- top-1 of 1 loop = 2 loops on 93.9 %, 2 = 4 loops on 97.7 %.
- NLL: 1 loop 0.366, 2 loops 0.345, 4 loops 0.347. Wiki 2.630, 2.596, 2.589.
- Zero loops (input block straight to output block): ppl 89229 on wiki, dead.
- Last-loop cache: ppl 14.519 against per-loop 14.600 on q4tp, read row by row: no cost.

Held-out half, top probability after one loop as the gate (a read-out through the output block and the head):
47 % of tokens: 99.99 % same top-1, NLL +0.000. 67 %: 99.89 %, +0.001. 78 %: 99.45 %, +0.004. AUC 0.916.

Patent-1 reconstruction over the state after one pass, two isolated PCA topologies (easy, hard), rank 16, no read-out
needed: AUC 0.735. 38 % of tokens: 99.61 %, NLL +0.001. 48 %: 99.17 %, +0.002. 60 %: 98.54 %, +0.004. Over the input block's
state AUC 0.656, the previous token's state 0.681, the previous token's top probability 0.652. Rank 64 changes nothing.

Request level, 80 prompts, leave-one-out: "the second loop matters more than the median" AUC 0.726 from Embryo's φ (rank 4,
384 numbers, 91 MB model), 0.681 from Dory's own input block. Type of request 67/80 and 76/80.

KV reuse bug found on the way: a replay outside the plan must size key dispatches by position.

## The far field: what it holds and what it costs
q4tp-v2, wiki, windows of 4096: exact 9.185; cut to 256 26.202, 1024 11.275, 2048 9.397; far field window 192 12.401,
1024 10.696, 2048 (seal 2068) 10.131. Windows of 16384: exact 8.098; cut to 2048 11.034; far field window 2048 sealed at
8192 8.815. KV at 256K rows, by arithmetic: per loop about 29 GB, last loop about 18 GB, far field under 1 GB.

The generic `o1_attend` held one workgroup per query head, and a workgroup stays on one SM. In the exact path's short
threads (`dory_os`, `dory_ol`, `dory_omax`, `dory_ofar`, `dory_osoft`, `dory_osum`, `dory_ogate`) a sealed token on L4 went
from 25.0 to 35.7 tok/s at two passes, ppl 8.774 both ways on the same rows, 8.765 on Metal. Metal 24.5 to 28.5 at 16000.

## A prompt in one pass
Under the last-loop cache a prompt row keeps only its cache rows, and the row's logits are not read. One pass over all
chunks but the last: ppl of the last 512 rows of 4096, 9.241 at two passes, 9.331 at one. Metal prefill 117 to 213 tok/s at
1000. Metal also seals at the prompt's end up to 32768 rows now: row-by-row sealed chunks made 16000 rows take 536 s, against
137 s now.

## Traps of this stage
- A compile on the Mac during a measurement read as "the gate gives nothing": 26.7 against 25.8, clean 32.2 against 25.5.
- A new binary with `CMF_NLL_SKIP` against an old one without it compares different rows: 16.8 against 12.0 was no bug.
- `suite.py` keeps its baseline per file: the agreement of a turbo file with the standard one needs `agree.py`-style compare.
- GSM8K in thinking mode spends 1500 tokens reasoning about the answer format: use direct answers.

## Exact rows: the owner's idea of illumination for facts
A far field loses verbatim facts: a pass key at the head of 100000 characters, exact 4/4, far field 0/4. The owner proposed
the method behind NVG's illumination: find the important stretches of the text and keep those exact.

- Illumination of heads (`illum_far.py`, 8 prompts): at the answer's rows 16 of 36 KV heads give a planted fact over half of
  their attention, the best 95 % (layer 46 head 1). Recall is spread over many heads: choosing heads will not do. Rows, then.
- Single signals (`illum_rows.py`), % of the prompt above the fact's best / mean row: key residual over landmarks 5.8 / 17.8,
  token surprise 14.7 / 28.0, patent-1 router 32.3 / 52.3, residual + surprise 3.6 / 10.5.
- Fit (`illum_feat.py`, `illum_fit.py`, 20 prompts, 109 features as ranks): leave one prompt out, mean 0.78, best 0.02, worst
  3.14. A kind of fact held out of the fit: pass key 0.06, code 0.10, password 0.22, token 0.19, a person's name 8.90. Residual
  and value norm without surprise: 1.06 / 0.08 / 3.36, and those need no read-out. `illum_budget.py`, budget 248 with 2
  neighbours: 97.1 % of a fact's rows, 90 % of facts whole. In the engine, on q4tp and 23000 rows: places 0 to 8.
- The rows were found and recall still failed, 1/8, and it did not move at 1024 rows. A far field without those rows at the
  prompt's end: still 1/8. The cause: a question's own rows attend to the fact and carry it on to the answer, and they were read
  before the choice. The choice before the last 512 rows: 7/8, the eighth one digit off. Metal 4/4, host 3/4 at 64 rows.
- `prefill_group` takes rows in any order: the kept rows go right behind the sinks and nothing in `nystrom` changes.

Budget 248 with a window of 1800 against 512 with 1536, the same near field of 2052 keys. Facts in 100000 characters: 7/8 and 8/8.
At 58000 rows through `cortiq serve`: 1/3 and 2/3, the miss 79086 for 790869. A name and a room at 59000 rows: miss. Two windows of
16384: 8.795 and 8.853. Speed the same. 512 went in.

## Final table
Vulkan L4, standard against turbo: real prompts 49.6 / 60.7; context 1000 41.1 / 60.6; 4000 35.1 / 51.8; 16000 21.5 / 52.2;
32000 14.1 / 54.0; 64000 not read in 25 min / 53.0. First token 78 s / 53 s at 16000, 360 s / 112 s at 32000, 228 s at 64000.
KV 998 / 642 MB at 4000, 4287 / 502 MB at 32000. Metal M4 Air: 25.7 / 32.9; 22.1 / 34.6; 20.3 / 31.0; 14.8 / 29.3. CPU M4 18.0 /
27.7 at 256, 12.1 / 18.3 at 2000; EPYC 9254 7.6 / 11.6, 5.3 / 8.2. ppl 13.957 / 14.219 with 84 % on one pass. GSM8K 60 questions,
direct answers: standard 53, turbo 52, one loop 55, conf.7 54, four loops 54, q8 51: no separation at this size. Two windows of
16384: exact 8.098, far field 8.815, far field with exact rows 8.853, cut to 2048 11.034. q8 on the EPYC prompt: 1.2 to 11.6 tok/s,
row by row. Six long requests in a row on one server: 6/6 and no growth in VRAM.

## More traps
- `pgrep -f NAME` inside a `bash -c` string matches that string: the wait never ended.
- A single prompt argument over 128 KB fails on Linux: long tests go through `serve`.
- A 64-row exact chunk needs 64 score planes of the whole context: 131 GB at 32000 rows. That is why the exact prefill falls
  from 362 to 89 tok/s, and why wgpu seals at 8192 and keeps rows for the choice.
