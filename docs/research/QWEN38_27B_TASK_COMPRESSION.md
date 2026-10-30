# Compressing Qwen3.8-27B to 14B for a task set on one RTX PRO 6000, step by step

*Languages: **English** · [Русский](QWEN38_27B_TASK_COMPRESSION.ru.md) · [中文](QWEN38_27B_TASK_COMPRESSION.zh.md)*

> The numbers come from the run logs of 29 October 2026 on a RunPod instance with an RTX PRO 6000.
> The model, its prompts and its teacher dataset are published as
> [infosave/Qwen3.8-27B-Code-14B-Experimental-cmf](https://huggingface.co/infosave/Qwen3.8-27B-Code-14B-Experimental-cmf).

## Abstract

We compressed Qwen3.8-27B (64 layers: 48 GatedDeltaNet and 16 full-attention,
26.9B parameters) to 14.4B parameters for a set of tasks: code in six
languages, page layout and visual tasks, and Russian and English questions on
knowledge, math, logic and tool use. The teacher was the 27B itself behind an
API, without thinking: 483 prompts with two answers each, 1.97M tokens for
$2.38. The `cortiq compress` pipeline does the rest in one command:
illumination of the FFN neurons on the task, osmotic folding of the dropped
neurons into the kept ones (Patent 15), healing on the teacher's answers
(Patents 15 and 16), GPTQ to q4tp, and scoring on held-out answers. The final
file takes 7.6 GB against 14.3 GB for the 27B in the same codec and decodes
77 tok/s against 28 for the 27B in q8_2f; with an MTP draft retrained for the
compressed trunk it reaches 89 tok/s (section 8). The held-out NLL on the task
rose from 0.117 to 0.396. The clean run took 1 h 34 min on one RTX PRO 6000
(the first pass took 2 h 20 min); with the fixes of section 6 the same recipe
took 1 h 10 min, including the scoring of five models in S6 (the run of
section 9).

Patents cited: Patent 15 is US application 19/731,402, Patent 16 is
19/738,763, Patent 2 (DTG-MA) is 19/452,464; see [PATENTS.md](../../PATENTS.md).

## 1. The task

Qwen3.8-27B in q4tp takes 14.3 GB and decodes about 50 tok/s on a 5090-class
card. Decoding is bound by memory bandwidth: every token reads all the weights.
A model with half the parameters reads half as much and fits cards from 12 GB.
The goal is a 14B that answers the tasks of our set as close to the 27B as
possible.

Qwen3.8-27B has 17.1B parameters in the FFN (64 layers × 3 × 5120 × 17408),
7.2B in attention and GatedDeltaNet, and 2.5B in the embeddings and the head.
Cutting the FFN width down to 14B leaves each layer about 25 % of its neurons
(4224 of 17408). We keep the depth, and attention and GatedDeltaNet stay whole.

## 2. Data: the task set

483 prompts (`cortiq-bot/benchmarks/calibration_prompts.jsonl`, published as
[`data/prompts.jsonl`](https://huggingface.co/infosave/Qwen3.8-27B-Code-14B-Experimental-cmf/blob/main/data/prompts.jsonl)):

| group | prompts |
|---|---|
| code, aider polyglot: JavaScript 48, Java 47, Go 38, Python 33, Rust 30, C++ 26 | 222 |
| code: generation, advanced programming, code completion, algorithms, systems, backend, frontend, DevOps, security, ML, embedded, networking, compilers, game development, refactoring, patterns, debugging, tests, databases; page layout and visual tasks (HTML, three.js) | 177 |
| RU/EN: knowledge, math, logic, tools, anti-looping | 84 |

The teacher was Qwen3.8-27B via OpenRouter, without thinking, sampled at
temperature 0.6 / top_p 0.95 / top_k 20, with a limit of 8192 tokens and two
answers per prompt. 160 answers came from the cache of earlier work; the 806
new ones cost $2.38 and took 15 minutes. 914 answers ended on their own and 52
hit the limit (long HTML/three.js pages); truncated answers are stored without
the end token. The split is 80/10/10 by prompt, not by answer, so the answers
to one prompt never land in both training and test: 789 / 88 / 88 records,
1.58 / 0.20 / 0.19M tokens. The finished dataset is published next to the model
as
[`data/q27-code-t27.tdata`](https://huggingface.co/infosave/Qwen3.8-27B-Code-14B-Experimental-cmf/tree/main/data/q27-code-t27.tdata),
with the answers as text in `data/q27-code-t27.tdata.answers.jsonl`.

**Use the tokenizer of the exact model you compress.** Qwen3.8-27B's
`tokenizer.json` has 7 more special tokens than Qwen3.5-0.8B's (audio/TTS,
ids 248070–248076) and its own `generation_config.json` with `<|im_end|>` as the
end of turn. Data tokenized by a smaller model of the family would fail the
vocabulary check in S0, and the answers would end with the wrong token. The
answers were re-imported with a tokenizer carrier: a CMF with the 0.8B's
weights and the 27B's tokenizer and template files.

## 3. Method

The stages are described in detail in the
[first article](TASK_COMPRESSION_ARTICLE.md); here, briefly, is what each one
does on the 27B.

| stage | what it does |
|---|---|
| S0 | bf16 checkpoint → working f16 and q8_2f copies; check of the data's tokenizer |
| S1 illumination | forward pass of the source over 200k tokens of the teacher's answers (train, plus a quarter as many for val); Grams of each layer's `down_proj` input (17408²); neuron importance `‖W_down[:,j]‖²·⟨a_j²⟩` |
| S2 plan | layer widths: a uniform 4224 (aligned to 128) |
| S3 osmotic folding (Patent 15) | the dropped neurons are regressed onto the kept ones, `W_down' = W_S + W_P·Πᵀ`, `Π = (Σ_SS+εI)⁻¹Σ_SP`; the ridge is chosen on val |
| S4 healing (Patent 15 §[0022], Patent 16) | the MLPs of the last layers train on the teacher's answers (CE), the rest stays frozen; the best step by val |
| S5 | GPTQ q4tp calibrated on the same data, full-attention k/v in q8_2f |
| S6 | NLL on 88 held-out answers, 16 greedy generations, quality gates |

## 4. Hardware

RunPod: an RTX PRO 6000 Blackwell (96 GB), a 116 GB container memory limit, a
27-core quota and 600 GB of local disk. Vulkan through wgpu with cooperative
matrices. The machine limits two things: host RAM (the 27B's Grams fit on the
host 15–17 layers at a time) and the wgpu buffer binding (2 GB per binding,
which bounds the trainer's long windows, section 6).

## 5. Step by step (the clean run)

```bash
# teacher data (Mac, no GPU)
OPENROUTER_API_KEY=… cortiq teacher-data generate openrouter:qwen/qwen3.8-27b \
    calibration_prompts.jsonl q27-code-t27.tdata --samples 2 --no-think \
    --tokenizer tok27.cmf --sampling temperature=0.6,top_p=0.95,top_k=20 --split 80/10/10

# pod: environment, checkpoint, binary (scripts/pod_setup_q27.sh), then one command
cortiq compress hf/Qwen3.8-27B q27-code.tdata q27-14b.cmf q4tp --size 14B \
    --min-width 0.2 --illum-tokens 200000 --keep-grams off \
    --heal fcd --heal-loss ce --heal-epochs 2 --heal-layers deep36
```

In this run `--heal-layers auto` picked 47 layers and ran out of memory: the
estimator counted 20 bytes per trained weight and 6 GB of working buffers.
36 layers peaked at 96.6 GB of 102.6; since 0.8.44 the estimator (25 bytes and
13 GB) picks these 36 by itself.

| stage | time | GPU load | output |
|---|---|---|---|
| S0 | 5.7 min | — | f16 54.7 GB, q8_2f 27.4 GB; 789 / 88 / 88 records |
| S1 | 6.4 min | — | layer energies and neuron importance over 200k tokens, 17 layers per pass; the Grams are not written (155 GB), the fold recomputes them |
| S3 | 10.2 min | — | 64 layers 17408 → 4224; read-out error on val 0.319 (plain truncation 0.412); f16 28.7 GB and q8_2f 14.4 GB |
| S4 | 48.2 min | 89 % | MLPs of layers 28–63 (2.34B weights), 2 epochs of 384 steps, 1.96 s per step; held-out loss 0.764 → 0.557 (best step 375, the end of the first epoch) |
| S5 | 3.8 min | 32 % | GPTQ q4tp on 65k tokens, full-attention k/v in q8_2f; 7.61 GB |
| S6 | 20 min | 52 % | 4 models: NLL on 88 records, 16 generations |

The GPU load of S0–S3 was not recorded: the `nvidia-smi` log was overwritten
when S4 restarted (the script is fixed). S6 also took two attempts: the weights
of the source 27B and of the fold stayed in the device cache, and the third
model ran out of memory; since 0.8.44 S6 frees them after each model.

## 6. What had to be sped up

The first pass ran on release 0.8.42 and showed where the GPU sat idle. Every
fix below gives the same result as the code it replaced: on Qwen3.5-0.8B the
illumination curves, the neuron order and all 48 Gram files matched bit for
bit, and all 24 folded layers matched; the streaming weight conversions are
covered by bit-exactness tests; GPTQ through the trainer gave the same ppl
(2.388 against 2.387).

| stage | first pass | what was wrong | after |
|---|---|---|---|
| S1 illumination | 16.5 min | the GPU waited while the host wrote 155 GB of Grams | the Grams are not written (`--keep-grams off`) or are written on a second thread; 6.4 min |
| S3 fold | 11 min (14B), 12 s per layer at ~0 % GPU | each layer waited for 2.4 GB of Grams to be read; f16 and q8 were written one after the other | the Grams are recomputed on the GPU, reads run one layer ahead, both writes run at once; 10.2 min with the recomputation |
| weight loading into the trainer | minutes per load | f16↔f32 conversion and transposition on one thread | on all threads |
| S4 healing | 30 min (1 epoch, 32 layers) | the second epoch never beat the best step; 31 checks over all 98 val windows took about a third of the stage | stop after 6 checks without improvement (`--heal-patience`), checks over 32 windows (`--heal-val-windows`); 28 min against 48 on the same 36 layers (the run of section 9) |
| S5 GPTQ | 20 min, 10–20 s of calibration per window at 6–9 % GPU | the runtime paid a round trip to the card for every projection of every step | calibration on the trainer's forward pass: 0.93 s per 2048-token window; 3.8 min |
| S6 scoring | 40+ min | 17 records longer than 4096 tokens (63 % of the test tokens) went through the runtime at ~200 tok/s | all on the GPU, long records scored on their first 4608 tokens; 3.5 min for the 14B, ~8 min for the 27B |
| MTP calibration | 57 tok/s on the CPU, ~17 min for 48 records | the trunk ran one token at a time on the host | the trunk in one GPU prefill, the draft's FFN in batches; 100 tok/s, the rest is the row-by-row pass of the draft layer on the host |

The 4608 limit comes from the buffer binding: the trainer's block of attention
scores for all 24 heads (24 · t² · 4 bytes) must fit in 2 GB. Long records are
scored on their first 4608 tokens, the same way for every candidate;
`CMF_COMPRESS_EVAL_LONG=runtime` scores them whole.

## 7. Results

First pass (release 0.8.42, CE healing, 1 epoch, last 32 layers):

| model | size | test NLL | loops | tok/s |
|---|---|---|---|---|
| source 27B (q8_2f) | 27.4 GB | 0.117 | 0/16 | 25.9 |
| 14B fold | 14.4 GB | 0.557 | 6/16 | 43.0 |
| healed 14B | 14.4 GB | 0.401 | 2/16 | 42.8 |
| 14B q4tp | 7.6 GB | 0.411 | 2/16 | 68.5 |

Clean run (CE healing, 2 epochs, last 36 layers):

| model | size | test NLL | ppl | loops | tok/s |
|---|---|---|---|---|---|
| source 27B (q8_2f) | 27.4 GB | 0.117 | 1.124 | 0/16 | 27.9 |
| 14B fold | 14.4 GB | 0.557 | 1.745 | 6/16 | 43.3 |
| healed 14B | 14.4 GB | 0.387 | 1.473 | 2/16 | 42.9 |
| **14B q4tp** | **7.6 GB** | **0.396** | **1.486** | **2/16** | **69.1** |

Healing removed 39 % of the loss the fold added (0.557 → 0.387, with the source
at 0.117), and GPTQ q4tp added 0.009. Against the first pass the result is
0.015 better (four more layers and a second epoch; the best step is the end of
the first).

The S6 generations are 16 greedy continuations, but each test prompt has two
teacher answers, so these are 8 distinct prompts, each twice (since 0.8.44 S6
takes 16 distinct prompts). Both "loops" are one prompt, `gen_java_27` (a rate
limiter in Java): the 14B starts with a long import list, invents
`java.util.concurrent.locks.RWLock` and `RWLockImpl`, and repeats the imports up
to the limit. On the other seven prompts the answers are coherent: a Go hash map
with the same error declarations and the same structure as the 27B's, and C++
perfect numbers ending by itself after 223 tokens (the 27B after 229). The 27B
solves the gigasecond exercise in 62 tokens (`return start + 1000000000;`); the
14B works through a conversion to hours in comments, gets the arithmetic wrong
and hits the 384-token limit. In sum, the 14B is close to the 27B in the form
and structure of its code and clearly weaker in the accuracy of the content.
In this run it did not pass the quality gates: 2 loops against an allowance of
"the source's count plus one", and 2 self-terminated answers against the
source's 4.

Re-scored on 16 distinct held-out prompts (the S6 of the section 9 run), the
same 14B loops on one prompt, the 27B on none, and the gates pass. The published
file is this trunk with the retrained draft of section 8; the draft does not
change greedy text.

## 8. MTP: the draft after compression

Qwen3.8 carries an MTP draft: one layer (0.42B parameters) that predicts the
token after next from the trunk's last hidden state. In greedy speculative
decoding the trunk verifies the drafts in one batch and accepts the matching
ones: the text is the same as without speculation, only the speed changes.
Compression does not touch the draft (it is not counted in `--size`), and GPTQ
encodes its linear layers without calibration.

Greedy decoding, 384 tokens, two test prompts (a Go hash map and C++ perfect
numbers), speculation on and off:

| model | draft acceptance | tok/s with speculation | tok/s without |
|---|---|---|---|
| 27B q8_2f, its own draft | 59 % / 80 % | 38.3 / 48.7 | 32.3 / 32.6 |
| 14B q4tp, the 27B's draft as is | 54 % / 53 % | 71.0 / 74.9 | 77.6 / 75.6 |
| 14B q4tp, draft after `mtp-compact` 0.9 | 54 % / 51 % | 78.2 / 74.3 | 78.3 / 75.9 |

On q8_2f the engine does not enable speculation by default (the verify runs
through tile GEMMs); for the 27B it was forced on (`CMF_GRAPH_SPEC=1`) and even
so gives ×1.19–1.49. On the 14B acceptance fell to 51–54 % and speculation
stopped paying: folding and healing shifted the hidden state the draft
predicts from. Compressing the draft's FFN with calibration through the 14B
trunk (`mtp-compact`) did not bring acceptance back, so in `compress` that stage
is opt-in (`--mtp-keep`, off by default). The gap between 71.0 and 78.2 tok/s at
equal acceptance on the first prompt is run-to-run spread on the machine.

The gain does not come back without training. We tried two analytic fixes: a
ridge regression of the 14B's hidden states onto what the draft saw on the 27B,
folded into the `eh_proj` input (input error 0.284 → 0.229), and a
least-squares solve of the draft's `down_proj` for the 14B's next hidden state
(error 0.96 → 0.21). Acceptance did not move: 56/48 % and 48/52 %. Closeness in
hidden-state space does not mean the best token matches.

What works is training the draft on the 14B's own predictions
(`cortiq mtp-heal`, the S5 stage `--mtp-heal`). The trunk runs once over 393k
tokens of answers on the trainer and returns its hidden states and the next
token's top-64. The draft block, taken as a one-layer model on its own input
`eh_proj·[enorm(e); hnorm(h)]`, learns the KL to the trunk's distribution at the
next position. The trap: in speculation the draft runs as a chain, and every
step after the first reads the block's own output, not the trunk. Training the
first step alone raised acceptance at depth 1 to 88–90 %, but at the working
depth of 3–5 it stayed at 58/42 %. So the whole chain is trained: the input of
step j is rebuilt from the output of step j − 1 with the current weights every
150 steps, and the target is the trunk's distribution j positions ahead. Three
steps took 38 minutes on the RTX PRO 6000; the KL of the chain's steps went from
0.51 / 1.16 / 1.92 to 0.20 / 0.42 / 0.69.

Decoding on an idle card, the same two prompts:

| draft | no speculation | adaptive depth | depth 2 |
|---|---|---|---|
| original | 77.1 / 75.7 | 77.7 / 74.9 (54 %) | 79.7 / 82.8 (73 / 77 %) |
| trained, chain of 3 | 77.9 / 75.7 | 81.8 / 76.7 (61 / 57 %) | **89.7 / 88.1** (87 / 87 %) |

With the trained draft at depth 2 (`CMF_GRAPH_SPEC_K=2`) decoding is 16 %
faster with the same text. The adaptive depth climbs to 3–5, where acceptance
falls, and loses almost all of the gain: a retrained draft needs a depth policy
driven by the acceptance at each depth. A draft confidence threshold (cutting
the chain at a draft below probability p) gave no gain.

**On a Mac.** On native Metal the batched verify graph used to decline this
file: S5 keeps the full-attention k/v projections in q8_2f, and the graph took
only q4tp and q2tp projections. Since cortiq 0.8.46 it takes q8_2f as well,
with the token graph's matvec reduction on every row. On a MacBook Air M4
(24 GB), greedy, 384 tokens, the Go prompt:

| mode | tok/s | draft acceptance |
|---|---|---|
| no speculation | 11.3–11.6 | — |
| `CMF_GRAPH_SPEC=1 CMF_GRAPH_SPEC_K=2` | **15.6–17.4** | 87 % |
| `CMF_GRAPH_SPEC=1`, adaptive depth | 13.2 | 46 % |

Native Metal runs speculation only when `CMF_GRAPH_SPEC=1` asks for it, and the
text is identical to plain decoding in every mode.

## 9. Skill before compression: DTG-MA baking

The hypothesis: if the task's skill is first baked into the model with DTG-MA
(Patent 2) and its accuracy goes up, the compression starts from a better
point. `cortiq dtgma-bake` trains soft gates σ(m) on the FFN neurons of the deep
layers with the task's LM loss under a small constant L1 (the best step on
held-out data, no sparsity target) and folds them into the columns of
`down_proj`, so the model's width does not change. For the 27B the mask covers
layers 32–63: a full backward pass of the 27B (f16 weights both ways) does not
fit in 96 GB, so layers 0–31 run once and are cached on the host (14.6 GB of
f16). Then comes the usual compression with the same recipe.

| test NLL | 0.8B: direct | 0.8B: bake → compress | 27B: direct | 27B: bake → compress |
|---|---|---|---|---|
| source | 0.571 | **0.561** | 0.117 | **0.110** |
| fold | 0.742 | **0.717** | **0.557** | 0.564 |
| healed | **0.604** | 0.607 | **0.387** | 0.389 |
| final q4tp | **0.646** | 0.649 | **0.396** | 0.398 |

0.8B: 75 % of the FFN; the teacher data is the 27B's answers re-tokenized for
the 0.8B; baking of layers 12–23, 300 steps, 3 minutes. 27B: 300 steps,
25 minutes. Baking raises the model's own accuracy (by 6 % on the 27B), but
after folding and healing the result matches direct compression. Healing
brings both models to the same point: its limit, not the starting point,
decides the result.

## 10. Order of methods: NVG ↔ DTG-MA

The question: is it better to encode the model with NVG first and then prune it
with a DTG-MA mask (Patent 2), or the other way round? Qwen3.5-0.8B, 75 % of the
FFN, both variants of the same size, test NLL on the same held-out answers (the
source scores 0.504 on this data and 0.189 on the data of 28 October):

| budget | DTG-MA → NVG | NVG → DTG-MA |
|---|---|---|
| 2.0 bits/weight (0.398 GB, 28.10) | 1.931, loops 16/16 | **1.618**, 14/16 |
| 3.0 bits/weight (0.437 GB, 28.10) | 1.340, 7/16 | **1.277**, 13/16 |
| 4.0 bits/weight (0.438 GB, 29.10) | 1.844, 8/16 | **1.339**, 15/16 |

NVG first is consistently better in NLL: DTG-MA masks on an encoded model close
the neurons that were encoded worst. But such models loop more often, and at
these budgets both are unusable on the 0.8B. At 4 bits the encoder reaches the
full rank of the 0.8B's matrices (r1024), so the size matches the 3-bit point.
For the 27B we keep the order "structure, then quantization": GPTQ q4tp is
calibrated on the already healed model.

## 11. Limitations

- One target point (14B). The 20B and 17B points were folded without healing in
  the first pass but not scored (scoring them was not part of the task).
- Healing is CE on the teacher's text only. KD to the teacher's distribution
  needs a pass of the 27B over 1.8M tokens and did not fit the pod budget.
- Long records are scored on their first 4608 tokens.
- The S6 generations of the first and the clean run cover 8 distinct prompts
  (each twice). The baking experiment was scored on 16 distinct prompts, and so
  was the earlier 14B in the same table.
- The MTP speed was measured on two prompts; no depth policy has been chosen
  for the retrained draft.
- NLL and the gates are properties of our task set, not of general benchmarks.

## 12. Reproduction

- `scripts/compress_q27_14b_pod.sh run` runs the whole chain in one command and
  logs the GPU load; `scripts/pod_setup_q27.sh` sets up the pod environment and
  the release binary; `scripts/pack_q27_pod.sh` packs the data for the pod.
- `scripts/order_nvg_dtgma.sh` is the NVG ↔ DTG-MA order experiment.
- `scripts/exp_q27_bake_then_compress.sh` and `scripts/exp_bake_then_compress.sh`
  bake before compressing on the 27B and the 0.8B.
- `cortiq mtp-heal MODEL.cmf --init SRC-f16.cmf --data TASK.tdata --chain 3
  --output OUT.cmf` trains a draft for the compressed trunk.
- The model, the prompts and the teacher dataset:
  [infosave/Qwen3.8-27B-Code-14B-Experimental-cmf](https://huggingface.co/infosave/Qwen3.8-27B-Code-14B-Experimental-cmf)
  (`data/`). With the published dataset the compression needs no API calls:
  `cortiq compress hf/Qwen3.8-27B data/q27-code-t27.tdata …`.
- Logs, reports and intermediate models (private repo):
  `infosave/cmf-experiments`, `runs/q27-14b-*`,
  `runs/q27-dtgma-bake-20261029/`, `q27-14b/`.
