# Compressing a language model for a task: illumination, osmotic folding, healing, DTG-MA masks and the NVG ↔ DTG-MA order

*Languages: **English** · [Русский](TASK_COMPRESSION_ARTICLE.ru.md) · [中文](TASK_COMPRESSION_ARTICLE.zh.md)*

*Oleg Urevich Kirichenko. Implementation and measurements: the cortiq engine and the CMF format (branch `compress`), 27–28 October 2026.*

## Abstract

A general language model spends most of its weights on things a given task
does not need. We build one command,
`cortiq compress SOURCE.cmf PROMPTS.jsonl OUTPUT.cmf --ratio R`, that turns a
general model into a specialist for a user's list of prompts. The pipeline:

1. A teacher answers the prompts: the model itself, a larger model, or a model
   behind an OpenAI-compatible API.
2. The answers show which FFN neurons the task uses ("illumination").
3. The surplus neurons are removed, and their contribution is moved into the
   kept ones by a closed-form solve: osmotic folding (Patent 15, USPTO
   application 19/731,402).
4. The model is trained further on the teacher's answers ("healing"; Patents 15
   and 16).
5. Held-out answers score the result.

Layer widths can also be chosen by a trained DTG-MA mask (Patent 2). For MoE
models the experts the task does not route to are removed. The heavy stages
run on a GPU trainer: on NVIDIA through Vulkan and tensor cores, on Apple
Silicon through Metal and simdgroup matrices.

The results on Qwen3.5-0.8B and 2B:

- **Osmotic folding is the main lever of structural compression.** Moving the
  contribution of the removed neurons into the kept ones halves the FFN output
  error compared with plain truncation.
- **The choice of neurons barely affects the result.** A trained DTG-MA mask
  gives a better fold than illumination, but the advantage disappears after
  healing: on these models healing erases the difference in selection.
- **Masks on a quantized model act as a quantization repair.** On a model
  encoded by NVG down to 2 bits per weight, DTG-MA masks close the most damaged
  neurons. The KL to the teacher falls from 2.17 to 1.23 without a single step
  on the weights. At equal size, "NVG first, then DTG-MA" beats the reverse
  order: test NLL 1.618 against 1.931 at 2 bits, 1.277 against 1.340 at 3 bits.
- **Small dense models should reach their compression ratio through
  quantization and apply pruning only on top of it.**

## 1. The problem

Given a model in CMF and a file of task prompts, produce a file of a given size
(a fraction `R` of the source in the same codec) that behaves like the source on
this task. The requirements:

- one command, with no manual stages;
- progress and an estimate of the remaining time are visible at every stage;
- an interrupted run resumes where it stopped;
- quantization is a separate user step: the output keeps the source's codec;
- the depth (number of layers) is not reduced, only the FFN width and the MoE
  experts;
- the CMF format stores the width of every layer itself, so the removed neurons
  physically leave the file. No defragmentation or width alignment is needed.

## 2. The author's patents and works this research builds on

| Source | What this work takes from it |
|---|---|
| **Patent 2**, USPTO application **19/452,464** (filed 19 January 2026): *System and Method for Dynamic Task-Guided Neural Network Compression with Catastrophic Forgetting Prevention* (DTG-MA, DFSC) | trainable masks σ(m) on the MLP neurons with frozen weights; L1 pressure toward sparsity; the hard threshold `1[σ(m) > τ]`; compiling the mask into a physically truncated model. Here the mask is built into the GPU trainer (§3.5) and applied to an NVG model (§5.6) |
| **Patent 15**, USPTO application **19/731,402** (filed 6 July 2026, CIP of 19/452,440 and 19/452,464): *Unified Execution Architecture for Serving a Plurality of Specialized Language Models from a Single Shared Backbone via Dynamically Overlaid Compressed Delta Representations Without Materializing Separate Models* | **osmotic folding** (§[0025]–[0026]): the removed FFN neurons flow into the kept ones by a covariance-weighted reconstruction `W_down' = W_down[:,kept] + W_down[:,pruned]·Σ_SS⁻¹Σ_PSᵀ` instead of a plain cut; **healing** (§[0022]): a distillation repair of the folded subnetwork; skill creation by a DTG-MA mask (§[0021]). The fold here is §3.3, with a ridge chosen on val |
| **Patent 16**, USPTO application **19/738,763** (filed 13 July 2026): *Training-Free Conversion of Softmax Attention Layers … into Constant-Memory Streaming Attention …, and Teacher-Anchored Surroundings-Only Quality Restoration* | the quality-restoration scheme: the norms and FFN tensors train on a mix of cross-entropy and a teacher-anchored divergence, the rest stays frozen. Healing here (§3.4) is built the same way: the MLPs train, with the loss `α·CE + (1−α)·KL` to the source |
| **Patent 12** (AWNP): *System and Method for Activation-Weighted Structured Compression of Neural Networks with Architecture-Preserving Layer Selection* | channel importance weighted by activations on task calibration; removal of a neuron as a triple (`gate`/`up` rows, `down` column); healing without distillation (§4.7), the `--heal-loss ce` mode. Layer-group selection (depth compression) was **not used**, by the author's decision |
| **Frozen Core Decomposition (FCD)**, Zenodo, DOI 10.5281/zenodo.18006952 | healing as training of the tail layers over a frozen core. GPU healing trains the layers from `train_from`, and the output of the frozen prefix is computed once and kept on the host |
| **DTG-MA**, Zenodo, DOI 10.5281/zenodo.18088857 | the phase-A recipe: mask initialization, the pressure schedule, checkpoint selection by the hard mask on held-out data |
| **NVG (Null-Vector Gravity)**, an encoder below 1 bit per weight: sign topology Ĉ, a closed-form metric Ť, discrete refinement of the signs; the `blr` container | the encoding in the experiment on method order (§5.6) and the metric training of `nvg-train` |
| **Patent 1**, USPTO application **19/452,440** (filed 19 January 2026): *System and Method for Unsupervised Multi-Task Routing via Signal Reconstruction Resonance* | the argmin-reconstruction-error rule that the NVG encoder applies when choosing candidate signs (indirectly, through NVG) |

## 3. Method

### 3.1 Teacher data

A teacher answers every prompt. By default it is the model itself. It can also
be another CMF with the same tokenizer or a model behind any OpenAI-compatible
API (`openrouter:`, `openai:`, `api:` + `--api-base`). API answers arrive as text
and are turned into tokens by the tokenizer of the model being compressed. The
reasoning (`reasoning` / `reasoning_content` / `<think>`) is placed before
`</think>`, the way the student's template continues it. An answer cut off by
the token limit is stored as a truncated record without the end token. Records
are split into train/val/test by prompt (80/10/10, stratified by category). No
compression stage sees the test split.

### 3.2 Illumination

The source model runs over the teacher's answers. For the `down_proj` input of
every layer it accumulates the Gram matrix `Σ = Σ_t a_t a_tᵀ` (separately on
train and val). The importance of neuron j is

`I_j = ‖W_down[:, j]‖² · ⟨a_j²⟩`,

that is, the neuron's contribution to the layer's output on the task data. This
is the activation-weighted channel importance of Patent 12, only without the
SVD: it scores the `down_proj` output itself.

### 3.3 Osmotic folding (Patent 15)

A layer keeps the neurons S and removes P. Instead of a plain cut, the
contribution of the removed neurons flows into the kept ones by a
covariance-weighted reconstruction, the osmotic folding of application
19/731,402 (§[0025]–[0026]):

`W_down' = W_S + W_P · Πᵀ`,  `Π = (Σ_SS + ε·I)⁻¹ Σ_SP`,

where Π holds the regression coefficients that best predict the activation of
each removed neuron from the kept ones, and ε is a relative ridge. In this
implementation the ridge, and the choice between folding and a plain cut, are
decided by the output error on val. The error of a candidate is computed
through the Gram of the removed columns `G = W_Pᵀ W_P`:

`‖W_P − A‖²_Σ = ⟨G, Σ_PP − 2·Πᵀ Σ_SP + Πᵀ Σ_SS Π⟩`.

This form costs k²p + 2kp² operations instead of h(k² + kp + p²) + hkp for
building `A = W_P Πᵀ` explicitly.

### 3.4 Healing (Patents 15, 16 and 12)

The folded model trains further. This is healing in the sense of application
19/731,402 (§[0022]): a repair of the perturbation the fold introduced, on the
task data. The MLP weights of the tail layers train: f32 copies of the weights
with Adam, with the f16 planes updated every step. The layers below are frozen
(FCD), and the backward pass does not reach them. The loss mixes `α·CE` on the
teacher's text with `(1−α)·KL` to the top-64 distribution of the uncompressed
source. This is the restoration scheme of application 19/738,763: the norms and
FFN tensors train on a mix of cross-entropy and a teacher-anchored divergence,
the rest stays frozen. With α = 1 it is healing without distillation (Patent 12,
§4.7). The best checkpoint is chosen on val.

### 3.5 DTG-MA masks inside the GPU trainer

A gate sits between SwiGLU and `down_proj` in every layer: `h ↦ h ⊙ s`, where
`s = σ(m)`. The forward and backward passes of the gate are two fused kernels.
The gradient `∂L/∂s_j = Σ_t ∂L/∂(h s)_{t,j} · h_{t,j}` is a column reduction on
the GPU. Adam over the logits m (86,016 numbers on the 0.8B) runs on the host.
The L1 pressure `λ·σ(m)` grows at every evaluation until the target sparsity is
reached. Every 25 steps the hard mask `σ(m) > 0.5` is checked on val, and the
first checkpoint at the target is taken. From the trained logits, one common
threshold sets the width of every layer (a multiple of 128), and the order of
the logits picks the neurons themselves. Folding and healing follow, as in
§3.3–3.4.

The gradient was checked on the real 0.8B. With open gates the network matches
the gateless network bit for bit. The gate gradient agrees with a finite
difference: within 0.3 % in the last layer and within 3.5 % in layer 0 at a step
of 0.08. A step of 0.02 in layer 0 spreads up to 33 % because of f16 rounding in
the forward pass, the same behaviour as the already verified norm gradient of
layer 0.

### 3.6 MoE models

Illumination counts how often the router picks each expert on the task's
answers. The file keeps the experts that gather the required share of the
routes; the threshold is found by bisection for `R`. The router is cut down to
the kept experts and chooses among them as before.

## 4. Infrastructure

- **Stages S0–S6** are separate processes with a cache keyed by their inputs. A
  repeated run resumes from the first unfinished stage.
- **The GPU trainer on NVIDIA** (Vulkan, 16×16 cooperative matrices), on
  Qwen3.5-0.8B and an RTX 5090:
  - healing takes 99 s against 6308 s on the CPU;
  - illumination 45 s against 320 s;
  - test NLL 10 s against 230 s, matching the runtime to within 2·10⁻⁴.
- **The GPU trainer on Apple Silicon** (Metal through wgpu, 8×8 simdgroup
  matrices). The GEMM and attention generators are generalized to the 8×8
  fragment. The GDN and sign-training kernels run on 32-wide SIMD groups. A
  healing step of the 0.8B (2 windows × 2048 tokens, the MLPs of all layers) on
  an M4:

  | what is on | s/step |
  |---|---|
  | f32 fallback kernels for attention and GDN | 47 |
  | + attention on 8×8 simdgroup matrices | 27.6 |
  | + GDN kernels on SIMD groups | 17.4 |
  | + a 64×64×32 GEMM tile (3.0 TFLOPS instead of 1.7) | 12.2 |

  The same `compress` on the Mac gives 0.1891 for the source and 0.3642 for the
  fold; on the 5090, 0.1891 and 0.3643.
- **The teacher behind an API.** Qwen3.8-27B via OpenRouter, coding prompts:
  - 160 answers without reasoning, 32 requests in parallel: 419k tokens in
    2 minutes for $0.91;
  - with reasoning and an 8192-token limit, 9 answers of 12 are cut off inside
    the reasoning;
  - providers of one model differ in price by 2.5×, and some serve it in fp4.
    Choosing the cheapest provider with 8/16-bit weights lowers the price per
    token by 30 %.

## 5. Experiments

Models: Qwen3.5-0.8B and Qwen3.5-2B. The task is 483 coding prompts, and the
teacher is the model itself. After filtering, the 0.8B kept 326 records, with a
test split of 34 records (39,315 answer tokens). The metric is NLL on the
teacher's test answers. The quality gates are looping, answer length relative to
the teacher, and the share of self-terminated answers on 16 test prompts.

### 5.1 The fold: moving the contribution, and width allocation

| comparison | result |
|---|---|
| mean FFN output error on val, 0.8B, 75 % of the FFN: cut → osmotic folding | 0.1053 → 0.0600 |
| the same, 2B | 0.1028 → 0.0625 |
| 0.8B without healing (an early dataset, source 0.504): plain cut / osmotic folding | 0.753 / 0.629 |
| width allocation: uniform / by error curves / weighted by layer energy | 0.621 / 0.629 / 0.635 |
| 2B fold: edge layers protected (full width) / unprotected | 0.330 / 0.320 |
| ridge chosen over 1130 layers | 0.1 — 86 %, 0.03 — 10 %, 0.01 — 3 %, 0.003 — never |

### 5.2 Healing

0.8B, 75 % of the FFN, fold 0.364.

| healing loss | test NLL |
|---|---|
| KL to the source's top-64 only | 0.303 |
| CE only (no distillation) | 0.292 (best val at step 150, overfitting after) |
| **mix 0.5·CE + 0.5·KL** | **0.281** |
| mix with an unfreezing schedule (genesis) | 0.285 |

### 5.3 One command: `--ratio`

| model | `--ratio` | parameters | test NLL: source → fold → healed | gates |
|---|---|---|---|---|
| 0.8B | 0.9 | 0.773B → 0.697B | 0.189 → 0.394 → 0.305 | PASS |
| 0.8B | 0.8 | 0.773B → 0.622B | 0.189 → 0.698 → 0.524 | FAIL (loops) |
| 2B | 0.8 | 1.943B → 1.565B | 0.185 → 0.464 → 0.354 | PASS |

A third of the 0.8B's parameters are embeddings, which are not compressed.
`--ratio 0.8` removes 30 % of the FFN width, and that is already the limit.

### 5.4 Quantization against pruning on the 2B

| file | size | test NLL |
|---|---|---|
| source | 1.97 GB (q8) | 0.185 |
| q4tp only | 1.03 GB | 0.225 |
| 75 % of the FFN + healing + q4tp | 0.92 GB | 0.288 |

A file 11 % smaller costs +0.063 NLL. Pruning is expensive for small dense
models: the compression ratio has to come from quantization, with pruning
applied on top of it.

### 5.5 Choosing neurons: illumination against a DTG-MA mask

Training the mask on the GPU takes 0.37 s per step against 16.7 s for the
`skillbake` implementation on the host (45 times faster). The whole mask of the
0.8B trains in ~3 minutes instead of 70. With a linear growth of the L1
pressure, pruning stalled at 15 %: the task gradient on the remaining gates
grows faster than the penalty. With growth of ×1.5 per evaluation, the 25 %
target is reached in 325 steps.

| model, target | fold: illumination / DTG-MA | after healing: illumination / DTG-MA |
|---|---|---|
| 0.8B, 75 % of the FFN | 0.364 / **0.339** | **0.281** / 0.282 |
| 0.8B, `--ratio 0.9` | 0.394 / **0.372** | **0.305** / 0.310 |
| 2B, 75 % of the FFN | 0.320 / 0.320 | **0.255** / 0.261 |

The mask chooses neurons better than illumination: the fold is better on the
0.8B. But healing levels the results. Illumination is cheaper and stays the
default.

### 5.6 Order of methods: NVG and DTG-MA

0.8B, 75 % of the FFN, both variants of the same size.

- **A, DTG-MA → NVG.** The mask-compressed f16 model is encoded by NVG, and then
  the metric trains further with `nvg-train`.
- **B, NVG → DTG-MA.** The whole model is encoded, then DTG-MA masks train on
  the NVG model with the same KL to the teacher; the closed neurons are cut
  straight out of the binary low-rank tensors (`gate`/`up` rows, `down` columns,
  without re-encoding); then the metric trains further the same way as in A.

To match the sizes, branch B encodes the FFN at the bit rate divided by 1.074.
After the cut, the tensor ranks then equal those of branch A. The teacher, the
data (task texts, 2 epochs × 262k tokens) and the top-k cache are shared by the
branches.

| bit rate, size | branch | test NLL | loops | KL on held-out data |
|---|---|---|---|---|
| 2 bits/weight, 0.398 GB | A, no further training | 2.587 | 16/16 | 2.217 |
| | A after `nvg-train` | 1.931 | 16/16 | 1.577 |
| | B right after the cut | 1.681 | 16/16 | 1.225 |
| | **B after `nvg-train`** | **1.618** | 14/16 | **1.160** |
| 3 bits/weight, 0.437 GB | A after `nvg-train` | 1.340 | 7/16 | 0.887 |
| | **B after `nvg-train`** | **1.277** | 13/16 | **0.774** |

Source model: test NLL 0.189.

**The main observation.** On an NVG model, DTG-MA masks do not damage, they
repair. Without a single step on the weights, only by closing 27 % of the
neurons, they lower the KL to the teacher from 2.17 to 1.23. The mask finds the
neurons the encoding distorted most and removes them. "NVG first" wins at equal
size, but the gain shrinks as the bit rate grows: −0.31 nats at 2 bits, −0.06 at
3. At 3 bits branch B also loops more often (13/16 against 7/16).

### 5.7 MoE: removing experts

LFM2.5-8B-A1B, 80 coding prompts. The test split has only 2 records, so the
numbers are indicative.

| `--ratio` | parameters | test NLL |
|---|---|---|
| source | 8.468B | 0.280 |
| 0.9 | 7.620B | 0.289 |
| 0.8 | 6.772B | 0.332 |
| 0.65 | 5.494B | 0.399 |

At `--ratio 0.8` the layers keep 20–30 experts of 32. Without healing, 10 % of
the parameters go at a cost of +0.009 nats.

### 5.8 Choosing the teacher

0.8B, `--ratio 0.9`. Answers of Qwen3.8-27B through an API (160 prompts) against
the model's own answers (483 prompts). Each model is scored on the test splits
of both teachers.

| compressed model | test NLL on the 27B's answers | test NLL on its own answers |
|---|---|---|
| teacher: 27B (API) | **0.492** | 0.442 |
| teacher: the model itself | 0.564 | **0.305** |

Each model is better on its own teacher's distribution, and the 27B variant also
had a third of the data. NLL does not tell which teacher is better. That needs a
check of the answers themselves, for example by executing the code.

## 6. Findings

1. **Osmotic folding (Patent 15), not the choice of neurons, decides the
   quality of compression.** Moving the contribution of the removed neurons in
   closed form halves the FFN output error. A plain cut in its place hurts NLL
   more than any other parameter.
2. **Healing erases the difference in neuron selection.** A trained DTG-MA mask
   selects better than illumination, but after healing the results match (0.8B)
   or illumination is slightly ahead (2B).
3. **Pruning as a quantization repair.** DTG-MA masks on a model encoded by NVG
   below the usual precision remove the neurons the encoding damaged most and
   nearly halve the KL to the teacher without training the weights. Hence the
   order for a hybrid: quantization first, then the mask and the cut.
4. **The mask pressure must grow geometrically.** A linear L1 schedule stops
   pruning at ~15 %.
5. **Small dense models get their compression from quantization.** Width
   pruning is expensive, and its place is on top of quantization.
6. **The fold's ridge is almost always 0.1.** This suggests the optimal ridge
   lies above the old grid. The new grid `0.03, 0.1, 0.3` gives the same result
   at one factorization less.

## 7. Limitations

- Only Qwen3.5-0.8B and 2B (plus one MoE), one task set (code), one seed. The
  test split is 34 records on the 0.8B; differences below ~0.01 nats are within
  the noise.
- NLL on the teacher's answers is not the quality of the solution. Code needs a
  check by execution.
- The branches of the order experiment at 2–3 bits per weight are broken in
  absolute terms: they loop almost everywhere. The order of the methods is
  measured, but there is no usable model at this bit rate yet.
- On an M4, healing the 0.8B is bound by a GEMM at ~3 TFLOPS and takes hours.
  The full runs were done on an RTX 5090.

## 8. Reproduction

```bash
# task → compressed model (the model is its own teacher)
cortiq compress Qwen3.5-0.8B.cmf prompts.jsonl q08-task.cmf --ratio 0.9

# neurons chosen by a trained DTG-MA mask (Patent 2)
cortiq compress Qwen3.5-0.8B.cmf prompts.jsonl q08-dtgma.cmf --ffn 0.75 --select dtgma

# a teacher behind an API
OPENROUTER_API_KEY=… cortiq compress Qwen3.5-0.8B.cmf prompts.jsonl q08-27b.cmf --ratio 0.9 \
    --teacher openrouter:qwen/qwen3.8-27b --no-think --api-concurrency 32

# the NVG → DTG-MA order: encoding, mask and cut, metric training
cortiq convert --model Qwen/Qwen3.5-0.8B --quant blr --bpw 2.0 --blr-paths 2 --calib task.txt --output nvg.cmf
cortiq nvg-train nvg.cmf --teacher q08-q8_2f.cmf --calib task.txt --tokens 262144 --epochs 2 \
    --mask-keep 0.75 --output nvg-dtgma.cmf

# several models compared on the same test records
cortiq compress Qwen3.5-0.8B.cmf task.tdata run.cmf --ffn 0.75 --compare nvg-dtgma.cmf --compare other.cmf
```

The follow-up on a large model, Qwen3.8-27B → 14B on one RTX PRO 6000, is
[QWEN38_27B_TASK_COMPRESSION.md](QWEN38_27B_TASK_COMPRESSION.md).

## References

1. O. U. Kirichenko. USPTO application 19/452,464 (Patent 2), filed 19 January
   2026: *System and Method for Dynamic Task-Guided Neural Network Compression
   with Catastrophic Forgetting Prevention*.
2. O. U. Kirichenko. USPTO application 19/731,402 (Patent 15), filed 6 July
   2026, CIP of 19/452,440 and 19/452,464: *Unified Execution Architecture for
   Serving a Plurality of Specialized Language Models from a Single Shared
   Backbone via Dynamically Overlaid Compressed Delta Representations Without
   Materializing Separate Models*.
3. O. U. Kirichenko. USPTO application 19/738,763 (Patent 16), filed 13 July
   2026: *Training-Free Conversion of Softmax Attention Layers of a Pre-Trained
   Language Model into Constant-Memory Streaming Attention … and
   Teacher-Anchored Surroundings-Only Quality Restoration*.
4. O. U. Kirichenko. Patent 12 (AWNP): *System and Method for
   Activation-Weighted Structured Compression of Neural Networks with
   Architecture-Preserving Layer Selection*.
5. O. U. Kirichenko. USPTO application 19/452,440 (Patent 1), filed 19 January
   2026: *System and Method for Unsupervised Multi-Task Routing via Signal
   Reconstruction Resonance*.
6. O. U. Kirichenko. *Frozen Core Decomposition (FCD): An Architectural Approach
   to Continual Learning Without Catastrophic Forgetting*. Zenodo, 2025.
   DOI 10.5281/zenodo.18006952.
7. O. U. Kirichenko. *DTG-MA: Hard Guardrails Against Catastrophic Forgetting
   Through Attention Masking and Task Graphs*. Zenodo, 2025.
   DOI 10.5281/zenodo.18088857.
8. O. U. Kirichenko. *Topology First: Sub-1-Bit LLM Compression by Separating
   Sign Topology from Metric (NVG)*, draft, 2026; *Null-Vector Gravity*,
   research notes.
9. Lee, Kim, You, Kim. *LittleBit: Ultra Low-Bit Quantization via Latent
   Factorization*. NeurIPS 2025 — the layout of the `blr` container.
10. Frantar et al. *GPTQ: Accurate Post-Training Quantization for Generative
    Pre-trained Transformers*. arXiv 2210.17323.
