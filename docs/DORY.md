# Nemotron-Labs Dory on cortiq

*Languages: **English** · [Русский](DORY.ru.md) · [中文](DORY.zh.md)*

[Dory 1.0 4B](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16) is NVIDIA's
looped transformer: 16 input sub-layers, 40 recurrent ones and 16 output ones, 4B parameters.
The recurrent block runs several times over the same weights, so one checkpoint serves
1, 2 or 4 loops with no retraining. It reasons in `<think>` and calls tools.

## Run

```bash
cargo install cortiq-cli
hf download infosave/Nemotron-Labs-Dory1.0-4B-cmf dory-turbo.cmf --local-dir .
cortiq run dory-turbo.cmf --prompt "Write a Rust function that reverses a list." --no-think
```

| file | weights | size | wikitext-2 ppl | use |
|---|---|---:|---:|---|
| `dory-turbo.cmf` | 4-bit GPTQ | 2.09 GB | 14.22 | the same weights as `q4tp-v2` with a router and a far field; **default** |
| `dory-q4tp-v2.cmf` | 4-bit GPTQ | 2.09 GB | 13.96 | two loops on every token |
| `dory-q8_2f.cmf` | 8-bit | 4.01 GB | 13.44 | closest to the original |

Perplexity: twelve 512-token windows of wikitext-2 test, read token by token. The bf16
original scores 13.42 on the same windows.

## Loop modes

| option | meaning |
|---|---|
| `--loops N` | a fixed loop count, 1 to 16; on a turbo file it also turns the router, the far field and the rest off |
| `--loops auto` | one or two passes per token: the file's router, or the top probability in a file without one |
| `--kv-cache per_loop\|last_loop` | one cache per loop, or one for all of them |
| `--o1 m,w,sink` | the far field on the full-attention layers, sealed at 8192 rows |
| `--full-window N` | cut the full-attention layers to a window instead |

`CMF_DORY_O1=m,w,sink,seal` sets the seal position too, and `CMF_DORY_O1=off` turns a turbo
file's far field off. `CMF_DORY_GATE=conf:0.7` or `recon:FILE:bias` chooses the gate.
`CMF_DORY_PROMPT_LOOPS=N` sets the passes over a prompt, `CMF_DORY_EXACT=FILE:rows` the choice of exact
rows, and `CMF_DORY_EXACT_LOG=FILE` writes every row's score.

## What the turbo file does

A turbo file holds the same weights under a header that switches on five things.

![One pass or two](https://huggingface.co/infosave/Nemotron-Labs-Dory1.0-4B-cmf/resolve/main/assets/router_en.png)

**Last-loop cache.** Each recurrent attention layer keeps one cache, and every pass overwrites this
token's row. The loop count can change from token to token and leave no holes.

**Resonance router.** After the first pass, the state is rebuilt by two isolated PCA subspaces of rank
16, fitted on tokens where a second pass changed the answer and on tokens where it did not. The lower
error decides whether the token leaves the loop. Nothing is trained by gradient, and the decision costs
no pass through the output block and the head. About 85 % of tokens take one pass.

**Far field.** Full-attention layers keep the first rows and the last 1536 rows exact, and everything
else in 32 landmarks per head (`nystrom`). Sliding layers already have a window of 1024. Device memory
stops growing, and so does the cost of a token.

![What a long prompt keeps on the card](https://huggingface.co/infosave/Nemotron-Labs-Dory1.0-4B-cmf/resolve/main/assets/far_field_en.png)

**Exact rows.** A far field alone loses a password: an exact model finds one in 100 000 characters of
text, and a far field does not. So the rows it would lose stay exact. For each KV head of each full
layer, two numbers per row: how far a key lies from the span of the landmarks, and the value's norm.
Each becomes a rank within the prompt, and a linear fit over the 72 ranks scores the row. Before the
prompt's last 512 rows the 512 best rows and their neighbours leave the far field, which is built anew
without them. That order matters: a question carries what it found in the text on to the answer, so it
has to read the exact fact itself. A password's rows land in the first places of tens of thousands.

![How passwords stay exact](https://huggingface.co/infosave/Nemotron-Labs-Dory1.0-4B-cmf/resolve/main/assets/exact_rows_en.png)

**One-pass prompt.** Under the last-loop cache a prompt row leaves only its cache rows behind, and nobody
reads its logits. All chunks but the last take one pass, which costs 1 % in the perplexity of what
follows.

## Numbers

![Standard and turbo](https://huggingface.co/infosave/Nemotron-Labs-Dory1.0-4B-cmf/resolve/main/assets/compare_en.png)

Decode in tok/s. The first row of each group is 12 real prompts of 200 tokens through `cortiq run`,
the others `cortiq bench --core` after a prompt of that length.

| | `dory-q4tp-v2` | `dory-turbo` |
|---|---:|---:|
| **Vulkan, NVIDIA L4 (24 GB)** | | |
| real prompts | 49.6 | **60.7** |
| context 1 000 | 41.1 | **60.6** |
| context 4 000 | 35.1 | **51.8** |
| context 16 000 | 21.5 | **52.2** |
| context 32 000 | 14.1 | **54.0** |
| context 64 000 | prompt not read in 25 min | **53.0** |
| first token at 16 000 / 32 000 / 64 000 | 78 s / 360 s / over 25 min | **53 s / 112 s / 228 s** |
| device KV memory at 4 000 / 32 000 | 1.0 GB / 4.3 GB | **0.64 GB / 0.50 GB** |
| **Metal, MacBook Air M4 (24 GB)** | | |
| real prompts | 25.7 | **32.9** |
| context 1 000 | 22.1 | **34.6** |
| context 4 000 | 20.3 | **31.0** |
| context 16 000 | 14.8 | **29.3** |
| first token at 16 000 | 211 s | **134 s** |
| **CPU** | | |
| M4, context 256 | 18.0 | **27.7** |
| M4, context 2 000 | 12.1 | **18.3** |
| EPYC 9254, 24 cores, context 256 | 7.6 | **11.6** |
| EPYC 9254, 24 cores, context 4 000 | 5.3 | **8.2** |

About 85 % of tokens take one pass, on Wikipedia and on a code answer alike. `dory-q8_2f.cmf` on
real prompts: 32.0 tok/s on Vulkan, 15.4 on Metal, 13.8 on the M4 CPU. Long windows, perplexity of
two windows of 16384 tokens: exact 8.10, turbo's near and far fields 8.85, a cut to 2048 rows 11.03.

## Building a turbo file

The router is fitted on the model's own answers. The scripts are in `scripts/dory/router`, and
each one's first lines give its arguments:

| script | what it does |
|---|---|
| `gen_samples.py` | answers of the model to a mixed prompt set |
| `loop_probe.py` | teacher-forced, the logits of 1, 2 and 4 loops and the state after a pass |
| `router_fit.py` | two topologies on half the answers, AUC on the other half |
| `make_bin.py` | the topologies as a flat file, and the bias of a few shares of one-pass tokens |
| `illum_feat.py` | prompts with a planted fact, each row's residuals and value norms |
| `illum_budget.py` | the fit, its coverage on prompts it did not see, the weights for the file |

```bash
cortiq dory-turbo dory-q4tp-v2.cmf dory-turbo.cmf --topologies phi1.bin --bias -0.000258 \
    --o1 32,1536,4,8192 --exact-weights keep.bin --exact-rows 512
```

A higher bias keeps more tokens on two passes. A window, sinks and exact rows up to 2052 keep sealed
prompt chunks on the batch kernels; past that they go row by row.

**What stays out.** Rows that arrive after a prompt ends can only stay in the window or go to the far
field. A turn that diverges from a sealed conversation reads the whole prompt again and chooses again.

## Conversion

```bash
hf download nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16 --local-dir bf16
cortiq convert --model bf16 --quant f16 --output dory-f16.cmf --also-output q8_2f=dory-q8_2f.cmf
cortiq quantize-gptq dory-f16.cmf --codec q4tp --calib-model dory-q8_2f.cmf \
    --calib wiki_train.txt=0.6 --calib data_chat.txt=0.25 --calib data_code.txt=0.15 \
    --tokens 32768 --window 2048 --lambda 0.01 --output dory-q4tp-v2.cmf
```

`scripts/dory/` holds the reference, the parity and the measurement scripts, and
`JOURNAL_DORY_2026-11-04.md` the log of what was tried.

## Parity

Greedy text on the raw prompt is identical to the f32 reference. Against vLLM on the vendor's
weights: 64 of 64 tokens, largest difference in a logit's log-probability 3e-4, including a
1500-token prompt. q4tp on Vulkan against the CPU: 8.9e-4; Metal against the CPU: 7.7e-3.
Attention-kernel changes hold the perplexity to three digits on Vulkan and Metal.

## License

Weights: [OpenMDW-1.1](https://openmdw.ai/license/1-1/). The router and the far field are subjects of
U.S. patent applications No. 19/452,440 and No. 19/738,763; see [PATENTS.md](../PATENTS.md).
