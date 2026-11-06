# Qwen3.8-27B Turbo, decode with drafts on a Mac, 06.11.2026

Goal: more than the 6.4 tok/s a Mac decodes this model at.
Stand: MacBook Air M4, 24 GB, fanless. Greedy, `--no-think`, 220 tokens, one run at a time with 30 s before each.
A pair's texts are the same in every row below.

## Where it stands

| answer | without drafts | drafts, seven deep (before) | drafts now |
|---|---:|---:|---:|
| an essay | 6.4 | 6.2 | 10.1 |
| an explanation in Russian | 6.4 | | 10.7 |
| a Rust function with tests | 6.4 | 9.8 to 12.5 | 12.1 |
| quotes from the prompt | 6.4 | | 15.0 |
| an edit of the prompt's code | 6.4 | | 20.5 |

After a prompt of 3 600 tokens read with the Neural Engine: 6.0 → 8.3 tok/s, first token 27.9 and 31.0 s, one run each.
Drafts are on with `CMF_GRAPH_SPEC=1`; without it native Metal decodes plain. `q27_spec_decode.sh` runs the table.

## What bounds it

A token reads 13.6 GB of q4tp weights. One vector goes at 92 to 95 GB/s, the memory's speed: 160 ms a token.
A round of drafts is one such pass over b = drafts + 1 rows, so the speed is tokens a pass over the pass's time.
The narrow GEMM (`q4tp_mul_mm_n8`) reads 60 to 67 GB/s at any batch up to eight: its 8 × 8 × 8 multiply takes 64
weights whatever the batch, so a verify of two rows cost as much as one of eight.

The draft chain of this model: the first draft holds in 0.80 to 0.86 of rounds, the second in 0.71 of those, the third
in 0.23 to 0.5. A chain of seven mostly pays for rows that are thrown away.

A round on the code prompt, medians in ms:

| drafts | chain | verify | commit | round |
|---:|---:|---:|---:|---:|
| 1 | | 164.4 | | 176.1 |
| 2 | 10.8 | 187.9 | 5.3 | 203.6 |
| 2, verify on the GEMM (`CMF_METAL_MVS=0`) | | 224.1 | | 241.0 |
| 3 | | 226 | | 246 |
| 7 | | | | 281 |

## What gave it

- **A batched matvec for two and three rows** (`q4tp_matvec_b2`, `_b3`). Four rows a simdgroup, a group of 32 columns
  in four steps of eight; a step's activations of every vector sit in registers, and a row's masked nibbles are
  unpacked once and multiplied into each. 95.7 GB/s for two vectors and 82.4 for three. Its rows against the plain
  path: logits equal to four decimal places and hidden states to five, where the GEMM's differ by 0.007 to 0.009 and
  0.05. On the code prompt 9.7 → 11.3 tok/s at two drafts.
- **The step loop has to stay a loop.** Unrolled, the compiler stages every step's activations ahead, the registers
  a thread holds grow, and fewer threads fit the GPU: 17.4 and 30.6 ms a pass for two and three vectors against 8.3
  and 9.6 with `#pragma clang loop unroll(disable)`. The GEMM takes 11.7 at any batch, one vector 8.6.
- **Depth by what the chain holds.** A hold rate per depth, given that the draft before it held, and a table of what
  a round of each depth costs in plain tokens (1.10, 1.275, 1.54, then 1.59 to 1.76 on the GEMM). The depth is the
  one with the most tokens per cost; a deeper one has to win by 4 %. Depths not tried drift towards the deepest
  tried one while it holds in nine rounds of ten. Two drafts on most text.
- **Prompt lookup.** Where the tip and the head's drafts repeat an earlier stretch of at least drafts + 3 tokens, the
  tokens that followed it there continue the chain up to seven. On the edit prompt 16 rounds of 34 gave 8 tokens in
  about 276 ms. It rests for 24 rounds when it adds less than one accepted draft a round. `CMF_SPEC_LOOKUP=0`
- **The chain's input stage as a command of its own**, below.

## The slow first rounds

The first five rounds of an essay took 362 ms in place of 204: the draft chain's command waited 165 to 198 ms for
9 ms of work, and the plain tokens timed after them took 242 ms in place of 160.

What it was not, each measured:
- other work on the GPU: 250 ms of sleep before the chain changed nothing, and the replay on the second queue made
  synchronous changed nothing;
- the working set: 14.1 GB allocated against 18.2 GB recommended, and no buffer was created in those rounds;
- the weights being prepared again: an empty command and a blit from the weights 1.5 and 13.2 GB into the file,
  right before the chain, took 0.1 to 0.7 ms;
- the K/V mirror: no resync.

What it was: `kernelStartTime` to `kernelEndTime` of that command was 150 to 170 ms, and the GPU's own time 8 to 10.
The command read the single 13.6 GB weight buffer in two places, the embedding table 1.5 GB into the file and the
draft block at 13.1 GB. Cut by stage:

| the chain as | chain's wait, ms |
|---|---:|
| one command | 165 to 198 |
| a cut after the input stage (the embedding and the input projection) | 8.5 to 9.8 |
| a cut after the block, after the head, or after the argmax | 170 to 190 |
| a command a stage | 9.4 to 11.3 |
| one command, weights in windows cut at tensors (`CMF_METAL_WINDOWS=1`) | 10 to 17 |

So the input stage is committed first. Why the driver holds the joined command is not known; the state ended by
itself at the first round without a warm-up of the draft head, and never began after a prompt of 3 300 characters.
An essay 9.4 → 10.1 tok/s, the edit 18.4 → 20.5. `CMF_MTP_CHAIN_JOIN=1` joins them again.

## What lost

In the kernel lab (`q27_batch_lab.sh`, `q27_batch_lab.metal`), against 8.3 and 9.6 ms a pass:
- four vectors on this kernel: 12.7 against the GEMM's 11.7, so four rows and more stay on the GEMM;
- activations as half or int16 in registers, row pointers hoisted out of the loop, the 16^-k weights folded into
  the activations: slower or equal;
- steps of 4, 16 and 32 columns, and 2, 3, 6, 8 and 16 rows a simdgroup: slower or equal.

In the depth rule: a survival rate per depth let a stale deep prior win when the shallow ones fell (depth 6 at round
7); without the 4 % margin a chain that never holds past two drafts spent 46 rounds of 400 on deeper ones.

## Measurement traps

- A wait on a build's log that was cut by `head` returned at once: the next runs fell on the compile and showed
  245 to 260 ms for the chain. Check the binary's time and that no `cargo` runs.
- A text's hash compares within one script only: the one taken in the shell and the one taken in Python differ.
- The trial's rule for a round that pays without timing plain tokens asks for 3.5 tokens a round, and a round of two
  drafts gives 3 at most, so two plain tokens are always timed: about 150 ms once an answer.

## Tools

`CMF_GRAPH_SPEC_TIME=1|2` prints every round and its host stamps. `CMF_METAL_GPUPROF=1 CMF_METAL_CMD_TIMES=1` prints
each token-graph command's time in the kernel and on the GPU, and what Metal holds for the process at every chain.
`CMF_METAL_VERIFY_CHECK=1` compares a verify's rows with the plain path. `q27_spec_rounds.sh` gives a round's medians
at a pinned depth, `q27_spec_prompts.sh` writes the five prompts.

## Not done

- Drafts on by default on native Metal: they still need `CMF_GRAPH_SPEC=1`.
- Training the draft head as a chain (`mtp-heal --chain 3`): its third draft holds in 0.23 to 0.5.
- The kernel for three vectors at the speed of two, 82 → 95 GB/s.
- Past the seal, at 16 000 tokens, drafts were not measured again.
- With drafts the first token at 3 600 tokens came 3 s later in the one run made.
- Windows cut at tensors for every model on a Mac: it removed the delay too, and was measured on this model only.
- The page on Hugging Face says nothing of this until a release has it.
- No Mac with a fan and no other chip: every number is one M4.
