# Qwen3.8-27B Turbo, the draft head trained as a chain, 06.11.2026

Goal: drafts that hold deeper, so that a round on a Mac gives more tokens for the same pass over the weights.
And an experiment the owner asked for: a draft made of resonance (patent 1) and NVG, against the MTP block.
Stand: one RTX PRO 6000 Blackwell, 96 GB, 27 cores, 188 GB of RAM. `/root/q27` on the pod's own disk.
The pod's own scripts are in `q27_mtp_pod/`: `base.sh` (packages, the model, a build with no LTO), `build.sh`,
`train.sh` and `train2.sh` (trainings A, B and C), `measure.sh` and `sets.sh` (a file's chain per data set),
`resq.sh` (the trained-resonance queue), `hfup.py` (a file into the private experiments repo, sha256 checked).

## What a round is worth

A round of k drafts costs 1.10, 1.275, 1.54, 1.59, 1.65, 1.70, 1.76 plain tokens on a MacBook Air M4 for k = 1..7
(`JOURNAL_Q27_MAC_DECODE_2026-11-06.md`): rows past the third are nearly free. It gives 1 + h1 + h1·h2 + … tokens,
where hj is the share of rounds whose draft j held among those whose draft j − 1 did. So the lever is hj at every
depth, and a chain that holds deep is worth more on a Mac than on a card whose verify grows with its rows.

## How it is measured

- **Real rounds.** `q27_spec_hold.sh CORTIQ MODEL PROMPT_DIR K TOKENS` answers twelve held-out prompts (the five of
  `q27_spec_prompts.sh` and seven more: prose, Russian, code, a list, a summary of wikitext-2 test) with drafts pinned
  K deep and no prompt lookup, and counts what the verify accepted in every round. Greedy, `--no-think`, 220 tokens.
- **In the trainer.** `cortiq mtp-heal` now prints, per chain step, the share of held-out rows where the draft's token
  is the trunk's among the rows where the steps before held and every token the chain was fed is the trunk's own
  choice. On the code set before training: 0.892 / 0.797 / 0.707. Real rounds on the four code prompts:
  0.89 / 0.83 / 0.77. The trainer's chain reads its own step's rows as context where a round reads the trunk's; the
  two numbers agree, so that difference does not decide anything here.

## The head as it is

Real rounds, drafts pinned three deep, the same twelve prompts:

| where | rounds | depth 1 | 2 | 3 | tokens a round |
|---|---:|---:|---:|---:|---:|
| the pod, wgpu | 550 | 0.836 | 0.739 | 0.715 | 2.90 |
| MacBook Air M4, Metal | 768 | 0.805 | 0.723 | 0.696 | 2.79 |

The Mac's chain holds as the pod's does. The 0.23 to 0.5 at depth three in the Mac journal came from the few rounds
its depth rule spent there on two prompts; there is nothing to repair in the Metal chain.

A first run seven deep here (299 rounds, 3.75 tokens a round) was not pinned: the trial kept the rounds that paid,
so it read high. The pinned numbers are in "Real rounds, pinned seven deep" below.

A round is counted only while speculation is on: the trial stops it where it does not pay, so the rounds of a weak
stretch are fewer than its tokens.

## Data

- `q27-code-t27.tdata`: answers of the bf16 model to code prompts, 400 000 tokens of its train split.
- `own.tdata`: the turbo file's own greedy answers to 276 prompts of `q27_mtp_prompts.py` (prose and explanations in
  English and Russian, code, lists, short questions, work on a text in the prompt, edits of code in the prompt),
  up to 256 tokens each: 116 942 tokens, 61 549 of them answers. 58 tok/s on this card with drafts.
- `own2.tdata`: its answers to 160 more (`q27_mtp_prompts.py … prose`: essays, explanations, histories and advice
  in English and Russian), 320 tokens each: 56 476 tokens, 51 200 of them answers. Prose is where the chain is weakest.
- 282 windows of 2048, every tenth held out. None of it repeats the twelve prompts that measure the result.

## The trainer

`cortiq mtp-heal` got, before any training:
- several data sets (`--data A.tdata@TOKENS,B.tdata`), their records packed into windows: a short answer shares its
  window with the next record, where a window a record used to leave most of 2048 rows as padding;
- `--trunk-cache DIR`: the trunk's pass (hidden rows, top-64) per data set stays on disk, and a later run on the same
  trunk, data and window reads it back in 8 to 29 s. The trunk reads 1 250 to 1 500 tok/s here beside a decode;
- the chain's hold rate per step, before and after, at every evaluation, and of the draft as each output file holds
  it. The checkpoint written is the one whose chain keeps the most drafts a round on held-out rows
  (h1 + h1·h2 + …), where it used to be the lowest KL: six steps of training lowered the KL from 1.54 to 1.26 and
  the hold rates from 0.820 / 0.744 to 0.794 / 0.724;
- `--hard F`: F of a target row's mass goes onto the trunk's own token, the one a greedy verify accepts;
- `--codec` and `--also-output CODEC=PATH`: the trained draft as q4tp, q8_2f or f16 from one training;
- `--dump DIR`: the chain's output rows and top-4 per step and the tensors a draft reads, for experiments;
- `--epochs 0`: the pass, the rates and the dump with no training, and with a codec named the draft re-encoded.

What a codec keeps of a small change: after those six steps the draft written back as q4tp held 0.820 / 0.742, the
untrained head's rates, and as q8_2f 0.811 / 0.733. A step of 2e-5 is far below q4tp's grid, so a short training
written as q4tp is the old head. The Metal chain takes q4tp and q8_2f draft blocks (`q1_parts`), not f16.

## Three ways to count a chain

- **Every start alike** (`--stride N`): a chain starts every N rows and at N/2, its steps read the trunk's rows as a
  round does. It over-counts the easy starts: a real round starts after a miss more often than a row picked evenly.
- **Rounds in order** (`--epochs 0` with a stride longer than the chain): the next round starts after what this one
  kept, so a round starts after a miss. This is what a decode does; the first draft holds less here (0.790 against
  0.852 for the stock head).
- **Real rounds** (`q27_spec_hold.sh` with `CMF_GRAPH_SPEC_PIN=1`, 36 fresh prompts, 300 tokens each): the engine
  itself, on prompts no training saw.

"Drafts a round" is h1 + h1·h2 + …; a round gives one token more than that.

## Trainings

All on the turbo file's own block, chain of five, 1 270 steps at 2e-5, the three sets, 282 windows; rates on held-out
windows, before → after:

| run | what differs | chain holds, steps 1..5 | drafts a round |
|---|---|---|---:|
| before, every row | | 0.860 / 0.761 / 0.683 / 0.613 / 0.544 | 2.39 |
| A | `--hard 0` | 0.893 / 0.830 / 0.793 / 0.766 / 0.745 | 3.01 |
| B | `--hard 0.5` | 0.892 / 0.832 / 0.793 / 0.766 / 0.743 | 3.01 |
| before, laid out as rounds | `--stride 8` | 0.847 / 0.783 / 0.748 / 0.716 / 0.664 | 2.76 |
| C | `--stride 8`, `--refresh 254` | 0.893 / 0.837 / 0.818 / 0.806 / 0.796 | 3.14 |

The hard target changes nothing. C, trained on rows laid out as a round reads them, is the one kept:
`out/C-f16.cmf`, and from it `C-q8.cmf` (the block as q8_2f; q4tp loses a training this small, see above).

## The same 28 held-out windows, seven deep

Drafts a round (all / code / own / own2):

| draft block | every start alike | rounds in order |
|---|---|---|
| stock, q4tp | 2.83 (3.11 / 2.32 / 1.88) | 2.38 (2.71 / 1.79 / 1.69) |
| stock, the same block as q8_2f | 2.92 (3.21 / 2.38 / 1.87) | 2.44 (2.79 / 1.83 / 1.71) |
| C, q8_2f | 3.67 (4.08 / 3.06 / 2.11) | 3.05 (3.53 / 2.40 / not read: the run was cut) |
| C, f16 | 3.67 (4.08 / 3.04 / 2.12) | 3.05 (3.53 / 2.39 / 1.88) |

Per step in order: stock 0.790 / 0.731 / 0.693 / 0.688 / 0.635 / 0.593 / 0.524, C 0.834 / 0.783 / 0.765 / 0.752 /
0.771 / 0.769 / 0.761. The stock block written as q8_2f in place of q4tp gives 2 to 3 % by itself.

## Real rounds, pinned seven deep

36 prompts no training saw (12 code, 12 English, 12 Russian), 300 tokens each, tokens a round:

| draft block | code | English | Russian | all |
|---|---:|---:|---:|---:|
| stock, q4tp | 4.15 | 2.64 | 2.61 | 2.97 |
| stock, q8_2f | 4.31 | 2.67 | 2.62 | 3.02 |
| C, q8_2f | 4.69 | 2.68 | 2.71 | 3.12 |
| C, f16 | 4.87 | 2.78 | 2.73 | 3.20 |

On prompts the training never saw C gives 13 % more tokens a round on code and 2 % on prose: about a third of what
the held-out windows promised. The windows come from the same prompt families as the training text.

## On the Mac

MacBook Air M4, the old file against the same trunk with C as q8_2f, decode tok/s (plain 6.4 to 6.6):

| prompt | stock q4tp | C q8_2f |
|---|---:|---:|
| code | 12.2 | 11.8 |
| code2 | 13.6 | 15.3 |
| code3 | 12.7 | 13.6 |
| edit | 20.6 | 18.8 |
| quote | 15.8 | 13.9 |
| prose | 10.1 | 10.0 |
| ru | 10.9 | 10.7 |

A round costs 4 to 13 % more with the q8_2f block (205 to 233 ms against 192 to 223), and the file (13 777 MB of
mapping) is over Metal's single-buffer limit on this machine (13 639 MB), so it is mapped as two windows. What the
block gains in tokens a round it pays back in the round's cost: about zero on the Mac as it is. The plain text of the
two files is the same (the trunk's tensors are byte-identical), and drafts of the new file give its own plain text.

Both files are in the private `infosave/cmf-experiments` under `q27-mtp-chain/` (sha256 checked by the API).

## Draft models from HF

Every "-MTP" repository of this model carries Qwen's stock block, quantized or bf16; none has a retrained one. Draft
models trained separately, run here by `q27_ext_draft.py` on the trunk's rows of the same 28 windows
(`mtp-heal --taps DIR=5,19,33,47,61` writes them), drafts a round, seven deep:

| drafter | every start alike | rounds in order (code / prose) |
|---|---|---|
| DFlash 2 (z-lab), 5 layers on 5 taps | 4.06 | 2.95 (3.73 / 1.74) |
| DFlash2-ru | 3.56 | – |
| DSpark v2 (RadixArk) | 3.57 | 2.43 (3.17 / 1.33) |
| stock block | 2.83 | 2.38 (2.71 / own 1.79, own2 1.69) |
| C | 3.67 | 3.05 (3.53 / own 2.39, own2 1.88) |

DFlash 2 holds 0.80 to 0.86 at every depth, but its first draft after a miss is weaker (0.745 against 0.790 stock
and 0.834 C). On code it is the best measured (3.73), on prose it equals the stock block. It would need a port: a
five-layer drafter, the taps of five trunk layers in the prefill and decode graphs, 1 to 2 GB more. EAGLE-4 is under
a closed licence and was deleted unmeasured; RedHatAI's speculator.dspark (8 taps) was not run.

## Resonance and NVG, first experiment

`q27_rn_draft.py` on the dump: resonance picks one of 16 regimes from the trunk's row, a low-rank NVG expert per
regime maps the row to the draft. Held-out rows, steps 1..3, against the block's 0.829 / 0.749 / 0.693 on the same
rows:

- resonance, 16 regimes, NVG experts of rank 512: 0.759 / 0.659 / 0.643; with float experts 0.768 / 0.683 / 0.668;
- no routing, one float expert of rank 2048: 0.784 / 0.699 / 0.668: the routing adds nothing to the expert;
- the block with an NVG correction of rank 1024 on top: 0.856 / 0.792 / 0.734;
- as a gate, resonance separates kept from dropped drafts with AUC 0.78 / 0.72 / 0.69, the block's own probability
  with 0.93 / 0.88 / 0.85.

## Resonance by cmf-decision's mechanism

The owner's order: the method of `infosave/cmf-decision` itself, in place of the head, trained. The mechanism
(`crates/cortiq-decision`): a signal of unit length, one isolated topology a label (the mean and the top K principal
directions of the label's own rows), the label with the smallest reconstruction error, p = softmax(−E/T), a gate.
Rows: 343 846 train, 43 170 calibration, 41 064 held out; the block is right on 0.871 / 0.766 / 0.654 of them at
steps 1..3, the token seen most after the anchor on 0.210.

**As fitted (pure), first draft** (`q27_res_head.py`):

| signal, a topology per next token | K=0 | K=4 | K=16 |
|---|---:|---:|---:|
| trunk's row and the anchor's embedding, 256 + 64 | 0.108 | 0.236 | 0.343 |
| the same, 1024 + 64 | 0.159 | 0.292 | 0.390 |
| the same at full width, 16 384 topologies | 0.099 | 0.237 | – |
| the block's output row, 256 | 0.364 | 0.541 | 0.588 |
| the block's output row, 1024 | 0.438 | 0.579 | 0.612 |

A skill per anchor token (9 090 skills, 35 516 topologies, the trunk's row picks among the tokens seen after the
anchor): an answer for 0.893 of the rows, 0.423 of all right. Choosing among the block's four options (the trunk's
token is among them on 0.969): the topologies alone 0.60 to 0.72, with the block's probability 0.872 against 0.871.

**Trained** (`q27_res_train.py`: centres, directions, gamma and a bias trained by cross-entropy on −E/T at the full
width, 16 384 topologies; no longer isolated, so this is the "trained" variant, and at K=0 it is a linear head):

| in place of | step 1 | 2 | 3 | drafts a round, in order, 3 deep |
|---|---:|---:|---:|---:|
| the block as it is | 0.871 | 0.766 | 0.654 | 1.85 |
| the vocabulary head, K=0 (80 MiB) | 0.887 | 0.796 | 0.697 | 1.98 |
| the vocabulary head, K=4 (400 MiB) | 0.887 | 0.797 | 0.698 | 1.98 |
| the whole block, K=0 | 0.587 | 0.427 | 0.371 | 0.85 |
| the whole block, K=4 | 0.571 | 0.429 | 0.373 | 0.83 |

In place of the head a topology starts as the head's own row, so the run starts from the block's answers
(checked: the same answer on 1.000 of the rows). The gain is 7 % more drafts a round on held-out windows of the
training sets (code 2.09 against 1.96, own 1.59 against 1.46), the directions add nothing to K=0, and on `own2`,
answers to prompts the topologies never met, there is no gain: 0.768 / 0.602 / 0.491 against the block's 0.775 /
0.601 / 0.489, 1.43 against 1.44 drafts a round. So it is what any head trained on this text gives on this text.

**Beside the block** (`q27_res_gate.py`, the stock block three deep, a round scored in the M4's cost):

- depth gate, four labels (drafts kept: 0.139 / 0.196 / 0.201 / 0.464): always three drafts 2.004 of plain speed,
  the depth known in advance 2.161, the block's own probability as the gate 2.036; resonance names the label on
  0.40 to 0.455 of the rows (the commonest label 0.464) and as the gate gives 1.59 to 1.69, with the depth of the
  best expected speed 1.91 to 1.94: below always three;
- trajectory memory (a train row is a key with the tokens that followed): the closest 2 % of the rows have the
  first token right on 0.812 and keep 4.0 of 8, the closest 10 % 0.763 and 3.2; used where the error is small, the
  block elsewhere: 2.007 against 2.004. The window's own past as the memory: no threshold beats the block.

**What it says.** cmf-decision chooses among tens of labels from a signal an encoder made for that choice. A draft
chooses among some 20 000 tokens from a row that holds the answer only after the block's attention layer has
worked on it: fitted topologies read 0.39 off the trunk's row and 0.61 off the block's, a trained no-block variant
0.59, the block 0.87. Trained on top of the block the topologies are a head tuned to the training text.

## Kept

Private `infosave/cmf-experiments`, `q27-mtp-chain/`, each checked by size and sha256 through the API:
`qwen38-27b-turbo-mtp5c-q8.cmf` and `-f16.cmf` (the trunk with head C), `data-own-own2-prompts.tgz` (the two data
sets made here and the prompts of the real rounds), `res-head-k0-topologies.tar` (the trained head topologies, three
steps), `logs-scripts-2026-11-06.tgz` (every log and script of the pod).

## Not done

- A port of DFlash 2 (the best on code): the owner's decision.
- A draft head of its own for the block in the engine (what the trained K=0 topologies are): not built; it gave
  nothing on text from other prompts, so more varied training text comes first.
- The trained head over C's rows, C's rounds in order on `own2`, RedHatAI's speculator.dspark.
- The Mac with the q8_2f block: the round's cost (4 to 13 %) and the two-window mapping eat the gain; the trained
  block as q4tp needs a training large enough to survive q4tp's grid.
