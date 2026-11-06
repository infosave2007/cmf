# Qwen3.8-27B Turbo, the DFlash 2 drafter in the engine, 06.11.2026

Goal: the published `z-lab/Qwen3.8-27B-DFlash2` drafter as a second draft source of the Turbo file, beside its own
MTP block. `JOURNAL_Q27_MTP_CHAIN_2026-11-06.md` measured it in its authors' code: 3.73 drafts a round on code,
seven deep, against the stock block's 2.71, and the stock block's level on prose.
Stands: MacBook Air M4, 24 GB (native Metal); one RTX 2000 Ada, 16 GB, Vulkan 1.4 (wgpu), 48 threads, a 31 GB
memory limit. Greedy, `--no-think`, 220 tokens, decode tok/s; five prompts (`q27_spec_prompts.sh`: code, an edit
of the prompt's code, quotes from the prompt, an essay, an explanation in Russian).

## What is in the engine

- `cortiq convert --model DFLASH2_DIR --output MODEL.cmf --dflash-sidecar --quant q4tp` writes `MODEL.dflash.cmf`
  beside the model: 5 layers, hidden 5120, the fc over five trunk rows, 1 288 MB. The embedding and the head stay
  the trunk's.
- `CMF_DFLASH=1` loads it (with `CMF_GRAPH_SPEC=1` on native Metal, where drafts are opt-in). A round's block is
  eight rows, the anchor and seven masks, over a context made of the trunk's rows after layers 5, 19, 33, 47, 61.
  Plain greedy only. `CMF_DFLASH_PICK=dflash|mtp` pins a drafter, `CMF_DFLASH_FUSED=0` runs a matrix a call,
  `CMF_DFLASH_TIME=1` prints the block's stages, `CMF_DFLASH_ROWS=FILE` logs every round for the router's fit.
- `cortiq dflash-check MODEL KIT` runs the engine's block against the published drafter's own numbers
  (`q27_dflash_kit.py` turns the reference run of `q27_ext_draft.py` into the kit). Lowest cosine of a block row
  0.98338 with the matrices as q4tp; the drafts a round keeps: 1316 of 1512, the reference's own count. The same on
  native Metal and on wgpu.
- A block on the M4: 70 ms against a plain token's 160 (0.44 of a token), one price at any depth.

## Which drafter takes a round

The block wins on new code and loses on prose, Russian and on text copied from the prompt. The choice is the
router of patent 1, in the sidecar's header (`dflash_route`): two rank-16 affine subspaces of the trunk's unit row
at the tip, one of the rounds the block won and one of the chain's; the row goes to the one that holds it better.
The margin is averaged over rounds (step 0.5), and where the last tokens repeat an earlier stretch of eight or
more (`tail_repeat`) the round is the chain's: its short round with the prompt lookup drafts a copy for less.

Fit: `q27_dflash_fit_prompts.py` writes 34 prompts (12 code, 5 edits, 5 quotes, 6 prose, 6 Russian), none of them
the five that measure. `q27_dflash_rows.sh` answers each with every round the block's, the chain drafting beside
it unverified: 2 225 rounds. `cortiq dflash-route-fit MODEL ROWS_DIR --edge 1.05` prices every round under both
drafters (each at the depth its own rule picks, the lookup going on after the drafts as in a round), labels a round
by the better drafter over four rounds on each side, fits and writes the router. Tokens in a plain token's time on
the M4's prices:

| | the chain alone | the block alone | the router | the better every round |
|---|---:|---:|---:|---:|
| all 34 prompts, fitted on them | 1.958 | 1.705 | 2.137 | 2.312 |
| 8 prompts held out of a first fit | 1.903 | | 2.001 | |

Rank 4 to 64 and a window of 2 to 8 rounds change little. Held-out code goes to the block (2.03 → 2.19), prose,
quotes and Russian to the chain.

## MacBook Air M4, native Metal

The MTP chain alone against the router (`q27_dflash_ab.sh`, arms in turn, a pause before each), the same text in
both arms:

| prompt | MTP | router | |
|---|---:|---:|---:|
| code | 11.7 | 13.3 | +14 % |
| edit | 20.3 | 19.8 | −2.5 % |
| quote | 14.8 | 14.4 | −3 % |
| prose | 9.6 | 9.9 | +3 % |
| Russian | 10.3 | 10.3 | 0 |

Edit and quote are inside the run-to-run spread of a fanless Mac. Before the copy rule the router gave the edit's
rounds to the block: −13 % there, −8 % on quotes.

## RTX 2000 Ada 16 GB, Vulkan

The Turbo file alone takes 14 181 MiB of the card's 16 380, with the sidecar 15 543. A plain token is 75 ms.

The chain, the default rule against a pinned depth (`CMF_GRAPH_SPEC_K`, `CMF_GRAPH_SPEC_PIN=1`):

| prompt | plain | default | 2 drafts | 3 | 4 |
|---|---:|---:|---:|---:|---:|
| code | 13.4 | 24.9 | 25.7 | 25.7 | 25.3 |
| edit | 13.1 | 28.3 | 26.8 | 31.9 | 31.0 |
| quote | 13.1 | 25.4 | 27.6 | 30.3 | 28.5 |
| prose | 13.3 | 20.3 | 21.6 | 21.7 | 20.5 |
| Russian | 13.4 | 21.4 | 22.2 | 22.0 | 20.9 |

A round of k drafts there, draft + verify + commit: 2: 6.5 + 89.7 + 4.3 ms, 3: 9.8 + 96.3 + 5.2, 4: 12.7 + 107.9 +
5.7, 5: 16 + 122 + 7; a verify of seven rows 143 ms, of eight 154. In plain tokens, verify and commit for 1..7
drafts: 1.19, 1.26, 1.36, 1.52, 1.71, 1.98, 2.12, and 0.044 a draft of the chain.

**The block there, pinned** (`CMF_DFLASH_PICK=dflash`; the wgpu graphs copy the five trunk rows out on their way,
`layer_taps`, so the context costs no extra pass): 141 ms a block in 18 submissions, 90 of 185 drafts kept
(3.0 of 6.2 a round), 13.3 to 13.8 tok/s — a plain token's speed. The sidecar is resident (a larger weight budget,
`CMF_GPU_VRAM_MB=15300`, changes nothing): the price is the per-matrix GEMM, which reads a 1.3 GB drafter in the
time the verify's batched kernel reads the trunk's 14 GB.

**What a block at its floor would give.** The logged rounds priced at this card's rounds
(`dflash-route-fit --dry-run --prices 1.19,1.26,1.36,1.52,1.71,1.98,2.12,BLOCK,0.044`), tokens in a plain token's
time, the chain alone → the router:

| a block costs | all 34 | held out | code, held out |
|---|---:|---:|---:|
| 0 | 1.83 → 2.12 | 1.77 → 2.02 | 1.91 → 2.39 |
| 0.18 (14 ms: one pass at the bus's speed) | 1.83 → 1.98 | 1.77 → 1.88 | 1.91 → 2.16 |

The table moves with the price list: a list 5 % cheaper at depths 3 and 4 gave the held-out code 2.12 → 2.15. On
the measuring prompt itself the block's rounds land 4.0 tokens in a 143 ms verify, which is the chain's 25 tok/s
with the block at 14 ms and less at any real price. A verify of eight rows costs two plain tokens on this card;
on the M4 rows past the third are nearly free, and that is where the block's deep rounds pay.

Verdict: on wgpu the chain keeps the rounds, and the block runs only pinned, for measuring. A block worth
turning on needs one graph for its five layers and a card whose deep verify is cheap; neither was built or
measured here. (RTX PRO 6000, earlier: 28 ms a block plus 5 ms of head against a plain token's 16.7.)

## The chain's depth by the card's own prices

The default rule moves the depth by the share of drafts kept: it climbs to five on text that holds (edit, quote),
where this card's fifth and sixth rows cost more than they land, and it was tuned on an RTX PRO 4000 where five
drafts win on code. Now a round's wall is averaged per depth (`spec_wall`), and once two depths are timed the
depth is the one that lands the most tokens in a unit of time at those prices and at the running hold of each
draft (`wgpu_spec_depth`; an untimed depth is priced on the line through the two nearest; a change stands for six
rounds and needs 4 %). Until two depths are timed the old rule moves it. And a plain token's time from a trial is
kept for the next request of the process while the context is of the same order, so its trial decides after five
rounds instead of decoding eight plain tokens again.

One answer a process (`cortiq run --prompt-file`), old → new binary, two passes: code 23.7, 24.5 → 25.4, 23.8;
edit 28.8, 28.0 → 28.5, 28.4; quote 25.2, 24.6 → 26.6, 26.6; prose 20.6, 20.5 → 20.6, 20.2; Russian 20.9, 20.8 →
21.2, 21.2. A first answer spends its rounds timing depths and still times the plain token: +2 % on average.

Five turns in one process (`cortiq run` reading `turns.txt`: a Rust function, the same in Python, two paragraphs
of prose, a repeat of the first answer, a Go function; 200 tokens each), old → new, two passes:

| turn | old | new | |
|---|---:|---:|---:|
| 1 | 25.5, 24.5 | 25.6, 26.1 | |
| 2 | 28.1, 26.8 | 31.9, 31.4 | +15 % |
| 3 | 20.2, 19.4 | 23.4, 22.1 | +15 % |
| 4 | 30.9, 29.9 | 34.8, 33.1 | +12 % |
| 5 | 28.7, 28.2 | 30.8, 31.2 | +9 % |

Turns 2 to 5: 26.5 → 29.8 tok/s, +12.5 %. The old binary timed plain tokens in every turn, the new one in the
first.

## Traps

- **wgpu's verify is not the token graph bit for bit.** The batched verify runs on int8 activations
  (`CMF_VERIFY_I8=0` for the exact one), so a near tie can turn: on the card the code prompt's text is one thing
  plain and under the default rule, another at any pinned depth, a third at seven drafts, whoever drafts.
  `CMF_COOP=0` changes nothing. Compare a drafter with the chain at the same depth, not with the plain text.
- **Means of per-round ratios** rate a drafter wrong: the chain's deep lookup rounds are few and long. Sum the
  tokens and the costs.
- **A lookup has to be modeled from the whole sequence** (`FILE.ids` beside the rows), not from the two drafts the
  record kept: the first router sent copied text to the block.
- **The router cannot tell copying from writing**: an edit of the prompt's code looks like code at the tip. The
  copy rule does.
- **A pinned trial.** Without `CMF_GRAPH_SPEC_PIN=1` the trial stops rounds that lose to plain tokens, and the run
  reports a mix: the first pinned-block run showed 13.0 tok/s with half its tokens plain.
- **A tar from macOS** brings `._name` twins unless packed with `COPYFILE_DISABLE=1`; `dflash-route-fit` skips
  dot-files now. The kit's `.f16` files are links on the Mac and arrive dangling.
- **The Mac's uplink** gave 0.65 MB/s to the pod: a 1.3 GB sidecar is converted there from the published drafter
  and the router fitted there from the rows (48 MB).
- **27B on wgpu on a 24 GB Mac swaps**; the wgpu round is checked there only against the kit.
- **A fanless Mac throttles** in a long collection (19 → 4 to 9 tok/s). Rows do not depend on speed; an A/B waits
  until it cools and takes its arms in turn.

## Where things are

`crates/cortiq-engine/src/dflash.rs` (the drafter), `pipeline.rs` (rounds, router's use, depth rules),
`gpu_metal.rs` / `gpu_wgpu.rs` (the block's matrices in one call, `layer_taps`), `crates/cortiq-cli/src/`
`dflash_check.rs`, `dflash_route.rs`, `convert.rs`. Scripts: `q27_dflash_kit.py`, `q27_dflash_fit_prompts.py`,
`q27_dflash_rows.sh`, `q27_dflash_ab.sh` (arms `plain`, `mtp`, `dflash`, `auto`, `k1`..`k7`),
`q27_mtp_pod/dflash-check.sh`, `dflash-acc.sh`, `evalq-measure.sh`, `dflash-wgpu.sh`.
Private HF `infosave/cmf-experiments/q27-dflash/`: `rows.tgz` (the 34 prompts' rounds) and
`pod-rtx2000ada-logs.tgz`, sha256 checked. The sidecar is rebuilt from the published drafter and the rows.

## Open

- No release loads the sidecar yet; the Turbo page on HF is as it was.
- A first answer on wgpu: the trial's eight plain tokens and the timing of depths cost it what a pinned depth of
  three does not (26.3 against 24.1 tok/s on the five prompts).
- The block as one wgpu graph, measured on a 24 GB card.
- The new depth rule is measured on one card. RTX PRO 4000, 3090 and 5090 have their own price lists; the rule
  reads them, the runs were not made.
