# Ripple-65B Preview → CMF, 06.11.2026

Goal: ricursiveAI/Ripple-65B-Preview in the format, three files (q4tp, q8_2f, a q2tp mix) with the turbo method of
the Dory and 27B pages, the published DFlash2 drafter as a speculative round, and the owner's experiment: the busiest
FFN neurons of a two-bit file kept at eight bits.
Stand: RunPod A40 46 GB (weight budget 41 972 MB), Vulkan, 96 host cores, 503 GB RAM, an overlay disk at about
400 MB/s. Scripts in `scripts/ripple/pod/`, the reference in `scripts/ripple/ripple_ref.py`.

## The model

65B dense: 80 layers, 16 full and 64 sliding (window 1024), hidden 5120, FFN 40 960, 80 heads × 128 over 8 KV
heads, vocabulary 100 352, embeddings scaled by √5120 on the way in, sandwich norms, a value residual toward the
scaled embedding, SwiGLU clamped at 20, no draft block. New in the engine: the `ricursive` mapping, the value
residual as one dispatch a layer in both wgpu graphs (`gpu_wgpu/vres.rs`), the clamp, the batch graph's arms for
the hot tube (kinds 6/7). A tiny model from the published code (`ripple_ref.py tiny`) checks every feature bit for
bit before the real file.

## Files as they go out

| file | GB | PPL, wikitext-2 test, first 2048 tokens (BF16 6.940) | top-1 as BF16 | mean \|ΔNLL\| |
|---|---:|---:|---:|---:|
| q8_2f | 64.82 | 6.974 (RTX PRO 6000; 14.644 vs 14.618 over 128 tokens on the host) | 98.4 % | 0.034 |
| q4tp: q4tp with K and V projections at eight bits | 34.61 | 7.089 | 91.1 % | 0.219 |
| q2tp: a q2tp mix, 5 % of each layer's FFN neurons and K, V at eight bits | 29.23 | 7.432 | 86.2 % | 0.384 |

The q2tp mix keeps layer 0 and the last two layers at eight bits (the owner's recipe), the head is q8_2f in every
file, the attention and the FFN's down matrix are q4tp in both low-bit files; the two differ in the gate and up
matrices.

## Hot neurons

A neuron is a gate row, an up row and a down column. `cortiq hot-probe` sums the squared activation of every neuron
over text (`hot/energy.mass`); `cortiq hot-bake COLD HOT STAT OUT --frac f` scores neurons by energy × ‖down
column‖², copies the cold neurons bit for bit from the low-bit file (`compact_encoded_tiled_rows`) and puts the
top `f` of each layer into a `…_proj.tube1.weight` triple in q8_2f from the eight-bit file. The engine's graphs
take a core and one always-on tube. `--take SUBSTR` copies whole tensors from the eight-bit file (`--frac 0`
allowed): the K and V projections here.

| file | GB | PPL | top-1 | \|ΔNLL\| |
|---|---:|---:|---:|---:|
| q2tp mix | 27.27 | 9.265 | | |
| + 5 % hot neurons | 28.85 | 7.724 | 85.3 % | 0.412 |
| + 10 % | | 7.601 | | |
| + 5 % random neurons (control) | | 8.636 | | |
| + 5 % hot, K and V at eight bits | 29.23 | 7.432 | 86.2 % | 0.384 |
| q4tp | 34.20 | 7.468 | 90.0 % | 0.243 |
| + 1 % hot (416 neurons a layer, 46.5 % of the score) | 34.45 | 7.461 | 89.5 % | 0.262 |
| + 2 % hot (832, 51.0 %) | 34.70 | 7.445 | 89.7 % | 0.261 |
| K and V at eight bits, no neuron cut | 34.61 | 7.089 | 91.1 % | 0.219 |
| the whole attention at eight bits | 40.65 | 7.195 | 93.7 % | 0.138 |
| + 2 % hot and K, V at eight bits | 35.10 | 7.116 | 90.7 % | 0.241 |

- At two bits the 5 % recover 86 % of the gap to q4tp (random 5 %: 41 %); at four bits 1–2 % recover almost
  nothing, and top-1 and |ΔNLL| get slightly worse: the cold down columns are re-cut from the eight-bit values
  over a different column set, and that costs about what the hot columns give. The q4tp file goes out without a
  neuron cut, with the K and V projections at eight bits (+1.2 % of size, 72 % of the gap to BF16 in PPL).
- Speed of the mix at 4000 tokens (A40): q2tp mix 58.2 tok/s prompt / 12.4 decode; with 5 % hot 52.7 / 11.7.
- A bake takes 230–340 s on the host; a PPL pass through the graphs 55–85 s with 0 decline lines.

## DFlash2

`ricursiveAI/Ripple-65B-Preview-DFlash2`: 5 Qwen3-shaped layers, blocks of 16, convolutions of kernel 4 and
group 16, a rank-256 selector over 32 candidates, mask token 100 278, context from the trunk's rows after layers
1, 20, 39, 58, 77; layers 0–3 sliding (4096), layer 4 full without RoPE. `cortiq convert --dflash-sidecar` reads
`layer_types` and `no_rope_layers` into the geometry (`windows`, `no_rope`); the drafter keeps a context per layer
with its own window and rotates only the layers that rotate.

A trunk without a draft block gets a round of its own on wgpu (`Pipeline::solo_spec_round`): the block's drafts
(or `CMF_SPEC_SCRIPT` ids for tests) are verified with the tip in one batch-graph submission with a `SpecTail`;
the kept prefix is committed, the far field told (`o1_spec_commit`), the drafter's context fed from the kept tap
rows, the graph mirrors rewound. Greedy without penalties only. A pay-off monitor times plain tokens against
rounds and makes a losing stretch sit out (32 tokens, doubling to 512). Exactness: the tiny model on Vulkan with
scripted drafts (all right, depth 4, every seventh wrong, depth 15) gives the plain run's 578 ids; the real file
with the drafter gives the plain run's ids in every run.

What went wrong, in order:

- The head's shortlist (`draft_head_rows`, the first 65 536 rows) is a Qwen-vocabulary assumption; Ripple's
  frequent ids lie above it. The whole head when the trunk has no draft block.
- Drafts past the first row were id 100 351 (the last of the vocabulary), 76 of 633 kept. The context was whole
  (`CMF_SPEC_DBG` shows the rows kept and the drafter's span). A reference run of the published forward on the
  engine's own tapped rows (`CMF_DFLASH_DUMP`, `scripts/ripple/ripple_dflash_ref.py`) kept 81 of 126 drafts over
  18 rounds — the drafter and its inputs were right. `cortiq dflash-check` against a kit from that run: on the host
  the block's rows at cosine ≥ 0.988 and 5.65 drafts a round against the kit's 5.90; on the device the rows at
  cosine ≥ 0.995 but 0.75 a round, and rows 1–3 onward of the block were NaN (a cosine fold over `f64::min`
  skips NaN).
- The cause: the fused q4tp FFN (`ffn_q4`) handed the cooperative f16 GEMM no activation scale (the uniform's
  fourth word was 0), so a SwiGLU row past 65 504 became inf. Every other caller scales its operand by its max;
  the fused FFN now does too — the host operand by `host_absmax`, the intermediate (never on the host) by the
  card's own reduction and the `q4tp_mm_coop_s` entry point. The drafter keeps 5.35 a round on the device after
  it. The 27B's drafter has blocks of 8 and never reached this kernel (`q4tp_rows8_*` up to eight rows).
- 260 of a 340 ms block were the convolutions' kernel projections: `sgemm_public` runs one thread on x86. The
  rows go across the pool now.
- A verify of nine rows costs 660 ms against 120 for three or four: the batch graph's narrow kernels take eight
  rows, so the round's depth is capped at seven drafts.

Prices and the decode with the drafter on the real file (`cortiq run --greedy`, 160 tokens, the q4tp file; `auto`
is the pay-off monitor, `k7`/`k4` a fixed depth):

| card | answer | plain, tok/s | auto | k7 | k4 | kept a round (k7) | draft / verify, ms (k7) |
|---|---|---:|---:|---:|---:|---:|---:|
| A40 | code | 12.1 | 11.6 | 12.6 | | 5.3 of 7 | |
| A40 | prose | 12.2 | 9.0 | 5.7 | | 1.3 of 7 | |
| RTX PRO 6000 | code | 30.2 | 26.9 | 24.5 | 24.0 | 4.6 of 7 | 119 / 106 |
| RTX PRO 6000 | prose | 30.6 | 21.6 | 10.4 | 13.0 | 1.2 of 7 | 103 / 106 |

The text is the plain run's in every run, and the drafter does not pay anywhere: on the RTX PRO 6000 a round is
160 ms against 32 ms for a plain token. `CMF_DFLASH_TIME=1` splits a block of 72–78 ms into kernel projections
22.7 ms (host convolutions), q k v 9.2 (five device round trips), attention 10.2 (host `sgemm_public` over up to
4096 context rows), o 7.0 (five more round trips), MLP 33.6 (five `q4tp_rows8_ffn` submissions); the head over the
whole vocabulary 7.8, the selector 3.7. The verify of four rows costs 69 ms, of seven 106 — two to three plain
tokens for a batch that reads the weights once. Both halves are orchestration, not arithmetic: the block wants to
be one device graph like the trunk's, and the verify the fused decode kernels for up to eight rows. Shipped as the
exact sidecar it is, off unless `CMF_DFLASH=1`; the card says so.

## Turbo

As on the 27B page: 32 landmarks per head, a window of 1800 exact rows, 4 sinks, 248 chosen exact rows, on for
prompts of 8192 rows and more (`scripts/ripple/pod/turbo.sh`, `fit.sh`, `facts.sh`).

## Tables

From `logs/final_q4tp.log`, `final_q4tp_redo.log`, `final_q2tp.log` (A40, `cortiq bench --core --tokens 60
--ignore-eos`, peak device memory by `nvidia-smi` over the run, MiB / 1000 as the card's label counts; two rows
of the q4tp table were re-run after a parallel check added its own buffers to the peak).

| q4tp, prompt | first token, s (standard / turbo) | decode, tok/s | peak VRAM, GB | RAM (turbo) |
|---:|---:|---:|---:|---:|
| 4 000 | 48.4 / 48.0 | 12.1 / 12.0 | 35.3 / 35.3 | 0.2 GB |
| 16 000 | 212.9 / 251.2 | 11.1 / 11.7 | 36.8 / 36.6 | 4.4 GB |
| 32 000 | 473.2 / 491.1 | 10.2 / 11.8 | 39.1 / 36.6 | 7.7 GB |
| 64 000 | 1138.2 / 987.8 | 8.8 / 11.8 | 43.2 / 36.6 | 14.9 GB |

q2tp (turbo): 4 000 tokens 77.8 s / 11.8 tok/s / 30.3 GB; 64 000 tokens 1459.2 s / 11.5 tok/s / 31.6 GB.

Facts (`facts.sh`, 60 000 characters, one trial a kind, 300 tokens of answer with reasoning): pass key, access
code and room number answered, the password missed — the model was still reasoning at the 300-token limit
(`finish: max_tokens`), not wrong.

## RTX PRO 6000 (96 GB)

A second pod for the eight-bit file and the decode speed the owner asked about: "16.2 tok/s for q8_2f on an RTX
PRO 6000 — did you optimize the engine at all?" The stand: Vulkan, 188 SMs, 1792 GB/s; `scripts/build.sh` builds
the synced tree without LTO; `CMF_GPU_TS=2 CMF_GRAPH_TS_ALL=1` prints the decode graph's stages a layer (every
stamp is `BOTTOM_OF_PIPE`, so a span is completion-accurate, and each stamp costs ~5 us, ~3.6 ms over a token;
`CMF_GRAPH_DEVICE_SPAN=1` gives the true span with two stamps).

Before and after, `cortiq bench --core --tokens 60 --ignore-eos --ctx 4000`:

| file | prompt, tok/s | decode, tok/s |
|---|---:|---:|
| q4tp | 209 → 209 | 23.4 → 29.8 |
| q8_2f | 43.4 → 156 | 16.2 → 18.3 |

Decode, q4tp, a layer at ctx 512 (us, stamps included): attend 49 → 29 (the K and V loops unrolled 8 and 4),
o projection 37 → 31 (the warp-per-row kernel instead of the generic 16-row one), rope + K/V projections on the
gated layers fused (−0.2 ms a token), the sandwich norms 30 + 25 → 26 + 17 (loads first, shuffle trees, phase
two in waves of eight), the narrow-grid cap by SM count (192 blocks were 4 an SM on the 48-SM PRO 4000; 4 × 188
here: 38.5 → 37.1 ms), a short first submit piece (`CMF_GRAPH_FIRST`, default 2 layers). Device span 38.5 →
32.6 ms. The ids of a 160-token greedy answer are the old build's bit for bit.

Decode, q8_2f: the FFN down ran at 1190 GB/s against the gate's 1420 — `q8_2f_matvec4` loads a step's
activations and column scales once per row, and the down's 160 KB activation vector does not sit in L1.
`q8_2f_matvec4k` (four rows a workgroup, the columns split across the four sub-blocks, the activations loaded
once for the four rows) 177 → 163 us a layer; `q8_2f_matvec16` (sixteen rows a workgroup) 169 and gate/up worse
(320 workgroups on 188 SMs), kept as a knob. 57.4 → 56.3 ms device, 16.4 → 16.7 at ctx 512 with stamps.

Prompt, q8_2f: 43 tok/s because the eight-bit prompt chunks took the scalar int8 GEMM (~5 TFLOP/s) for the
wide matrices and the eight-row matvec for the rest (the down read its 210 MB once per eight tokens).
`q82_mm_coop_s` is the four-bit cooperative GEMM with its weight tile read from q8_2f (f16 operands, f32 sums,
the activation scale from the device); wide matrices only (rows ≥ 4096) — on the K and V projections of 1024
rows the 64-row tiles and the scale reduction cost the four-bit file's prompt 209 → 196. 43 → 156 tok/s;
PPL 6.973 → 6.974, top-1 98.34 → 98.39 %.

Measured and rejected on this card (the code keeps them as knobs with the numbers): a column-split q4tp matvec
(`CMF_MV_SPLIT`), two or four warps a workgroup (`CMF_MV_SG_W`), two rows a warp for the thin grids
(`CMF_MV_SG_THIN=1`: down 87 → 97 us, o 31 → 35), eight warps on eight column slices of the same four rows
(`CMF_MV_SG_KS=1`: q 48 → 72, o 31 → 43, down 87 → 97). The 160 workgroups of a 5120-row matrix leave 28 SMs
idle, and every cure that spreads the rows thinner loses more than that: four rows a warp with eight warps a
workgroup keeps the most bytes in flight per SM. The host is not the wall: the encode of 80 layers takes 12.4 ms
and overlaps the card; setup 0.4 ms, the first piece and the readback about a millisecond together.

What is left on the table: the two sandwich norm dispatches (~2.2 ms a token, one workgroup each, their
256-lane phase kept for bit parity with the unfused kernels), the q8 gate + up + SiLU as three dispatches
(~0.7 ms), the q8 o projection at 1090 GB/s, the drafter's orchestration above. The eight-bit file's floor on
this card is 64.8 GB a token at 1.79 TB/s — 36 ms, 27 tok/s — against 54.6 ms now.

## Traps

- `grep | cut` on a long-running bake buffers until the end; watch the process and the file instead.
- `tinyspec.sh` summed with `bc`, absent on the pod; awk.
- The first decode token's time is warm-up; the pay-off monitor skips the first plain sample.
- A parallel GPU job pollutes `vram_peak.sh`'s peaks (it reads the card's total); nothing runs beside a table.
- `kill` by PID from a separate ssh call; `pkill -f` matches the remote shell itself.
- `read -r HF_TOKEN` over ssh returns non-zero when the token file has no trailing newline; `|| true` and test
  the variable, or the chained upload never starts.
- A wgpu pipeline's auto layout is per pipeline: a bind group made with the row-split kernel's layout is refused
  by the column-split pipeline of the same bindings.
- `dflash.sh NAME` without `plain` turns the drafter on and defaults to `hf/ripple-65b-q4tp.cmf`; pass `FILE=`.
