# Qwen3.8-Flash-Next Turbo, 07.11.2026

Goal: the Flash-Next q2tp-v2 file as a turbo (the qwen4 device path learns the O(1) far field) and a "hot" variant
that keeps the busiest expert neurons at eight bits, measured honestly; HF page `infosave/Qwen3.8-Flash-Next-turbo-cmf`
(private until the release that reads the tubes).
Stand: RunPod RTX 5090 32 GB (weight budget 29 000 MB), Vulkan, 120 host cores, 134 GB cgroup, overlay disk.
Scripts in `scripts/flashnext/pod/` (`env.sh` holds BIN/FILE/PROFILE; `ppl.sh NAME TOKENS` is the code perplexity on
`expert_store_v089.rs`, the card's protocol; `bench.sh NAME CTX`; `rebake.sh`, `research2.sh` the bake queues).

## The model

48 layers (36 GDN + 12 attention), hidden 2560, 512 routed experts top-10 + 1 shared expert of 640 neurons, routed
gate/up q2tp (GPTQ-calibrated, "v2"), down q4tp, attention / indexer / head q8_2f. 76.9 GB; the device arena pins
the shared expert in the routed experts' format. Code ppl 1536 = 4.438 (q4tp file 4.111; the gap is the two-bit
gate/up of the routed AND the shared expert).

## Far field (qwen4 O(1))

The qwen4 device path had no far field: `qwen4_o1.rs` seals the first chunk that reaches `seal` rows (prompt or
decode), all rows so far become sinks + 32 landmarks per KV head; later rows attend window 1800 + far field.
`cortiq far-turbo` writes the header (`o1_turbo`; `--exact-weights` optional). 6 045-token prompt, 160 tokens:
exact 113 s first token / 26.5 tok/s, sealed at 4096 105 s / 29.3 tok/s, both answers a working page. WikiText-2
8 192 tokens: 2.241 exact vs 2.323 sealed (+3.7 %). Flash-Next's KV is small (12 layers × 2 × 256), so the seal is
about attention cost, not memory: at 16k there is no speed gain (40.0 vs 41.3 prompt, 23.5 vs 24.5 decode).

## Hot neurons in the routed experts

`hot-probe --experts` (host path, CMF_GPU=0) sums (w·act)² per layer, expert and neuron over 16 KB code + 16 KB wiki;
`hot-bake --experts COLD HF STAT OUT --frac f` zeroes the hot rows in the q2tp gate/up (rung 0) and writes one q8_2f
tube per expert (`mlp.experts.{e}.*_proj.tube1.weight`) from the bf16 checkpoint; the device runs the winners'
tubes in two dispatches a layer (`q4t_tube_gu` / `q4t_tube_dn`), the host paths add them after the triple.

| variant | ppl 1536 | ppl 128 | size |
|---|---:|---:|---:|
| turbo (no tubes) | 4.438 | 23.241 | 76.88 GB |
| 5 % top, bf16 down | 4.422 | 25.709 | 83.57 GB |
| 5 % top, cold (GPTQ) down | 4.425 | 25.788 | 83.57 GB |

Host and device agree on the hot file, so it is not a path bug: on a GPTQ-calibrated q2tp the error is not in the
hot neurons — their rows are the best-rendered (the Hessian weighs the frequent inputs), the residual is spread over
the cold mass. The Ripple gain (9.27 → 7.72 at 5 %) was on a plain q2tp without a shared expert.

The lever is the **shared expert**: q2tp, every token, weight σ — 100 % hot. (17.10: a q4tp shared expert measured
~4.27 but the arena could not pin a foreign format.) Now `hot-bake --shared-q8` writes the whole shared expert as a
q8_2f tube (`mlp.shared_expert.*_proj.tube1.weight`, +4.9 MB a layer) beside its zeroed triple; the device runs it
as a one-column tube with σ read from the route's weight table (steps 722/723), the host paths add it after the
triple (`shared_tubes_add`), and the per-op GPU MoE block (`moe_ffn_gpu`) now adds the tubes too — before, it
silently dropped them.

## Whole experts (the owner's idea, 08:35)

The arena's resident set under a small VRAM budget is the busiest experts; the owner asked whether lifting THOSE
whole to eight bits is the right grain for a MoE. The statistic already ranks experts (Σ over their neurons):

| in q8 per layer | share of the layer's Σ(w·act)², mean over 48 layers (min / max) |
|---|---:|
| 5 % of every expert's neurons (32 × 512 rows) | 31.7 % |
| the 28 busiest experts whole (28 × 640 rows, the same bytes) | 43.2 % (27.1 / 77.8) |
| the 51 busiest (10 %) | 56.2 % |
| the 103 busiest (20 %) | 74.4 % |

998 of 24 576 experts never fire on the probe. A whole expert also takes the error spread over its cold rows, which
the neuron tube cannot. `hot-bake --whole-experts N` (one tube of all 640 neurons for the N busiest experts of a
layer, the others untouched; `--pick random` the control); the device bank packs the experts that have a tube and a
slot table (`tt_slot[e]`, 0xFFFFFFFF = none) sends the kernels to the right entry.

Queue (`research5.sh`, binary ft7): shared tube alone (device 1536 / 128, host 128 for parity), whole28 (the same),
whole28 + shared, random 28 experts control, `--pick err` 5 %. The 25 % cold-down ceiling (107.7 GB, 59.7 % of the
score) could not run on the card: 30 GB of tube banks → wgpu "Out of Memory" (the banks now refuse what the budget
cannot hold and the model stays on the host path).

## Results

| variant | ppl 1536 | ppl 128 (device / host) | size |
|---|---:|---:|---:|
| turbo (no tubes) | 4.438 | 23.241 | 76.88 GB |
| 5 % neurons, bf16 down | 4.422 | 25.709 | 83.57 GB |
| 5 % neurons, cold down | 4.425 | 25.788 | 83.57 GB |
| shared expert as a q8 tube | **4.241** | 23.824 / 23.758 | 77.17 GB (+225 MB resident) |
| 28 busiest experts a layer whole + shared | **4.167** | | 83.81 GB (+6.5 GB resident; the expert pool 23.4 → 17.1 GB on the 32 GB card) |
| 28 random experts a layer whole (control, 5.7 % of the statistic) | 4.388 | | 83.57 GB |
| 28 busiest experts a layer whole, no shared tube | 4.327 | | 83.57 GB |

The whole-expert grain pays: the 5 % neuron tubes bought 0.013 of ppl for 6.7 GB, the 28 whole experts 0.074 on top
of the shared tube for the same bytes. Shared + 28 experts closes 83 % of the gap to the 97 GB q4tp file (4.111) at
83.8 GB. That file is the page's hot file (`--whole-experts 28 --shared-q8`); a random-28 control and the 28
experts alone follow.

Device against host on the same 25-token prompt (`CMF_QWEN_DEVICE_CHECK=1 CMF_QWEN_PREFILL_CHUNK=1`, the last
prompt row): turbo cos 0.9962 / max |Δ| 0.87, hot cos 0.9938 / 1.30, the argmax the same token in both — the tube
kernels add no discrepancy of their own beyond the arena's two-bit kernels. (The check panics on the first decode
step in `selected_tokens`, a check-mode host-cache bug independent of the tubes; the prompt row is what it yields.)

Two WGSL traps on the way: `shared` is a reserved word even as a struct field (the module was rejected and the model
fell to the host path — `cargo test -p cortiq-engine --features gpu --lib qwen4_shaders_validate` catches it without
a card), and the tube banks were allocated past the VRAM budget.

## Final pass (fnx/final2.sh, final3.sh)

Hot file `qwen38-flash-next-q2tp-hot-turbo.cmf` (83 805 883 496 B, sha256 eabe117c…): code ppl 1536 4.167, WikiText-2
12 × 512 5.782 (turbo 6.062). Device/host parity on the prompt row: turbo cos 0.9962, hot 0.9938. Greedy `--no-think`
generation of the two files on "a Rust function that reverses a string, with a test": both a correct function and a
test module (the hot file's test checks "hello" → "olleh", the turbo file's the empty string). cortiq 0.8.60 on the hot
file: ppl 200.7 over 256 tokens — an engine that does not read the tubes sees zeroed rows, so the page stays private
until the release that does.

`cortiq bench --core --ctx N --tokens 60`, turbo file, `CMF_QWEN_O1=off` (exact) against the header's seal at 8 192:

| rows | first token, s exact / sealed | decode, tok/s exact / sealed | peak MiB exact / sealed |
|---:|---:|---:|---:|
| 1 000 | 3.4 / 3.7 | 59.6 / 62.3 | 30 070 / 30 239 |
| 4 000 | 19.8 / 21.9 | 54.6 / 44.6 | 29 902 / 30 935 |
| 16 000 | 93.7 / 126.7 | 37.7 / 25.8 | 30 586 / 30 905 |

Below the seal the two runs are the same code path and differ within the stand's noise (the host is shared, load
average 65 on 120 cores). At 16 000 the sealed run is slower on both axes, as the 07:00 pair already hinted (exact
41.3 / 24.5, sealed 40.0 / 23.5): Flash-Next's attention is 12 layers × 2 KV heads × 256, exact attention over 16 000
rows is cheap, and the far field's extra dispatches a layer cost more than the rows they skip. On this model the far
field bounds the attention state (the KV grows no further after the seal) and costs +3.7 % wiki ppl; it is not a speed
device, and `CMF_QWEN_O1=off` is the setting while the prompt fits. The first final2 run was cut off in the 32 000 exact
bench (no OOM: memory.events oom_kill 0; a stray kill); final3.sh reran hot 1k/4k; the 32k pair was stopped when the owner closed the pod work (11:20).

## The hot file alone (11:20)

The owner's verdict: the far field buys no speed here, so the page is not "turbo" — renamed to
`infosave/Qwen3.8-Flash-Next-hot-cmf` (private), and the hot file is rebaked over the plain q2tp-v2 file (final4.sh),
so its default is exact attention; the far field stays an option (`CMF_QWEN_O1=32,1800,4,8192`). The turbo files
leave the page. Hot file on the 32 GB card, `bench --core`: 1 000 rows first token 9.0 s, decode 27.8 tok/s;
4 000 rows 43.4 s, 29.9 tok/s — against the 2-bit file's 3.4 s / 59.6 and 19.8 s / 54.6: the resident banks take
6.5 GB from the expert cache (23.4 → 17.1 GB) and every layer runs two more dispatches, so the file is about half the
speed on a card this size.
