# Engineering notes

## Pipeline "genesis + tooling" (2026-10-17): metric genesis, escort, exact ppl as tools

Goal: turn the q2tp / q4tp studies' Python scripts (MiniCPM5-2B, `cmf-lb-local-runs/
{q4tp_genesis_minicpm5_20261017, q2tp_nvg_minicpm5_20261016T1800Z, q4tp_nvg_minicpm5_20261016T1215Z}`)
into production tools that scale to Qwen3.8-27B (hybrid GDN, `qwen3_5`) on one 32 GB GPU with
≥ 128 GB host RAM, without touching the CMF format or the kernels.

### What changed

- **cortiq CLI** (`crates/cortiq-cli/src/main.rs`): `cortiq dequant --dtype raw` writes a tensor's
  encoded payload verbatim; `cortiq patch-tensor --raw` stores an already encoded payload verbatim
  (the entry's dtype, or `--dtype` as the encoder would emit it for that shape), length-checked by
  the reader's own `validate_payload`, hashes recomputed; in place when the slot fits, else
  `--output`. Unit test `dequant_raw_and_patch_raw_round_trip_payloads_verbatim`.
- **tools/quant/** (new, see its README): `cmfio.py` (read-only CMF reader over mmap + write-back
  via `patch-tensor --raw` + `verify`), `cmfcodec.py` (byte-exact q4tp / q2tp / q8_2f / float
  payloads, field splicing), `qmodel.py` (torch model over a real CMF: llama family + `qwen3_5`
  hybrid; compact weights decoded on the device inside a memory-light custom autograd matmul;
  chunked KL head with an analytic backward; `resident` / `stream` residency with side-stream
  prefetch, bit-identical), `calib.py` (corpus specs, the studies' window placement, the cached
  teacher top-K), `metric_refit.py` (metric genesis), `escort.py` (sensitivity + water-filling,
  greedy re-measure, `--tensor-quant` output, escorted file), `ppl.py` (exact perplexity with
  paired SE), `tests/`.

### How to use

See `tools/quant/README.md` (commands, memory model, 27B sizing, recipes q4tp-v2 / q2tp-v2).

### Tests

- `cargo test -p cortiq-cli --bin cortiq -- dequant_raw` (5090 pod):
  `dequant_raw_and_patch_raw_round_trip_payloads_verbatim` ok.
- `python -m unittest discover -s tools/quant/tests` (5090 pod, CPU + two CUDA tests): 20 tests ok
  in 12.7 s — codecs (round trip, field splice, sequential f32 ladder), device decode == the numpy
  decoder bit for bit, row-blocked == one-block matmul, the custom matmul and the chunked KL head
  against dense autograd, batched head == per-window head (also over streamed, host-parked
  boundaries on CUDA), the chunked gated delta rule against
  the token recurrence (f64), stream == resident bit for bit over 4 training steps on tiny llama
  and qwen3_5 models (CPU and CUDA), escort candidates / cached block inputs / water-fill, the
  studies' window placement verbatim, mixed corpora, snapped write-back touches only f16 fields.
- `tests/cortiq_parity.py --refit` (tiny random llama + qwen3_5 checkpoints converted by cortiq to
  f16 / q4tp / q2tp): torch NLL vs `cortiq ppl` (CPU exact) per window, worst relative difference
  5.2e-8; then `metric_refit.py` end to end on the q4tp and q2tp files of both architectures (f16
  teacher, `patch-tensor --raw` write-back + verify): held-out CE falls in all four, 0 bytes changed
  outside the f16 fields, the written files score the same in torch and cortiq (≤ 1.1e-7).

### Measured (MiniCPM5-2B, pod B RTX 2000 Ada shared under a 3 GB GPU cap)

WikiText-2 test, 160 × 512 windows, exact evaluator (`ppl.py`, every payload decoded from the file);
bf16 18.889. Journal and scripts: `cmf-lb-local-runs/genesis_tooling_20261017/`.

| check | result |
|---|---|
| evaluator parity | C2 19.5613 / f16 18.8890 / R2 21.8868 / W128dh 25.6222 = the studies' numbers; cortiq window 0 of G-C2: 1326.206278 vs torch 1326.206228 |
| teacher cache (f16.cmf through qmodel) vs the studies' teacher.pt | same windows, held CE 3.097050 = 3.097050, top-1 100 %, top-64 sets 99.999 % |
| **G-C2 reproduction** (`metric_refit.py --model C2.cmf --teacher f16.cmf --corpus train.ids,train=1024,held=65536:8+200000:16 --step --epochs 2 --mb 2`) | **19.1319** (12 w 18.1454), −2.195 % ± 0.089 vs C2; study 19.1344 (18.1460), −2.18 % ± 0.09. Best held-out CE 3.10660 at step 144 (study 3.10684 at 144); lo changed in 604 333 rows (study 604 286) |
| **streaming == resident** | the streaming file (one layer on the GPU, 0.85 GiB peak) and the resident file (2.00 GiB) are **byte-identical**; all 17 held-out evaluations equal to the last bit (resident run also resumed from a mid-run state) |
| escort inside C2 (`escort.py --hi q8_2f.cmf --corpus train.ids,held=8192:8 --bpw 0.1 0.27`) | +0.1 bpw: 24 v, 17 k, 2 o, 2 q, 1 down (study in A2: 23 v, 18 k, 3 o, 1 q, 1 up); v_proj median gain/MiB 10× other kinds; escorted file 19.3621 (−1.02 % ± 0.05 vs C2; study's C4 19.437) |
| q4tp-v2 chain end to end (C2 → escort → genesis, `--mb 8 --head-windows 8`) | **19.1092**, +1.17 % vs bf16 (study G-C4 19.0983); escort after genesis −0.12 % ± 0.04 |
| q2tp path (`metric_refit.py --model W128dh.cmf ... --step --epochs 2`, the study's R2 recipe) | **21.8833** (12 w 20.1603), −14.59 % ± 0.46 vs W128dh; study R2 21.8867 (20.1584), −14.58 % ± 0.46; best held-out CE 3.22974 at step 176 (study 3.2301 at 176) |

Speed on the shared pod: 28–39 s/step (streaming, mb 2) while three other jobs used the GPU and
~4 of 5.1 CPUs; 13.9 s/step resident mb 2 and 8.5 s/step streaming mb 8 + 8-window head when
alone (the study's script: 6.5 s/step at 7.9 GiB). Qwen3.8-27B geometry (scale probe on the 5090,
1.95 GB cap): 1.62 GiB peak for one 512-token window through a training step, 0.25–0.32 s/layer.

### Open issues

- GPTQ on 32 k mixed tokens with cortiq (`quantize-gptq --tokens 32768`) was not run here; C1/C2
  came from the q4tp study's torch GPTQ (same rule). The H-metric scale refit (C1 → C2, −0.31 %)
  is still a study script; q2tp GPTQ (placed ladder, layer-7 rule, debias) is not in cortiq.
- The genesis runs here train on WikiText only (to reproduce the studies); at 4 bits the gain is
  mostly in-domain, so a release needs the broad mixed corpus the tool supports (not measured).
- Qwen3.8-27B was not run (no checkpoint on the stands): the qwen3_5 path is validated on tiny
  random checkpoints against cortiq (≤ 1.1e-7) and end to end through the write-back; whether
  `quantize-gptq` captures Hessians for all 400 hybrid linears is unverified.
- Not done: decode caching across micro-batches, bf16 matmuls, CUDA graphs (the shared pod was
  launch-bound).
