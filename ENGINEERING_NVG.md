# q4tp-nvg — NVG-кванты в CMF / NVG quants in CMF

## Кратко (для владельца)

**Что в ветке.** `q4tp-nvg` = 0.8.19 (master 5d6f1830) + три проверенные ветки воркфлоу:

1. **q2tp v2 + GPTQ-q2tp** — новый кодировщик q2tp в `cortiq convert` (без данных: MiniCPM5 206.6 → 89.5)
   и `cortiq quantize-gptq --codec q2tp` (2-битные gate/up через GPTQ, остальное q4tp; 32k токенов → 27.76).
   Калибровка из нескольких корпусов (`--calib PATH=SHARE`), по частям (`--calib-chunk-layers`),
   защита от нехватки ОЗУ для 27B.
2. **Инструменты genesis** — `cortiq dequant --dtype raw` / `patch-tensor --raw` и `tools/quant/`:
   metric genesis (`metric_refit.py`, дообучение f16-полей строк и норм под учителя, коды не трогает),
   escort (`escort.py`, кому дать лишние биты), точная ppl (`ppl.py`). Llama-семейство и гибрид
   `qwen3_5` (Qwen3.8), потоковый режим по слоям для 27B на одной GPU.
3. **q1g** — новый dtype 20: 1 бит + одна f16-шкала на 128 весов. Bonsai-1.7B 334 → 254 МБ без потерь
   (PPL 28.3705), CPU-декод ~2× быстрее q1.

Формат CMF: добавлен только dtype 20 (q1g). Ридеры и ядра q4tp/q2tp не менялись — старые файлы читаются
как раньше, новые q4tp/q2tp-файлы читаются старыми релизами.

**Проверка слитой ветки.**
- `cargo test --workspace` (5090, GPU-тесты на карте): 1578 прошли, 3 упали, 47 пропущены. Все три падения
  не из-за слияния: два падают и на master 5d6f1830, третий (mimo_moe) проходит отдельно на обоих деревьях.
  clippy без ошибок, проверка сборки под macOS (aarch64-apple-darwin) без ошибок, Python-тесты 21/21,
  паритет torch↔cortiq 9e-8.
- Слитый бинарь пишет **побитово те же файлы**, что и ветки по отдельности: q2tp-конвертер (83d36e66),
  GPTQ-q2tp W0cd (81f35e49), GPTQ-q4tp Q4c (0b44a6f9), свежая калибровка (кэш гессианов a1096e4c),
  Bonsai q1g (724a34dc) и обратно q1 (f5abb584).

**Числа (MiniCPM5-2B, WikiText-2 test 160×512, bf16 = 18.889; ppl реальных CMF-файлов):**

| квант | ppl | файл |
|---|---:|---|
| q4tp как выпускали (GPTQ 2048 токенов) | 20.226 | 1.58 ГБ |
| q4tp GPTQ 32k (cortiq, Q4c) | 19.619 | 1.58 ГБ |
| q4tp-v2: Q4c + metric genesis (G-Q4c) | **19.187** | 1.58 ГБ |
| q4tp-v2 + escort +0.1 бит/вес (G-Q4c-esc) | **19.170** | 1.60 ГБ |
| q2tp старый конвертер | 206.6 | 1.31 ГБ |
| q2tp v2 конвертер (без данных) | 89.46 | 1.31 ГБ |
| q2tp GPTQ 32k (cortiq, W0cd) | 27.76 | 1.31 ГБ |
| q2tp-v2: W0cd + metric genesis (G-W0cd) | **22.48** | 1.31 ГБ |

q4tp-v2 при том же размере: +1.6 % к bf16 вместо +7.1 % у выпущенного рецепта (−5.1 % ppl); escort после
genesis добавляет лишь −0.09 %. q2tp-v2: +19 % к bf16 (старый конвертер был ×11). Вся цепочка — только cortiq
(GPTQ) + tools/quant; без H-refit из исследований (он стоит ещё ~0.3 % у q4tp). Конвертер и Bonsai мерились на
поде B; под B выключила основная сессия, поэтому ppl GPTQ-файлов и genesis досчитаны на 5090 (кэп 2 ГБ GPU).

**Что нужно для Qwen3.8-27B.** Под с ≥ 128 ГБ ОЗУ (лимит cgroup, не `free`) и GPU ≥ 32 ГБ, локальный диск ≥ 300 ГБ.
Команды — ниже в разделе «Qwen3.8-27B». Главное ограничение этой ветки: калибровка GPTQ здесь идёт
на CPU (по частям, 16 слоёв за проход). GPU-калибровка и GPU-свёртка уже сделаны в ветке `q27-build`
(основная сессия) — её стоит слить с `q4tp-nvg`, чтобы она получила исправления проверяющих (список ниже).
По правилу владельца 27B делается только в Rust/CMF: metric genesis пока есть только на Python
(`tools/quant`), для 27B её нужно перенести в Rust (в `q27-build` есть начало: NVG-тренер на устройстве).

---

## English details

### Branch contents

Base master 5d6f1830 (0.8.19). Merges (`--no-ff`, not pushed, no tag):

| merge | branch | what |
|---|---|---|
| bec72a6a | worktree-wf_6ce4b01d-f88-1 (15 commits) | q2tp encoder v2 (default), `quantize-gptq --codec q2tp`, `--q2tp-rule`, `--q2tp-debias on\|off` (default on), `--q4tp-debias`, multi-corpus + chunked calibration, whole-model Hessian cache, threaded cache-blocked fold, widest-first groups, RAM guards, `tools/q2tp_eval/` |
| 24ed78fa | worktree-wf_6ce4b01d-f88-2 (14 commits) | `dequant --dtype raw`, `patch-tensor --raw`, `tools/quant/` (cmfio, cmfcodec, qmodel, calib, metric_refit, escort, ppl, tests) |
| 76cc179b | worktree-wf_6ce4b01d-f88-3 (10 commits) | dtype 20 `q1g`, `requant --quant q1g [--allow-tol]` / `--quant q1`, q1g kernels CPU / wgpu / Metal (Metal: explicit `fma`), docs |
| c137d7e2 | — | tools/quant README: GPTQ-q2tp is in cortiq now; 27B GPTQ command with `--calib-chunk-layers 16` |

Detailed engineering notes of each pipeline (what changed, tests, measurements, independent verification,
open issues) are in `ENGINEERING.md` (three sections); recipes and the 27B memory model in
`tools/quant/README.md`. Run journal and every pod script of this integration:
`cmf-lb-local-runs/merge_nvg_20261017/`.

### Merge notes

- `crates/cortiq-cli/src/main.rs`: one conflict, the `quantize-gptq --tensor-quant` doc. Branch 2 documented
  master's behaviour ("an overridden linear skips GPTQ"); branch 1 changed `gptq.rs` so that a q4tp/q2tp
  override stays GPTQ (an escort on a q2tp profile) and any other codec goes to the converter. The merged doc
  states branch 1's behaviour, which is what the merged `gptq.rs` does; `tools/quant/README.md` (escort
  paragraph) was corrected to match: escort.py's q8_2f lists skip GPTQ, its q4tp lists on a q2tp profile stay
  GPTQ. The q1g `requant` arms merged without conflict.
- `ENGINEERING.md` (add/add in both merges): the three sections concatenated. `CHANGELOG.md`: q1g entry placed
  under Added after the q2tp entries; the raw CLI and tools/quant entries (branch 2 had none) added.
- `tools/quant/cmfio.py`: dtype id 20 → `Q1G`, so the Python reader lists q1g files instead of refusing them
  (the codec still decodes only q4tp / q2tp / q8_2f / float, as before).
- The uncommitted `main.rs` diff in this worktree (the raw dequant / patch CLI) was already discarded before
  this step (`git reflog`: reset to HEAD, clean tree). The source pod B's q4x build came from
  (`/root/q4x/src`) still holds it; branch 2's commit 8d3b9736 contains the same two features, reworked
  (reader's `validate_payload`, `raw_patch_dtype`, unit test).

### Tests on the merged tree (5090 pod, rustc 1.99.1)

| check | result |
|---|---|
| `cargo build --locked --release -p cortiq-cli` | ok, 5 m 48 s, binary e3e78308… |
| `cargo test --locked --workspace --no-fail-fast` (debug, wgpu tests on the RTX 5090, `XDG_RUNTIME_DIR=/tmp`) | **1578 passed, 3 failed, 47 ignored** (51 min) |
| the 3 failures re-run alone, merged vs master 5d6f1830 | `gpu_wgpu::tests::wgpu_q8_mul_mm_matches_cpu_reference` (max\|Δ\| 1.87e-3): fails on master too. `decision_cli::decide_oracle_leaves_a_signal_it_inherited_as_ignored_ignored`: fails on master too (known on this pod). `mimo_moe::bank_tests::dynamic_attention_graph_batches_preserve_layer_keys_and_rewind`: passes alone on both (fails only under the parallel run). None caused by the merge. |
| `cargo clippy --locked --workspace --all-targets` | 0 errors |
| `cargo check --locked -p cortiq-embryo --features vulkan --all-targets` | ok |
| `cargo check --locked --workspace --all-targets --target aarch64-apple-darwin` (C shims stubbed) | 0 errors (Metal host code and the CLI's macOS paths type-check) |
| `python -m unittest discover -s tools/quant/tests` | 21 ok (CPU + CUDA) |
| `tools/quant/tests/cortiq_parity.py --refit` with the merged binary | torch vs `cortiq ppl` worst 8.98e-8; refit chains llama / qwen3_5 × q4tp / q2tp: held-out CE falls, 0 bytes outside the f16 fields, written files torch vs cortiq ≤ 1.4e-7 |

Not run here: the Metal kernels on Apple hardware (the q1g verifier ran them on an M4 with a Swift harness;
no file they touch changed in the merge).

### Re-measurement with the merged binary

Byte identity against the pipelines' evaluated files (CPU; the MiniCPM5 files on the 5090 pod, Bonsai on
pod B; inputs read only):

| file | command | result |
|---|---|---|
| Q0h_v2 | `convert --quant q2tp` + head/embed q8_2f | **83d36e66… identical** (39.6 s) |
| detA | fresh calibration, 4096 tokens, one pass, q2tp, down_proj to the converter | Hessian cache **a1096e4c… identical**; with `--q2tp-debias off` the file is **5c862b8e… identical** (the pipeline wrote detA before the debias became the default; with the default on the file differs by design) |
| W0cd | `quantize-gptq --codec q2tp` (CLI defaults) from the 32k cache | **81f35e49… identical** (870 s on 7 threads, peak RSS 7.8 GB) |
| Q4c | `quantize-gptq --codec q4tp` from the same cache | **0b44a6f9… identical** |
| Bonsai q1g | `requant --quant q1g --allow-tol` (strict mode refuses: 393 of 13 436 752 groups, nothing written) | **724a34dc… identical**; `--quant q1` back: **f5abb584… identical**; `verify` ok |

Exact perplexity with the merged binary and `tools/quant/ppl.py` (exact f32, every payload decoded from the
file), WikiText-2 test 160 × 512, bf16 18.889. Pod B (RTX 2000 Ada) ran the converter file and Bonsai; it was
shut down by the main session (backed up at 19:22 UTC) before the rest, which then ran on the 5090 under a
1.2–2.0 GB allocator cap (token ids regenerated with the merged cortiq: 2 400 483 / 285 116, the studies' counts):

| file | ppl (160 w) | 12 w | reference |
|---|---:|---:|---|
| Q0h_v2 (converted on pod B) | **89.4643** | 72.9158 | pipeline 89.4643 |
| W0cd (5090) | **27.7562** | 24.9453 | pipeline 27.7562 / 24.9453 |
| Q4c (5090) | **19.6191** | 18.5185 | pipeline 19.6191 / 18.5184 |
| Bonsai q1 / q1g / q1u (`cortiq ppl`, wgpu, 16 × 511 windows) | 28.3705 / 28.3705 / 28.3705 | | NLL 27351.593365 / 27351.592607 / 27351.592607 = pipeline to the last digit; q1g == q1u on 16/16 windows |

The pod B convert file hashes 4acae0ec…, not 83d36e66…: the header records `"source_model"` (the input
directory: /root/q4x/hf/… vs /root/cmfbuild/q2tp_work/hf/…); all 381 tensor hashes are identical.

New end-to-end numbers (cortiq GPTQ instead of the studies' torch GPTQ, then the merged tools):

| file | recipe | ppl (160 w) | Δ vs input | vs bf16 |
|---|---|---:|---:|---:|
| **G-Q4c** | Q4c + `metric_refit.py --step --epochs 2` (G-C2 recipe, streaming, mb 2) | **19.1869** | −2.203 % ± 0.085 | +1.58 % (59.2 % of the gap closed) |
| Q4c-esc | Q4c + `escort.py --bpw 0.1` (list measured in Q4c: 23 v, 18 k, 4 q, 3 o; +24.7 MB) | 19.4755 | −0.732 % ± 0.050 | +3.10 % |
| **G-Q4c-esc** | Q4c-esc + genesis (q4tp-v2 chain, cortiq only) | **19.1702** | −2.288 % ± 0.095 vs Q4c; −0.087 % ± 0.042 vs G-Q4c | +1.49 % (61.5 % of the gap closed) |
| **G-W0cd** | W0cd + genesis (q2tp-v2 chain, cortiq only) | **22.4801** | −19.01 % ± 0.62 | +19.0 % (59.5 % of the gap closed) |

Setup: 5090 under a 2.0 GB cap, `--mb 2 --eval-mb 4`, 1024 train windows + held 65536:8+200000:16 of the
WikiText-2 train ids, teacher cache rebuilt from `f16.cmf` (train CE 2.99293 / held CE 3.09705, the studies'
teacher exactly), 2.8–3.0 s/step, GPU peak 0.93 GiB (q4tp) / 0.77 GiB (q2tp), ~15 min per refit.
- G-Q4c: best held-out CE 3.10377 at step 208 (step 0 3.12562), lo changed in 613 125 / 795 648 rows; written
  file sha256 0243f1da…, `cortiq ppl` CPU-exact window 0 1343.202567 vs torch 1343.202446.
- Paired against the study chains: G-Q4c vs G-C2 (the study's C2 = C1 + its H-metric scale refit, then genesis):
  +0.29 % ± 0.14, i.e. the unported H-metric refit is still worth ~0.3 % after genesis. G-W0cd vs R2t (W128dh:
  131k tokens + H-refit + code/scale alternation, then genesis): +2.73 % ± 0.31.
- Against the shipped q4tp recipe A1 (2048-token GPTQ, 20.226) the equal-size G-Q4c is −5.1 % (unpaired).

### How to make the quants

Everything below is CMF files written by cortiq; head and embeddings stay q8_2f (as shipped).

**q4tp-v2**
```sh
cortiq convert --model HF_DIR --quant f16   --output f16.cmf     # teacher, GPTQ input
cortiq convert --model HF_DIR --quant q8_2f --output q8_2f.cmf   # calibration forward, escort hi file
cortiq quantize-gptq f16.cmf --codec q4tp --calib-model q8_2f.cmf \
  --calib wiki.train.raw=0.5 --calib c4.jsonl=0.3 --calib code.txt=0.2 --tokens 32768 --window 2048 \
  --tensor-quant lm_head.weight=q8_2f --tensor-quant model.embed_tokens.weight=q8_2f \
  --hessians q4.hess --output q4tp-gptq.cmf
python tools/quant/escort.py --model q4tp-gptq.cmf --hi q8_2f.cmf --corpus calib.ids,held=0:8 \
  --bpw 0.1 --residency stream --write q4tp-esc.cmf --cortiq cortiq          # optional, +0.1 bpw
python tools/quant/metric_refit.py --model q4tp-esc.cmf --teacher f16.cmf --out run \
  --corpus wiki.ids,train=512 --corpus c4.ids,train=256 --corpus code.ids,train=256 \
  --step --epochs 1 --residency stream --mb 8 --head-windows 8 --write q4tp-v2.cmf --cortiq cortiq
python tools/quant/ppl.py --model q4tp-v2.cmf --ids test.ids --windows 160 --also 12 --out pw/v2 --ref pw/base
```
The teacher corpus must be broad and de-duplicated against every evaluation set: at 4 bits the genesis gain
is mostly in-domain (the q4tp study: −0.3 … −0.8 % on docs / code against −2.2 % on WikiText).

**q2tp-v2** (gate/up 2-bit, other linears q4tp)
```sh
cortiq convert --model HF_DIR --quant q2tp \
  --tensor-quant lm_head.weight=q8_2f --tensor-quant model.embed_tokens.weight=q8_2f --output q2tp.cmf  # no data
cortiq quantize-gptq f16.cmf --codec q2tp --calib-model q8_2f.cmf --calib ... --tokens 32768 --window 2048 \
  --tensor-quant lm_head.weight=q8_2f --tensor-quant model.embed_tokens.weight=q8_2f \
  --hessians q2.hess --output q2tp-gptq.cmf          # placed rule + debias are the defaults
python tools/quant/escort.py --model q2tp-gptq.cmf --hi q4tp-gptq.cmf --hi q8_2f.cmf --greedy 2 --bpw 0.1 ...
python tools/quant/metric_refit.py --model q2tp-gptq.cmf --teacher f16.cmf --step ... --write q2tp-v2.cmf
```
A 2-bit super-weight layer (MiniCPM5 layer 7) can be escorted inside GPTQ:
`--tensor-quant model.layers.7.mlp.gate_proj.weight=q4tp` (stays GPTQ, in q4tp).

**q1g** (1-bit models with a native 128-group, e.g. Bonsai)
```sh
cortiq requant bonsai-q1.cmf --quant q1g --allow-tol --output bonsai-q1g.cmf   # strict without --allow-tol
cortiq requant bonsai-q1g.cmf --quant q1 --output bonsai-q1.cmf                # exact inverse, old readers
```

### Qwen3.8-27B (`infosave/Qwen3.8-27B-cmf`)

Published baseline: q4tp 14.27 GB, WikiText-2 12 × 512 ppl 8.793 (plain `convert`, no GPTQ).

**Stand.** ≥ 128 GB host RAM by the cgroup limit (`/sys/fs/cgroup/memory.max` or
`memory/memory.limit_in_bytes`; `free` lies in containers and `/dev/shm` counts against it), one GPU with
≥ 32 GB, ≥ 300 GB local disk (f16 export 54.7 GB, q8_2f 27.4 GB, each quant 11–15 GB; the optional
`--hessians` cache stores upper triangles, by arithmetic ~1.8 GB per layer, ~115 GB for all 64). Run models
from local disk, not a network volume.

**Commands with this branch** (`--calib-chunk-layers 16`: ~3.6 GB of dense f64 Hessians per layer; the RAM
guard refuses a pass that cannot fit and names a chunk size that does):
```sh
cortiq convert --model Qwen/Qwen3.8-27B --quant f16   --output q27-f16.cmf
cortiq convert --model Qwen/Qwen3.8-27B --quant q8_2f --output q27-q8_2f.cmf
cortiq quantize-gptq q27-f16.cmf --codec q4tp --calib-model q27-q8_2f.cmf \
  --calib calib.mix.txt=0.8 --calib code.calib.txt=0.2 --tokens 65536 --window 2048 --lambda 0.01 \
  --hessians q27.hess --calib-chunk-layers 16 \
  --tensor-quant lm_head.weight=q8_2f --tensor-quant model.embed_tokens.weight=q8_2f --output q27-q4tp-v2.cmf
cortiq quantize-gptq q27-f16.cmf --codec q2tp ... --output q27-q2tp-v2.cmf     # same flags, q2tp profile
cortiq ppl q27-q4tp-v2.cmf --file wiki.test.raw --windows 12 --window-len 512   # published protocol
```
Check in the log that the runner captured all 400 linears of the hybrid (64 × 3 FFN + 48 × 3 GDN in_proj_qkv /
in_proj_z / out_proj + 16 × 4 attention; the 8 MTP-head linears never execute and stay converter RTN).

**What limits this branch at 27B, and where it is already solved.** Here the calibration forward is CPU-only
(the Hessian hook is in the CPU matmat) and the fold runs on the host. The main session's branch `q27-build`
(worktree `/Users/oleg/dev/cmf-q27`, journal `cmf-lb-local-runs/q27v2_pod_20261017/`) adds GPU calibration
(device Hessians, packed triangle, input aliasing, one pass for all 64 layers), the device fold and device
q4tp / q2tp sweeps, and a device NVG trainer; on a PRO 6000 it measured (wiki12): published q4tp 8.794 →
q4tp-v2 8.707, q4tp-v2e (attention k/v/o q8_2f) 8.690, q2tp-v3 9.574 (old RTN q2tp 14.33). `q27-build` merged
earlier states of the three pipeline branches; it lacks these verified commits, which merging `q4tp-nvg` into
it would bring:
- branch 1: 570c01b5 (RAM guards), a5e406d7 (test env guard), c799283b (row-blocked debias), b77b3130
  (`--q4tp-debias`), a099f3d7 (`--q2tp-debias on|off`, default on), e6580236, f41c7abf (act-order warning), docs;
- branch 2: 1217929a (batched head moves whole boundaries — the NaN fix), 046f105e (forward-only passes in
  place: ~16 GB less pinned memory at 27B), docs;
- branch 3: 7c95fb01 (Metal q1g `fma` — without it Metal q1g differs from q1 by rounding), 27a655e9,
  401cbd1e (q1g matvec weight prefetch), 440e731e (x86 group reduce), docs.

**Owner rule for 27B: Rust/CMF only.** Metric genesis exists only as `tools/quant/metric_refit.py` (Python);
for 27B it needs the Rust/device trainer (`q27-build` has a start). Re-measure MTP acceptance and tok/s of any
refit 27B file before release: the refit leaves `mtp.*` byte-identical but retunes the trunk it drafts from.

### Open issues

- 3 workspace test failures, none from the merge (see Tests); `wgpu_q8_mul_mm_matches_cpu_reference` and the
  decision_cli interrupt test fail on master on this pod and deserve their own look.
- Calibration is CPU-only in this branch (see above); the H-metric refit and code/scale alternation of the q2tp
  study are not ported (the remaining gap W0cd 27.76 → W128dh 25.62 before genesis).
- Mixed-corpus calibration and genesis were checked functionally only; every quality number here is WikiText.
- q1g Metal host path not run on a Mac; q1g has no arm in the GPU MoE expert path, Metal chunked prefill and
  the q1-only wgpu attention drop-in (those stay on the CPU / q1).
- The f16 `lo` quantum (0.27–0.54 % row gain) is the size of the trained gains: snapping costs part of them.
