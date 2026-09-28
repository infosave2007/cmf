# Embryo-O1 в рантайме: как запускать, что измерять, что не поддержано

Статус: 24.09.2026, по коду `crates/cortiq-engine` / `crates/cortiq-cli` (контракт оператора —
`docs/EMBRYO_BOUNDED_ANCHOR.md`). Файл Embryo-O1 — это `.cmf` с
`LayerType::BoundedAttention` + `arch.anchor_core = {kind:"swa_sink_v1", window, sink, …}`
и миксерами `linear_core` вида `vmf_phase` (Embryo-0, вариант A) либо
`gated_delta_net` (вариант B). Состояние потока — запись фиксированного
размера, выводимая из заголовка; ничего в пайплайне не растёт с диалогом.

## 1. Что показывает `cortiq info`

```
cortiq info bounded-500-f32.cmf
  Layers:      8 (0 full / 1 bounded / 7 linear)
  Anchor:      swa_sink_v1 window=128 sink=4 rope=relative_in_window sink_scores=nope train_windows=[64, 128] far=null
               ring per layer: 262144 B (2 kv_heads x 128 x 128 f32, K+V)
  State/seq:   2129408 B fixed (262144 B bounded rings + 1867264 B recurrent) — O(1) in context
```

`State/seq` = `per_sequence_state_bytes()` (`loader.rs`): кольца якорей
`2·kvh·W·hd·4` на bounded-слой + рекуррентная запись миксера
(`vmf`: `heads·2·nphase·dv + (k−1)·hidden`; `gated_delta_net`:
`GdnCfg::state_len() = (kk−1)·c_dim + nv·dk·dv`). Вариант B (6×GDN 4×128×128 +
2 якоря): 1 683 456 + 524 288 = **2 207 744 B**. `growing layers 0` — ни один слой
не хранит ничего на позицию. `cortiq info <f> --tensors <prefix>` печатает dtype
каждого тензора (нужно для аудита q4tp, §7).

## 2. CPU-путь (по умолчанию; единственный на macOS)

```
CMF_GPU=0 cortiq run bounded-500-f32.cmf --prompt "…"
CMF_GPU=0 cortiq ppl bounded-500-f32.cmf --file heldout-en.txt --windows 8 --window-len 512
CMF_GPU=0 cortiq ppl bounded-500-f32.cmf --file heldout-en.txt --tokens 16384
```

- Оператор якоря — `bounded.rs` (`BoundedState::attend`, кольцо сырых k̂/v, поворот
  запроса на Δ по таблице `BoundedRope`); батчевый prefill — `bounded_attention_batch`
  (проекции GEMM, оператор по позициям, скоры только по кольцу ∪ чанку).
- `ppl` идёт штатным оператором: `--o1`/`CMF_O1*` на bounded-файле — **ошибка**
  («anchor is native bounded»), не warn. Метка результата: `PPL = … (bounded operator
  swa_sink_v1 W=128 S=4)`.
- Пул потоков: `CMF_THREADS=N`; после исправления гонки дескриптора
  (`pool.rs`, seqlock + эпоха в дескрипторе) ограниченные рассылки малых матриц
  Embryo больше не зависают/не падают (регрессионные тесты `pool.rs`
  `uninvited_straggler_*`, `tests/legacy_pool_ppl.rs`).
- Диагностика: `CMF_PPL_TRACE=1` — NLL каждого токена (`BTRACE pos … nll …`);
  `CMF_STATE_TRACE=1` — RMS/max рекуррентной записи каждого слоя каждые 256 позиций
  (`STATE pos=… layer=… kind=… S_rms=…`); `CMF_PREFILL_PROF=1` — строка
  `kv-reuse: R of N prompt positions already cached` на каждом ходе с переиспользованием.

## 3. Резидентный wgpu-граф (Linux/Windows, дискретная карта)

```
export CMF_GPU=wgpu CMF_GPU_PROBE=0 CMF_GPU_WGPU_GRAPH=1 CMF_EMBRYO_RESIDENT=parallel
cortiq bench bounded-500-f32.cmf --tokens 128 --ctx 32768 --core --json
```

- Весь токен — один submit (`wgpu_submits_per_token = 1.0`): веса, рекуррентное
  состояние, кольца якорей и иерархическая голова живут на устройстве
  (`gpu_wgpu.rs` `EMBRYO_CORE_SRC`, слои kind 0/1 = vmf_phase, 3 = bounded, 4 = GDN;
  legacy full-anchor kind 2 — только для старых файлов).
- Prefill — чанками по `CMF_EMBRYO_CHUNK` позиций (по умолчанию 64 = `EMBRYO_CHUNK_MAX`):
  проекции/FFN/роутер — чанковые GEMM с порядком редукции токен-пути, рекуррентные
  слои и якорь идут по чанку во времени на устройстве; логиты — только последней
  позиции. `CMF_EMBRYO_CHUNK=0` — потокенный prefill (референс). Чанковый и потокенный
  дают **одинаковые биты** (`tests/embryo_resident_parity.rs`, `CMF_EMBRYO_CHUNK_TOL`).
- Граф **f32-only**: q4tp/q8-файл → `eligible=false` → per-op путь (медленнее CPU, §7).
- Телеметрия: `bench --json` → `device_state_bytes`, `device_kv_bytes` (читаются из
  живых буферов), `wgpu_submits_per_token`; `CMF_EMBRYO_DBG=1 RUST_LOG=warn` печатает
  причину отказа от графа; `CMF_EMBRYO_PROFILE=1` — тайминги стадий (рвёт слитый pass,
  не для замеров).
- Паритет с CPU: `embryo_resident_parity` (двухпроцессный: `CMF_EMBRYO_PARITY_MODE=cpu`
  пишет референс, `=parallel` сравнивает; `CMF_EMBRYO_PARITY_IDS=synth:512|natural`,
  `CMF_EMBRYO_PARITY_TOKENIZER` + `_PROMPT_FILE` для натурального префикса).

## 4. Команды-зонды

| команда | что меряет |
|---|---|
| `cortiq bench <f> --tokens 128 --ctx {64,512,4096,32768} --core --json` | декод tok/s (медиана из 3), prefill tok/s, байты состояния на устройстве |
| `cortiq ppl <f> --file heldout-en.txt --windows 8 --window-len 512` | протокол 8×512 (окна не пересекают 4k) |
| `cortiq ppl <f> --file … --tokens 16384` | один префикс: видно обрыв за пределами обучающей длины |
| `cortiq probe-recall <f…> --pairs 4 --distances 256,1024,4096,16384 --trials 32 [--json]` | MQAR/NIAH-lite через рантайм: acc@1/acc@5/any_value/mean_logp по дистанции |
| `cortiq probe-utility <f…> [--prompts f.tsv | --prompts-jsonl f.jsonl] --max-tokens 128 --rep-penalty 1.0 [--json]` | гейт S10: контракт cmf-im-v1 (`<\|im_start\|>user\n…<\|im_end\|>\n<\|im_start\|>assistant\n`, терминальный `\n` в префиксе, без BOS), exact/keyword, distinct-5-gram последних 128 токенов, период ≤16 / run ≥64 = loop, top-5 первого токена, TTFT, tok/s |

Форматы промптов: TSV `LABEL<TAB>EXPECTED[|ALT]<TAB>PROMPT` (11 замороженных промптов
аудита встроены по умолчанию) или JSONL `{"lang","prompt","expect":[…],"src"}`
(keyword = любая строка `expect` без учёта регистра в ответе).

## 5. Диалог и состояние (serve, wire v2, сплит)

- `cortiq serve <f> --port 8080` — OpenAI-совместимый `/v1/chat/completions`;
  `GET /v1/cortiq/status` отдаёт `slot_attention_kv_bytes` / `slot_recurrent_state_bytes`
  по слотам — у bounded-файла обе константы на любом ходе.
- Переиспользование префикса: `KvPrefix` (`pipeline.rs`) — длина + скользящий хеш
  всего съеденного + последние 128 id литерально; ход, строго продолжающий
  (промпт + сгенерированный ответ), перепрогоняет **только новые токены**
  (инвариант из телеметрии `CMF_PREFILL_PROF=1`: `R = prompt_prev + completion_prev − 1`
  — последний выданный токен модель никогда не прогоняет, он входит в новый prefill
  вместе с новым ходом); расхождение в истории — ровно один полный re-prefill, без
  строки `kv-reuse`. Измерено на `recovery-1500-f32.cmf` (CPU): 20 ходов, R = 28/55,
  60/87, …, 619/648; состояние 262 152 + 1 867 264 B на каждом ходе; расхождение на
  ходе 21 → полный prefill 671 токена; ход 22 → R = 676 из 697. Тест:
  `CMF_SERVE_TEST_MODEL=<bounded.cmf> cargo test -p cortiq-cli --release --test embryo_serve_multiturn -- --nocapture`
  (без env — механика на синтетическом геноме).
- Wire v2 (`kv_cache.rs`, `export_wire/import_wire`, магия `CMFS`): версия 2, хеш
  оператора, kind, позиция + запись фиксированной длины по kind (bounded: len/head +
  ring_k/ring_v, f32 или f16; linear: S + кольцо conv); `import(export(state))` → следующие
  64 логита Δ = 0 (`tests/bounded_runtime.rs`). Сетевой сплит (`cortiq-net`) шлёт тот же
  блоб; старый unversioned wire читается для старых kind.

## 6. Что НЕ поддержано

- **Metal-резидентный граф** — нет; на macOS Embryo-O1 идёт CPU-путём, Metal
  block/chunk-пути отказывают чисто (`tests/embryo_metal_refusal.rs`).
- **q4tp на графе** — граф f32-only; q4tp-файл = CPU-артефакт (§7).
- `ppl` и `prefill_span_ids` на резидентном графе — потокенно (чанковый prefill выдаёт
  только последние логиты); legacy full-anchor (kind 2) — потокенный prefill.
- MTP/спекуляция, task-маски, `--o1` — не сочетаются с резидентным графом (отказ до
  первого токена).

## 7. q4tp

```
cortiq requant bounded-500-f32.cmf --quant q4tp-quantize --output bounded-500-q4tp.cmf
```
`in_proj_a/b`, `sink_k/sink_v`, `landmarks_*` остаются f32 **по имени**
(`requant.rs` `embryo_keep_f32`), плюс исторические исключения (embed, lm_head, desc,
k_gate, 1-D). Измерено (S4-экспорт, CPU): ppl 8×512 f32 55.010 → q4tp 56.893 (×1.034),
декод 167 → 201 tok/s; файл 226 → 76 МБ.

## 8. Измеренные числа (RTX PRO 4000 Blackwell, Vulkan; 24.09.2026)

Синтетические геномы 56M (`tests/common/embryo_synth.rs`): `embryo0_bounded`
(7×vmf_phase + 1 bounded) и `embryo3_gdn_bounded` (6×GDN + 2 bounded). Медианы из 3,
`bench --tokens 128 --core --json`.

| геном | ctx 64 | 512 | 4096 | 32768 | max/min | state (B) | prefill 4096 chunk / per-pos |
|---|---|---|---|---|---|---|---|
| GDN+bounded 56M | 632.4 | 627.3 | 626.2 | 649.7 | 1.038 | 1 683 456 + 524 288 | 4418 / 657 tok/s (**6.7×**) |
| vmf+bounded 56M | 610.6 | — | — | 615.0 | — | 1 867 264 + 262 144 | 3792 / 628 tok/s (**6.0×**) |

Паритет GPU↔CPU: 2.86e-5 (vmf), 3.81e-5 (GDN), 3.81e-5 (реальный S4-экспорт, натуральный
префикс 512 + 32 сгенерированных токена бит-в-бит); reset Δ = 0; chunked vs потокенный
Δ = 0. Реальные экспорты (`recovery-1500-f32.cmf`, `fam-a-f32.cmf`) — та же геометрия,
что vmf+bounded 56M; их матрица bench снимается только на свободной карте
(`s7rt/bench2/run.log`) и дописывается сюда по готовности.

CPU (28 ядер, 27 воркеров): f32 ≈ 167 tok/s, q4tp ≈ 201 tok/s; ppl 8×512 бит-в-бит
с GPU (55.010).

## 9. Обрыв за обучающей длиной — это модель, не рантайм

На экспортах без переноса состояния (`control-500`, `bounded-500`) NLL рушится на
~3600–4100 позициях от старта **любого** окна (`CMF_STATE_TRACE`: RMS рекуррентной
записи vmf-слоёв растёт монотонно и взрывается в 50–100× между 4096 и 4608);
`carry-r16` выходит на плато (ppl@8192 = 81.1). Рантайм тут ни при чём:
`tests/prefix_chunk_agreement.rs` — 6000 токенов, чанки 1024/2048/4096 и потокенно
Δ = 0. Это гейт S6b/S8 тренера.
