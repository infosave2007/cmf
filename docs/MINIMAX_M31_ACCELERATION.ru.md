# MiniMax-M3.1-Flash — ускорение на одном GPU (RTX PRO 6000, 97 ГБ)

## Актуальный статус (после аудита корректности)

**Корректно:** пул экспертов внутри токен-графа (`CMF_MINIMAX_DYNAMIC_TAIL=1`).
Вывод совпадает с эталоном: «The capital of France is Paris.», косинус hidden
0.9987 на шаге 1. Устройственный аттеншн хвоста совпадает с хостом (≈1%).

**Скорость (честная):** 4–7.5 tok/s на корректном пути при 344 слотах банка
(диапазон из-за общего хоста; парные A/B-замеры, см. ниже). Прежняя цифра
«16.9 tok/s» **недействительна**: она получена на вырожденном тексте
(«initi initi…»), где повторяющиеся маршруты давали почти полные попадания
 в пул (4 холодных эксперта на токен против 75 на реальном тексте).

**Узкие места (измерено, `CMF_CPU_PROF`, bench JSON):**
- 75 холодных экспертов на токен → ~83 загрузки по PCIe (≈1.6 ГБ/токен);
- холодные вычисления: **87.3 мс/токен на устройстве** (было 131.9 на CPU —
  A/B `bench --tokens 120`: 7.51 vs 5.35 tok/s парными замерами);
- холодный TTFT: ~23 с на 41-токенный промпт — префилл проходит хвост по одному токену.

**Что сделано в этом шаге (0.8.33):**
- **Cold-эксперты на устройстве** (`minimax_cold_device`): адмишн в банк, затем
  один `prepared_frame` строго по холодным слотам; shared едет с весом 0.
  CPU-путь остаётся fallback'ом и эталоном. Корректность: cos 0.9987, текст идентичен.
- **Профиль-прогрев банка** (`CMF_M3_PROFILE_SAVE` / `CMF_M3_PROFILE` /
  `CMF_M3_PREWARM_N`) — **замерено: эффекта нет** (75.0 → 74.9 холодных/токен,
  скорость в пределах шума). Причина: банк 344 слота при 128×26 экспертах,
  админленные 300 вытесняются LRU. Фича осталась под env, выключена по умолчанию.
- Ветка `feat/nemotron-gemma4e` смержена в релиз; workspace **0.8.33**
  (0.8.32 уже на crates.io без этой ветки). Под 69.8.146.147 выключен;
  артефакты (`prof_story.bin`, bench JSON, digest'ы sidecar'ов) сохранены локально
  в `pod-artifacts-20261024/`, sidecar'ы 2×1.8 ГБ — в `hf-outbox/minimax/`
  (sha256: `65e5e321…4a1f` q2tp-mix, `0a587b61…2a68` q4tp).

**Что дальше (по убыванию отдачи, по данным Strata и замеров):**
- **Пакетный префилл** (Strata читает промпт 1000+ tok/s; у нас ~1.5 tok/s, TTFT ~23 с).
- **DSpark guess-then-check** (Strata: +1.6–1.8x): чекпоинт драфта уже сконвертирован
  (`*.dspark.cmf`), нужен исполнитель (5 блоков, верификация 8 строк).

**Ошибки, найденные и исправленные при аудите:**
1. `segment_slots` передавался количеством сегментов (8) вместо слотов в сегменте
   (43): shared-эксперт и часть routed читались как нули (вывод — мусор).
2. Знак sigmoid-скоринга лежал в битах `flags`, пересекающихся с полем shared-слота;
   перенесён в старший бит `n`.
3. Пакетные cold-кернели (`moe_gate_up_many`) считают только SiLU; OAI-эксперты
   идут по пути с собственной активацией (иначе холодные веса неверны).
4. Отсутствовал `ffn_pre` (h += attn; n1 = norm) в pooled-руке — исправлено.
5. Тесты, закрепляющие контракты: `minimax_device_route_matches_host_sigmoid_contract`,
   `minimax_device_cold_weights_match_host_contract`, `main_wgsl_source_validates`.

## 1. Измеренные корни тормозов (под 69.8.146.147, q2tp-mix, CMF_THREADS=24)

Профиль `CMF_CPU_PROF=1`, `bench --tokens 60 --core`, мс/токен:

| слот | пул ВЫКЛ | пул ВКЛ | что это |
|---|---:|---:|---|
| `ffn_gate_up` + `ffn_down` | 85.1 | **0** | CPU-MoE хвоста — ушло на GPU ✓ |
| `qkv` + `attn_o` (внутри `layers`) | 56.0 | 58.4 | **перезаливка** весов аттеншна хвоста каждый токен |
| `head` | 20.7 | 21.0 | **перезаливка** lm_head (1.26 ГБ) каждый токен |
| `layers(total)` | 150.8 | 141.7 | включая ~83 мс кадра пула |
| allocs/токен | 9 102 | 11 524 | — |
| decode | 5.6 | **5.1** | итог |

Счётчики кадра (добавлены в bench-JSON):

```
moe_frame_wait_ms_per_tok = 20.9   # 26 GPU-синков × 0.8 мс
moe_pool_fills_per_tok    = 66     # 66 новых адмиссий/токен!
moe_pool_fill_mb          = 77 934 / 60 токенов = 1.3 ГБ/токен по PCIe
dynamic-pool capacity=344 segment_slots=43 gu=q2tp
```

**Три корня:**

1. **VRAM голодает по кругу.** Префикс 34 слоя × 2.5 ГБ ≈ 86 ГБ baked-экспертов;
   арена резидентности на пределе, поэтому `qkv`/`attn_o` хвоста и `lm_head`
   вытесняются и **заливаются заново каждый токен** (79 мс ≈ PCIe). Пул в
   оставшихся ~5 ГБ (344 слота) голодает: 66 miss/токен → 1.3 ГБ/токен загрузок,
   которые съедают выигрыш от device-MoE.
2. **Синк на слой.** Подготовленный кадр читает hidden обратно: 26 × 0.8 мс = 21 мс
   (пол — DSV4 делает ~1.5 мс на слой при 60 слоях).
3. **Бюджетный баланс тупиковый**: `CMF_GRAPH_LIVE_MB=70000/55000/40000`
   (префикс 29/24/18) даёт 4.2–4.5 — выигрыш в резидентности аттеншна проигрывает
   росту хвоста. Точечной настройкой не лечится.

## 2. Физика цели 40 tok/s

Эксперт-payload на токен (60 слоёв × 5 экспертов × 19.6 МБ) ≈ 5.9 ГБ.
Всё resident → чтение из VRAM @1.8 ТБ/с = 3.3 мс → потолок ~300 tok/s.
То есть стена — **не полоса, а разрыв резидентности + синки**, и её лечит только
пул внутри графа: MoE и аттеншн одного слоя в ОДНОМ submit (1 синк/слой),
холодные эксперты — readback + CPU параллельно (контракт DSV4).

Расчётный результат этапа 3: 60 слоёв × ~1.0–1.3 мс + head в графе ≈ **60–80 мс/токен
→ 12–16 tok/s**, далее чанкинг (см. §5) → 25–40.

## 3. Что уже сделано (в этой ветке)

1. **Динамический пул для хвоста minimax** — `minicpm_tail.rs` расширен на второй
   флейвор (`CMF_MINIMAX_DYNAMIC_TAIL=1`):
   - `PoolKey{gu_q2, minimax}`, банк `QwenGpuPool::create_explicit(gu_q2)`
     (кодек `Q2tpGateUp` = q2tp gate/up + q4tp down — совпадает с q2tp-mix),
   - размер слотов из СВОБОДНОГО бюджета (`minimax_tail_slots`, `CMF_MINIMAX_TAIL_SLOTS`,
     отказ < 64 слотов),
   - `m3_triple`: act = SwiGLU-OAI(1.702, 7.0), dtype-проверки,
   - гейты архитектуры: top-4, sigmoid+norm_topk, `expert_bias` разрешён
     (роутинг host-side уже применил bias к выбору),
   - prepared-фрейм только для minimax (legacy-рука без act-кода запрещена).
2. **Шейдер `minicpm_tail_prepared_frame` генерализован**: `act`-параметр
   (`_p1` в gu-uniform), top_k ∈ {4,8}, `gu_q2` стриды (Q2TiledP), сверка кодека
   банка (+ отказ от mxfp4).
3. **Шейдер `DSV4_GLOBAL_MOE_SRC`**: ветка `_p1 == 3` = SwiGLU-OAI
   (`min(g,7)·sigmoid(1.702·g)·(clamp(u,±7)+1)`), `_p0` (bf16-fold) ортогонален.
4. **`moe_route` (роутер DSV4/цепочки)**: флаг `256` = sigmoid-скоринг minimax
   (`p=sigmoid(l)`, выбор по `p+bias`, норма `p·scale/(sum+1e-20)` — бит-в-бит
   с host `moe_route_with_eps`). Пока никем не включается — это фундамент
   пул-руки графа.
5. **Счётчики bench-JSON**: `moe_frame_{enc,wait,up}_ms_per_tok`,
   `moe_pool_fills{,_per_tok}`, `moe_pool_fill_mb`.
6. **Тест валидации главного шейдера** `main_wgsl_source_validates` (naga parse
   `WGSL`) — раньше опечатка в общем шейдере падала только на старте на устройстве.
7. **DSpark-драфтер возвращён в формат**: `cortiq convert --dspark-sidecar`
   теперь принимает ОБЕ базы (архитектура `--output` выбирает контракт):
   - `DsparkFlavor::{Minicpm,Minimax}`, reader `DsparkLayout::minimax` с полной
     проверкой опубликованного конфига (тексты под `text_config`, `dspark_block_size=7`,
     тапы `[3,17,31,45,59]`, mlp=12288 (DENSE ширина, не 3072!), swigluoai 1.702/7.0),
   - source: 97 тензоров = 62 логических + 35 MXFP8-скейлов; decode через
     `unpack_mxfp8_rowwise` → квантование в сторонний quant (q8_2f по умолчанию),
   - на поде созданы и прошли `model.verify()`:
     `minimax-m3.1-flash-q2tp-mix.dspark.cmf` и `minimax-m3.1-flash-q4tp.dspark.cmf`
     (62 тензора, 1.9 ГБ, arch `minimax_m31_dspark_sidecar`),
   - тест `minimax_dspark_contract_is_locked_to_the_published_draft`.

Статус DSpark = **упаковка, как у MiniCPM P0** (provenance `runtime.status:
conversion_only`): исполнителя (DFlash executor) в cortiq нет ни для одной
архитектуры кроме DSV4 — см. §6.

## 4. Ключевое решение: `GraphFfn::MoePooled` (этап 3, следующий)

Цель: **весь хвост на устройстве, один синк на слой**. Уже заложено:

- вариант `GraphFfn::MoePooled` в `gpu.rs` (router/sgate/n_exp/top_k/inter/act/
  gu_q2/sigmoid/bias/has_shared/route_scale/pool_uid/segment_slots);
- в `graph_layer_payload_bytes` пул-слой считается ТОЛЬКО по роутинговым весам
  (эксперты в банке) — это и есть разрешение бюджета;
- точки отказа (build/batch) помечены `single-layer arm pending` — поведение
  не изменилось, компиляция зелёная.

Доделать (в порядке шагов):

1. **Build**: в `forward_token_graph`, когда бюджет закончился (`prefix=true`),
   если пул-режим включён (`CMF_MINIMAX_POOL_GRAPH=1` + банк существует/создаётся)
   И это одиночный слой (`layers.is_empty()` до добавления) → собрать
   `LFfn::MoePooled` вместо обрыва.
2. **Encode**: новая рука в match `&lw.ffn` (существующий `LFfn::Moe` НЕ трогать):
   `emat(router, n1→mlogit)` → `c.moe_route` (bindings как в
   `encode_moe_chain_p`: bs/mk/fc/rp/rmb/coldb; rflags = bias|pin_shared(8)|
   subset(16)|sigmoid(256)|shared_slot<<8) → глобальные `dsv4_global_gu(_q2)` /
   `dsv4_global_dn` с descriptor-биндами (эталон — блок `if let Some(gl) =
   w.moe.global` в цепочке) и gu-uniform `[…, segment_slots, 0, act]`.
   Всё в ОДНОМ compute-pass.
3. **Синхронизация состояния пула** — thread-local `POOLED_LAYER{remap, shared_slot,
   pool_uid, segment_slots}` + `POOLED_COLD` (чтобы не раздувать сигнатуру
   `forward_token_graph` у 4 вызывающих). Хост перед вызовом:
   `ensure(layer, picks=прошлого токена)` → remap в thread-local; после вызова —
   drain cold.
4. **Cold-контракт**: хвост слоя для пул-руки выводится RAW (`axpy`: h+attn+об),
   readback складывает hidden + cold(id,weight) + вход FFN (n1); хост:
   `moe_cold_experts_cpu` → сумма в h → норма следующего слоя (контракт хост-цикла:
   `h += attn → norm(post_norm) → FFN → h += ffn_out` — сверить по полям
   GraphLayer `input_norm/post_norm`, включая Gemma-sandwich).
5. **Драйвер в `pipeline.rs`**: в хост-цикле хвоста вместо attn+MoE — по одному
   graph-вызову `[li, li+1)` (`layer_base=li`), на последнем слое запросить
   `lm_head` (не-prefix → фолд в графе → **head 21 мс уходит**), KV-зеркала слоёв
   хвоста переходят на устройство.
6. Проверка: `allocs<100`, `fills/token≈0` после прогрева, `decode≥12`.

## 5. Этап 4 (после измерений): чанкинг

Один submit на K слоёв + маленький флаг-чтение cold. Если флаг пуст — слои
корректны (K-слойный прогресс за один синк); если нет — пересчёт чанка
по-слойно. С прогревом пула cold≈0 → 4–5 синков на токен → 25–40 tok/s.
Сначала чанк=1 (корректно всегда), потом адаптивно.

## 6. DSpark-исполнитель (этап 5)

Sidecar готов; нужен исполнитель. Контракт M3.1 (из официального конфига):

- 5 блоков Qwen3-GQA, hidden **6144 = hidden цели** (тапы без проекции),
  mlp 12288, swigluoai, gemma-norm, q/k-norm, sliding 4096;
- тапы целевых слоёв `[3, 17, 31, 45, 59]`, `fc.weight [6144, 30720]`,
  `hidden_norm`, `final_norm`;
- Markov-голова `markov_w1/w2 [200064, 256]` + `confidence_head.proj [1, 6400]`;
- `block_size 7` → verify-окно 8 строк (anchor + 7), `mask_token_id 200058`.

Что сделать: захват тапов на устройстве (слои 3/17/31 внутри префикса — нужен
выход промежуточного hidden из графа), драфт-проход (5 блоков × 7 шагов — на
GPU, банк sidecar-тензоров ~500 МБ resident), верификация 8 строк через
`try_batch_graph_wgpu(spec)` + батчевый хвост. Дать ×2 к декоду ТОЛЬКО после
того, как этап 3 опустил стоимость токена (сейчас 8-строчная верификация при
196 мс/токен всё убивает).

## 7. Полезные команды

```bash
# бенч со счётчиками пула
CMF_THREADS=24 CMF_MINIMAX_DYNAMIC_TAIL=1 ./target/release/cortiq bench \
  /workspace/models/minimax-m3.1-flash-q2tp-mix.cmf --tokens 60 --core --json

# отладка отказов хвоста / кадра
CMF_M3_TAIL_DEBUG=1 CMF_DSV4_FRAME_DEBUG=1 CMF_QWEN_PROF=1 ...
# профиль CPU по слотам
CMF_CPU_PROF=1 ...
# размер пула и префикс
CMF_MINIMAX_TAIL_SLOTS=512 CMF_GRAPH_LIVE_MB=...

# DSpark sidecar
cortiq convert --model /workspace/mm-src/dspark \
  --output /workspace/models/minimax-m3.1-flash-q2tp-mix.cmf --dspark-sidecar
```