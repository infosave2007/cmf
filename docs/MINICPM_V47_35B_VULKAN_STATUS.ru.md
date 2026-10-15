# MiniCPM-V-4.7-35B-A3B q4tp: статус Vulkan-оптимизации

**Статус на 2026-10-15:** работы на удалённом GPU остановлены по явному
запросу пользователя. Это журнал измеренных результатов, а не модельная
карточка и не заявление о готовом релизе.

## Что реализовано в рабочем дереве

Для текстового `MiniCPM-V-4.7-35B-A3B-q4tp.cmf` добавлен opt-in точный
динамический MoE-tail для Vulkan:

- GPU исполняет device-prefix, а 13 не помещающихся tail-MoE-слоёв используют
  bounded GPU-bank выбранных экспертов; router, top-8, веса route и shared
  expert остаются исходными;
- cold routed experts считаются на CPU, resident experts — fused DSV4/WGPU
  frame, после чего результаты суммируются в исходном порядке;
- admission проверяется до исполнения; при ошибке, несовпадении карты слотов
  или сбое GPU путь отказывается целиком и используется прежний точный CPU
  MoE, без частичного результата;
- shared physical bank привязан к `(model UID, WGPU device)`, а логическая
  карта сбрасывается при overlay/eviction, чтобы не читать перезаписанный
  slot;
- добавлены два безопасных варианта, которые **сохраняются только как
  корректностные улучшения**, а не как заявленное ускорение: предварительное
  резервирование/перекрытие CPU cold-expert работы с H2D fill и floor
  резидентности по фактически достижимым 13 tail-слоям, а не по всем 40.

Ключевые файлы: `crates/cortiq-engine/src/minicpm_tail.rs`,
`crates/cortiq-engine/src/qwen4_exp.rs` и
`crates/cortiq-engine/src/pipeline.rs`.

## Проверенная конфигурация

На выделенной RTX 4090 / Vulkan использовался следующий точный режим:

```sh
CMF_GPU=1 CMF_GPU_PROBE=0 WGPU_BACKEND=vulkan \
CMF_GPU_WGPU_GRAPH=1 CMF_GRAPH_SPLIT=8 CMF_GRAPH_BGCACHE=0 \
CMF_THREADS=24 CMF_SDOT=0 \
CMF_MINICPM_DYNAMIC_TAIL=1 CMF_MINICPM_PREPARED_FRAME=1 \
CMF_MINICPM_TAIL_SLOTS=512 CMF_QWEN_FETCH_MAX=4 \
CMF_QWEN_FETCH_MIN_SEEN=1
```

`CMF_QWEN_ADMIT_THREADS=2` — допустимое явное значение; если переменная не
задана, MiniCPM-tail задаёт те же два admission workers. Это не рекомендация
для других моделей: для них сохраняется прежняя общая политика.

## Измеренные результаты

Все цифры ниже получены одним бинарником на prompt `The capital of France is`,
greedy decode, после успешной локальной release-сборки рабочей ветки. Публичный `v0.8.18` предшествует этому path и не должен использоваться для воспроизведения данных ниже.

| сценарий | результат |
|---|---:|
| CPU-tail / dynamic cache выключен, короткий контрольный прогон 16 token | 13.9 tok/s decode |
| Dynamic tail, 512 token, прогон 1 | 23.8 tok/s; first token 18.1 s |
| Dynamic tail, 512 token, прогон 2 | 24.3 tok/s; first token 17.9 s |
| Dynamic tail, 512 token, прогон 3 | 23.8 tok/s; first token 17.9 s |

Таким образом подтверждённый рабочий диапазон для этой машины — **23.8–24.3
 tok/s** (среднее около 24.0). Короткий 16-token CPU-tail контроль не является
сопоставимым baseline для этого 512-token замера; эти данные не дают права
заявлять универсальный процент ускорения или обещать такую скорость на другом
GPU, другой длине контекста или Mac.

### Корректность

Для нового path проведён независимый strict run против обычного path на 16
сгенерированных токенах:

- тело ответа совпало;
- `16/16` greedy `argmax` совпали;
- максимальная абсолютная разница logit: `4.00543212890625e-05`;
- implicit default admission policy и явный `CMF_QWEN_ADMIT_THREADS=2`
  совпали друг с другом (`argmax` и logits, max delta `0.0`).

Локально успешно выполнены `cargo check --locked -p cortiq-engine --features
gpu`, целевые unit tests для `qwen4_exp`/`minicpm_tail`, shader tests,
`cargo build --release --locked -p cortiq-cli` и `git diff --check`.

## Что было проверено и отвергнуто

Эти варианты не следует рекламировать как ускорения:

| вариант | наблюдение | решение |
|---|---|---|
| Prepared frame вместо legacy frame | 512-token A/B: 23.7 → 24.0 tok/s | различие в пределах шума |
| `CMF_GRAPH_BGCACHE=1` | strict parity есть, 512-token 23.8 против ~23.7 | нет устойчивого выигрыша |
| `CMF_QWEN_FETCH_MIN_SEEN=2` | 19.7 tok/s | регрессия |
| 4 admission workers | 23.8 tok/s против 24.3 при 1–2 | регрессия/нет выигрыша |
| отключение spin | 23.8 tok/s | регрессия/нет выигрыша |
| CPU cold-expert ↔ H2D overlap и tail-specific residency floor | strict parity есть, последние 512-token прогоны 23.8/24.3/23.8 | не доказан прирост сверх рабочего профиля |

## Почему потолок пока около 24 tok/s

Профиль на 128-token decode показал около **20.6 GPU readback/token** и
**16.5 ms ожидания readback/token**. У модели 13 CPU-tail слоёв; hidden state
нужен на CPU между последовательными слоями, поэтому простой GPU expert frame
добавляет fence/readback на каждом таком шаге. Перекрытие cold-expert CPU
работы с upload полезно для корректной латентности, но не убирает эту
последовательную границу.

Полный статический WGPU graph на 24-GiB карте намеренно не включался: q4tp
MoE stack имеет 17.5 GiB логических весов, но во время Vulkan graph setup
пакетные expert planes и staging дают существенно больший физический пик.
Существующая консервативная граница `9/16` VRAM дала стабильный 27-layer
prefix; снятие этого лимита ранее приводило к OOM. Поднимать VRAM budget ради
красивого бенчмарка было бы небезопасно и не было сделано после остановки.

Реальный следующий большой шаг — полноценная GPU-tail chain с сохранением
hidden/KV/GDN на устройстве и корректной обработкой cold experts, либо
runtime для отдельного DSpark sidecar. Это отдельная архитектурная работа,
а не безопасный env-тюнинг. Конвертер DSpark sidecar существует, но runtime
его **не запускает**, поэтому ему не приписывается ускорение.

## Чего не было сделано

- не опубликованы и не загружены новые веса на Hugging Face;
- публичный `v0.8.18` уже существовал до этого path; для runtime нужен отдельный новый релиз, а не переиспользование его тега;
- не заявлена скорость Metal как готовая: текущий Metal путь и его ограничения
  описаны отдельно в `docs/MINICPM_V47_35B_METAL_MOE.ru.md`;
- не сделана небезопасная попытка снять VRAM safety reserve или удержать на
  GPU полный q4tp stack.

## Остановка

Перед завершением проверено, что на pod отсутствуют процессы `cortiq`,
`cmf-safe-run.py` и `cargo build` этого задания. `nvidia-smi` показал
`1 MiB` used и `0%` GPU utilization. Все запущенные исследовательские
подзадачи также прерваны.
