# Teacher-data v1: практическое использование

`cortiq teacher-data` создаёт проверяемый набор данных `.tdata` из ответов
локальной CMF-модели-учителя. Версия 0.8.38 реализует **text-only direct-answer
v1** и подготовку данных для поддерживаемой MTP-compaction: один горячий
pipeline обслуживает весь набор, а данные записываются в
стриминговый формат с индексом, манифестом и SHA-256 для каждого файла.

Это подготовка датасета, а не общий компрессор модели. Команды `compress` и
`illuminate ffn-compact` из исследовательского ТЗ пока не являются частью
этого релиза. Исключение — готовый путь для **MTP draft FFN**:
`cortiq mtp-compact SOURCE --data DATA --teacher TEACHER --prompts PROMPTS
--output OUT` сам создаёт отсутствующий корень `DATA.tdata` (selector вроде
`:train=0.5` допустим), а при наличии готового
каталога переиспользует его без перезаписи. Полный пример и границы этого
оркестратора приведены в [MTP_COMPACTION.ru.md](MTP_COMPACTION.ru.md). Для
самого частого случая есть короче: `cortiq compact SOURCE.cmf PROMPTS.jsonl
OUTPUT.cmf` использует source как teacher и создаёт/переиспользует
`OUTPUT.tdata`; он всё равно компактирует только существующий MTP draft, не
делает generic fine-tuning trunk.

## 1. Подготовьте JSONL

Каждая непустая строка — один JSON-объект. Нужен ровно один из ключей
`prompt` или `messages`:

```jsonl
{"id":"knapsack","category":"algorithms","prompt":"Write a dynamic-programming solution for the 0/1 knapsack problem in Python with type hints."}
{"id":"review","messages":[{"role":"system","content":"Answer concisely."},{"role":"user","content":"Explain stable sorting and give one example."}]}
```

- `id`, `category`, `samples`, `max_tokens` и `meta` необязательны;
- если `id` не задан, он детерминированно выводится из нормализованного
  содержимого запроса;
- пустые строки недопустимы; слишком короткие и дублирующие запросы не
  становятся обучающими примерами и отражаются в манифесте;
- текст запроса сохраняется без переписывания: это важно для повторяемой
  токенизации.

## 2. Сгенерируйте `.tdata`

Учитель должен быть локальным CMF с chat template. В v1 параметр
`--no-think` обязателен: так в записи хранится одна точная граница
answer-token, а не догадка о скрытой reasoning-части.

```bash
cortiq teacher-data generate \
  /models/teacher.cmf \
  calibration_prompts.jsonl \
  teacher-calibration.tdata \
  --no-think \
  --max-tokens 512 \
  --samples 1 \
  --seed 42 \
  --split 90/5/5
```

Генерация greedy и детерминированна для заданного seed. `--topk` пока должен
оставаться `0`: v1 намеренно отказывается писать придуманные распределения,
пока runtime не выдаёт сырые logits.

### Контроль без MTP

Если требуется воспроизводимый plain-greedy контроль либо draft-путь
неподходящ для длинного прогона, добавьте `--disable-mtp`:

```bash
cortiq teacher-data generate /models/teacher.cmf calibration_prompts.jsonl \
  teacher-plain.tdata --no-think --disable-mtp --max-tokens 512
```

Этот флаг отключает только speculative/MTP draft decoding. Trunk остаётся на
выбранном backend (включая Metal), поэтому это **не** CPU fallback. В
`manifest.json` записываются `generation.speculative_mtp` и
`generation.mtp_control`, чтобы два режима нельзя было перепутать.

## 3. Проверьте результат

`inspect` потоково проверяет каждую запись, индекс и диапазон токенов;
`--verify` добавляет отдельный проход проверки целостности:

```bash
cortiq teacher-data inspect teacher-calibration.tdata --verify
cortiq teacher-data inspect teacher-calibration.tdata --verify --json
```

Успешный каталог содержит `manifest.json`, `prompts.jsonl`, `records.jsonl`,
`tokens.bin` и `tokens.idx`. Не редактируйте их вручную: `--verify` должен
проходить до передачи набора в последующие эксперименты.

## Ограничения v1

- Только текстовый direct-answer путь; изображение, audio, top-k logits,
  resume/sharding и перенос `.tdata` в итоговое сжатие ещё не реализованы.
- Учитель без сохранённого chat template отклоняется, как и `--raw`.
- Скорость зависит от модели и backend; данный релиз не заявляет ускорение.

Подробное проектное ТЗ и экспериментальные результаты находятся в
[`TZ_TEACHER_DATA_COMPRESS.ru.md`](TZ_TEACHER_DATA_COMPRESS.ru.md).
